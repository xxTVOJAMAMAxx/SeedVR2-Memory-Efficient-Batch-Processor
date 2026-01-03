import torch
import numpy as np
import os
import gc
import folder_paths
import subprocess
import threading
import tempfile
import json
import time
import queue
import cv2
from pathlib import Path
from datetime import datetime


class TrueBatchedVideoLoader:
    """
    Loads ONLY the current batch from video file into RAM.
    True memory efficiency - loads one batch at a time.
    
    NO BATCH SIZE LIMIT - Use any batch size your VRAM can handle!
    """
    
    def __init__(self):
        self.video_path_cache = None
        self.video_info_cache = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
                "batch_size": ("INT", {"default": 16, "min": 1, "max": 9999, "step": 1}),
                "overlap": ("INT", {"default": 4, "min": 0, "max": 9999, "step": 1}),
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "force_size": (["Disabled", "256x?", "?x256", "512x?", "?x512", "1080x?", "?x1080"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT", "BOOLEAN", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("batch_frames", "batch_index", "total_batches", "is_last", "fps", "start_frame_number", "video_source_path")
    FUNCTION = "load_batch"
    CATEGORY = "video/loaders"
    
    def load_batch(self, video_path, batch_size, overlap, batch_index, force_size):
        """Load only the specified batch from video"""
        
        # Find video file
        full_path = self.find_video(video_path)
        
        # Get video info (cached, but invalidate if path changed)
        if self.video_path_cache != full_path:
            cap = cv2.VideoCapture(full_path)
            self.video_info_cache = {
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            }
            cap.release()
            self.video_path_cache = full_path
            print(f"Video: {self.video_info_cache['total_frames']} frames, "
                  f"{self.video_info_cache['fps']:.2f} fps, "
                  f"{self.video_info_cache['width']}x{self.video_info_cache['height']}")
        
        info = self.video_info_cache
        total_frames = info['total_frames']
        stride = batch_size - overlap if overlap < batch_size else batch_size
        total_batches = (total_frames + stride - 1) // stride
        
        # Validate batch index
        if batch_index >= total_batches:
            raise ValueError(f"Batch {batch_index} exceeds total batches {total_batches}. "
                           f"Video has {total_frames} frames, calculated {total_batches} batches "
                           f"(batch_size={batch_size}, overlap={overlap}, stride={stride})")
        
        # Calculate frame range
        start_frame = batch_index * stride
        end_frame = min(start_frame + batch_size, total_frames)
        
        print(f"[LOADER] Batch {batch_index + 1}/{total_batches}: Loading frames {start_frame}-{end_frame-1} (count: {end_frame-start_frame})")
        
        # Load batch
        cap = cv2.VideoCapture(full_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Target size
        target_width, target_height = info['width'], info['height']
        if force_size != "Disabled":
            if "x?" in force_size:
                target_width = int(force_size.split('x')[0])
                target_height = int(info['height'] * target_width / info['width'])
            elif "?x" in force_size:
                target_height = int(force_size.split('x')[1])
                target_width = int(info['width'] * target_height / info['height'])
        
        frames = []
        for i in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if (target_width, target_height) != (info['width'], info['height']):
                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
            
            frames.append(frame)
        
        cap.release()
        
        frames_np = np.stack(frames).astype(np.float32) / 255.0
        frames_tensor = torch.from_numpy(frames_np)
        
        is_last = (batch_index >= total_batches - 1)
        ram_mb = frames_tensor.element_size() * frames_tensor.nelement() / (1024 * 1024)
        print(f"[OK] Loaded {len(frames)} frames (~{ram_mb:.1f} MB)")
        
        gc.collect()
        
        return (frames_tensor, batch_index, total_batches, is_last, info['fps'], start_frame, full_path)
    
    def find_video(self, video_path):
        """Find video file"""
        if os.path.exists(video_path):
            return video_path
        
        input_dir = folder_paths.get_input_directory()
        test_path = os.path.join(input_dir, video_path)
        if os.path.exists(test_path):
            return test_path
        
        for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            test_path = os.path.join(input_dir, video_path + ext)
            if os.path.exists(test_path):
                return test_path
        
        raise ValueError(f"Video not found: {video_path}")


# Update node mappings to include the loader
NODE_CLASS_MAPPINGS.update({
    "TrueBatchedVideoLoader": TrueBatchedVideoLoader,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "TrueBatchedVideoLoader": "True Batched Video Loader (Memory Efficient)",
})
    """Background thread for non-blocking video encoding"""
    
    def __init__(self):
        super().__init__(daemon=True)
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()
        
    def run(self):
        """Process encoding tasks in background"""
        while not self.stop_event.is_set():
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:  # Stop signal
                    break
                
                task_type, params = task
                
                if task_type == "encode":
                    result = self._encode_video(**params)
                    self.result_queue.put(result)
                
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Encoder thread error: {e}")
                self.result_queue.put({"success": False, "error": str(e)})
    
    def _encode_video(self, temp_dir, output_path, fps, codec, quality, container, frame_count, start_number):
        """Encode video using FFmpeg"""
        try:
            codec_settings = self._get_codec_settings(codec, quality)
            
            # FIXED: Use correct frame pattern - 6 digits not 8!
            input_pattern = os.path.join(temp_dir, "frame_%06d.png")
            
            cmd = [
                "ffmpeg",
                "-y",
                "-start_number", str(start_number),  # CRITICAL: Tell FFmpeg which frame to start from
                "-framerate", str(fps),
                "-i", input_pattern,
                "-c:v", codec_settings['codec'],
                "-crf", str(codec_settings['crf']),
                "-preset", codec_settings['preset'],
                "-pix_fmt", "yuv420p",
            ]
            
            if container == "MOV":
                cmd.extend(["-movflags", "+faststart"])
            
            cmd.append(output_path)
            
            print(f"     [ENCODE] [Background] Encoding {frame_count} frames...")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=300
            )
            
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"     [OK] [Background] Encoded: {file_size_mb:.1f} MB - {os.path.basename(output_path)}")
            
            return {
                "success": True,
                "path": output_path,
                "size_mb": file_size_mb
            }
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "FFmpeg timeout after 5 minutes"}
        except subprocess.CalledProcessError as e:
            return {"success": False, "error": f"FFmpeg error: {e.stderr}"}
        except FileNotFoundError:
            return {"success": False, "error": "FFmpeg not found in PATH"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _get_codec_settings(self, codec, quality):
        """Get FFmpeg codec settings"""
        codec_map = {
            "H.264": "libx264",
            "H.265": "libx265"
        }
        
        if codec == "H.264":
            quality_map = {
                "Fast": {"crf": 25, "preset": "veryfast"},
                "Balanced": {"crf": 23, "preset": "medium"},
                "High Quality": {"crf": 20, "preset": "slow"},
                "Very High Quality": {"crf": 18, "preset": "slower"},
            }
        else:
            quality_map = {
                "Fast": {"crf": 28, "preset": "fast"},
                "Balanced": {"crf": 26, "preset": "medium"},
                "High Quality": {"crf": 24, "preset": "slow"},
                "Very High Quality": {"crf": 22, "preset": "slower"},
            }
        
        return {
            'codec': codec_map[codec],
            'crf': quality_map[quality]['crf'],
            'preset': quality_map[quality]['preset']
        }
    
    def stop(self):
        """Stop the encoder thread"""
        self.stop_event.set()


class DirectVideoWriterNonBlocking:
    """
    Non-blocking direct video writer - upscaling continues while encoding happens in background!
    
    Key improvements:
    - Fixed frame numbering (6 digits not 8)
    - Background encoding thread
    - Upscaling starts immediately after previous batch completes
    - Progress monitoring without blocking
    """
    
    # Shared resources
    _encoder_thread = None
    _encoder_lock = threading.Lock()
    _video_sessions = {}
    _session_lock = threading.Lock()
    
    def __init__(self):
        # Start encoder thread if needed
        with DirectVideoWriterNonBlocking._encoder_lock:
            if DirectVideoWriterNonBlocking._encoder_thread is None or \
               not DirectVideoWriterNonBlocking._encoder_thread.is_alive():
                DirectVideoWriterNonBlocking._encoder_thread = VideoEncoderThread()
                DirectVideoWriterNonBlocking._encoder_thread.start()
                print("[VIDEO] Background video encoder started")
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "processed_batch": ("IMAGE",),
                "batch_index": ("INT",),
                "total_batches": ("INT",),
                "start_frame_number": ("INT",),
                "overlap": ("INT", {"default": 3, "min": 0, "max": 9999}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                
                "video_source_path": ("STRING", {"default": "", "tooltip": "Original video path for naming"}),
                "codec": (["H.264", "H.265"],),
                "quality": (["Fast", "Balanced", "High Quality", "Very High Quality"],),
                "container": (["MP4", "MOV"],),
                
                "progressive_updates": ("BOOLEAN", {"default": True}),
                "update_every_n_batches": ("INT", {"default": 1, "min": 1, "max": 50}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("video_path", "status", "frames_written", "is_complete", "output_folder")
    FUNCTION = "write_video"
    CATEGORY = "video/export"
    OUTPUT_NODE = True
    
    def write_video(self, processed_batch, batch_index, total_batches, start_frame_number,
                   overlap, fps, video_source_path, codec, quality, container,
                   progressive_updates, update_every_n_batches):
        """NON-BLOCKING video writing with parallel upscaling"""
        
        # Get batch size from processed_batch
        batch_size = processed_batch.shape[0] + overlap if batch_index > 0 else processed_batch.shape[0]
        
        # Generate smart folder and filename
        if video_source_path and os.path.exists(video_source_path):
            video_name = os.path.splitext(os.path.basename(video_source_path))[0]
        else:
            video_name = "video"
        
        # Clean video name
        video_name = "".join(c for c in video_name if c.isalnum() or c in (' ', '-', '_'))
        video_name = video_name.strip().replace(' ', '_')
        if len(video_name) > 50:
            video_name = video_name[:50]
        
        # Create base name: videoname_b{batch_size}_o{overlap}
        base_name = f"{video_name}_b{batch_size}_o{overlap}"
        
        # Find next available folder number
        output_base_dir = folder_paths.get_output_directory()
        counter = 1
        folder_name = base_name
        
        while os.path.exists(os.path.join(output_base_dir, folder_name)) and counter <= 999:
            folder_name = f"{base_name}_{counter:03d}"
            counter += 1
        
        session_id = folder_name
        
        # Initialize session
        with DirectVideoWriterNonBlocking._session_lock:
            if session_id not in DirectVideoWriterNonBlocking._video_sessions:
                output_dir = os.path.join(output_base_dir, folder_name)
                os.makedirs(output_dir, exist_ok=True)
                
                temp_dir = tempfile.mkdtemp(prefix='seedvr2_video_')
                
                log_path = os.path.join(output_dir, "processing_log.txt")
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write("="*70 + "\n")
                    f.write("SeedVR2 Non-Blocking Video Export\n")
                    f.write("="*70 + "\n\n")
                    f.write(f"Session: {session_id}\n")
                    f.write(f"Output Folder: {output_dir}\n")
                    f.write(f"Source Video: {video_source_path if video_source_path else 'N/A'}\n")
                    f.write(f"Batch Size: {batch_size}\n")
                    f.write(f"Overlap: {overlap}\n")
                    f.write(f"FPS: {fps}\n")
                    f.write(f"Codec: {codec} ({quality})\n")
                    f.write(f"Container: {container}\n")
                    f.write(f"Total Batches: {total_batches}\n")
                    f.write(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                DirectVideoWriterNonBlocking._video_sessions[session_id] = {
                    'output_dir': output_dir,
                    'temp_dir': temp_dir,
                    'log_path': log_path,
                    'base_name': base_name,
                    'fps': fps,
                    'codec': codec,
                    'quality': quality,
                    'container': container,
                    'last_saved_frame': -1,
                    'total_frames_written': 0,
                    'last_update_batch': -1,
                    'start_time': time.time(),
                    'batch_times': [],
                    'encoding_jobs': [],
                    'first_frame_number': None,  # Track first frame for FFmpeg
                }
                print(f"\n[VIDEO] Starting session: {session_id}")
                print(f"   Output folder: {output_dir}")
                print(f"   Temp: {temp_dir}")
            
            session = DirectVideoWriterNonBlocking._video_sessions[session_id]
        
        # Check results from background encoding (non-blocking)
        self._check_encoding_results(session)
        
        batch_start = time.time()
        
        # Process frames
        batch_frames_count = processed_batch.shape[0]
        
        if batch_index == 0:
            frames_to_save = processed_batch
            skip_frames = 0
            save_start_frame = start_frame_number
        else:
            skip_frames = min(overlap, batch_frames_count)
            frames_to_save = processed_batch[skip_frames:]
            save_start_frame = start_frame_number + skip_frames
        
        frames_to_save_count = frames_to_save.shape[0]
        
        print(f"\n{'='*70}")
        print(f"[VIDEO WRITER] Non-Blocking - Batch {batch_index + 1}/{total_batches}")
        print(f"{'='*70}")
        print(f"  Frames: {save_start_frame} to {save_start_frame + frames_to_save_count - 1} (count: {frames_to_save_count})")
        print(f"  Skipped overlap: {skip_frames}")
        
        # Save frames to temp directory - FIXED: Use 6-digit numbering
        frames_np = (frames_to_save.cpu().numpy() * 255).astype(np.uint8)
        temp_dir = session['temp_dir']
        
        from PIL import Image
        frames_written = 0
        
        # Track the first frame number for FFmpeg
        if session['first_frame_number'] is None:
            session['first_frame_number'] = save_start_frame
        
        for i, frame in enumerate(frames_np):
            frame_number = save_start_frame + i
            
            if frame_number <= session['last_saved_frame']:
                continue
            
            # FIXED: Save with 6-digit format to match FFmpeg pattern
            temp_frame_path = os.path.join(temp_dir, f"frame_{frame_number:06d}.png")
            img = Image.fromarray(frame)
            img.save(temp_frame_path, "PNG", compress_level=1)
            
            session['last_saved_frame'] = frame_number
            frames_written += 1
        
        session['total_frames_written'] += frames_written
        
        batch_time = time.time() - batch_start
        session['batch_times'].append(batch_time)
        
        print(f"  [OK] Buffered {frames_written} frames (Total: {session['total_frames_written']})")
        print(f"  [TIME] Batch time: {batch_time:.2f}s")
        
        # Log batch
        with open(session['log_path'], 'a', encoding='utf-8') as f:
            f.write(f"Batch {batch_index + 1}/{total_batches}\n")
            f.write(f"  Frames: {save_start_frame}-{save_start_frame+frames_to_save_count-1}\n")
            f.write(f"  Total: {session['total_frames_written']}\n")
            f.write(f"  Time: {batch_time:.2f}s\n\n")
        
        # Decide if we should trigger encoding
        is_complete = (batch_index >= total_batches - 1)
        should_encode = False
        
        if progressive_updates:
            batches_since_update = batch_index - session['last_update_batch']
            if batches_since_update >= update_every_n_batches or is_complete:
                should_encode = True
        elif is_complete:
            should_encode = True
        
        video_path = ""
        status = ""
        
        # Queue encoding job (NON-BLOCKING)
        if should_encode:
            ext = "mp4" if container == "MP4" else "mov"
            
            # Video filename: videoname_b{batch_size}_o{overlap}_progress.mp4 or _final.mp4
            if is_complete:
                video_name = f"{base_name}_final.{ext}"
            else:
                video_name = f"{base_name}_progress.{ext}"
            
            video_path = os.path.join(session['output_dir'], video_name)
            
            # Queue the encoding task
            encode_task = ("encode", {
                "temp_dir": temp_dir,
                "output_path": video_path,
                "fps": fps,
                "codec": codec,
                "quality": quality,
                "container": container,
                "frame_count": session['total_frames_written'],
                "start_number": session['first_frame_number']  # CRITICAL: Pass the first frame number
            })
            
            DirectVideoWriterNonBlocking._encoder_thread.task_queue.put(encode_task)
            session['last_update_batch'] = batch_index
            session['encoding_jobs'].append(video_path)
            
            print(f"  [ENCODE] Queued encoding: {video_name} ({session['total_frames_written']} frames)")
            
            if is_complete:
                status = f"[VIDEO] Final encoding queued (will complete in background)"
            else:
                status = f"[PROGRESS] Progress encoding queued"
        else:
            frames_until = update_every_n_batches - (batch_index - session['last_update_batch'])
            status = f"[BUFFER] Buffering... ({session['total_frames_written']} frames, updates in {frames_until} batches)"
        
        # Check queue status
        queue_size = DirectVideoWriterNonBlocking._encoder_thread.task_queue.qsize()
        if queue_size > 0:
            status += f" [Queue: {queue_size} jobs]"
        
        print(f"  Status: {status}")
        print(f"{'='*70}\n")
        
        # Cleanup if complete
        if is_complete:
            # Don't wait for encoding - let it finish in background
            print(f"[COMPLETE] Upscaling complete! Video encoding continues in background.")
            print(f"   Check: {session['output_dir']}")
            
            with open(session['log_path'], 'a', encoding='utf-8') as f:
                f.write("\n" + "="*70 + "\n")
                f.write("UPSCALING COMPLETE\n")
                f.write("="*70 + "\n")
                f.write(f"Total frames: {session['total_frames_written']}\n")
                f.write(f"Encoding jobs queued: {len(session['encoding_jobs'])}\n")
                f.write(f"Check output folder for final video\n")
        
        # Memory cleanup
        del frames_to_save, frames_np
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return (video_path, status, session['total_frames_written'], is_complete, session['output_dir'])
    
    def _check_encoding_results(self, session):
        """Check for completed encoding jobs (non-blocking)"""
        try:
            while True:
                result = DirectVideoWriterNonBlocking._encoder_thread.result_queue.get_nowait()
                
                if result.get("success"):
                    print(f"[OK] [Background] Video ready: {os.path.basename(result['path'])} ({result['size_mb']:.1f} MB)")
                    
                    with open(session['log_path'], 'a', encoding='utf-8') as f:
                        f.write(f"Video completed: {os.path.basename(result['path'])} ({result['size_mb']:.1f} MB)\n")
                else:
                    print(f"[ERROR] [Background] Encoding failed: {result.get('error', 'Unknown')}")
                    
                    with open(session['log_path'], 'a', encoding='utf-8') as f:
                        f.write(f"Encoding failed: {result.get('error', 'Unknown')}\n")
                        
        except queue.Empty:
            pass


# Node registration
NODE_CLASS_MAPPINGS = {
    "DirectVideoWriterNonBlocking": DirectVideoWriterNonBlocking,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DirectVideoWriterNonBlocking": "Direct Video Writer (Non-Blocking, Parallel)",
}