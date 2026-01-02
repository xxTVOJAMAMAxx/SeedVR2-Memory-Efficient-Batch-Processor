import torch
import numpy as np
from PIL import Image
import os
import gc
import folder_paths
import comfy.utils
import cv2

class TrueBatchedVideoLoader:
    """
    Loads ONLY the current batch from video file into RAM.
    True memory efficiency - loads one batch at a time.
    """
    
    def __init__(self):
        self.video_path_cache = None
        self.video_info_cache = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
                "batch_size": ("INT", {"default": 16, "min": 1, "max": 64, "step": 1}),
                "overlap": ("INT", {"default": 4, "min": 0, "max": 16, "step": 1}),
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "force_size": (["Disabled", "256x?", "?x256", "512x?", "?x512", "1080x?", "?x1080"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT", "BOOLEAN", "FLOAT", "INT")
    RETURN_NAMES = ("batch_frames", "batch_index", "total_batches", "is_last", "fps", "start_frame_number")
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
        
        print(f"🎬 LOADER: Batch {batch_index + 1}/{total_batches}: Loading frames {start_frame}-{end_frame-1} (count: {end_frame-start_frame})")
        
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
        print(f"✓ Loaded {len(frames)} frames (~{ram_mb:.1f} MB)")
        
        gc.collect()
        
        return (frames_tensor, batch_index, total_batches, is_last, info['fps'], start_frame)
    
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


class BatchFrameToDiskSaver:
    """
    Saves each processed batch directly to disk as image sequence.
    NO accumulation in RAM - truly memory efficient!
    Handles overlap by skipping overlapping frames from subsequent batches.
    Supports PNG and WebP formats with multi-threading.
    
    WITH DIAGNOSTIC LOGGING
    
    NOTE: ComfyUI creates new instances for each batch, so we use file-based state tracking.
    """
    
    def __init__(self):
        # These get reset each batch, so we track state in files
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "processed_batch": ("IMAGE",),
                "batch_index": ("INT",),
                "total_batches": ("INT",),
                "start_frame_number": ("INT",),
                "overlap": ("INT", {"default": 4, "min": 0, "max": 16}),
                "output_folder_name": ("STRING", {"default": "upscaled_output"}),
                "image_format": (["PNG", "WebP"], {"default": "WebP"}),
                "compression_quality": ("INT", {"default": 95, "min": 0, "max": 100, "step": 5}),
            }
        }
    
    RETURN_TYPES = ("STRING", "INT", "BOOLEAN")
    RETURN_NAMES = ("output_folder", "frames_saved", "is_complete")
    FUNCTION = "save_batch_to_disk"
    CATEGORY = "video/batch"
    OUTPUT_NODE = True
    
    def save_batch_to_disk(self, processed_batch, batch_index, total_batches, start_frame_number, 
                           overlap, output_folder_name, image_format, compression_quality):
        """Save batch directly to disk, skipping overlapping frames"""
        
        # Create output folder in ComfyUI output directory
        output_dir = folder_paths.get_output_directory()
        output_path = os.path.join(output_dir, output_folder_name)
        
        # State files for tracking across batch instances
        state_file = os.path.join(output_path, ".batch_state.txt")
        log_file = os.path.join(output_path, "frame_save_diagnostic.log")
        
        # Create folder and initialize state if first batch
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"\n📁 Created output folder: {output_path}")
            
            # Initialize state file
            with open(state_file, 'w') as f:
                f.write("-1\n0")  # last_saved_frame, total_frames_saved
            
            # Create diagnostic log file
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("=== FRAME SAVE DIAGNOSTIC LOG ===\n")
                f.write(f"Output folder: {output_path}\n")
                f.write(f"Format: {image_format}\n\n")
        
        # Read current state
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                lines = f.read().strip().split('\n')
                last_saved_frame = int(lines[0])
                total_frames_saved = int(lines[1]) if len(lines) > 1 else 0
        else:
            last_saved_frame = -1
            total_frames_saved = 0
        
        # Determine which frames to save from this batch
        batch_frames_count = processed_batch.shape[0]
        
        if batch_index == 0:
            # First batch: save all frames
            frames_to_save = processed_batch
            skip_frames = 0
            save_start_frame = start_frame_number
            save_end_frame = start_frame_number + batch_frames_count - 1
        else:
            # Subsequent batches: skip overlapping frames
            # The overlap frames are at the BEGINNING of this batch
            skip_frames = min(overlap, batch_frames_count)
            frames_to_save = processed_batch[skip_frames:]
            save_start_frame = start_frame_number + skip_frames
            save_end_frame = save_start_frame + frames_to_save.shape[0] - 1
        
        frames_to_save_count = frames_to_save.shape[0]
        
        # DIAGNOSTIC OUTPUT
        print(f"\n{'='*70}")
        print(f"💾 SAVER DIAGNOSTIC - Batch {batch_index + 1}/{total_batches}")
        print(f"{'='*70}")
        print(f"  Batch loaded frames:    {start_frame_number} to {start_frame_number + batch_frames_count - 1} (count: {batch_frames_count})")
        print(f"  Overlap to skip:        {skip_frames} frames")
        print(f"  Frames to save:         {save_start_frame} to {save_end_frame} (count: {frames_to_save_count})")
        print(f"  Last saved frame:       {last_saved_frame}")
        print(f"  Expected next frame:    {last_saved_frame + 1}")
        
        # Check for gaps or overlaps
        if last_saved_frame >= 0:
            expected_next = last_saved_frame + 1
            actual_next = save_start_frame
            if actual_next > expected_next:
                gap_size = actual_next - expected_next
                print(f"  ⚠️  WARNING: GAP DETECTED! Missing frames {expected_next} to {actual_next - 1} ({gap_size} frames)")
            elif actual_next < expected_next:
                overlap_size = expected_next - actual_next
                print(f"  ⚠️  WARNING: OVERLAP DETECTED! Duplicate frames {actual_next} to {expected_next - 1} ({overlap_size} frames)")
            else:
                print(f"  ✓ Frame sequence is continuous")
        
        print(f"{'='*70}\n")
        
        # Log to file (with UTF-8 encoding for emoji support)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n--- Batch {batch_index + 1}/{total_batches} ---\n")
            f.write(f"Loader provided: frames {start_frame_number}-{start_frame_number + batch_frames_count - 1} (count: {batch_frames_count})\n")
            f.write(f"Overlap skip: {skip_frames} frames\n")
            f.write(f"Saving: frames {save_start_frame}-{save_end_frame} (count: {frames_to_save_count})\n")
            f.write(f"Last saved: {last_saved_frame}\n")
            
            if last_saved_frame >= 0:
                expected_next = last_saved_frame + 1
                if save_start_frame > expected_next:
                    f.write(f"WARNING: GAP: Missing frames {expected_next}-{save_start_frame - 1}\n")
                elif save_start_frame < expected_next:
                    f.write(f"WARNING: OVERLAP: Duplicate frames {save_start_frame}-{expected_next - 1}\n")
                else:
                    f.write(f"OK: Continuous\n")
        
        # Convert tensors to numpy
        frames_np = (frames_to_save.cpu().numpy() * 255).astype(np.uint8)
        
        # Save frames with multi-threading
        if image_format == "WebP":
            saved_count = self.save_frames_webp_threaded(frames_np, output_path, save_start_frame, last_saved_frame)
        else:  # PNG
            saved_count = self.save_frames_png_threaded(frames_np, output_path, save_start_frame, last_saved_frame)
        
        # Update state
        new_last_saved_frame = save_start_frame + saved_count - 1
        total_frames_saved += saved_count
        
        with open(state_file, 'w') as f:
            f.write(f"{new_last_saved_frame}\n{total_frames_saved}")
        
        is_complete = (batch_index >= total_batches - 1)
        
        # Progress info
        print(f"💾 Batch {batch_index + 1}/{total_batches}: Saved {saved_count} frames as {image_format} "
              f"(Cumulative total: {total_frames_saved} frames)")
        
        if is_complete:
            print(f"\n{'='*60}")
            print(f"✓✓✓ ALL BATCHES SAVED!")
            print(f"Total frames saved: {total_frames_saved}")
            print(f"Format: {image_format}")
            print(f"Output folder: {output_path}")
            print(f"Diagnostic log: {log_file}")
            print(f"{'='*60}\n")
            
            # Final summary to log
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"FINAL SUMMARY\n")
                f.write(f"{'='*60}\n")
                f.write(f"Total frames saved: {total_frames_saved}\n")
                f.write(f"Total batches: {total_batches}\n")
                f.write(f"Last frame number: {new_last_saved_frame}\n")
                f.write(f"{'='*60}\n")
        
        # Clean up GPU memory
        del frames_to_save, frames_np
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return (output_path, saved_count, is_complete)
    
    def save_frames_webp_threaded(self, frames_np, output_path, start_frame_number, last_saved_frame):
        """Save frames as WebP using multi-threading"""
        from concurrent.futures import ThreadPoolExecutor
        import threading
        
        saved_count = 0
        lock = threading.Lock()
        
        def save_single_frame(args):
            nonlocal saved_count
            frame, frame_number = args
            
            # Skip if already saved
            if frame_number <= last_saved_frame:
                return
            
            # Create filename
            filename = f"frame_{frame_number:06d}.webp"
            filepath = os.path.join(output_path, filename)
            
            # Save as WebP (lossless if quality=100, near-lossless otherwise)
            img = Image.fromarray(frame)
            img.save(filepath, "WebP", 
                    quality=95,  # 95 = excellent quality, good compression
                    method=4,    # 4 = good balance of speed/compression
                    lossless=False)  # False = lossy but much faster
            
            with lock:
                saved_count += 1
        
        # Prepare frame list with numbers
        frame_args = [(frame, start_frame_number + i) for i, frame in enumerate(frames_np)]
        
        # Use ThreadPoolExecutor for parallel saving
        # CPU count * 2 is usually optimal for I/O bound tasks
        max_workers = min(os.cpu_count() * 2, 16)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(save_single_frame, frame_args)
        
        return saved_count
    
    def save_frames_png_threaded(self, frames_np, output_path, start_frame_number, last_saved_frame):
        """Save frames as PNG using multi-threading"""
        from concurrent.futures import ThreadPoolExecutor
        import threading
        
        saved_count = 0
        lock = threading.Lock()
        
        def save_single_frame(args):
            nonlocal saved_count
            frame, frame_number = args
            
            if frame_number <= last_saved_frame:
                return
            
            filename = f"frame_{frame_number:06d}.png"
            filepath = os.path.join(output_path, filename)
            
            # Save as PNG with compression level 9
            img = Image.fromarray(frame)
            img.save(filepath, "PNG", compress_level=9)
            
            with lock:
                saved_count += 1
        
        frame_args = [(frame, start_frame_number + i) for i, frame in enumerate(frames_np)]
        
        max_workers = min(os.cpu_count() * 2, 16)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(save_single_frame, frame_args)
        
        return saved_count


class PNGSequenceToVideo:
    """
    Converts saved PNG sequence back to video file.
    Use this after all batches are processed.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_folder": ("STRING", {"default": ""}),
                "output_filename": ("STRING", {"default": "upscaled_video.mp4"}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "codec": (["h264", "h265", "prores", "ffv1"], {"default": "h264"}),
                "quality": (["low", "medium", "high", "lossless"], {"default": "high"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "convert_to_video"
    CATEGORY = "video/conversion"
    OUTPUT_NODE = True
    
    def convert_to_video(self, input_folder, output_filename, fps, codec, quality):
        """Convert PNG sequence to video using ffmpeg"""
        
        try:
            import subprocess
        except ImportError:
            raise Exception("subprocess module required")
        
        # Find input folder
        if not os.path.isabs(input_folder):
            output_dir = folder_paths.get_output_directory()
            input_folder = os.path.join(output_dir, input_folder)
        
        if not os.path.exists(input_folder):
            raise ValueError(f"Input folder not found: {input_folder}")
        
        # Count frames
        png_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])
        if not png_files:
            raise ValueError(f"No PNG files found in {input_folder}")
        
        print(f"\n🎬 Converting {len(png_files)} frames to video...")
        print(f"FPS: {fps}, Codec: {codec}, Quality: {quality}")
        
        # Output path
        output_dir = folder_paths.get_output_directory()
        output_path = os.path.join(output_dir, output_filename)
        
        # Build ffmpeg command
        input_pattern = os.path.join(input_folder, "frame_%06d.png")
        
        # Codec settings
        codec_params = {
            "h264": ["-c:v", "libx264"],
            "h265": ["-c:v", "libx265"],
            "prores": ["-c:v", "prores_ks"],
            "ffv1": ["-c:v", "ffv1"]
        }
        
        # Quality settings
        quality_params = {
            "h264": {
                "low": ["-crf", "28"],
                "medium": ["-crf", "23"],
                "high": ["-crf", "18"],
                "lossless": ["-crf", "0"]
            },
            "h265": {
                "low": ["-crf", "32"],
                "medium": ["-crf", "28"],
                "high": ["-crf", "24"],
                "lossless": ["-crf", "0"]
            },
            "prores": {
                "low": ["-profile:v", "0"],
                "medium": ["-profile:v", "2"],
                "high": ["-profile:v", "3"],
                "lossless": ["-profile:v", "4"]
            },
            "ffv1": {
                "low": ["-level", "1"],
                "medium": ["-level", "3"],
                "high": ["-level", "3"],
                "lossless": ["-level", "3"]
            }
        }
        
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-framerate", str(fps),
            "-i", input_pattern,
            "-pix_fmt", "yuv420p"
        ]
        
        cmd.extend(codec_params[codec])
        cmd.extend(quality_params[codec][quality])
        cmd.append(output_path)
        
        # Run ffmpeg
        try:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✓ Video saved: {output_path}")
            return (output_path,)
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr}")
            raise Exception(f"FFmpeg conversion failed: {e}")
        except FileNotFoundError:
            raise Exception("FFmpeg not found. Please install FFmpeg and add it to PATH")


# Node registration
NODE_CLASS_MAPPINGS = {
    "TrueBatchedVideoLoader": TrueBatchedVideoLoader,
    "BatchFrameToDiskSaver": BatchFrameToDiskSaver,
    "PNGSequenceToVideo": PNGSequenceToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TrueBatchedVideoLoader": "True Batched Video Loader (Memory Efficient)",
    "BatchFrameToDiskSaver": "Batch Frame to Disk Saver (No RAM Accumulation) [DIAGNOSTIC]",
    "PNGSequenceToVideo": "PNG Sequence to Video Converter",
}
