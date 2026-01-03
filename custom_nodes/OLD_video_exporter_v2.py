import os
import subprocess
import threading
import time
import json
import glob
import queue
from pathlib import Path


class VideoEncoderWorker(threading.Thread):
    """Background thread for video encoding"""
    
    def __init__(self, task_queue, result_queue):
        super().__init__(daemon=True)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.stop_event = threading.Event()
        
    def run(self):
        """Process encoding tasks in background"""
        while not self.stop_event.is_set():
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:  # Poison pill
                    break
                
                task_type, params = task
                
                if task_type == "progress":
                    result = self.encode_progress_video(**params)
                elif task_type == "final":
                    result = self.encode_final_videos(**params)
                
                self.result_queue.put(result)
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Encoder thread error: {e}")
                self.result_queue.put({"error": str(e)})
    
    def encode_progress_video(self, output_folder, frame_info, fps, codec, quality, container):
        """Encode progress video"""
        try:
            codec_name, codec_settings = get_codec_settings(codec, quality, is_progress=True)
            container_ext = "mp4" if "MP4" in container else "mov"
            output_name = f"video_progress_{codec_name.lower()}.{container_ext}"
            output_path = os.path.join(output_folder, output_name)
            input_pattern = os.path.join(output_folder, frame_info['pattern'])
            
            print(f"🎬 [Background] Encoding progress video: {frame_info['count']} frames")
            
            cmd = [
                "ffmpeg", "-y", "-framerate", str(fps),
                "-i", input_pattern,
                "-c:v", codec_settings['codec'],
                "-crf", str(codec_settings['crf']),
                "-preset", codec_settings['preset'],
                "-pix_fmt", "yuv420p",
                output_path
            ]
            
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"   ✅ Progress video updated: {output_name}")
            
            return {"success": True, "path": output_path, "type": "progress"}
            
        except Exception as e:
            print(f"   ❌ Progress encoding failed: {e}")
            return {"success": False, "error": str(e), "type": "progress"}
    
    def encode_final_videos(self, output_folder, frame_info, fps, codec_choice, quality, container_choice):
        """Encode final video(s)"""
        try:
            codecs = []
            if codec_choice == "Both H.264+H.265":
                codecs = ["H.264", "H.265"]
            else:
                codecs = [codec_choice]
            
            containers = []
            if container_choice == "Both MP4+MOV":
                containers = ["mp4", "mov"]
            elif "MP4" in container_choice:
                containers = ["mp4"]
            else:
                containers = ["mov"]
            
            output_paths = []
            input_pattern = os.path.join(output_folder, frame_info['pattern'])
            
            print(f"\n🎥 [Background] Creating {len(codecs) * len(containers)} final video(s)...")
            
            for codec in codecs:
                codec_name, codec_settings = get_codec_settings(codec, quality, is_progress=False)
                
                for container in containers:
                    output_name = f"video_final_{codec_name.lower()}.{container}"
                    output_path = os.path.join(output_folder, output_name)
                    
                    print(f"   🎥 {output_name} ({quality}, CRF {codec_settings['crf']})")
                    
                    cmd = [
                        "ffmpeg", "-y", "-framerate", str(fps),
                        "-i", input_pattern,
                        "-c:v", codec_settings['codec'],
                        "-crf", str(codec_settings['crf']),
                        "-preset", codec_settings['preset'],
                        "-pix_fmt", "yuv420p",
                        output_path
                    ]
                    
                    subprocess.run(cmd, capture_output=True, text=True, check=True)
                    file_size = os.path.getsize(output_path) / (1024 * 1024)
                    print(f"      ✅ {file_size:.1f} MB")
                    output_paths.append(output_path)
            
            print(f"   ✅ All final videos complete!")
            return {"success": True, "paths": output_paths, "type": "final"}
            
        except Exception as e:
            print(f"   ❌ Final encoding failed: {e}")
            return {"success": False, "error": str(e), "type": "final"}
    
    def stop(self):
        """Stop the worker thread"""
        self.stop_event.set()


def get_codec_settings(codec, quality, is_progress=False):
    """Get FFmpeg codec settings"""
    
    codec_map = {
        "H.264": "libx264",
        "H.265": "libx265"
    }
    
    if is_progress:
        quality_map = {
            "Fast": {"crf": 25, "preset": "veryfast"},
            "Medium": {"crf": 23, "preset": "medium"},
            "High": {"crf": 20, "preset": "fast"},
        }
    else:
        quality_map = {
            "High": {"crf": 18, "preset": "medium"},
            "Very High": {"crf": 15, "preset": "slow"},
            "Lossless": {"crf": 0, "preset": "medium"},
        }
    
    codec_name = codec
    settings = {
        'codec': codec_map[codec],
        'crf': quality_map[quality]['crf'],
        'preset': quality_map[quality]['preset']
    }
    
    return codec_name, settings


class BatchInfoReader:
    """Read batch configuration from TrueBatchedVideoLoader log or auto-detect"""
    
    @staticmethod
    def read_batch_loader_log(output_folder):
        """
        Read the log created by TrueBatchedVideoLoader
        Looks for batch_loader_log.json or similar diagnostic files
        """
        possible_log_names = [
            "batch_loader_log.json",
            "batch_info.json",
            ".batch_info.json",
            "loader_diagnostic.json",
            "batch_progress.json"
        ]
        
        # Check in output folder
        for log_name in possible_log_names:
            log_path = os.path.join(output_folder, log_name)
            if os.path.exists(log_path):
                try:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Extract batch info if present
                    if 'batch_size' in data or 'batch_info' in data:
                        if 'batch_info' in data:
                            data = data['batch_info']
                        
                        return {
                            'batch_size': data.get('batch_size'),
                            'overlap': data.get('overlap'),
                            'total_batches': data.get('total_batches'),
                            'current_batch': data.get('current_batch', 0),
                            'source': f'log:{log_name}'
                        }
                except:
                    continue
        
        # Check in segments folder (sometimes logs are stored there)
        segments_folder = os.path.join(output_folder, "segments")
        if os.path.exists(segments_folder):
            for log_name in possible_log_names:
                log_path = os.path.join(segments_folder, log_name)
                if os.path.exists(log_path):
                    try:
                        with open(log_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        if 'batch_size' in data or 'batch_info' in data:
                            if 'batch_info' in data:
                                data = data['batch_info']
                            
                            return {
                                'batch_size': data.get('batch_size'),
                                'overlap': data.get('overlap'),
                                'total_batches': data.get('total_batches'),
                                'current_batch': data.get('current_batch', 0),
                                'source': f'log:segments/{log_name}'
                            }
                    except:
                        continue
        
        return None
    
    @staticmethod
    def extract_from_folder_name(folder_name):
        """
        Extract batch info from smart naming format
        Example: my_video_seedvr2_ema_3b_b21_o3_001
        Returns: {'batch_size': 21, 'overlap': 3}
        """
        import re
        
        # Pattern: _b{batch_size}_o{overlap}_
        pattern = r'_b(\d+)_o(\d+)_'
        match = re.search(pattern, folder_name)
        
        if match:
            return {
                'batch_size': int(match.group(1)),
                'overlap': int(match.group(2)),
                'source': 'folder_name'
            }
        return None
    
    @staticmethod
    def read_workflow_file(output_folder):
        """
        Try to find and read workflow JSON in parent directories
        Looks for .json files that might contain workflow data
        """
        current_dir = Path(output_folder)
        
        # Search up to 3 levels up
        for _ in range(3):
            # Look for workflow files
            for json_file in current_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        workflow = json.load(f)
                    
                    # Check if it's a ComfyUI workflow
                    if isinstance(workflow, dict):
                        for node_id, node_data in workflow.items():
                            if isinstance(node_data, dict) and "class_type" in node_data:
                                if "TrueBatchedVideoLoader" in node_data["class_type"]:
                                    inputs = node_data.get("inputs", {})
                                    return {
                                        'batch_size': inputs.get('batch_size'),
                                        'overlap': inputs.get('overlap'),
                                        'source': f'workflow:{json_file.name}'
                                    }
                except:
                    continue
            
            current_dir = current_dir.parent
        
        return None
    
    @staticmethod
    def get_batch_info(output_folder):
        """
        Get batch info using multiple strategies (priority order):
        1. Read from TrueBatchedVideoLoader diagnostic log
        2. Extract from folder name
        3. Search for workflow JSON
        """
        
        # Strategy 1: Read diagnostic log from batch loader
        info = BatchInfoReader.read_batch_loader_log(output_folder)
        if info and info.get('batch_size'):
            print(f"📊 Batch info from {info.get('source', 'log')}: batch_size={info['batch_size']}, overlap={info['overlap']}")
            return info
        
        # Strategy 2: Extract from folder name
        folder_name = os.path.basename(output_folder)
        info = BatchInfoReader.extract_from_folder_name(folder_name)
        if info and info.get('batch_size'):
            print(f"📊 Batch info from {info.get('source', 'folder_name')}: batch_size={info['batch_size']}, overlap={info['overlap']}")
            return info
        
        # Strategy 3: Search for workflow file
        info = BatchInfoReader.read_workflow_file(output_folder)
        if info and info.get('batch_size'):
            print(f"📊 Batch info from {info.get('source', 'workflow')}: batch_size={info['batch_size']}, overlap={info['overlap']}")
            return info
        
        # Default fallback
        print(f"⚠️  Could not detect batch info, using defaults: batch_size=21, overlap=3")
        return {'batch_size': 21, 'overlap': 3, 'source': 'default'}


class VideoExporterFromFrames:
    """
    All-in-one video exporter for frame sequences.
    
    Features:
    - Real-time progress video (updates during processing)
    - High-quality final video (created at completion)
    - Multiple codec support (H.264, H.265)
    - Multiple container support (MP4, MOV)
    - Automatic frame detection (PNG, WebP, JPG)
    - MULTITHREADED - Non-blocking background encoding
    - AUTO UPDATE FREQUENCY - Calculates optimal update interval based on batch size
    """
    
    # Shared encoder worker pool
    _task_queue = queue.Queue()
    _result_queue = queue.Queue()
    _worker_thread = None
    _worker_lock = threading.Lock()
    
    def __init__(self):
        self.last_encoded_frame = -1
        self.batch_info = None
        
        # Start worker thread if not already running
        with VideoExporterFromFrames._worker_lock:
            if VideoExporterFromFrames._worker_thread is None or not VideoExporterFromFrames._worker_thread.is_alive():
                VideoExporterFromFrames._worker_thread = VideoEncoderWorker(
                    VideoExporterFromFrames._task_queue,
                    VideoExporterFromFrames._result_queue
                )
                VideoExporterFromFrames._worker_thread.start()
                print("🎬 Video encoder worker thread started")
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_folder": ("STRING", {"default": ""}),
                "fps": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                
                # Export options
                "export_mode": (["Both", "Progress Only", "Final Only", "Disabled"],),
                
                # Progress video settings
                "progress_codec": (["H.264", "H.265"],),
                "progress_quality": (["Fast", "Medium", "High"],),
                
                # Final video settings
                "final_codec": (["H.264", "H.265", "Both H.264+H.265"],),
                "final_quality": (["High", "Very High", "Lossless"],),
                
                # Container format
                "container": (["MP4", "MOV", "Both MP4+MOV"],),
                
                # Update settings
                "update_mode": (["Auto (Smart)", "Every Batch", "Every 2 Batches", "Every 3 Batches", "Manual"],),
                "manual_min_frames": ("INT", {"default": 18, "min": 1, "max": 1000, "step": 1, 
                                             "tooltip": "Only used when update_mode is Manual"}),
                
                # Advanced
                "delete_segments": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "is_complete": ("BOOLEAN", {"default": False}),
                "frames_saved": ("INT", {"default": 0}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("progress_video_path", "final_video_paths", "status", "encoding_active", "batch_info")
    FUNCTION = "export_videos"
    CATEGORY = "video/export"
    OUTPUT_NODE = True
    
    def calculate_min_frames(self, update_mode, manual_min_frames, batch_info):
        """
        Calculate optimal minimum frames for progress video updates
        
        Strategies:
        - Auto: batch_size - overlap (one effective batch)
        - Every Batch: batch_size - overlap
        - Every 2 Batches: 2 * (batch_size - overlap)
        - Every 3 Batches: 3 * (batch_size - overlap)
        - Manual: user-specified value
        """
        
        if update_mode == "Manual":
            return manual_min_frames
        
        batch_size = batch_info.get('batch_size', 21)
        overlap = batch_info.get('overlap', 3)
        
        # Effective frames per batch (accounting for overlap)
        effective_batch = batch_size - overlap
        
        if update_mode == "Auto (Smart)":
            # Smart mode: Update every effective batch
            # This means you see progress after each new batch is processed
            return effective_batch
        
        elif update_mode == "Every Batch":
            return effective_batch
        
        elif update_mode == "Every 2 Batches":
            return effective_batch * 2
        
        elif update_mode == "Every 3 Batches":
            return effective_batch * 3
        
        # Fallback
        return effective_batch
    
    def export_videos(self, output_folder, fps, export_mode, progress_codec, progress_quality,
                     final_codec, final_quality, container, update_mode, manual_min_frames,
                     delete_segments, is_complete=False, frames_saved=0):
        """Main export function - NON-BLOCKING with AUTO update frequency"""
        
        if export_mode == "Disabled":
            return ("", "", "Video export disabled", False, "")
        
        # Validate output folder
        if not output_folder or not os.path.exists(output_folder):
            return ("", "", f"Error: Folder not found: {output_folder}", False, "")
        
        # Get batch info (auto-detect or read from log)
        if self.batch_info is None:
            self.batch_info = BatchInfoReader.get_batch_info(output_folder)
            print(f"📊 Batch Info Detected: batch_size={self.batch_info['batch_size']}, overlap={self.batch_info['overlap']}")
        
        # Calculate optimal min_frames based on mode
        min_frames_for_update = self.calculate_min_frames(update_mode, manual_min_frames, self.batch_info)
        
        # Detect frame format and count
        frame_info = self.detect_frames(output_folder)
        if not frame_info:
            return ("", "", "No frames found in output folder", False, json.dumps(self.batch_info))
        
        # Check results from previous encoding tasks (non-blocking)
        self.check_encoding_results()
        
        # Determine if we should trigger encoding
        should_encode_progress = False
        should_encode_final = False
        
        if export_mode in ["Both", "Progress Only"] and not is_complete:
            # Update progress video when enough new frames are available
            frames_since_last = frame_info['count'] - self.last_encoded_frame
            if frames_since_last >= min_frames_for_update:
                should_encode_progress = True
                self.last_encoded_frame = frame_info['count']
                print(f"🎬 Triggering progress update: {frames_since_last} new frames (threshold: {min_frames_for_update})")
        
        if is_complete and export_mode in ["Both", "Final Only"]:
            should_encode_final = True
        
        # Queue encoding tasks (non-blocking)
        progress_path = ""
        final_paths = []
        encoding_active = False
        
        if should_encode_progress:
            print(f"\n🎬 Queueing progress video update ({frame_info['count']} frames)...")
            task = ("progress", {
                "output_folder": output_folder,
                "frame_info": frame_info,
                "fps": fps,
                "codec": progress_codec,
                "quality": progress_quality,
                "container": container
            })
            VideoExporterFromFrames._task_queue.put(task)
            encoding_active = True
            
            # Generate expected path
            codec_name = progress_codec
            container_ext = "mp4" if "MP4" in container else "mov"
            progress_path = os.path.join(output_folder, f"video_progress_{codec_name.lower()}.{container_ext}")
        
        if should_encode_final:
            print(f"\n🎥 Queueing final video creation...")
            task = ("final", {
                "output_folder": output_folder,
                "frame_info": frame_info,
                "fps": fps,
                "codec_choice": final_codec,
                "quality": final_quality,
                "container_choice": container
            })
            VideoExporterFromFrames._task_queue.put(task)
            encoding_active = True
            
            # Cleanup if requested
            if delete_segments:
                self.cleanup_segments(output_folder)
        
        # Build status message
        queue_size = VideoExporterFromFrames._task_queue.qsize()
        
        if is_complete:
            if queue_size > 0:
                status = f"🎥 Encoding final video(s)... ({queue_size} tasks in queue)"
            else:
                status = f"✅ Complete! Video export finished"
        else:
            if encoding_active:
                status = f"🔄 Encoding in background... ({frame_info['count']} frames, {queue_size} tasks queued)"
            else:
                frames_until_update = min_frames_for_update - (frame_info['count'] - self.last_encoded_frame)
                status = f"⏸️  Waiting for {frames_until_update} more frames... ({frame_info['count']} frames so far, update every {min_frames_for_update})"
        
        return (progress_path, "\n".join(final_paths), status, encoding_active, json.dumps(self.batch_info))
    
    def check_encoding_results(self):
        """Check for completed encoding tasks (non-blocking)"""
        try:
            while True:
                result = VideoExporterFromFrames._result_queue.get_nowait()
                
                if result.get("success"):
                    if result["type"] == "progress":
                        print(f"✅ Progress video ready: {os.path.basename(result['path'])}")
                    elif result["type"] == "final":
                        print(f"✅ Final videos ready: {len(result['paths'])} file(s)")
                else:
                    print(f"❌ Encoding error: {result.get('error', 'Unknown error')}")
                    
        except queue.Empty:
            pass  # No results yet, that's fine
    
    def detect_frames(self, output_folder):
        """Detect frame format and count frames"""
        
        # Try different patterns
        patterns = [
            ("frame_*.webp", "WebP"),
            ("frame_*.png", "PNG"),
            ("frame_*.jpg", "JPEG"),
            ("*.webp", "WebP"),
            ("*.png", "PNG"),
            ("*.jpg", "JPEG"),
        ]
        
        for pattern, format_name in patterns:
            frames = sorted(glob.glob(os.path.join(output_folder, pattern)))
            if frames:
                # Extract frame numbers to find pattern
                first_frame = os.path.basename(frames[0])
                
                # Determine pattern
                if first_frame.startswith("frame_"):
                    frame_pattern = f"frame_%06d.{frames[0].split('.')[-1]}"
                else:
                    frame_pattern = f"%06d.{frames[0].split('.')[-1]}"
                
                return {
                    'format': format_name,
                    'pattern': frame_pattern,
                    'extension': frames[0].split('.')[-1],
                    'count': len(frames),
                    'first_frame': frames[0],
                    'frames': frames
                }
        
        return None
    
    def cleanup_segments(self, output_folder):
        """Clean up temporary segment files"""
        segments_folder = os.path.join(output_folder, "segments")
        if os.path.exists(segments_folder):
            import shutil
            try:
                shutil.rmtree(segments_folder)
                print(f"🗑️  Cleaned up temporary segments\n")
            except Exception as e:
                print(f"⚠️  Could not clean up segments: {e}\n")


# Node registration
NODE_CLASS_MAPPINGS = {
    "VideoExporterFromFrames": VideoExporterFromFrames,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoExporterFromFrames": "Video Exporter from Frames (Auto-Update)",
}
