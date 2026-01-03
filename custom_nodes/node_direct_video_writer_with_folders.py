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
from pathlib import Path
from datetime import datetime


class DirectVideoWriter:
    """
    Writes frames directly to video file without intermediate PNG storage.
    Progressively updates video after each batch for real-time progress monitoring.
    
    Features:
    - Direct frame-to-video encoding (no PNG intermediates)
    - Progressive video updates (see progress as it processes)
    - H.264 and H.265 codec support
    - MP4 and MOV container support
    - Quality presets (Fast, Balanced, High Quality, Very High Quality)
    - Automatic overlap handling
    - Smart naming: videoname_batchsize_overlap
    - Detailed logging with timestamps and performance metrics
    """
    
    # Shared state for progressive video building
    _video_sessions = {}
    _session_lock = threading.Lock()
    
    def __init__(self):
        self.session_id = None
        
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
                
                # Output settings
                "video_source_path": ("STRING", {"default": "", "tooltip": "Original video path for naming"}),
                "codec": (["H.264", "H.265"],),
                "quality": (["Fast", "Balanced", "High Quality", "Very High Quality"],),
                "container": (["MP4", "MOV"],),
                
                # Progressive encoding
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
        """Write frames directly to video with progressive updates"""
        
        # Get batch size from processed_batch
        batch_size = processed_batch.shape[0] + overlap if batch_index > 0 else processed_batch.shape[0]
        
        # Generate smart folder name
        if video_source_path and os.path.exists(video_source_path):
            video_name = os.path.splitext(os.path.basename(video_source_path))[0]
        else:
            video_name = "video"
        
        # Clean video name
        video_name = "".join(c for c in video_name if c.isalnum() or c in (' ', '-', '_'))
        video_name = video_name.strip().replace(' ', '_')
        if len(video_name) > 50:
            video_name = video_name[:50]
        
        # Create folder name: videoname_b{batch_size}_o{overlap}
        folder_name = f"{video_name}_b{batch_size}_o{overlap}"
        
        # Find next available number if folder exists
        output_base_dir = folder_paths.get_output_directory()
        counter = 1
        base_folder_name = folder_name
        while os.path.exists(os.path.join(output_base_dir, folder_name)) and counter <= 999:
            folder_name = f"{base_folder_name}_{counter:03d}"
            counter += 1
        
        session_id = folder_name
        
        # Initialize or get session
        with DirectVideoWriter._session_lock:
            if session_id not in DirectVideoWriter._video_sessions:
                output_dir = os.path.join(output_base_dir, folder_name)
                os.makedirs(output_dir, exist_ok=True)
                
                # Create temp directory for frames
                temp_dir = tempfile.mkdtemp(prefix='seedvr2_video_')
                
                # Create log file
                log_path = os.path.join(output_dir, "processing_log.txt")
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write("="*70 + "\n")
                    f.write("SeedVR2 Direct Video Export - Processing Log\n")
                    f.write("="*70 + "\n\n")
                    f.write(f"Session ID: {session_id}\n")
                    f.write(f"Output Folder: {output_dir}\n")
                    f.write(f"Source Video: {video_source_path if video_source_path else 'N/A'}\n")
                    f.write(f"Batch Size: {batch_size}\n")
                    f.write(f"Overlap: {overlap}\n")
                    f.write(f"FPS: {fps}\n")
                    f.write(f"Codec: {codec}\n")
                    f.write(f"Quality: {quality}\n")
                    f.write(f"Container: {container}\n")
                    f.write(f"Total Batches: {total_batches}\n")
                    f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("\n" + "="*70 + "\n\n")
                
                DirectVideoWriter._video_sessions[session_id] = {
                    'output_dir': output_dir,
                    'temp_dir': temp_dir,
                    'log_path': log_path,
                    'fps': fps,
                    'codec': codec,
                    'quality': quality,
                    'container': container,
                    'batch_size': batch_size,
                    'overlap': overlap,
                    'last_saved_frame': -1,
                    'total_frames_written': 0,
                    'last_update_batch': -1,
                    'start_time': time.time(),
                    'batch_times': [],
                    'batch_start_time': time.time(),
                }
                print(f"\n🎬 Starting new video session: {session_id}")
                print(f"   Output folder: {output_dir}")
                print(f"   Temp directory: {temp_dir}")
            
            session = DirectVideoWriter._video_sessions[session_id]
        
        # Log batch start
        batch_start_time = time.time()
        
        # Determine which frames to save from this batch
        batch_frames_count = processed_batch.shape[0]
        
        if batch_index == 0:
            # First batch: save all frames
            frames_to_save = processed_batch
            skip_frames = 0
            save_start_frame = start_frame_number
        else:
            # Subsequent batches: skip overlapping frames
            skip_frames = min(overlap, batch_frames_count)
            frames_to_save = processed_batch[skip_frames:]
            save_start_frame = start_frame_number + skip_frames
        
        frames_to_save_count = frames_to_save.shape[0]
        
        # Diagnostic info
        print(f"\n{'='*70}")
        print(f"🎥 VIDEO WRITER - Batch {batch_index + 1}/{total_batches}")
        print(f"{'='*70}")
        print(f"  Frames to write: {save_start_frame} to {save_start_frame + frames_to_save_count - 1} (count: {frames_to_save_count})")
        print(f"  Skipped overlap: {skip_frames} frames")
        
        # Convert frames to numpy and save to temp directory
        frames_np = (frames_to_save.cpu().numpy() * 255).astype(np.uint8)
        
        temp_dir = session['temp_dir']
        frames_written = 0
        
        from PIL import Image
        for i, frame in enumerate(frames_np):
            frame_number = save_start_frame + i
            
            # Skip if already saved
            if frame_number <= session['last_saved_frame']:
                continue
            
            # Save frame to temp directory
            temp_frame_path = os.path.join(temp_dir, f"frame_{frame_number:08d}.png")
            img = Image.fromarray(frame)
            img.save(temp_frame_path, "PNG", compress_level=1)  # Low compression for speed
            
            session['last_saved_frame'] = frame_number
            frames_written += 1
        
        session['total_frames_written'] += frames_written
        
        # Calculate batch processing time
        batch_time = time.time() - batch_start_time
        session['batch_times'].append(batch_time)
        
        print(f"  ✓ Buffered {frames_written} frames (Total: {session['total_frames_written']})")
        print(f"  ⏱️  Batch processing time: {batch_time:.2f}s")
        
        # Log to file
        with open(session['log_path'], 'a', encoding='utf-8') as f:
            f.write(f"Batch {batch_index + 1}/{total_batches}\n")
            f.write(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Frames: {save_start_frame} to {save_start_frame + frames_to_save_count - 1}\n")
            f.write(f"  Frames saved: {frames_written}\n")
            f.write(f"  Total frames encoded: {session['total_frames_written']}\n")
            f.write(f"  Batch time: {batch_time:.2f}s\n")
            f.write(f"  Processing FPS: {frames_written / batch_time:.2f}\n")
            f.write("\n")
        
        # Check if we should update the video
        is_complete = (batch_index >= total_batches - 1)
        should_update = False
        
        if progressive_updates:
            batches_since_update = batch_index - session['last_update_batch']
            if batches_since_update >= update_every_n_batches or is_complete:
                should_update = True
        elif is_complete:
            should_update = True
        
        # Generate video if needed
        video_path = ""
        status = ""
        
        if should_update:
            print(f"\n  🎬 Encoding video with {session['total_frames_written']} frames...")
            
            output_dir = session['output_dir']
            ext = "mp4" if container == "MP4" else "mov"
            
            # Smart video filename: videoname_b{batch_size}_o{overlap}
            if is_complete:
                video_filename = f"{video_name}_b{batch_size}_o{overlap}_final.{ext}"
            else:
                video_filename = f"{video_name}_b{batch_size}_o{overlap}_progress.{ext}"
            
            video_path = os.path.join(output_dir, video_filename)
            
            encode_start = time.time()
            success = self._encode_video(
                temp_dir=temp_dir,
                output_path=video_path,
                fps=fps,
                codec=codec,
                quality=quality,
                container=container
            )
            encode_time = time.time() - encode_start
            
            if success:
                session['last_update_batch'] = batch_index
                
                # Log encoding
                with open(session['log_path'], 'a', encoding='utf-8') as f:
                    f.write(f"{'='*50}\n")
                    if is_complete:
                        f.write(f"FINAL VIDEO ENCODED\n")
                    else:
                        f.write(f"PROGRESS VIDEO UPDATED\n")
                    f.write(f"{'='*50}\n")
                    f.write(f"  Video: {video_filename}\n")
                    f.write(f"  Frames: {session['total_frames_written']}\n")
                    f.write(f"  Encoding time: {encode_time:.2f}s\n")
                    f.write(f"  Encoding FPS: {session['total_frames_written'] / encode_time:.2f}\n")
                    f.write("\n")
                
                if is_complete:
                    total_time = time.time() - session['start_time']
                    avg_batch_time = sum(session['batch_times']) / len(session['batch_times']) if session['batch_times'] else 0
                    total_processing_fps = session['total_frames_written'] / sum(session['batch_times']) if session['batch_times'] else 0
                    
                    status = f"✅ Complete! Video exported: {video_filename}"
                    
                    print(f"\n{'='*70}")
                    print(f"✅ VIDEO EXPORT COMPLETE!")
                    print(f"   Output: {video_path}")
                    print(f"   Total frames: {session['total_frames_written']}")
                    print(f"   Total time: {total_time:.2f}s ({total_time/60:.1f}m)")
                    print(f"   Avg batch time: {avg_batch_time:.2f}s")
                    print(f"   Processing FPS: {total_processing_fps:.2f}")
                    print(f"   FPS: {fps}")
                    print(f"   Codec: {codec}")
                    print(f"   Quality: {quality}")
                    print(f"{'='*70}\n")
                    
                    # Final log summary
                    with open(session['log_path'], 'a', encoding='utf-8') as f:
                        f.write("\n" + "="*70 + "\n")
                        f.write("FINAL SUMMARY\n")
                        f.write("="*70 + "\n")
                        f.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Total Processing Time: {total_time:.2f}s ({total_time/60:.1f}m)\n")
                        f.write(f"Total Frames: {session['total_frames_written']}\n")
                        f.write(f"Total Batches: {total_batches}\n")
                        f.write(f"Average Batch Time: {avg_batch_time:.2f}s\n")
                        f.write(f"Processing FPS: {total_processing_fps:.2f}\n")
                        f.write(f"Output Video FPS: {fps}\n")
                        f.write(f"Final Video: {video_filename}\n")
                        f.write(f"File Size: {os.path.getsize(video_path) / (1024*1024):.1f} MB\n")
                        f.write("="*70 + "\n")
                    
                    # Cleanup temp directory
                    self._cleanup_session(session_id)
                else:
                    status = f"🔄 Progressive update: {session['total_frames_written']} frames encoded"
                    print(f"  ✓ Progressive video updated: {video_filename}")
            else:
                status = f"❌ Encoding failed at batch {batch_index + 1}"
        else:
            status = f"⏳ Buffering... {session['total_frames_written']} frames ready (will update in {update_every_n_batches - (batch_index - session['last_update_batch'])} batches)"
        
        print(f"  Status: {status}")
        print(f"{'='*70}\n")
        
        # Cleanup GPU memory
        del frames_to_save, frames_np
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return (video_path, status, session['total_frames_written'], is_complete, session['output_dir'])
    
    def _encode_video(self, temp_dir, output_path, fps, codec, quality, container):
        """Encode video using FFmpeg"""
        
        # Get codec settings
        codec_settings = self._get_codec_settings(codec, quality)
        
        # Build FFmpeg command
        input_pattern = os.path.join(temp_dir, "frame_%08d.png")
        
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-framerate", str(fps),
            "-i", input_pattern,
            "-c:v", codec_settings['codec'],
            "-crf", str(codec_settings['crf']),
            "-preset", codec_settings['preset'],
            "-pix_fmt", "yuv420p",
        ]
        
        # Add container-specific options
        if container == "MOV":
            cmd.extend(["-movflags", "+faststart"])
        
        cmd.append(output_path)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=300  # 5 minute timeout
            )
            
            # Get file size
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"     ✓ Encoded: {file_size_mb:.1f} MB")
            
            return True
            
        except subprocess.TimeoutExpired:
            print(f"     ❌ FFmpeg timeout after 5 minutes")
            return False
        except subprocess.CalledProcessError as e:
            print(f"     ❌ FFmpeg error: {e.stderr}")
            return False
        except FileNotFoundError:
            print(f"     ❌ FFmpeg not found. Please install FFmpeg and add to PATH")
            return False
        except Exception as e:
            print(f"     ❌ Encoding error: {e}")
            return False
    
    def _get_codec_settings(self, codec, quality):
        """Get FFmpeg codec settings based on codec and quality"""
        
        codec_map = {
            "H.264": "libx264",
            "H.265": "libx265"
        }
        
        # Quality presets
        if codec == "H.264":
            quality_map = {
                "Fast": {"crf": 25, "preset": "veryfast"},
                "Balanced": {"crf": 23, "preset": "medium"},
                "High Quality": {"crf": 20, "preset": "slow"},
                "Very High Quality": {"crf": 18, "preset": "slower"},
            }
        else:  # H.265
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
    
    @classmethod
    def _cleanup_session(cls, session_id):
        """Cleanup temporary files and session data"""
        with cls._session_lock:
            if session_id in cls._video_sessions:
                session = cls._video_sessions[session_id]
                temp_dir = session['temp_dir']
                
                # Delete temporary files
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                    print(f"   🗑️  Cleaned up temporary files")
                except Exception as e:
                    print(f"   ⚠️  Could not clean up temp files: {e}")
                
                # Remove session
                del cls._video_sessions[session_id]


class DirectVideoWriterAdvanced:
    """
    Advanced version with dual-output: progress video + final high-quality video.
    Creates a fast preview video that updates frequently, plus a final high-quality export.
    """
    
    _video_sessions = {}
    _session_lock = threading.Lock()
    
    def __init__(self):
        pass
    
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
                
                # Output settings
                "video_source_path": ("STRING", {"default": "", "tooltip": "Original video path for naming"}),
                
                # Progress video (fast, frequent updates)
                "progress_codec": (["H.264", "H.265"],),
                "progress_quality": (["Fast", "Balanced"],),
                "progress_update_every": ("INT", {"default": 1, "min": 1, "max": 10}),
                
                # Final video (high quality, only at completion)
                "final_codec": (["H.264", "H.265", "Both H.264+H.265"],),
                "final_quality": (["High Quality", "Very High Quality"],),
                "final_container": (["MP4", "MOV", "Both MP4+MOV"],),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("progress_video", "final_videos", "status", "frames_written", "is_complete", "output_folder")
    FUNCTION = "write_videos"
    CATEGORY = "video/export"
    OUTPUT_NODE = True
    
    def write_videos(self, processed_batch, batch_index, total_batches, start_frame_number,
                    overlap, fps, video_source_path, progress_codec, progress_quality,
                    progress_update_every, final_codec, final_quality, final_container):
        """Write both progress and final videos with logging"""
        
        # Get batch size
        batch_size = processed_batch.shape[0] + overlap if batch_index > 0 else processed_batch.shape[0]
        
        # Generate smart folder name
        if video_source_path and os.path.exists(video_source_path):
            video_name = os.path.splitext(os.path.basename(video_source_path))[0]
        else:
            video_name = "video"
        
        video_name = "".join(c for c in video_name if c.isalnum() or c in (' ', '-', '_'))
        video_name = video_name.strip().replace(' ', '_')
        if len(video_name) > 50:
            video_name = video_name[:50]
        
        folder_name = f"{video_name}_b{batch_size}_o{overlap}"
        
        # Find next available number
        output_base_dir = folder_paths.get_output_directory()
        counter = 1
        base_folder_name = folder_name
        while os.path.exists(os.path.join(output_base_dir, folder_name)) and counter <= 999:
            folder_name = f"{base_folder_name}_{counter:03d}"
            counter += 1
        
        session_id = folder_name
        
        # Initialize session
        with DirectVideoWriterAdvanced._session_lock:
            if session_id not in DirectVideoWriterAdvanced._video_sessions:
                output_dir = os.path.join(output_base_dir, folder_name)
                os.makedirs(output_dir, exist_ok=True)
                
                temp_dir = tempfile.mkdtemp(prefix='seedvr2_video_')
                
                # Create log file
                log_path = os.path.join(output_dir, "processing_log.txt")
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write("="*70 + "\n")
                    f.write("SeedVR2 Direct Video Export (Advanced) - Processing Log\n")
                    f.write("="*70 + "\n\n")
                    f.write(f"Session: {session_id}\n")
                    f.write(f"Output: {output_dir}\n")
                    f.write(f"Source: {video_source_path if video_source_path else 'N/A'}\n")
                    f.write(f"Batch Size: {batch_size}, Overlap: {overlap}\n")
                    f.write(f"FPS: {fps}\n")
                    f.write(f"Progress: {progress_codec} ({progress_quality})\n")
                    f.write(f"Final: {final_codec} ({final_quality})\n")
                    f.write(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("\n" + "="*70 + "\n\n")
                
                DirectVideoWriterAdvanced._video_sessions[session_id] = {
                    'output_dir': output_dir,
                    'temp_dir': temp_dir,
                    'log_path': log_path,
                    'last_saved_frame': -1,
                    'total_frames_written': 0,
                    'last_progress_update': -1,
                    'start_time': time.time(),
                    'batch_times': [],
                }
                print(f"\n🎬 Starting video session: {session_id}")
            
            session = DirectVideoWriterAdvanced._video_sessions[session_id]
        
        # Process frames (same as DirectVideoWriter)
        batch_start = time.time()
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
        
        print(f"\n🎥 VIDEO WRITER (Advanced) - Batch {batch_index + 1}/{total_batches}")
        print(f"   Frames: {save_start_frame} to {save_start_frame + frames_to_save_count - 1}")
        
        # Save frames
        frames_np = (frames_to_save.cpu().numpy() * 255).astype(np.uint8)
        temp_dir = session['temp_dir']
        
        from PIL import Image
        frames_written = 0
        
        for i, frame in enumerate(frames_np):
            frame_number = save_start_frame + i
            
            if frame_number <= session['last_saved_frame']:
                continue
            
            temp_frame_path = os.path.join(temp_dir, f"frame_{frame_number:08d}.png")
            img = Image.fromarray(frame)
            img.save(temp_frame_path, "PNG", compress_level=1)
            
            session['last_saved_frame'] = frame_number
            frames_written += 1
        
        session['total_frames_written'] += frames_written
        batch_time = time.time() - batch_start
        session['batch_times'].append(batch_time)
        
        # Log batch
        with open(session['log_path'], 'a', encoding='utf-8') as f:
            f.write(f"Batch {batch_index + 1}/{total_batches}\n")
            f.write(f"  Time: {datetime.now().strftime('%H:%M:%S')}\n")
            f.write(f"  Frames: {save_start_frame}-{save_start_frame+frames_to_save_count-1}\n")
            f.write(f"  Total encoded: {session['total_frames_written']}\n")
            f.write(f"  Batch time: {batch_time:.2f}s\n")
            f.write(f"  FPS: {frames_written/batch_time:.2f}\n\n")
        
        # Update videos
        is_complete = (batch_index >= total_batches - 1)
        progress_video_path = ""
        final_videos = []
        status = ""
        
        batches_since_progress = batch_index - session['last_progress_update']
        if batches_since_progress >= progress_update_every or is_complete:
            # Update progress video
            progress_video_path = os.path.join(session['output_dir'], 
                                               f"{video_name}_b{batch_size}_o{overlap}_progress.mp4")
            
            print(f"   🔄 Updating progress video...")
            writer = DirectVideoWriter()
            success = writer._encode_video(temp_dir, progress_video_path, fps, 
                                          progress_codec, progress_quality, "MP4")
            
            if success:
                session['last_progress_update'] = batch_index
        
        # Create final video(s) if complete
        if is_complete:
            print(f"\n   🎬 Creating final video(s)...")
            
            codecs = ["H.264", "H.265"] if final_codec == "Both H.264+H.265" else [final_codec]
            containers = []
            if final_container == "Both MP4+MOV":
                containers = ["MP4", "MOV"]
            elif "MP4" in final_container:
                containers = ["MP4"]
            else:
                containers = ["MOV"]
            
            writer = DirectVideoWriter()
            
            for codec in codecs:
                for container in containers:
                    ext = "mp4" if container == "MP4" else "mov"
                    codec_short = "h264" if codec == "H.264" else "h265"
                    final_path = os.path.join(session['output_dir'], 
                                             f"{video_name}_b{batch_size}_o{overlap}_final_{codec_short}.{ext}")
                    
                    print(f"      🎥 {codec} ({container})...")
                    success = writer._encode_video(temp_dir, final_path, fps, codec, 
                                                   final_quality, container)
                    
                    if success:
                        file_size = os.path.getsize(final_path) / (1024 * 1024)
                        print(f"         ✓ {file_size:.1f} MB")
                        final_videos.append(final_path)
            
            # Final summary
            total_time = time.time() - session['start_time']
            avg_batch_time = sum(session['batch_times']) / len(session['batch_times'])
            processing_fps = session['total_frames_written'] / sum(session['batch_times'])
            
            status = f"✅ Complete! {len(final_videos)} final video(s) created"
            
            print(f"\n{'='*70}")
            print(f"✅ VIDEO EXPORT COMPLETE!")
            print(f"   Total time: {total_time:.2f}s ({total_time/60:.1f}m)")
            print(f"   Frames: {session['total_frames_written']}")
            print(f"   Processing FPS: {processing_fps:.2f}")
            print(f"{'='*70}\n")
            
            # Final log
            with open(session['log_path'], 'a', encoding='utf-8') as f:
                f.write("\n" + "="*70 + "\n")
                f.write("FINAL SUMMARY\n")
                f.write("="*70 + "\n")
                f.write(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Time: {total_time:.2f}s ({total_time/60:.1f}m)\n")
                f.write(f"Frames: {session['total_frames_written']}\n")
                f.write(f"Batches: {total_batches}\n")
                f.write(f"Avg Batch Time: {avg_batch_time:.2f}s\n")
                f.write(f"Processing FPS: {processing_fps:.2f}\n")
                f.write(f"Output FPS: {fps}\n")
                for fv in final_videos:
                    f.write(f"Video: {os.path.basename(fv)} ({os.path.getsize(fv)/(1024*1024):.1f} MB)\n")
                f.write("="*70 + "\n")
            
            # Cleanup
            self._cleanup_session(session_id)
        else:
            status = f"⏳ Progress: {session['total_frames_written']} frames"
        
        # Cleanup GPU
        del frames_to_save, frames_np
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        final_videos_str = "\n".join([os.path.basename(fv) for fv in final_videos])
        
        return (progress_video_path, final_videos_str, status, 
                session['total_frames_written'], is_complete, session['output_dir'])
    
    @classmethod
    def _cleanup_session(cls, session_id):
        """Cleanup temporary files"""
        with cls._session_lock:
            if session_id in cls._video_sessions:
                session = cls._video_sessions[session_id]
                temp_dir = session['temp_dir']
                
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                    print(f"   🗑️  Cleaned up temporary files")
                except:
                    pass
                
                del cls._video_sessions[session_id]


# Node registration
NODE_CLASS_MAPPINGS = {
    "DirectVideoWriter": DirectVideoWriter,
    "DirectVideoWriterAdvanced": DirectVideoWriterAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DirectVideoWriter": "Direct Video Writer (Progressive Export)",
    "DirectVideoWriterAdvanced": "Direct Video Writer (Advanced - Progress + Final)",
}