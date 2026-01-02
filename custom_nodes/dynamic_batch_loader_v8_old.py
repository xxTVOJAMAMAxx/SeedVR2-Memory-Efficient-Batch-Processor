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
        
        # Get video info (cached)
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
        
        # Calculate frame range
        start_frame = batch_index * stride
        end_frame = min(start_frame + batch_size, total_frames)
        
        if start_frame >= total_frames:
            raise ValueError(f"Batch {batch_index} exceeds total batches {total_batches}")
        
        print(f"Loading batch {batch_index + 1}/{total_batches}: frames {start_frame}-{end_frame-1}")
        
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
        print(f"Loaded {len(frames)} frames (~{ram_mb:.1f} MB)")
        
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
    Saves each processed batch directly to disk as PNG sequence.
    NO accumulation in RAM - truly memory efficient!
    Handles overlap by skipping overlapping frames from subsequent batches.
    """
    
    def __init__(self):
        self.last_saved_frame = -1
        self.output_folder = None
    
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
                "png_compression": ("INT", {"default": 9, "min": 0, "max": 9, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("STRING", "INT", "BOOLEAN")
    RETURN_NAMES = ("output_folder", "frames_saved", "is_complete")
    FUNCTION = "save_batch_to_disk"
    CATEGORY = "video/batch"
    OUTPUT_NODE = True
    
    def save_batch_to_disk(self, processed_batch, batch_index, total_batches, start_frame_number, 
                           overlap, output_folder_name, png_compression):
        """Save batch directly to disk, skipping overlapping frames"""
        
        # Create output folder in ComfyUI output directory
        output_dir = folder_paths.get_output_directory()
        output_path = os.path.join(output_dir, output_folder_name)
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"\n📁 Created output folder: {output_path}")
            self.output_folder = output_path
            self.last_saved_frame = -1
        
        # Determine which frames to save from this batch
        if batch_index == 0:
            # First batch: save all frames
            frames_to_save = processed_batch
            skip_frames = 0
        else:
            # Subsequent batches: skip overlapping frames
            skip_frames = min(overlap, processed_batch.shape[0])
            frames_to_save = processed_batch[skip_frames:]
        
        # Convert tensors to numpy and save as PNG
        frames_np = (frames_to_save.cpu().numpy() * 255).astype(np.uint8)
        
        saved_count = 0
        for i, frame in enumerate(frames_np):
            frame_number = start_frame_number + skip_frames + i
            
            # Skip if we already saved this frame (shouldn't happen, but safety check)
            if frame_number <= self.last_saved_frame:
                continue
            
            # Create filename with padding (e.g., frame_000001.png)
            filename = f"frame_{frame_number:06d}.png"
            filepath = os.path.join(output_path, filename)
            
            # Save as PNG with specified compression
            img = Image.fromarray(frame)
            img.save(filepath, "PNG", compress_level=png_compression)
            
            saved_count += 1
            self.last_saved_frame = frame_number
        
        is_complete = (batch_index >= total_batches - 1)
        
        # Progress info
        total_saved = self.last_saved_frame + 1
        print(f"💾 Batch {batch_index + 1}/{total_batches}: Saved {saved_count} frames "
              f"(Total saved: {total_saved} frames)")
        
        if is_complete:
            print(f"\n{'='*60}")
            print(f"✓✓✓ ALL BATCHES SAVED!")
            print(f"Total frames: {total_saved}")
            print(f"Output folder: {output_path}")
            print(f"{'='*60}\n")
        
        # Clean up GPU memory
        del frames_to_save, frames_np
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return (output_path, saved_count, is_complete)


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
    "BatchFrameToDiskSaver": "Batch Frame to Disk Saver (No RAM Accumulation)",
    "PNGSequenceToVideo": "PNG Sequence to Video Converter",
}
