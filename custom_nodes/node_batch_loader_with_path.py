import torch
import numpy as np
from PIL import Image
import os
import gc
import folder_paths
import comfy.utils
import cv2

class TrueBatchedVideoLoaderWithPath:
    """
    Loads ONLY the current batch from video file into RAM.
    True memory efficiency - loads one batch at a time.
    
    NOW WITH SOURCE PATH OUTPUT - for automatic naming in video export!
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


# Node registration
NODE_CLASS_MAPPINGS = {
    "TrueBatchedVideoLoaderWithPath": TrueBatchedVideoLoaderWithPath,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TrueBatchedVideoLoaderWithPath": "True Batched Video Loader (with Source Path)",
}