import torch
import numpy as np
from PIL import Image
import os
import gc
import folder_paths
from concurrent.futures import ThreadPoolExecutor
import threading


class BatchFrameToDiskSaver:
    """
    Saves each processed batch directly to disk as image sequence.
    NO accumulation in RAM - truly memory efficient!
    Handles overlap by skipping overlapping frames from subsequent batches.
    Supports PNG and WebP formats with multi-threading.
    """
    
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
                "overlap": ("INT", {"default": 4, "min": 0, "max": 9999}),
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
        log_file = os.path.join(output_path, "frame_save_log.txt")
        
        # Create folder and initialize state if first batch
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"\n[DISK SAVER] Created output folder: {output_path}")
            
            # Initialize state file
            with open(state_file, 'w') as f:
                f.write("-1\n0")  # last_saved_frame, total_frames_saved
            
            # Create log file
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("="*70 + "\n")
                f.write("Frame Save Log\n")
                f.write("="*70 + "\n")
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
            skip_frames = min(overlap, batch_frames_count)
            frames_to_save = processed_batch[skip_frames:]
            save_start_frame = start_frame_number + skip_frames
            save_end_frame = save_start_frame + frames_to_save.shape[0] - 1
        
        frames_to_save_count = frames_to_save.shape[0]
        
        # Log output
        print(f"\n{'='*70}")
        print(f"[DISK SAVER] Batch {batch_index + 1}/{total_batches}")
        print(f"{'='*70}")
        print(f"  Batch loaded frames:    {start_frame_number} to {start_frame_number + batch_frames_count - 1} (count: {batch_frames_count})")
        print(f"  Overlap to skip:        {skip_frames} frames")
        print(f"  Frames to save:         {save_start_frame} to {save_end_frame} (count: {frames_to_save_count})")
        print(f"  Last saved frame:       {last_saved_frame}")
        
        # Check for gaps or overlaps
        if last_saved_frame >= 0:
            expected_next = last_saved_frame + 1
            actual_next = save_start_frame
            if actual_next > expected_next:
                gap_size = actual_next - expected_next
                print(f"  [WARNING] GAP DETECTED! Missing frames {expected_next} to {actual_next - 1} ({gap_size} frames)")
            elif actual_next < expected_next:
                overlap_size = expected_next - actual_next
                print(f"  [WARNING] OVERLAP DETECTED! Duplicate frames {actual_next} to {expected_next - 1} ({overlap_size} frames)")
            else:
                print(f"  [OK] Frame sequence is continuous")
        
        print(f"{'='*70}\n")
        
        # Log to file
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
        print(f"[DISK SAVER] Batch {batch_index + 1}/{total_batches}: Saved {saved_count} frames as {image_format} "
              f"(Cumulative total: {total_frames_saved} frames)")
        
        if is_complete:
            print(f"\n{'='*60}")
            print(f"[COMPLETE] ALL BATCHES SAVED!")
            print(f"Total frames saved: {total_frames_saved}")
            print(f"Format: {image_format}")
            print(f"Output folder: {output_path}")
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
        
        saved_count = 0
        lock = threading.Lock()
        
        def save_single_frame(args):
            nonlocal saved_count
            frame, frame_number = args
            
            if frame_number <= last_saved_frame:
                return
            
            filename = f"frame_{frame_number:06d}.webp"
            filepath = os.path.join(output_path, filename)
            
            img = Image.fromarray(frame)
            img.save(filepath, "WebP", quality=95, method=4, lossless=False)
            
            with lock:
                saved_count += 1
        
        frame_args = [(frame, start_frame_number + i) for i, frame in enumerate(frames_np)]
        max_workers = min(os.cpu_count() * 2, 16)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(save_single_frame, frame_args)
        
        return saved_count
    
    def save_frames_png_threaded(self, frames_np, output_path, start_frame_number, last_saved_frame):
        """Save frames as PNG using multi-threading"""
        
        saved_count = 0
        lock = threading.Lock()
        
        def save_single_frame(args):
            nonlocal saved_count
            frame, frame_number = args
            
            if frame_number <= last_saved_frame:
                return
            
            filename = f"frame_{frame_number:06d}.png"
            filepath = os.path.join(output_path, filename)
            
            img = Image.fromarray(frame)
            img.save(filepath, "PNG", compress_level=9)
            
            with lock:
                saved_count += 1
        
        frame_args = [(frame, start_frame_number + i) for i, frame in enumerate(frames_np)]
        max_workers = min(os.cpu_count() * 2, 16)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(save_single_frame, frame_args)
        
        return saved_count


# Node registration
NODE_CLASS_MAPPINGS = {
    "BatchFrameToDiskSaver": BatchFrameToDiskSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchFrameToDiskSaver": "Batch Frame to Disk Saver (No RAM Accumulation)",
}