# SeedVR2-Memory-Efficient-Batch-Processor
Solves ComfyUI's inherent RAM bottleneck where videos are converted to uncompressed 32-bit float tensors (~24MB per frame for 1080p), causing exponential memory usage for longer videos. This workflow loads and processes frames in configurable batches, saving processed frames directly to disk as compressed PNGs to prevent accumulation.

✅ Loads only one batch at a time (~156 MB instead of 100+ GB (for 3minute and longer videos))
✅ Saves directly to disk (no RAM accumulation)
✅ Automatically calculates batches needed
✅ Queues everything with one command
✅ Works with any video via command line


# ComfyUI Batch Video Upscaling Tutorial

## Installation

1. **Paste `dynamic_batch_loader_v8.py`** into ComfyUI's `custom_nodes` folder
2. **Paste both files** into `python_embeded` folder:
   - `auto_batch_script_automatic_batches_v2.py` (the Python script)
   - `SeedVR2_HD_video_upscale_RAM_save_v2.json` (the workflow JSON)

## How to Run

### Step 1: Start ComfyUI
```
run_nvidia_gpu.bat
```

### Step 2: Place your video
Put your video file in ComfyUI's `input` folder:
```
ComfyUI\input\your_video.mp4
```

### Step 3: Run the script

Open CMD in `python_embeded` folder and run:

**Option A: Process video from workflow (use existing video path in JSON):**
```bash
python auto_batch_script_automatic_batches_v2.py SeedVR2_HD_video_upscale_RAM_save_v2.json
```

**Option B: Process a different video (recommended):**
```bash
python auto_batch_script_automatic_batches_v2.py SeedVR2_HD_video_upscale_RAM_save_v2.json your_video.mp4
```

**Option C: Specify custom output folder:**
```bash
python auto_batch_script_automatic_batches_v2.py SeedVR2_HD_video_upscale_RAM_save_v2.json your_video.mp4 output_folder_name
```

### Step 4: Confirm and wait
- The script will auto-detect the number of batches needed
- Type `yes` to confirm
- The script will queue all batches automatically
- ComfyUI will process them one by one
- Monitor progress in ComfyUI browser window

## Output

Upscaled frames will be saved as PNG sequence in:
```
ComfyUI\output\[output_folder_name]\
```

Files will be named: `frame_000001.png`, `frame_000002.png`, etc.

## Notes

- **You do NOT need to click "Run" in ComfyUI** - the script does everything
- Each batch uses ~156 MB RAM for input (memory efficient!)
- Output is saved directly to disk as PNG files (no RAM accumulation)
- PNG compression level 9 = smallest files, lossless quality
- Make sure ComfyUI stays running - don't close the terminal!

## Troubleshooting

**"Video not found" error:**
- Make sure video is in `ComfyUI\input\` folder
- Use just the filename, not full path (unless video is elsewhere)

**Script can't find workflow:**
- Make sure JSON file is in `python_embeded` folder
- Check filename matches exactly

**Wrong number of batches detected:**
- Manually specify: add number at the end
  ```bash
  python auto_batch_script_automatic_batches_v2.py workflow.json video.mp4 output 100
  ```
