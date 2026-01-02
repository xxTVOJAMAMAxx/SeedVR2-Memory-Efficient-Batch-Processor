# SeedVR2 Memory-Efficient Batch Processor

**Solve ComfyUI's RAM bottleneck for video upscaling with SeedVR2**

ComfyUI converts videos to uncompressed 32-bit float tensors (~24MB per frame), causing massive RAM usage for longer videos. This workflow processes videos in configurable batches, loading only what's needed and saving directly to disk.

## 🎯 Features

- ✅ **Memory Efficient**: Load only one batch at a time (~156MB vs 40GB+)
- ✅ **Direct-to-Disk Saving**: No RAM accumulation during processing
- ✅ **Auto-Detection**: Automatically calculates batches from video properties
- ✅ **Dynamic Configuration**: Adjust batch size, overlap, and format via command line
- ✅ **WebP Support**: Multi-threaded saving (10x faster than PNG)
- ✅ **Unique Output Folders**: Each run creates timestamped folders
- ✅ **Overlap Blending**: Maintains temporal coherence between batches
- ✅ **Resume Support**: Restart from any batch if interrupted

## 📋 Requirements

- ComfyUI
- SeedVR2 Video Upscaler nodes
- Python 3.8+
- OpenCV: `pip install opencv-python` (for auto-detection)

## 🚀 Installation

1. **Install custom nodes:**
   ```bash
   # Copy to ComfyUI/custom_nodes/
   ComfyUI/custom_nodes/diagnostic_batch_loader_v3.py
   ```

2. **Install automation script:**
   ```bash
   # Copy to ComfyUI/python_embeded/ (or your Python directory)
   scripts/auto_batch_script_for_seedvr2.py
   ```

3. **Import example workflow:**
   ```bash
   # Load in ComfyUI
   examples/SeedVR2_HD_video_upscale_RAM_save_v2.json
   ```

4. **Restart ComfyUI**

## 📖 Usage

### Basic Usage (Auto-detect everything)

```bash
python auto_batch_script_for_seedvr2.py workflow.json
```

### Specify Video

```bash
python auto_batch_script_for_seedvr2.py workflow.json --video my_video.mp4
```

### Custom Batch Settings

```bash
python auto_batch_script_for_seedvr2.py workflow.json --video my_video.mp4 --batch-size 16 --overlap 4
```

### Full Control

```bash
python auto_batch_script_for_seedvr2.py workflow.json \
  --video "path/to/my video.mp4" \
  --batch-size 16 \
  --overlap 4 \
  --start 5
```

**Important for videos with spaces in filename:** Use full path with quotes!

```bash
python auto_batch_script_for_seedvr2.py workflow.json --video "E:\path\to\my video file.mp4"
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--video` | Video file path | From workflow |
| `--batch-size` | Frames per batch | From workflow |
| `--overlap` | Frame overlap between batches | From workflow |
| `--start` | Start from batch N | 0 |

## ⚙️ Configuration

### Batch Size & Overlap

**Batch Size**: Number of frames processed together
- Smaller = less RAM, more batches, slower overall
- Larger = more RAM, fewer batches, faster overall
- Recommended: 16-33 frames

**Overlap**: Frames shared between batches for temporal coherence
- Recommended: 3-4 frames
- Higher overlap = smoother transitions, more processing

### Image Format

**WebP (Recommended):**
- Multi-threaded encoding (~10x faster than PNG)
- Quality 95 = excellent quality, good compression
- Smaller file sizes

**PNG:**
- Lossless, universally compatible
- Single-threaded (slower)
- Larger file sizes

Set in ComfyUI workflow: `BatchFrameToDiskSaver` → `image_format`

## 🔄 Workflow

1. **Start ComfyUI**
   ```bash
   run_nvidia_gpu.bat  # or your start command
   ```

2. **Run the script**
   ```bash
   python auto_batch_script_for_seedvr2.py workflow.json --video my_video.mp4 -b 16 --overlap 4
   ```

3. **Monitor progress**
   - Check ComfyUI browser interface
   - Watch terminal output for batch completion

4. **Find output**
   ```
   ComfyUI/output/[videoname_timestamp]/
   ├── frame_000001.webp
   ├── frame_000002.webp
   └── ...
   ```

## ⚠️ Important Notes

### Queue Management

**After each complete job, the queue is automatically cleared.** 

**Only manually clear the queue if:**
- You want to cancel/abort a running job
- You accidentally queued the wrong video
- Something went wrong and you want to restart

**To clear the queue:**
1. Open ComfyUI browser interface (http://127.0.0.1:8188)
2. Look for **Queue** menu or button
3. Click **"Clear Queue"** or **"Cancel All"**

**OR restart ComfyUI:**
```bash
# Press Ctrl+C to stop ComfyUI
# Then start again
run_nvidia_gpu.bat
```

### Resume from Specific Batch

If processing is interrupted, resume from where you left off:

```bash
python auto_batch_script_for_seedvr2.py workflow.json --video my_video.mp4 --start 10
```

This will start from batch 10 (skipping batches 0-9).

### Output Folder Mixing

Each run creates a **unique timestamped folder** automatically:
```
my_video_20251231_143022/
my_video_20251231_150045/
```

This prevents mixing frames from different runs.

## 🐛 Troubleshooting

### "Video not found" error

Use the **full path** with quotes for videos with spaces:
```bash
python auto_batch_script_for_seedvr2.py workflow.json --video "E:\full\path\to\my video.mp4"
```

### Processing wrong video

The queue might still have old batches. **Clear the queue**:
1. Open ComfyUI browser (http://127.0.0.1:8188)
2. Clear all queued items
3. Or restart ComfyUI completely

### Frames saving slowly

1. **Use WebP format** instead of PNG (10x faster)
2. **Lower compression quality**: 85-90 instead of 95
3. **Check CPU usage**: Should see multiple cores active during saving

### Out of RAM

1. **Reduce batch size**: Use 8 or 12 instead of 16+
2. **Close other applications**
3. **Check if accumulation is disabled**: Make sure using `BatchFrameToDiskSaver`, not `BatchFrameAccumulator`

### "Module not found" errors

Make sure custom nodes are installed:
```bash
# Check if file exists
ComfyUI/custom_nodes/dynamic_batch_loader.py
```

Restart ComfyUI after adding custom nodes.

## 📊 Performance Comparison

### RAM Usage
- **Without batching**: 24MB × frames (e.g., 10,000 frames = 240GB)
- **With batching**: ~156MB per batch (e.g., 16 frames = 384MB)

### Saving Speed
- **PNG compression 9**: ~1-3 seconds/frame (single-threaded)
- **WebP quality 95**: ~0.1-0.3 seconds/frame (multi-threaded)

## 🤝 Contributing

Contributions welcome! Please feel free to submit issues or pull requests.

## 📜 License

Apache 2.0

## 🙏 Credits

- **SeedVR2**: Original video upscaling model by ByteDance Seed
- **ComfyUI**: Node-based UI framework
- **Community**: Thank you to everyone who tested and provided feedback!

## 📚 Additional Resources

- [SeedVR2 Original Repository](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)
- [ComfyUI Documentation](https://docs.comfy.org/)
- [Report Issues](https://github.com/yourusername/SeedVR2-Memory-Efficient-Batch-Processor/issues)

---

**Made with ❤️ for the ComfyUI community**
