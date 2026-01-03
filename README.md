# SeedVR2 Memory-Efficient Batch Processor

**Process long videos in ComfyUI without running out of RAM**

ComfyUI loads entire videos as uncompressed 32-bit float tensors (~24MB per frame), causing massive RAM usage. This workflow processes videos in configurable batches, dramatically reducing memory requirements while maintaining quality.

## 🎯 Key Features

- ✅ **Memory Efficient**: Process videos in batches - load only what you need (~400MB vs 40GB+)
- ✅ **Automatic Video Export**: Real-time progress video + high-quality final video
- ✅ **Smart Naming**: Output folders named automatically: `videoname_modelname_b21_o3_001`
- ✅ **Auto-Detection**: Calculates batches from video properties automatically
- ✅ **Multi-threaded**: Parallel encoding while upscaling continues
- ✅ **Resume Support**: Restart from any batch if interrupted

---

## 🚀 Quick Start

### Installation

1. **Copy custom nodes to ComfyUI:**
   ```bash
   ComfyUI/custom_nodes/
   ├── batch_loader_with_path.py
   ├── batch_disk_saver.py
   ├── node_video_exporter_v2.py
   └── node_video_exporter_v3.py  # includes TrueBatchedVideoLoader
   ```

2. **Copy automation script:**
   ```bash
   # Place in ComfyUI/python_embeded/ or your Python directory
   script_auto_batch_smart_naming_v9.py
   ```

3. **Restart ComfyUI**

4. **Load workflow:**
   - Import `SeedVR2_HD_video_upscale_RAM_save_v5_video_export_nodes.json` in ComfyUI
   - Or use the API version: `SeedVR2_HD_video_upscale_RAM_save_v5_video_export_nodes_api.json`

### Basic Usage

**1. Configure your workflow once:**
   - Open workflow in ComfyUI
   - Set your video path in `TrueBatchedVideoLoaderWithPath` node
   - Set batch size (e.g., 21) and overlap (e.g., 3)
   - Save workflow as JSON (e.g., `my_workflow.json`)

**2. Run the script:**
   ```bash
   python script_auto_batch_smart_naming_v9.py my_workflow.json
   ```

**3. The script will:**
   - ✅ Auto-detect video properties
   - ✅ Calculate total batches
   - ✅ Generate smart output folder name
   - ✅ Queue all batches to ComfyUI
   - ✅ Ask for confirmation before starting

**4. Monitor progress:**
   - Watch ComfyUI browser interface for real-time processing
   - Progress video updates automatically during processing
   - Final high-quality video created when complete

**5. Find your output:**
   ```
   ComfyUI/output/videoname_modelname_b21_o3_001/
   ├── frame_000001.png
   ├── frame_000002.png
   ├── frame_000003.png
   ├── ...
   ├── video_progress_h265.mp4    # Updates during processing
   ├── video_final_h265.mp4       # High quality final video
   └── processing_log.txt         # Detailed processing log
   ```

---

## ⚙️ Configuration

### What's Automatic

✅ **Batch calculation** - Script reads video and calculates batches  
✅ **Output naming** - Format: `videoname_modelname_b21_o3_001`  
✅ **Progress video** - Updates every batch (or every N batches)  
✅ **Final video** - Created automatically when complete  
✅ **Frame numbering** - Handles overlap correctly  
✅ **Folder incrementation** - Creates `_002`, `_003` if folder exists

### What You Control

**In ComfyUI Workflow:**
- **Batch size** - Frames per batch (e.g., 21, 61)
- **Overlap** - Frames shared between batches (e.g., 3, 4)
- **Upscale resolution** - Target resolution (e.g., 1080p)
- **Model settings** - DiT model, VAE, quality presets

**Via Script Arguments:**
```bash
python script_auto_batch_smart_naming_v9.py workflow.json \
  --video "path/to/video.mp4" \        # Override video path
  --batch-size 21 \                     # Override batch size
  --overlap 3 \                         # Override overlap
  --format PNG                          # Image format (PNG or WebP)
```

---

## 📖 Workflow Nodes Explained

### Core Processing Chain

```
TrueBatchedVideoLoaderWithPath
    ↓
SeedVR2 Video Upscaler
    ↓
BatchFrameToDiskSaver  ← Saves frames to disk
    ↓
VideoExporterFromFrames  ← Creates videos automatically
```

### Node Details

**`TrueBatchedVideoLoaderWithPath`**
- Loads ONLY current batch into RAM
- Outputs: frames, batch_index, total_batches, fps, start_frame_number, video_source_path
- No batch size limit - use what your VRAM can handle

**`BatchFrameToDiskSaver`**
- Saves processed frames directly to disk as PNG/WebP
- Handles overlap automatically (skips duplicate frames)
- Creates diagnostic log with frame ranges
- Output: folder path, frames_saved, is_complete

**`VideoExporterFromFrames`**
- Monitors saved frames and creates videos automatically
- Progress video: Updates during processing (fast encoding)
- Final video: High quality export when complete (configurable codec/quality)
- Multi-threaded background encoding
- Auto-detects batch info from folder name or logs

---

## 🎛️ Advanced Settings

### Batch Size & Overlap

**Batch Size** affects:
- RAM usage: Larger batch = more RAM needed
- Processing speed: Larger batch = fewer batch loads
- **Recommended**: 16-33 frames (21 is a good default)

**Overlap** affects:
- Temporal coherence between batches
- Higher overlap = smoother transitions
- **Recommended**: 3-4 frames

### Video Export Settings

**Progress Video** (in `VideoExporterFromFrames` node):
- **Codec**: H.264 (compatible) or H.265 (better compression)
- **Quality**: Fast, Medium, High
- **Update Mode**: 
  - `Auto (Smart)` - Updates every effective batch
  - `Every Batch` - Update after each batch
  - `Every 2/3 Batches` - Less frequent updates
  - `Manual` - Specify minimum frames

**Final Video** (created at completion):
- **Codec**: H.264, H.265, or Both
- **Quality**: High, Very High, Lossless
- **Container**: MP4, MOV, or Both

### Image Format

**PNG** (default):
- Lossless quality
- Compatible everywhere
- Slower saving (~1-3 sec/frame)

**WebP**:
- Near-lossless at quality 95
- 10x faster saving (multi-threaded)
- Smaller file sizes

Change in `BatchFrameToDiskSaver` node → `image_format`

---

## 🔧 Troubleshooting

### Video not found
Use full path with quotes for paths with spaces:
```bash
python script.py workflow.json --video "E:\path\to\my video.mp4"
```

### Out of RAM
1. Reduce batch size (try 16 or 12)
2. Close other applications
3. Check SeedVR2 model settings (reduce blocks_to_swap if needed)

### Wrong video processing
Clear ComfyUI queue:
- Open http://127.0.0.1:8188
- Click "Clear Queue" or restart ComfyUI

### Resume from specific batch
If interrupted, resume from batch N:
```bash
python script.py workflow.json --video video.mp4 --start 10
```

### Slow frame saving
1. Use WebP format instead of PNG
2. Lower compression_quality to 85-90
3. Ensure multi-threading is working (check CPU usage)

---

## 📊 Memory Comparison

**Without batching:**
- 10,000 frames × 24MB = **240GB RAM** required

**With batching (batch_size=21):**
- 21 frames × 24MB = **~500MB RAM** per batch
- Process 10,000 frames with minimal memory

---

## 🤝 Contributing

Contributions welcome! Feel free to submit issues or pull requests.

---

## 📜 License

Apache 2.0

---

## 🙏 Credits

- **SeedVR2**: ByteDance Seed team
- **SeedVR2 ComfyUI**: [Nymz](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)
- **ComfyUI**: [comfyanonymous](https://github.com/comfyanonymous/ComfyUI)

---

**Made with ❤️ for the ComfyUI community**
