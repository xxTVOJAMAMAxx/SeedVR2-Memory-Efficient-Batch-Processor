eedVR2 Memory-Efficient Batch Processor
Process long videos in ComfyUI without running out of RAM
ComfyUI loads entire videos as uncompressed 32-bit float tensors (~24MB per frame), causing massive RAM usage. This workflow processes videos in configurable batches, dramatically reducing memory requirements while maintaining quality.
🎯 Key Features

✅ Memory Efficient: Process videos in batches - load only what you need (~400MB vs 40GB+)
✅ Automatic Video Export: Real-time progress video + high-quality final video
✅ Smart Naming: Output folders named automatically: videoname_modelname_b21_o3_001
✅ Auto-Detection: Calculates batches from video properties automatically
✅ Multi-threaded: Parallel encoding while upscaling continues
✅ Resume Support: Restart from any batch if interrupted


🚀 Quick Start
Installation

Copy custom nodes to ComfyUI:

