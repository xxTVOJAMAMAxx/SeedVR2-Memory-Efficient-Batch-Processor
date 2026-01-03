#!/usr/bin/env python3
"""
ComfyUI Auto Batch Queue Script with Smart Naming
Part of SeedVR2-Memory-Efficient-Batch-Processor

Output folder format: videoname_modelname_b21_o3_001
Example: my_video_seedvr2_ema_3b_b21_o3_001

Auto-increments number if same settings run multiple times.

License: Apache 2.0
"""

import json
import sys
import os
import urllib.request
import time


def generate_smart_output_folder(video_path, workflow, batch_size, overlap, output_base_dir):
    """Generate intelligent output folder name"""
    
    # Get video name
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_name = "".join(c for c in video_name if c.isalnum() or c in (' ', '-', '_'))
    video_name = video_name.strip().replace(' ', '_')
    
    if len(video_name) > 50:
        video_name = video_name[:50]
    
    # Extract model name
    model_name = "unknown_model"
    for node_id, node_data in workflow.items():
        if "class_type" in node_data:
            if "SeedVR2LoadDiTModel" in node_data["class_type"]:
                if "inputs" in node_data and "model" in node_data["inputs"]:
                    full_model = node_data["inputs"]["model"]
                    model_name = os.path.splitext(full_model)[0]
                    model_name = model_name.replace("_fp16", "").replace("_fp32", "")
                    break
    
    # Get batch settings
    if batch_size is None or overlap is None:
        for node_id, node_data in workflow.items():
            if "class_type" in node_data and "TrueBatchedVideoLoader" in node_data["class_type"]:
                if "inputs" in node_data:
                    if batch_size is None:
                        batch_size = node_data["inputs"].get("batch_size", 16)
                    if overlap is None:
                        overlap = node_data["inputs"].get("overlap", 0)
                break
    
    # Build base name
    base_name = f"{video_name}_{model_name}_b{batch_size}_o{overlap}"
    base_name = base_name.replace("__", "_")
    
    # Find next available number
    counter = 1
    while counter <= 999:
        folder_name = f"{base_name}_{counter:03d}"
        full_path = os.path.join(output_base_dir, folder_name)
        
        if not os.path.exists(full_path):
            return folder_name
        
        counter += 1
    
    # Fallback
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}"


class ComfyUIBatchQueue:
    """Manages queueing of batched video processing jobs to ComfyUI"""
    
    def __init__(self, server_address="127.0.0.1:8188"):
        self.server_address = server_address
        self.client_id = "seedvr2_batch_processor"
        
    def queue_prompt(self, prompt):
        """Queue a single prompt to ComfyUI"""
        url = f"http://{self.server_address}/prompt"
        
        payload = {
            "prompt": prompt,
            "client_id": self.client_id
        }
        
        data = json.dumps(payload).encode('utf-8')
        
        try:
            req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result
        except Exception as e:
            print(f"Error queueing prompt: {e}")
            return None
    
    def find_batch_loader_node(self, workflow):
        """Find the TrueBatchedVideoLoader node"""
        for node_id, node_data in workflow.items():
            if "class_type" in node_data:
                if "TrueBatchedVideoLoader" in node_data["class_type"]:
                    if "inputs" in node_data and "batch_index" in node_data["inputs"]:
                        return node_id
        return None
    
    def update_batch_index(self, workflow, batch_index):
        """Update the batch_index parameter"""
        node_id = self.find_batch_loader_node(workflow)
        if node_id and "inputs" in workflow[node_id]:
            workflow[node_id]["inputs"]["batch_index"] = batch_index
            return True
        return False
    
    def queue_all_batches(self, workflow_path, num_batches, start_batch=0):
        """Queue all batches to ComfyUI"""
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
        
        print(f"\n{'='*60}")
        print(f"ComfyUI Auto Batch Queue")
        print(f"Batches to queue: {start_batch} to {num_batches-1} (total: {num_batches - start_batch})")
        print(f"{'='*60}\n")
        
        queued_prompts = []
        
        for batch_idx in range(start_batch, num_batches):
            workflow_copy = json.loads(json.dumps(workflow))
            
            if not self.update_batch_index(workflow_copy, batch_idx):
                print(f"✗ Failed to update batch {batch_idx}")
                continue
            
            print(f"Queueing batch {batch_idx + 1}/{num_batches}...", end=" ")
            result = self.queue_prompt(workflow_copy)
            
            if result and "prompt_id" in result:
                queued_prompts.append((batch_idx, result["prompt_id"]))
                print(f"✓")
            else:
                print(f"✗ Failed")
            
            time.sleep(0.3)
        
        print(f"\n{'='*60}")
        print(f"✓ Successfully queued: {len(queued_prompts)} batches")
        print(f"{'='*60}\n")
        
        return queued_prompts


def calculate_batches(workflow):
    """Calculate number of batches from video"""
    loader_node = None
    for node_id, node_data in workflow.items():
        if "class_type" in node_data and "TrueBatchedVideoLoader" in node_data["class_type"]:
            loader_node = node_data
            break
    
    if not loader_node or "inputs" not in loader_node:
        return None
    
    inputs = loader_node["inputs"]
    video_path = inputs.get("video_path", "")
    batch_size = inputs.get("batch_size", 16)
    overlap = inputs.get("overlap", 0)
    
    if not video_path or not os.path.exists(video_path):
        return None
    
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        stride = batch_size - overlap if overlap < batch_size else batch_size
        total_batches = (total_frames + stride - 1) // stride
        
        print(f"\n📹 Video Analysis:")
        print(f"  File: {os.path.basename(video_path)}")
        print(f"  Total frames: {total_frames}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Batch size: {batch_size}")
        print(f"  Overlap: {overlap}")
        print(f"  ➜ Total batches: {total_batches}\n")
        
        return total_batches
    except:
        return None


def update_workflow(workflow, video_path=None, output_folder=None, batch_size=None, overlap=None, image_format=None):
    """Update workflow parameters"""
    for node_id, node_data in workflow.items():
        if "class_type" not in node_data:
            continue
        
        if "TrueBatchedVideoLoader" in node_data["class_type"]:
            if "inputs" in node_data:
                if video_path:
                    node_data["inputs"]["video_path"] = video_path
                    print(f"📹 Video: {os.path.basename(video_path)}")
                if batch_size is not None:
                    node_data["inputs"]["batch_size"] = batch_size
                    print(f"⚙️  Batch size: {batch_size}")
                if overlap is not None:
                    node_data["inputs"]["overlap"] = overlap
                    print(f"⚙️  Overlap: {overlap}")
        
        if "BatchFrameToDiskSaver" in node_data["class_type"]:
            if "inputs" in node_data:
                if output_folder:
                    node_data["inputs"]["output_folder_name"] = output_folder
                    print(f"📁 Output: {output_folder}")
                if image_format:
                    node_data["inputs"]["image_format"] = image_format
                    print(f"🖼️  Format: {image_format}")


def main():
    """Main entry point"""
    
    if len(sys.argv) < 2:
        print("Usage: python script.py workflow.json --video PATH --batch-size N --overlap N --format FORMAT")
        print("\nOptions:")
        print("  --video PATH         Video file path")
        print("  --batch-size N       Batch size (e.g., 21)")
        print("  --overlap N          Overlap (e.g., 3)")
        print("  --format FORMAT      Image format: PNG or WebP (default: from workflow)")
        sys.exit(1)
    
    workflow_path = sys.argv[1]
    
    if not os.path.exists(workflow_path):
        print(f"ERROR: Workflow not found: {workflow_path}")
        sys.exit(1)
    
    # Parse arguments
    args = {}
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--video" and i + 1 < len(sys.argv):
            args['video'] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--batch-size" and i + 1 < len(sys.argv):
            args['batch_size'] = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--overlap" and i + 1 < len(sys.argv):
            args['overlap'] = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--format" and i + 1 < len(sys.argv):
            args['format'] = sys.argv[i + 1]
            i += 2
        else:
            i += 1
    
    # Load workflow
    print(f"Loading workflow: {workflow_path}")
    with open(workflow_path, 'r', encoding='utf-8') as f:
        workflow = json.load(f)
    
    # Find output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try multiple possible paths
    possible_paths = [
        os.path.join(script_dir, "..", "..", "ComfyUI", "output"),  # From python_embeded
        os.path.join(script_dir, "..", "ComfyUI", "output"),
        "E:\\comfy trellis 2.0\\ComfyUI_windows_portable\\ComfyUI\\output",  # Direct path
    ]
    
    output_base_dir = None
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            output_base_dir = abs_path
            break
    
    if not output_base_dir:
        print(f"ERROR: Could not find output directory.")
        print(f"Tried:")
        for path in possible_paths:
            print(f"  - {os.path.abspath(path)}")
        sys.exit(1)
    
    # Get video path
    video_path = args.get('video')
    if not video_path:
        for node_id, node_data in workflow.items():
            if "class_type" in node_data and "TrueBatchedVideoLoader" in node_data["class_type"]:
                if "inputs" in node_data:
                    video_path = node_data["inputs"].get("video_path")
                    break
    
    if not video_path:
        print("ERROR: No video path specified!")
        sys.exit(1)
    
    # Generate output folder
    print("\nGenerating output folder name...")
    output_folder = generate_smart_output_folder(
        video_path,
        workflow,
        args.get('batch_size'),
        args.get('overlap'),
        output_base_dir
    )
    
    print(f"\n{'='*60}")
    print(f"OUTPUT FOLDER: {output_folder}")
    print(f"{'='*60}\n")
    
    # Update workflow
    update_workflow(
        workflow,
        video_path=args.get('video'),
        output_folder=output_folder,
        batch_size=args.get('batch_size'),
        overlap=args.get('overlap'),
        image_format=args.get('format')
    )
    
    # Save workflow
    print(f"\n💾 Saving updated workflow...")
    with open(workflow_path, 'w', encoding='utf-8') as f:
        json.dump(workflow, f, indent=2)
    
    # Calculate batches
    print("\nCalculating batches...")
    num_batches = calculate_batches(workflow)
    
    if num_batches is None:
        num_batches = int(input("Enter number of batches: "))
    
    # ALWAYS ask for confirmation
    confirm = input(f"\n{'='*60}\nQueue {num_batches} batches? (yes/no): ").lower()
    if confirm not in ['yes', 'y']:
        print("Cancelled.")
        sys.exit(0)
    
    # Queue batches
    queue = ComfyUIBatchQueue()
    queued = queue.queue_all_batches(workflow_path, num_batches)
    
    if queued:
        print(f"✓ Success! {len(queued)} batches queued.\n")
    else:
        print("✗ Failed to queue batches.\n")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
