#!/usr/bin/env python3
"""
ComfyUI Auto Batch Queue Script
Part of SeedVR2-Memory-Efficient-Batch-Processor

Automatically queues multiple batch processing runs in ComfyUI for memory-efficient
video upscaling with SeedVR2. Uses only Python standard library - no pip install needed!

This script solves ComfyUI's RAM bottleneck by:
- Automatically detecting video properties and calculating required batches
- Dynamically updating workflow with new video paths
- Queueing all batches to ComfyUI for hands-free processing

Usage:
    python auto_batch_queue.py workflow.json [video_path] [output_folder] [num_batches] [start_batch]

Examples:
    # Auto-detect everything from workflow
    python auto_batch_queue.py workflow.json
    
    # Specify video path (auto-detect batches)
    python auto_batch_queue.py workflow.json my_video.mp4
    
    # Specify video and output folder
    python auto_batch_queue.py workflow.json my_video.mp4 output_folder
    
    # Full control with manual batch count
    python auto_batch_queue.py workflow.json my_video.mp4 output_folder 100 0

License: Apache 2.0
"""

import json
import sys
import os
import urllib.request
import urllib.parse
import time


class ComfyUIBatchQueue:
    """Manages queueing of batched video processing jobs to ComfyUI"""
    
    def __init__(self, server_address="127.0.0.1:8188"):
        """
        Initialize the batch queue manager.
        
        Args:
            server_address: ComfyUI server address (default: 127.0.0.1:8188)
        """
        self.server_address = server_address
        self.client_id = "seedvr2_batch_processor"
        
    def queue_prompt(self, prompt):
        """
        Queue a single prompt to ComfyUI.
        
        Args:
            prompt: Workflow dictionary to queue
            
        Returns:
            Response dictionary with prompt_id if successful, None otherwise
        """
        url = f"http://{self.server_address}/prompt"
        
        payload = {
            "prompt": prompt,
            "client_id": self.client_id
        }
        
        data = json.dumps(payload).encode('utf-8')
        
        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result
                
        except urllib.error.URLError as e:
            print(f"Error connecting to ComfyUI: {e}")
            print(f"Make sure ComfyUI is running at {self.server_address}")
            return None
        except Exception as e:
            print(f"Error queueing prompt: {e}")
            return None
    
    def find_batch_loader_node(self, workflow):
        """
        Find the TrueBatchedVideoLoader node in workflow.
        
        Args:
            workflow: Workflow dictionary
            
        Returns:
            Node ID string if found, None otherwise
        """
        for node_id, node_data in workflow.items():
            if "class_type" in node_data:
                class_type = node_data["class_type"]
                # Look for batch loader nodes
                if "TrueBatchedVideoLoader" in class_type:
                    if "inputs" in node_data and "batch_index" in node_data["inputs"]:
                        return node_id
        
        # Debug: print available nodes if not found
        print("\nSearching for batch loader node...")
        print("Available nodes in workflow:")
        for node_id, node_data in workflow.items():
            if "class_type" in node_data:
                print(f"  - Node {node_id}: {node_data['class_type']}")
                if "inputs" in node_data:
                    print(f"    Inputs: {list(node_data['inputs'].keys())}")
        
        return None
    
    def update_batch_index(self, workflow, batch_index):
        """
        Update the batch_index parameter in the workflow.
        
        Args:
            workflow: Workflow dictionary
            batch_index: New batch index value
            
        Returns:
            True if successful, False otherwise
        """
        node_id = self.find_batch_loader_node(workflow)
        
        if node_id is None:
            print("\nERROR: Could not find TrueBatchedVideoLoader node!")
            return False
        
        if "inputs" not in workflow[node_id]:
            print(f"ERROR: Node {node_id} has no inputs!")
            return False
        
        # Update batch_index
        workflow[node_id]["inputs"]["batch_index"] = batch_index
        return True
    
    def queue_all_batches(self, workflow_path, num_batches, start_batch=0):
        """
        Queue all batches to ComfyUI for processing.
        
        Args:
            workflow_path: Path to workflow JSON file
            num_batches: Total number of batches to process
            start_batch: Starting batch index (default: 0)
            
        Returns:
            List of (batch_index, prompt_id) tuples for queued batches
        """
        # Load workflow
        try:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                workflow = json.load(f)
        except Exception as e:
            print(f"ERROR: Could not load workflow: {e}")
            return []
        
        print(f"\n{'='*60}")
        print(f"ComfyUI Auto Batch Queue")
        print(f"Workflow: {workflow_path}")
        print(f"Batches to queue: {start_batch} to {num_batches-1} (total: {num_batches - start_batch})")
        print(f"Server: {self.server_address}")
        print(f"{'='*60}\n")
        
        queued_prompts = []
        failed_batches = []
        
        # Queue each batch
        for batch_idx in range(start_batch, num_batches):
            # Create a copy of the workflow for this batch
            workflow_copy = json.loads(json.dumps(workflow))
            
            # Update batch index
            if not self.update_batch_index(workflow_copy, batch_idx):
                print(f"✗ Failed to update batch {batch_idx}")
                failed_batches.append(batch_idx)
                continue
            
            # Queue to ComfyUI
            print(f"Queueing batch {batch_idx + 1}/{num_batches}...", end=" ")
            result = self.queue_prompt(workflow_copy)
            
            if result and "prompt_id" in result:
                prompt_id = result["prompt_id"]
                queued_prompts.append((batch_idx, prompt_id))
                print(f"✓ Queued (ID: {prompt_id})")
            else:
                print(f"✗ Failed")
                failed_batches.append(batch_idx)
            
            # Small delay to avoid overwhelming the server
            time.sleep(0.3)
        
        # Summary
        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"  ✓ Successfully queued: {len(queued_prompts)} batches")
        if failed_batches:
            print(f"  ✗ Failed batches: {failed_batches}")
        print(f"{'='*60}\n")
        
        if queued_prompts:
            print("All batches are now in the ComfyUI queue.")
            print("Check your ComfyUI interface to monitor progress.")
            print("\nThe workflow will process all batches automatically!")
        else:
            print("No batches were queued. Check the errors above.")
        
        return queued_prompts


def calculate_batches_from_video(workflow):
    """
    Automatically calculate number of batches needed from video in workflow.
    
    Args:
        workflow: Workflow dictionary containing video path
        
    Returns:
        Number of batches required, or None if detection fails
    """
    # Find the batch loader node
    batch_loader_node = None
    for node_id, node_data in workflow.items():
        if "class_type" in node_data:
            if "TrueBatchedVideoLoader" in node_data["class_type"]:
                batch_loader_node = node_data
                break
    
    if not batch_loader_node or "inputs" not in batch_loader_node:
        return None
    
    inputs = batch_loader_node["inputs"]
    
    # Extract parameters
    video_path = inputs.get("video_path", "")
    batch_size = inputs.get("batch_size", 16)
    overlap = inputs.get("overlap", 0)
    
    if not video_path:
        print("⚠ No video path found in workflow")
        return None
    
    # Try to locate video file
    if not os.path.exists(video_path):
        # Try relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        comfy_root = os.path.dirname(os.path.dirname(script_dir))
        comfy_input = os.path.join(comfy_root, "ComfyUI", "input")
        
        test_path = os.path.join(comfy_input, video_path)
        if os.path.exists(test_path):
            video_path = test_path
        else:
            # Try common extensions
            for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                test_path = video_path + ext if not video_path.endswith(ext) else video_path
                if os.path.exists(test_path):
                    video_path = test_path
                    break
            else:
                print(f"⚠ Video file not found: {video_path}")
                return None
    
    # Read video properties using OpenCV
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"⚠ Could not open video: {video_path}")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Calculate total batches needed
        stride = batch_size - overlap if overlap < batch_size else batch_size
        total_batches = (total_frames + stride - 1) // stride
        
        # Display video info
        print(f"\n📹 Video Analysis:")
        print(f"  File: {os.path.basename(video_path)}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Total frames: {total_frames}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Batch size: {batch_size}")
        print(f"  Overlap: {overlap}")
        print(f"  ➜ Total batches needed: {total_batches}\n")
        
        return total_batches
        
    except ImportError:
        print("⚠ OpenCV not available - cannot auto-detect batch count")
        print("Install with: pip install opencv-python")
        return None
    except Exception as e:
        print(f"⚠ Error reading video: {e}")
        return None


def update_video_path_in_workflow(workflow, new_video_path):
    """
    Update video path in the workflow.
    
    Args:
        workflow: Workflow dictionary
        new_video_path: New video path to set
        
    Returns:
        True if successful, False otherwise
    """
    for node_id, node_data in workflow.items():
        if "class_type" in node_data:
            if "TrueBatchedVideoLoader" in node_data["class_type"]:
                if "inputs" in node_data:
                    old_path = node_data["inputs"].get("video_path", "")
                    node_data["inputs"]["video_path"] = new_video_path
                    print(f"📝 Updated video path: {os.path.basename(old_path)} → {os.path.basename(new_video_path)}")
                    return True
    return False


def update_output_folder_in_workflow(workflow, new_folder):
    """
    Update output folder name in the workflow.
    
    Args:
        workflow: Workflow dictionary
        new_folder: New output folder name
        
    Returns:
        True if successful, False otherwise
    """
    for node_id, node_data in workflow.items():
        if "class_type" in node_data:
            if "BatchFrameToDiskSaver" in node_data["class_type"]:
                if "inputs" in node_data:
                    old_folder = node_data["inputs"].get("output_folder_name", "")
                    node_data["inputs"]["output_folder_name"] = new_folder
                    print(f"📁 Updated output folder: {old_folder} → {new_folder}")
                    return True
    return False


def update_batch_settings_in_workflow(workflow, batch_size=None, overlap=None):
    """
    Update batch_size and overlap in the workflow.
    
    Args:
        workflow: Workflow dictionary
        batch_size: New batch size (None to keep existing)
        overlap: New overlap value (None to keep existing)
        
    Returns:
        True if node found, False otherwise
    """
    for node_id, node_data in workflow.items():
        if "class_type" in node_data:
            if "TrueBatchedVideoLoader" in node_data["class_type"]:
                if "inputs" in node_data:
                    if batch_size is not None:
                        old_batch = node_data["inputs"].get("batch_size", "")
                        node_data["inputs"]["batch_size"] = batch_size
                        print(f"⚙️ Updated batch_size: {old_batch} → {batch_size}")
                    if overlap is not None:
                        old_overlap = node_data["inputs"].get("overlap", "")
                        node_data["inputs"]["overlap"] = overlap
                        print(f"⚙️ Updated overlap: {old_overlap} → {overlap}")
                    return True
    return False


def parse_arguments(args):
    """
    Parse command line arguments (supports both positional and flag-based).
    
    Args:
        args: sys.argv[2:] (arguments after workflow path)
        
    Returns:
        Dictionary with parsed arguments
    """
    result = {
        'video_path': None,
        'output_folder': None,
        'batch_size': None,
        'overlap': None,
        'start_batch': 0
    }
    
    i = 0
    while i < len(args):
        arg = args[i]
        
        # Flag-based arguments
        if arg in ['--video', '-v']:
            if i + 1 < len(args):
                result['video_path'] = args[i + 1]
                i += 2
                continue
        elif arg in ['--output', '-o']:
            if i + 1 < len(args):
                result['output_folder'] = args[i + 1]
                i += 2
                continue
        elif arg in ['--batch-size', '-b']:
            if i + 1 < len(args):
                result['batch_size'] = int(args[i + 1])
                i += 2
                continue
        elif arg in ['--overlap']:
            if i + 1 < len(args):
                result['overlap'] = int(args[i + 1])
                i += 2
                continue
        elif arg in ['--start']:
            if i + 1 < len(args):
                result['start_batch'] = int(args[i + 1])
                i += 2
                continue
        
        # Positional arguments (backward compatibility)
        # Order: video_path, output_folder
        if result['video_path'] is None and not arg.isdigit():
            result['video_path'] = arg
        elif result['output_folder'] is None and not arg.isdigit():
            result['output_folder'] = arg
        
        i += 1
    
    return result
    """
    Generate a unique output folder name based on video and timestamp.
    
    Args:
        video_path: Path to the video file
        custom_name: Optional custom folder name
        
    Returns:
        Unique folder name string
    """
    from datetime import datetime
    
    if custom_name:
        return custom_name
    
    # Get video filename without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Clean filename (remove special characters)
    video_name = "".join(c for c in video_name if c.isalnum() or c in (' ', '-', '_'))
    video_name = video_name.strip().replace(' ', '_')
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Combine: videoname_timestamp
    folder_name = f"{video_name}_{timestamp}"
    
    return folder_name


def print_usage():
    """Print usage instructions"""
    print("ComfyUI Auto Batch Queue Script")
    print("Part of SeedVR2-Memory-Efficient-Batch-Processor")
    print("="*60)
    print("\nUsage:")
    print("  python auto_batch_queue.py <workflow.json> [options]")
    print("\nOptions (in order):")
    print("  --video <path>        Video file path")
    print("  --output <folder>     Output folder name")
    print("  --batch-size <n>      Frames per batch (default: from workflow)")
    print("  --overlap <n>         Frame overlap between batches (default: from workflow)")
    print("  --start <n>           Start from batch N (default: 0)")
    print("\nSimple Usage (positional args):")
    print("  workflow.json [video_path] [output_folder]")
    print("\nExamples:")
    print("  # Auto-detect everything")
    print("  python auto_batch_queue.py workflow.json")
    print()
    print("  # Specify video")
    print("  python auto_batch_queue.py workflow.json --video my_video.mp4")
    print()
    print("  # Custom batch settings")
    print("  python auto_batch_queue.py workflow.json --video my_video.mp4 --batch-size 9 --overlap 3")
    print()
    print("  # Positional (backward compatible)")
    print("  python auto_batch_queue.py workflow.json my_video.mp4 output_folder")
    print("\nFeatures:")
    print("  ✓ Automatic video analysis and batch calculation")
    print("  ✓ Dynamic workflow updates (video, batch size, overlap)")
    print("  ✓ Memory-efficient processing (loads one batch at a time)")
    print("  ✓ Direct-to-disk saving (no RAM accumulation)")
    print("\nRequirements:")
    print("  - ComfyUI running at 127.0.0.1:8188")
    print("  - Workflow exported in API format from ComfyUI")
    print("  - OpenCV (for auto-detection): pip install opencv-python")


def main():
    """Main entry point"""
    
    # Check arguments
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    workflow_path = sys.argv[1]
    
    # Validate workflow file exists
    if not os.path.exists(workflow_path):
        print(f"ERROR: Workflow file not found: {workflow_path}")
        print(f"Current directory: {os.getcwd()}")
        sys.exit(1)
    
    # Load workflow
    try:
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load workflow: {e}")
        sys.exit(1)
    
    # Parse command line arguments
    args = parse_arguments(sys.argv[2:])
    
    video_path_arg = args['video_path']
    output_folder_arg = args['output_folder']
    batch_size_arg = args['batch_size']
    overlap_arg = args['overlap']
    start_batch = args['start_batch']
    
    # Update video path if provided
    if video_path_arg:
        if not update_video_path_in_workflow(workflow, video_path_arg):
            print("⚠ Warning: Could not find video loader node to update")
    
    # Update batch settings if provided
    if batch_size_arg is not None or overlap_arg is not None:
        if not update_batch_settings_in_workflow(workflow, batch_size_arg, overlap_arg):
            print("⚠ Warning: Could not find video loader node to update batch settings")
    
    # Generate output folder name
    # Priority: 1) Command line arg, 2) Auto-generate from video, 3) Use workflow default
    if output_folder_arg:
        # User specified output folder
        final_output_folder = output_folder_arg
    else:
        # Auto-generate from video path
        # Get video path from either command line or workflow
        video_path_for_folder = video_path_arg
        if not video_path_for_folder:
            # Extract from workflow
            for node_id, node_data in workflow.items():
                if "class_type" in node_data and "TrueBatchedVideoLoader" in node_data["class_type"]:
                    if "inputs" in node_data:
                        video_path_for_folder = node_data["inputs"].get("video_path", "")
                        break
        
        if video_path_for_folder:
            final_output_folder = generate_output_folder_name(video_path_for_folder)
        else:
            # Fallback: use timestamp only
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_output_folder = f"upscaled_{timestamp}"
    
    # Update output folder in workflow
    if not update_output_folder_in_workflow(workflow, final_output_folder):
        print("⚠ Warning: Could not find disk saver node to update")
    
    # Determine number of batches
    num_batches = None
    
    # Auto-detect from video (will use updated batch_size/overlap if provided)
    print("Attempting to auto-detect batch count from video...")
    num_batches = calculate_batches_from_video(workflow)
    
    if num_batches is None:
        # Fallback to manual input
        try:
            num_batches = int(input("\nCould not auto-detect. Enter total number of batches: "))
        except (ValueError, KeyboardInterrupt):
            print("\nCancelled.")
            sys.exit(1)
    
    # Get start batch
    start_batch = args['start_batch']
    
    # Confirmation for large batch counts
    if num_batches - start_batch > 10:
        print(f"\nYou are about to queue {num_batches - start_batch} batches.")
        try:
            confirm = input("Continue? (yes/no): ").lower()
            if confirm not in ['yes', 'y']:
                print("Cancelled.")
                sys.exit(0)
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)
    
    # Queue all batches
    queue = ComfyUIBatchQueue(server_address="127.0.0.1:8188")
    queued = queue.queue_all_batches(workflow_path, num_batches, start_batch)
    
    if queued:
        print(f"\n✓ Success! {len(queued)} batches queued.")
        print("Your video will be processed automatically in ComfyUI.")
    else:
        print("\n✗ Failed to queue batches. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
