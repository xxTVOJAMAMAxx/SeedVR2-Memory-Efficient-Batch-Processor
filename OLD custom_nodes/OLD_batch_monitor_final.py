#!/usr/bin/env python3
"""
Batch Progress Monitor for SeedVR2 Memory-Efficient Batch Processor
WITH SEPARATE MONITORING WINDOW SUPPORT!

License: Apache 2.0
"""

import os
import json
import time
import glob
import csv
from datetime import datetime, timedelta
from collections import deque
import threading
import queue as Queue
import subprocess
import sys
import platform
from pathlib import Path

# Try importing monitoring libraries
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  psutil not available - RAM monitoring disabled")

try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except:
    PYNVML_AVAILABLE = False
    print("⚠️  pynvml not available - VRAM monitoring disabled")


# ============================================================================
# NEW: SEPARATE MONITOR WINDOW
# ============================================================================

class MonitorWindowThread(threading.Thread):
    """Background thread that manages a separate monitoring console"""
    
    def __init__(self):
        super().__init__(daemon=True)
        self.queue = Queue.Queue()
        self.process = None
        self.running = False
        
    def run(self):
        if platform.system() != 'Windows':
            print("⚠️  Separate monitor window only works on Windows")
            return
        
        try:
            # Monitor display script (ASCII-only for Windows compatibility)
            monitor_script = '''
import sys, os, time
os.system("title SeedVR2 Batch Monitor - Live Updates")
os.system("mode con: cols=80 lines=45")
os.system("color 0A")
print("\\n" + "="*78)
print(" "*25 + "SEEDVR2 BATCH MONITOR")
print(" "*22 + "Waiting for first update...")
print("="*78 + "\\n")
sys.stdout.flush()

try:
    while True:
        line = sys.stdin.readline()
        if not line: break
        if line.startswith("CLEAR"): 
            os.system("cls")
        elif line.startswith("COMPLETE"):
            # Keep window open after completion
            os.system("cls")
            print("\\n" + "="*78)
            print(" "*30 + "[COMPLETE]")
            print("="*78 + "\\n")
            print("Processing finished successfully!")
            print("\\nThis window will close in 10 seconds...")
            print("Or press any key to close now.")
            sys.stdout.flush()
            time.sleep(10)
            break
        else:
            print(line, end='', flush=True)
except: pass

print("\\n" + "="*78)
print(" "*28 + "Monitor Closed")
print("="*78)
time.sleep(1)
'''
            
            # Save temp script
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                script_path = f.name
                f.write(monitor_script)
            
            # Launch new console
            self.process = subprocess.Popen(
                [sys.executable, script_path],
                stdin=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                text=True,
                bufsize=1
            )
            
            self.running = True
            print("✅ Monitoring window opened")
            
            # Process queue
            while self.running:
                try:
                    text = self.queue.get(timeout=0.1)
                    if text is None: break
                    if self.process and self.process.stdin:
                        self.process.stdin.write(text + "\n")
                        self.process.stdin.flush()
                except Queue.Empty:
                    continue
                except:
                    break
            
            # Cleanup
            if self.process:
                try:
                    self.process.stdin.close()
                    self.process.wait(timeout=2)
                except:
                    self.process.kill()
            try:
                os.unlink(script_path)
            except:
                pass
                
        except Exception as e:
            print(f"⚠️  Could not start monitor window: {e}")
            self.running = False
    
    def write(self, text, clear_first=False):
        if not self.running:
            return False
        try:
            if clear_first:
                self.queue.put("CLEAR")
            self.queue.put(text)
            # Debug: confirm write
            print(f"[WINDOW] Queued {len(text)} chars (queue size: {self.queue.qsize()})")
            return True
        except Exception as e:
            print(f"[WINDOW] Write failed: {e}")
            return False
    
    def close(self):
        self.running = False
        self.queue.put(None)


# ============================================================================
# RESOURCE LOGGER
# ============================================================================

class ResourceLogger:
    """Comprehensive logging system for resource usage tracking"""
    
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        self.session_id = None
        self.session_log_path = None
        self.csv_log_path = None
        self.summary_log_path = None
        self.session_data = {}
        self.sample_count = 0
        os.makedirs(log_dir, exist_ok=True)
        
    def start_session(self, workflow_name="", model_name="", batch_size=0, 
                     overlap=0, resolution=0, total_batches=0):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"session_{timestamp}"
        
        self.session_data = {
            'session_id': self.session_id,
            'start_time': datetime.now().isoformat(),
            'workflow_name': workflow_name,
            'model_name': model_name,
            'batch_size': batch_size,
            'overlap': overlap,
            'resolution': resolution,
            'total_batches': total_batches,
            'samples': [],
            'peak_vram_gb': 0.0,
            'peak_ram_gb': 0.0,
            'avg_vram_gb': 0.0,
            'avg_ram_gb': 0.0,
            'total_frames_processed': 0,
            'total_processing_time': 0.0,
            'avg_batch_time': 0.0,
            'avg_fps': 0.0
        }
        
        session_folder = os.path.join(self.log_dir, self.session_id)
        os.makedirs(session_folder, exist_ok=True)
        
        self.session_log_path = os.path.join(session_folder, "session_data.json")
        self.csv_log_path = os.path.join(session_folder, "resource_timeline.csv")
        self._init_csv_log()
        self.summary_log_path = os.path.join(self.log_dir, "sessions_summary.csv")
        
        print(f"📊 Logging session started: {self.session_id}")
        
    def _init_csv_log(self):
        headers = [
            'timestamp', 'elapsed_seconds', 'current_batch', 'progress_percent',
            'vram_used_gb', 'vram_total_gb', 'vram_percent',
            'ram_used_gb', 'ram_total_gb', 'ram_percent',
            'ram_cached_gb', 'ram_used_excluding_cache_gb', 'ram_available_gb',
            'process_rss_gb', 'process_vms_gb',
            'gpu_utilization_percent', 'gpu_temperature_c',
            'frames_saved', 'instantaneous_fps', 'avg_batch_time_seconds'
        ]
        with open(self.csv_log_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(headers)
    
    def log_sample(self, stats):
        if not self.session_id:
            return
        
        self.sample_count += 1
        vram = stats.get('vram', {})
        ram = stats.get('ram', {})
        gpu = stats.get('gpu', {})
        process = stats.get('process', {})
        
        vram_used = vram.get('used', 0.0)
        ram_used = ram.get('used', 0.0)
        
        if vram_used > self.session_data['peak_vram_gb']:
            self.session_data['peak_vram_gb'] = vram_used
        if ram_used > self.session_data['peak_ram_gb']:
            self.session_data['peak_ram_gb'] = ram_used
        
        n = self.sample_count
        self.session_data['avg_vram_gb'] += (vram_used - self.session_data['avg_vram_gb']) / n
        self.session_data['avg_ram_gb'] += (ram_used - self.session_data['avg_ram_gb']) / n
        
        sample = {
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': stats['elapsed_seconds'],
            'current_batch': stats['current_batch'],
            'progress_percent': stats['progress_percent'],
            'vram_used_gb': vram_used,
            'vram_total_gb': vram.get('total', 0.0),
            'vram_percent': vram.get('percent', 0.0),
            'ram_used_gb': ram_used,
            'ram_total_gb': ram.get('total', 0.0),
            'ram_percent': ram.get('percent', 0.0),
            'ram_cached_gb': ram.get('cached', 0.0),
            'ram_used_excluding_cache_gb': ram.get('used_excluding_cache', 0.0),
            'ram_available_gb': ram.get('available', 0.0),
            'process_rss_gb': process.get('rss_gb', 0.0),
            'process_vms_gb': process.get('vms_gb', 0.0),
            'gpu_utilization_percent': gpu.get('utilization', 0) if gpu else 0,
            'gpu_temperature_c': gpu.get('temperature', 0) if gpu else 0,
            'frames_saved': stats['frames_saved'],
            'instantaneous_fps': stats['fps'],
            'avg_batch_time_seconds': stats['avg_batch_time']
        }
        
        self.session_data['samples'].append(sample)
        
        with open(self.csv_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([sample[k] for k in [
                'timestamp', 'elapsed_seconds', 'current_batch', 'progress_percent',
                'vram_used_gb', 'vram_total_gb', 'vram_percent',
                'ram_used_gb', 'ram_total_gb', 'ram_percent',
                'ram_cached_gb', 'ram_used_excluding_cache_gb', 'ram_available_gb',
                'process_rss_gb', 'process_vms_gb',
                'gpu_utilization_percent', 'gpu_temperature_c',
                'frames_saved', 'instantaneous_fps', 'avg_batch_time_seconds'
            ]])
    
    def end_session(self, stats):
        if not self.session_id:
            return
        
        self.session_data['end_time'] = datetime.now().isoformat()
        self.session_data['total_frames_processed'] = stats['frames_saved']
        self.session_data['total_processing_time'] = stats['elapsed_seconds']
        self.session_data['avg_batch_time'] = stats['avg_batch_time']
        self.session_data['avg_fps'] = stats['fps']
        
        with open(self.session_log_path, 'w', encoding='utf-8') as f:
            json.dump(self.session_data, f, indent=2)
        
        self._append_to_summary()
        print(f"\n📊 Session log saved: {self.session_log_path}")
    
    def _append_to_summary(self):
        file_exists = os.path.exists(self.summary_log_path)
        headers = [
            'session_id', 'start_time', 'end_time', 'duration_seconds',
            'workflow_name', 'model_name', 'batch_size', 'overlap', 'resolution',
            'total_batches', 'total_frames', 'avg_batch_time_seconds', 'avg_fps',
            'peak_vram_gb', 'avg_vram_gb', 'peak_ram_gb', 'avg_ram_gb'
        ]
        
        with open(self.summary_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow([
                self.session_data['session_id'],
                self.session_data['start_time'],
                self.session_data['end_time'],
                self.session_data['total_processing_time'],
                self.session_data['workflow_name'],
                self.session_data['model_name'],
                self.session_data['batch_size'],
                self.session_data['overlap'],
                self.session_data['resolution'],
                self.session_data['total_batches'],
                self.session_data['total_frames_processed'],
                self.session_data['avg_batch_time'],
                self.session_data['avg_fps'],
                self.session_data['peak_vram_gb'],
                self.session_data['avg_vram_gb'],
                self.session_data['peak_ram_gb'],
                self.session_data['avg_ram_gb']
            ])


# ============================================================================
# RESOURCE MONITOR
# ============================================================================

class ResourceMonitor:
    def __init__(self, gpu_index=0):
        self.gpu_index = gpu_index
        self.gpu_handle = None
        if PYNVML_AVAILABLE:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except:
                pass
    
    def get_vram_usage(self):
        if not self.gpu_handle:
            return None
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            return {
                'used': info.used / (1024**3),
                'total': info.total / (1024**3),
                'free': info.free / (1024**3),
                'percent': (info.used / info.total) * 100
            }
        except:
            return None
    
    def get_ram_usage(self):
        if not PSUTIL_AVAILABLE:
            return None
        try:
            memory = psutil.virtual_memory()
            ram_data = {
                'used': memory.used / (1024**3),
                'total': memory.total / (1024**3),
                'available': memory.available / (1024**3),
                'percent': memory.percent,
                'free': memory.free / (1024**3),
            }
            
            if hasattr(memory, 'cached'):
                ram_data['cached'] = memory.cached / (1024**3)
            if hasattr(memory, 'buffers'):
                ram_data['buffers'] = memory.buffers / (1024**3)
            
            if hasattr(memory, 'cached') and hasattr(memory, 'buffers'):
                ram_data['used_excluding_cache'] = (memory.used - memory.cached - memory.buffers) / (1024**3)
            elif hasattr(memory, 'cached'):
                ram_data['used_excluding_cache'] = (memory.used - memory.cached) / (1024**3)
            else:
                ram_data['used_excluding_cache'] = ram_data['used']
            
            if not hasattr(memory, 'cached'):
                ram_data['cached'] = max(0, ram_data['total'] - ram_data['used'] - ram_data['free'])
            
            return ram_data
        except:
            return None
    
    def get_gpu_stats(self):
        if not self.gpu_handle:
            return None
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            return {'utilization': util.gpu, 'temperature': temp}
        except:
            return None
    
    def get_process_memory(self):
        if not PSUTIL_AVAILABLE:
            return None
        try:
            process = psutil.Process()
            mem_info = process.memory_info()
            return {
                'rss_gb': mem_info.rss / (1024**3),
                'vms_gb': mem_info.vms / (1024**3),
            }
        except:
            return None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_progress_bar(percent, width=20):
    """Create ASCII-only progress bar"""
    filled = int(width * percent / 100)
    empty = width - filled
    return '#' * filled + '-' * empty  # ASCII characters only

def format_time(seconds):
    if seconds is None or seconds < 0:
        return "calculating..."
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s:02d}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m:02d}m"

def format_completion_time(eta_seconds):
    if eta_seconds is None or eta_seconds < 0:
        return "calculating..."
    completion_time = datetime.now() + timedelta(seconds=eta_seconds)
    return completion_time.strftime("%I:%M %p")


# ============================================================================
# BATCH PROGRESS MONITOR
# ============================================================================

class BatchProgressMonitor:
    """Monitors batch processing progress with optional separate window"""
    
    def __init__(self):
        self.resource_monitor = ResourceMonitor()
        self.resource_logger = ResourceLogger()
        self.start_time = None
        self.last_update_time = 0
        self.update_interval = 2.0
        self.session_started = False
        self.session_completed = False
        self.batch_info = None
        self.monitor_window = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_folder": ("STRING", {"default": ""}),
                "enable_monitoring": ("BOOLEAN", {"default": True}),
                "enable_logging": ("BOOLEAN", {"default": True}),
                "log_directory": ("STRING", {"default": "logs"}),
            },
            "optional": {
                "separate_window": ("BOOLEAN", {"default": True}),
                "is_complete": ("BOOLEAN", {"default": False}),
                "frames_saved": ("INT", {"default": 0}),
                "workflow_name": ("STRING", {"default": ""}),
                "model_name": ("STRING", {"default": ""}),
                "batch_size": ("STRING", {"default": "0"}),
                "overlap": ("STRING", {"default": "0"}),
                "resolution": ("STRING", {"default": "0"}),
                "total_batches": ("STRING", {"default": "0"}),
                "gpu_index": ("INT", {"default": 0, "min": 0, "max": 7}),
                "start_time": ("FLOAT", {"default": 0.0}),
            }
        }
    
    RETURN_TYPES = ("INT", "FLOAT", "FLOAT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("eta_seconds", "progress_percent", "vram_used_gb", "vram_total_gb", "stats_json", "status_text")
    FUNCTION = "monitor_progress"
    CATEGORY = "video/monitoring"
    OUTPUT_NODE = True
    
    def monitor_progress(self, output_folder, enable_monitoring, enable_logging, 
                        log_directory, separate_window=True, is_complete=False, 
                        frames_saved=0, workflow_name="", model_name="", 
                        batch_size="0", overlap="0", resolution="0", 
                        total_batches="0", gpu_index=0, start_time=0.0):
        
        if not enable_monitoring:
            return (0, 0.0, 0.0, 0.0, "{}", "Monitoring disabled")
        
        # Initialize FIRST (before opening window)
        if self.start_time is None:
            self.start_time = time.time() if start_time == 0.0 else start_time
            print(f"🎬 Monitoring started at {datetime.now().strftime('%H:%M:%S')}")
        
        # OPEN MONITOR WINDOW (only once, on very first call)
        if separate_window and self.monitor_window is None:
            self.monitor_window = MonitorWindowThread()
            self.monitor_window.start()
            print("🎬 Opening monitor window...")
            time.sleep(1.5)  # Wait for window to fully open
            
            # Send welcome message
            if self.monitor_window.running:
                welcome = "="*78 + "\n"
                welcome += " "*25 + "SEEDVR2 BATCH MONITOR\n"
                welcome += " "*20 + "Monitoring session started\n"
                welcome += "="*78 + "\n\n"
                welcome += "Waiting for first batch data...\n"
                self.monitor_window.write(welcome, clear_first=True)
                print("✅ Monitor window ready")
        
        # Convert strings to ints
        try:
            batch_size = int(batch_size) if batch_size and str(batch_size).strip() else 0
            overlap = int(overlap) if overlap and str(overlap).strip() else 0
            resolution = int(resolution) if resolution and str(resolution).strip() else 0
            total_batches = int(total_batches) if total_batches and str(total_batches).strip() else 0
        except:
            batch_size, overlap, resolution, total_batches = 0, 0, 0, 0
        
        # Auto-detect from folder name
        if self.batch_info is None:
            self.batch_info = self.read_batch_info(output_folder)
            print(f"📊 Detected batch info: {self.batch_info}")
        
        # Use detected or provided values
        actual_batch_size = batch_size if batch_size > 0 else self.batch_info.get('batch_size', 21)
        actual_overlap = overlap if overlap > 0 else self.batch_info.get('overlap', 3)
        actual_total_batches = total_batches if total_batches > 0 else self.batch_info.get('total_batches', 0)
        
        # Get resources
        self.resource_monitor.gpu_index = gpu_index
        vram = self.resource_monitor.get_vram_usage()
        ram = self.resource_monitor.get_ram_usage()
        gpu_stats = self.resource_monitor.get_gpu_stats()
        process = self.resource_monitor.get_process_memory()
        
        # Count frames (use provided value OR count from disk)
        if frames_saved > 0:
            frame_count = frames_saved
        else:
            frame_count = self.count_frames(output_folder)
        
        # Detect current batch
        current_batch = self.detect_current_batch(output_folder)
        if current_batch is None:
            # Estimate from frame count if we can't detect from segments
            if actual_batch_size > 0 and frame_count > 0:
                stride = actual_batch_size - actual_overlap if actual_overlap < actual_batch_size else actual_batch_size
                if stride > 0:
                    current_batch = max(0, (frame_count + stride - 1) // stride)
                else:
                    current_batch = 0
            else:
                current_batch = 0
        
        print(f"[MONITOR] Batch {current_batch}/{actual_total_batches}, Frames: {frame_count}, Window: {self.monitor_window.running if self.monitor_window else 'not started'}")
        
        # Calculate progress
        progress = (current_batch / actual_total_batches * 100) if actual_total_batches > 0 else 0
        elapsed = time.time() - self.start_time
        
        batches_left = actual_total_batches - current_batch if actual_total_batches > 0 else 0
        avg_batch_time = elapsed / current_batch if current_batch > 0 else 0
        eta = avg_batch_time * batches_left if current_batch > 0 else 0
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Build stats
        stats = {
            'current_batch': current_batch,
            'total_batches': actual_total_batches,
            'progress_percent': progress,
            'elapsed_seconds': elapsed,
            'eta_seconds': eta,
            'avg_batch_time': avg_batch_time,
            'fps': fps,
            'frames_saved': frame_count,
            'vram': vram,
            'ram': ram,
            'gpu': gpu_stats,
            'process': process,
        }
        
        # Logging
        if enable_logging and not self.session_started and actual_total_batches > 0:
            self.resource_logger.start_session(
                workflow_name=workflow_name or "seedvr2",
                model_name=model_name,
                batch_size=actual_batch_size,
                overlap=actual_overlap,
                resolution=resolution or 1080,
                total_batches=actual_total_batches
            )
            self.session_started = True
        
        if enable_logging and self.session_started:
            self.resource_logger.log_sample(stats)
        
        if is_complete and self.session_started and not self.session_completed:
            self.resource_logger.end_session(stats)
            self.session_completed = True
        
        # UPDATE DISPLAY (force update every time if we have a window)
        current_time = time.time()
        should_update = (
            (self.monitor_window and self.monitor_window.running) and  # Window is open
            (current_time - self.last_update_time >= self.update_interval or  # Time elapsed
             self.last_update_time == 0)  # First update
        )
        
        if should_update:
            display_text = self.format_status(stats)
            
            # Write to monitor window with clear
            if self.monitor_window and self.monitor_window.running:
                success = self.monitor_window.write(display_text, clear_first=True)
                if not success:
                    print("⚠️  Monitor window is not responding")
            
            # Also print brief update to main console
            if current_time - self.last_update_time >= 10.0 or self.last_update_time == 0:
                print(f"📊 Progress: Batch {current_batch}/{actual_total_batches} ({progress:.1f}%) - ETA: {format_time(eta)}")
            
            self.last_update_time = current_time
        
        # Debug: Always log what we're seeing
        if current_batch > 0 and frames_saved > 0:
            print(f"[DEBUG] Monitor sees: batch={current_batch}/{actual_total_batches}, frames={frame_count}, window_running={self.monitor_window.running if self.monitor_window else False}")
        
        # Completion message
        if is_complete and self.monitor_window and self.monitor_window.running:
            completion = "COMPLETE"  # Signal to monitor script
            self.monitor_window.write(completion)
            time.sleep(0.5)  # Give it time to display
        
        # Return
        vram_used = vram['used'] if vram else 0.0
        vram_total = vram['total'] if vram else 0.0
        status_text = f"Batch {current_batch}/{actual_total_batches} ({progress:.1f}%)"
        
        return (int(eta), float(progress), float(vram_used), float(vram_total), 
                json.dumps(stats, indent=2), status_text)
    
    def format_status(self, stats):
        """Format display text"""
        lines = []
        lines.append("="*78)
        lines.append(" "*28 + "SEEDVR2 BATCH MONITOR")
        lines.append("="*78)
        lines.append("")
        
        # Progress
        cur = stats['current_batch']
        tot = stats['total_batches']
        pct = stats['progress_percent']
        bar = create_progress_bar(pct, 30)
        lines.append(f"Progress:  Batch {cur}/{tot} ({pct:.1f}%)")
        lines.append(f"           {bar}")
        lines.append("")
        
        # Time
        lines.append(f"Elapsed:   {format_time(stats['elapsed_seconds'])}")
        lines.append(f"ETA:       {format_time(stats['eta_seconds'])} (at {format_completion_time(stats['eta_seconds'])})")
        lines.append(f"Speed:     {stats['fps']:.2f} fps | {stats['avg_batch_time']:.1f}s/batch")
        lines.append("")
        
        # Resources
        lines.append("RESOURCES:")
        vram = stats.get('vram')
        if vram:
            vbar = create_progress_bar(vram['percent'], 20)
            lines.append(f"  VRAM: {vram['used']:5.1f}/{vram['total']:5.1f} GB  {vbar} {vram['percent']:5.1f}%")
        
        ram = stats.get('ram')
        if ram:
            rbar = create_progress_bar(ram['percent'], 20)
            lines.append(f"  RAM:  {ram['used']:5.1f}/{ram['total']:5.1f} GB  {rbar} {ram['percent']:5.1f}%")
            if 'cached' in ram:
                lines.append(f"    -> Cached: {ram['cached']:5.1f} GB")
                lines.append(f"    -> Active: {ram['used_excluding_cache']:5.1f} GB")
        
        proc = stats.get('process')
        if proc:
            lines.append(f"  ComfyUI: {proc['rss_gb']:5.1f} GB")
        
        gpu = stats.get('gpu')
        if gpu:
            lines.append(f"  GPU: {gpu['utilization']:3d}% util | {gpu['temperature']:3d}°C")
        
        lines.append("")
        lines.append(f"Frames: {stats['frames_saved']} saved")
        lines.append("="*78)
        
        return "\n".join(lines)
    
    def read_batch_info(self, output_folder):
        """Auto-detect batch info from folder name"""
        import re
        info = {'batch_size': 21, 'overlap': 3, 'total_batches': 0}
        if not output_folder:
            return info
        
        folder_name = os.path.basename(output_folder)
        match = re.search(r'_b(\d+)_o(\d+)_', folder_name)
        if match:
            info['batch_size'] = int(match.group(1))
            info['overlap'] = int(match.group(2))
            print(f"📊 Auto-detected: batch_size={info['batch_size']}, overlap={info['overlap']}")
        return info
    
    def count_frames(self, output_folder):
        """Count frames in folder"""
        if not output_folder or not os.path.exists(output_folder):
            return 0
        total = 0
        for ext in ["*.png", "*.webp", "*.jpg"]:
            total += len(glob.glob(os.path.join(output_folder, ext)))
        return total
    
    def detect_current_batch(self, output_folder):
        """Detect current batch from segments folder"""
        if not output_folder or not os.path.exists(output_folder):
            return None
        
        segments_folder = os.path.join(output_folder, "segments")
        if os.path.exists(segments_folder):
            segment_folders = [d for d in os.listdir(segments_folder) 
                             if os.path.isdir(os.path.join(segments_folder, d))]
            if segment_folders:
                return len(segment_folders)
        return None


# Node registration
NODE_CLASS_MAPPINGS = {
    "BatchProgressMonitor": BatchProgressMonitor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchProgressMonitor": "Batch Progress Monitor (SeedVR2)",
}
