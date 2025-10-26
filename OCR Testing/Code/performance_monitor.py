"""
Performance Monitor for LaTeX-OCR Benchmarking
Tracks CPU, memory, GPU usage during inference
"""

import psutil
import time
import logging
from typing import Dict, Optional, Any
import threading
from collections import deque

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    GPUtil = None

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Comprehensive system performance monitoring during inference."""
    
    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self.process = psutil.Process()
        
    def start_monitoring(self) -> Dict[str, Any]:
        """Start performance monitoring and return initial state."""
        initial_state = {
            'start_time': time.time(),
            'initial_memory': self.process.memory_info().rss / 1024 / 1024,  # MB
            'initial_cpu_percent': self.process.cpu_percent(),
            'system_cpu_count': psutil.cpu_count(),
            'system_memory_total': psutil.virtual_memory().total / 1024 / 1024,  # MB
        }
        
        # GPU monitoring if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            initial_state.update({
                'gpu_available': True,
                'gpu_count': torch.cuda.device_count(),
                'initial_gpu_memory_allocated': torch.cuda.memory_allocated() / 1024 / 1024,  # MB
                'initial_gpu_memory_cached': torch.cuda.memory_reserved() / 1024 / 1024,  # MB
                'gpu_device_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'Unknown'
            })
        else:
            initial_state['gpu_available'] = False
        
        # Start continuous monitoring if needed
        initial_state['monitoring_data'] = {
            'cpu_samples': deque(maxlen=100),
            'memory_samples': deque(maxlen=100),
            'timestamp_samples': deque(maxlen=100)
        }
        
        return initial_state
    
    def stop_monitoring(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Stop monitoring and calculate performance statistics."""
        end_time = time.time()
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        final_cpu_percent = self.process.cpu_percent()
        
        # Calculate basic metrics
        performance_stats = {
            'total_time_seconds': end_time - initial_state['start_time'],
            'memory_used_mb': final_memory - initial_state['initial_memory'],
            'final_memory_mb': final_memory,
            'cpu_usage_before': initial_state['initial_cpu_percent'],
            'cpu_usage_after': final_cpu_percent,
            'cpu_usage_delta': final_cpu_percent - initial_state['initial_cpu_percent'],
            'device': 'GPU' if initial_state.get('gpu_available', False) else 'CPU',
            'system_info': {
                'cpu_count': initial_state['system_cpu_count'],
                'total_memory_mb': initial_state['system_memory_total']
            }
        }
        
        # Add GPU metrics if available
        if initial_state.get('gpu_available', False):
            try:
                final_gpu_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                final_gpu_cached = torch.cuda.memory_reserved() / 1024 / 1024  # MB
                
                performance_stats.update({
                    'gpu_memory_allocated_mb': final_gpu_allocated,
                    'gpu_memory_cached_mb': final_gpu_cached,
                    'gpu_memory_used_mb': final_gpu_allocated - initial_state['initial_gpu_memory_allocated'],
                    'gpu_device_name': initial_state['gpu_device_name'],
                    'gpu_count': initial_state['gpu_count']
                })
                
                # Get GPU utilization if GPUtil is available
                if GPUTIL_AVAILABLE:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]  # Use first GPU
                            performance_stats.update({
                                'gpu_utilization_percent': gpu.load * 100,
                                'gpu_memory_utilization_percent': gpu.memoryUtil * 100,
                                'gpu_temperature_c': gpu.temperature
                            })
                    except Exception as e:
                        logger.debug(f"Could not get GPU utilization: {e}")
                
            except Exception as e:
                logger.error(f"Error collecting GPU metrics: {e}")
        
        # Add system-wide metrics
        system_memory = psutil.virtual_memory()
        performance_stats['system_memory_usage'] = {
            'percent': system_memory.percent,
            'available_mb': system_memory.available / 1024 / 1024,
            'used_mb': system_memory.used / 1024 / 1024
        }
        
        return performance_stats
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current system performance statistics."""
        stats = {
            'timestamp': time.time(),
            'cpu_percent': self.process.cpu_percent(),
            'memory_mb': self.process.memory_info().rss / 1024 / 1024,
            'system_cpu_percent': psutil.cpu_percent(),
            'system_memory_percent': psutil.virtual_memory().percent
        }
        
        # Add GPU stats if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                stats.update({
                    'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                    'gpu_memory_cached_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                })
                
                if GPUTIL_AVAILABLE:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        stats.update({
                            'gpu_utilization_percent': gpu.load * 100,
                            'gpu_memory_utilization_percent': gpu.memoryUtil * 100
                        })
            except Exception as e:
                logger.debug(f"Could not collect GPU stats: {e}")
        
        return stats
    
    def monitor_continuously(self, duration: float, callback: Optional[callable] = None) -> Dict[str, Any]:
        """Monitor system performance continuously for a specified duration."""
        start_time = time.time()
        samples = {
            'timestamps': [],
            'cpu_percent': [],
            'memory_mb': [],
            'system_cpu_percent': [],
            'system_memory_percent': []
        }
        
        # Add GPU monitoring arrays if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            samples.update({
                'gpu_memory_allocated_mb': [],
                'gpu_memory_cached_mb': []
            })
            
            if GPUTIL_AVAILABLE:
                samples.update({
                    'gpu_utilization_percent': [],
                    'gpu_memory_utilization_percent': []
                })
        
        while time.time() - start_time < duration:
            current_stats = self.get_current_stats()
            
            # Store samples
            samples['timestamps'].append(current_stats['timestamp'])
            samples['cpu_percent'].append(current_stats['cpu_percent'])
            samples['memory_mb'].append(current_stats['memory_mb'])
            samples['system_cpu_percent'].append(current_stats['system_cpu_percent'])
            samples['system_memory_percent'].append(current_stats['system_memory_percent'])
            
            # Store GPU samples if available
            if 'gpu_memory_allocated_mb' in current_stats:
                samples['gpu_memory_allocated_mb'].append(current_stats['gpu_memory_allocated_mb'])
                samples['gpu_memory_cached_mb'].append(current_stats['gpu_memory_cached_mb'])
            
            if 'gpu_utilization_percent' in current_stats:
                samples['gpu_utilization_percent'].append(current_stats['gpu_utilization_percent'])
                samples['gpu_memory_utilization_percent'].append(current_stats['gpu_memory_utilization_percent'])
            
            # Call callback if provided
            if callback:
                callback(current_stats)
            
            time.sleep(self.sampling_interval)
        
        # Calculate summary statistics
        summary = self._calculate_monitoring_summary(samples)
        return {
            'samples': samples,
            'summary': summary,
            'duration': time.time() - start_time
        }
    
    def _calculate_monitoring_summary(self, samples: Dict[str, list]) -> Dict[str, Any]:
        """Calculate summary statistics from monitoring samples."""
        summary = {}
        
        # Calculate stats for numeric arrays
        for key, values in samples.items():
            if key == 'timestamps' or not values:
                continue
            
            try:
                summary[key] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'median': sorted(values)[len(values) // 2] if values else 0
                }
            except (TypeError, ZeroDivisionError):
                logger.debug(f"Could not calculate summary for {key}")
        
        return summary
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        system_info = {
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'cpu_freq_current': psutil.cpu_freq().current if psutil.cpu_freq() else 'Unknown',
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'platform': psutil.Platform if hasattr(psutil, 'Platform') else 'Unknown'
        }
        
        # GPU information
        if TORCH_AVAILABLE and torch.cuda.is_available():
            system_info['gpu_available'] = True
            system_info['gpu_count'] = torch.cuda.device_count()
            system_info['gpu_devices'] = []
            
            for i in range(torch.cuda.device_count()):
                gpu_props = torch.cuda.get_device_properties(i)
                system_info['gpu_devices'].append({
                    'name': gpu_props.name,
                    'memory_gb': gpu_props.total_memory / (1024**3),
                    'compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
                })
        else:
            system_info['gpu_available'] = False
        
        return system_info
    
    def check_resource_limits(self, memory_limit_mb: float = 2048,
                            cpu_limit_percent: float = 95.0) -> Dict[str, bool]:
        """Check if system resources are within specified limits."""
        current_stats = self.get_current_stats()
        
        return {
            'memory_ok': current_stats['memory_mb'] < memory_limit_mb,
            'cpu_ok': current_stats['system_cpu_percent'] < cpu_limit_percent,
            'system_memory_ok': current_stats['system_memory_percent'] < 90.0,
            'current_memory_mb': current_stats['memory_mb'],
            'current_cpu_percent': current_stats['system_cpu_percent'],
            'current_system_memory_percent': current_stats['system_memory_percent']
        }
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory if available."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.debug("GPU memory cleanup completed")
            except Exception as e:
                logger.debug(f"GPU memory cleanup failed: {e}")
