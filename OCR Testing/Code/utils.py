"""
Utility Functions for LaTeX-OCR Benchmarking
Common utilities, logging setup, and helper functions
"""

import logging
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import hashlib

def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """Set up comprehensive logging for the benchmark system."""
    
    # Configure root logger
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()  # Remove any existing handlers
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always debug level for files
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress some noisy loggers
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    if not verbose:
        logging.getLogger('transformers').setLevel(logging.WARNING)
        logging.getLogger('torch').setLevel(logging.WARNING)

def validate_paths(images_dir: str, ground_truth_file: str, output_dir: str) -> Tuple[bool, List[str]]:
    """Validate that all required paths exist and are accessible."""
    errors = []
    
    # Check images directory
    if not os.path.exists(images_dir):
        errors.append(f"Images directory does not exist: {images_dir}")
    elif not os.path.isdir(images_dir):
        errors.append(f"Images path is not a directory: {images_dir}")
    else:
        # Check if directory contains images
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
        image_files = [f for f in os.listdir(images_dir) 
                      if os.path.splitext(f.lower())[1] in image_extensions]
        if not image_files:
            errors.append(f"No image files found in directory: {images_dir}")
    
    # Check ground truth file
    if not os.path.exists(ground_truth_file):
        errors.append(f"Ground truth file does not exist: {ground_truth_file}")
    elif not os.path.isfile(ground_truth_file):
        errors.append(f"Ground truth path is not a file: {ground_truth_file}")
    else:
        # Validate JSON format
        try:
            with open(ground_truth_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    errors.append("Ground truth file must contain a JSON object (dictionary)")
                elif not data:
                    errors.append("Ground truth file is empty")
        except json.JSONDecodeError as e:
            errors.append(f"Ground truth file is not valid JSON: {e}")
        except Exception as e:
            errors.append(f"Error reading ground truth file: {e}")
    
    # Check output directory (create if doesn't exist)
    try:
        os.makedirs(output_dir, exist_ok=True)
        # Test write permissions
        test_file = os.path.join(output_dir, '.write_test')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
    except Exception as e:
        errors.append(f"Cannot write to output directory: {output_dir} - {e}")
    
    return len(errors) == 0, errors

def load_and_validate_ground_truth(ground_truth_file: str, images_dir: str) -> Tuple[Dict[str, str], List[str]]:
    """Load ground truth and validate against available images."""
    try:
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
    except Exception as e:
        return {}, [f"Failed to load ground truth: {e}"]
    
    warnings = []
    validated_ground_truth = {}
    
    # Get available image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
    available_images = set()
    
    for filename in os.listdir(images_dir):
        if os.path.splitext(filename.lower())[1] in image_extensions:
            available_images.add(filename)
    
    # Validate each ground truth entry
    for image_name, latex_code in ground_truth.items():
        if not isinstance(latex_code, str):
            warnings.append(f"Ground truth for {image_name} is not a string")
            continue
        
        if image_name not in available_images:
            warnings.append(f"Image file not found: {image_name}")
            continue
        
        if not latex_code.strip():
            warnings.append(f"Empty ground truth for: {image_name}")
            continue
        
        validated_ground_truth[image_name] = latex_code.strip()
    
    # Check for images without ground truth
    images_without_gt = available_images - set(ground_truth.keys())
    if images_without_gt:
        warnings.append(f"Images without ground truth: {sorted(list(images_without_gt))}")
    
    return validated_ground_truth, warnings

def generate_dataset_summary(images_dir: str, ground_truth: Dict[str, str]) -> Dict[str, Any]:
    """Generate comprehensive dataset summary statistics."""
    summary = {
        'total_images': len(ground_truth),
        'dataset_path': images_dir,
        'generation_time': datetime.now().isoformat()
    }
    
    if not ground_truth:
        return summary
    
    # Analyze ground truth LaTeX strings
    latex_strings = list(ground_truth.values())
    lengths = [len(s) for s in latex_strings]
    
    summary.update({
        'latex_statistics': {
            'total_characters': sum(lengths),
            'avg_length': sum(lengths) / len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'median_length': sorted(lengths)[len(lengths) // 2]
        }
    })
    
    # Analyze LaTeX command usage
    command_counts = {}
    structure_counts = {'braces': 0, 'brackets': 0, 'parentheses': 0}
    
    for latex_str in latex_strings:
        # Count LaTeX commands
        import re
        commands = re.findall(r'\\[a-zA-Z]+', latex_str)
        for cmd in commands:
            command_counts[cmd] = command_counts.get(cmd, 0) + 1
        
        # Count structural elements
        structure_counts['braces'] += latex_str.count('{')
        structure_counts['brackets'] += latex_str.count('[')  
        structure_counts['parentheses'] += latex_str.count('(')
    
    # Get top 10 most common commands
    top_commands = sorted(command_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    summary.update({
        'latex_analysis': {
            'total_unique_commands': len(command_counts),
            'top_commands': top_commands,
            'structure_elements': structure_counts,
            'avg_commands_per_expression': sum(command_counts.values()) / len(latex_strings)
        }
    })
    
    # Analyze image files
    image_info = analyze_image_directory(images_dir, list(ground_truth.keys()))
    summary['image_analysis'] = image_info
    
    return summary

def analyze_image_directory(images_dir: str, image_names: List[str]) -> Dict[str, Any]:
    """Analyze image files in the dataset directory."""
    image_analysis = {
        'total_files': len(image_names),
        'file_extensions': {},
        'total_size_mb': 0,
        'avg_size_mb': 0
    }
    
    sizes = []
    
    try:
        from PIL import Image
        PIL_AVAILABLE = True
        dimensions = []
    except ImportError:
        PIL_AVAILABLE = False
        dimensions = []
    
    for image_name in image_names:
        image_path = os.path.join(images_dir, image_name)
        
        if os.path.exists(image_path):
            # File extension
            ext = os.path.splitext(image_name.lower())[1]
            image_analysis['file_extensions'][ext] = image_analysis['file_extensions'].get(ext, 0) + 1
            
            # File size
            size_bytes = os.path.getsize(image_path)
            size_mb = size_bytes / (1024 * 1024)
            sizes.append(size_mb)
            image_analysis['total_size_mb'] += size_mb
            
            # Image dimensions (if PIL available)
            if PIL_AVAILABLE:
                try:
                    with Image.open(image_path) as img:
                        dimensions.append(img.size)  # (width, height)
                except Exception:
                    pass  # Skip problematic images
    
    if sizes:
        image_analysis.update({
            'avg_size_mb': sum(sizes) / len(sizes),
            'min_size_mb': min(sizes),
            'max_size_mb': max(sizes)
        })
    
    if dimensions and PIL_AVAILABLE:
        widths = [d[0] for d in dimensions]
        heights = [d[1] for d in dimensions]
        
        image_analysis['dimensions'] = {
            'avg_width': sum(widths) / len(widths),
            'avg_height': sum(heights) / len(heights),
            'min_width': min(widths),
            'max_width': max(widths),
            'min_height': min(heights),
            'max_height': max(heights),
            'aspect_ratios': [w/h for w, h in dimensions if h > 0]
        }
    
    return image_analysis

def create_sample_ground_truth(images_dir: str, output_file: str, sample_entries: int = 5):
    """Create a sample ground truth file for testing purposes."""
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
    image_files = []
    
    for filename in os.listdir(images_dir):
        if os.path.splitext(filename.lower())[1] in image_extensions:
            image_files.append(filename)
    
    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")
    
    # Create sample entries with placeholder LaTeX
    sample_latex_expressions = [
        r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}",
        r"\int_{0}^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}",
        r"\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}",
        r"E = mc^2",
        r"\lim_{x \to 0} \frac{\sin x}{x} = 1",
        r"\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}",
        r"f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x-a)^n",
        r"\mathbf{F} = m\mathbf{a}",
        r"\left| \sum_{i=1}^n a_i b_i \right| \leq \sqrt{\sum_{i=1}^n a_i^2} \sqrt{\sum_{i=1}^n b_i^2}",
        r"\frac{d}{dx} \ln(x) = \frac{1}{x}"
    ]
    
    # Select images and create ground truth
    selected_images = image_files[:min(sample_entries, len(image_files))]
    ground_truth = {}
    
    for i, image_name in enumerate(selected_images):
        latex_expr = sample_latex_expressions[i % len(sample_latex_expressions)]
        ground_truth[image_name] = latex_expr
    
    # Save ground truth file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)
    
    print(f"Created sample ground truth file: {output_file}")
    print(f"Contains {len(ground_truth)} entries")
    print("⚠️  Note: This contains placeholder LaTeX expressions!")
    print("Please replace with actual ground truth for your images.")
    
    return ground_truth

def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of a file for integrity checking."""
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logging.error(f"Failed to calculate hash for {file_path}: {e}")
        return ""

def format_time_duration(seconds: float) -> str:
    """Format time duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"

def format_memory_size(size_mb: float) -> str:
    """Format memory size in human-readable format."""
    if size_mb < 1024:
        return f"{size_mb:.1f} MB"
    else:
        size_gb = size_mb / 1024
        return f"{size_gb:.2f} GB"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    return numerator / denominator if denominator != 0 else default

def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information for benchmarking context."""
    import platform
    import psutil
    
    system_info = {
        'platform': platform.platform(),
        'system': platform.system(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'cpu_count_physical': psutil.cpu_count(logical=False),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'timestamp': datetime.now().isoformat()
    }
    
    # Add GPU information if available
    try:
        import torch
        if torch.cuda.is_available():
            system_info['gpu_available'] = True
            system_info['gpu_count'] = torch.cuda.device_count()
            system_info['gpu_name'] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'Unknown'
        else:
            system_info['gpu_available'] = False
    except ImportError:
        system_info['gpu_available'] = False
    
    return system_info

def create_benchmark_metadata(images_dir: str, ground_truth_file: str, 
                             models_tested: List[str]) -> Dict[str, Any]:
    """Create comprehensive metadata for benchmark run."""
    metadata = {
        'benchmark_id': hashlib.md5(
            f"{images_dir}_{ground_truth_file}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16],
        'timestamp': datetime.now().isoformat(),
        'dataset': {
            'images_dir': images_dir,
            'ground_truth_file': ground_truth_file,
            'ground_truth_hash': calculate_file_hash(ground_truth_file)
        },
        'models_tested': models_tested,
        'system_info': get_system_info(),
        'environment': {
            'working_directory': os.getcwd(),
            'command_line': ' '.join(sys.argv)
        }
    }
    
    return metadata

def print_benchmark_header(models: List[str], dataset_info: Dict[str, Any]):
    """Print formatted benchmark header with key information."""
    print("\n" + "="*80)
    print("🔤 LATEX-OCR MULTI-MODEL BENCHMARKING SYSTEM")
    print("="*80)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖼️  Dataset: {dataset_info.get('total_images', 0)} images")
    print(f"🤖 Models: {', '.join(models)}")
    print(f"💻 System: {dataset_info.get('system', 'Unknown')}")
    print(f"🔧 Device: {'GPU' if dataset_info.get('gpu_available', False) else 'CPU'}")
    print("="*80)

def print_benchmark_footer(total_time: float, total_predictions: int):
    """Print formatted benchmark completion summary."""
    print("\n" + "="*80)
    print("✅ BENCHMARKING COMPLETED")
    print("="*80)
    print(f"⏱️  Total Time: {format_time_duration(total_time)}")
    print(f"📊 Total Predictions: {total_predictions}")
    print(f"⚡ Average Speed: {safe_divide(total_predictions, total_time, 0):.2f} predictions/second")
    print(f"🏁 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

class ProgressTracker:
    """Simple progress tracking for benchmark operations."""
    
    def __init__(self, total_items: int, description: str = "Processing"):
        self.total_items = total_items
        self.current_item = 0
        self.description = description
        self.start_time = time.time()
        self.last_update = 0
    
    def update(self, increment: int = 1):
        """Update progress counter."""
        self.current_item += increment
        current_time = time.time()
        
        # Update every second or on completion
        if current_time - self.last_update > 1.0 or self.current_item == self.total_items:
            self._print_progress()
            self.last_update = current_time
    
    def _print_progress(self):
        """Print current progress."""
        if self.total_items == 0:
            return
        
        percentage = (self.current_item / self.total_items) * 100
        elapsed_time = time.time() - self.start_time
        
        if self.current_item > 0:
            avg_time_per_item = elapsed_time / self.current_item
            estimated_remaining = (self.total_items - self.current_item) * avg_time_per_item
            remaining_str = f" | ETA: {format_time_duration(estimated_remaining)}"
        else:
            remaining_str = ""
        
        print(f"\r{self.description}: {self.current_item}/{self.total_items} "
              f"({percentage:.1f}%) | Elapsed: {format_time_duration(elapsed_time)}"
              f"{remaining_str}", end="", flush=True)
        
        if self.current_item == self.total_items:
            print()  # New line when complete

def check_dependencies() -> Dict[str, bool]:
    """Check availability of all required and optional dependencies."""
    dependencies = {
        'required': {
            'PIL': 'PIL (Pillow)',
            'torch': 'PyTorch',
            'numpy': 'NumPy',
            'pandas': 'Pandas',
            'psutil': 'psutil',
            'Levenshtein': 'python-Levenshtein'
        },
        'optional': {
            'pix2tex': 'Pix2Tex',
            'transformers': 'Transformers',
            'GPUtil': 'GPUtil',
            'cv2': 'OpenCV'
        }
    }
    
    results = {'required': {}, 'optional': {}, 'all_required_available': True}
    
    for category, deps in dependencies.items():
        for module, description in deps.items():
            try:
                __import__(module)
                results[category][description] = True
            except ImportError:
                results[category][description] = False
                if category == 'required':
                    results['all_required_available'] = False
    
    return results

def print_dependency_status():
    """Print status of all dependencies."""
    deps = check_dependencies()
    
    print("\n📦 DEPENDENCY STATUS")
    print("-" * 50)
    
    print("Required Dependencies:")
    for dep, available in deps['required'].items():
        status = "✅" if available else "❌"
        print(f"  {status} {dep}")
    
    print("\nOptional Dependencies:")
    for dep, available in deps['optional'].items():
        status = "✅" if available else "⚠️ "
        print(f"  {status} {dep}")
    
    if not deps['all_required_available']:
        print("\n❌ Some required dependencies are missing!")
        print("Please install missing dependencies before running benchmarks.")
        return False
    else:
        print("\n✅ All required dependencies are available!")
        return True

def validate_model_config(model_name: str, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate model configuration."""
    errors = []
    
    # Basic validation
    if not isinstance(config, dict):
        errors.append("Configuration must be a dictionary")
        return False, errors
    
    # Model-specific validation
    if model_name == 'pix2tex':
        # Pix2Tex doesn't require special config
        pass
    
    elif model_name == 'im2markup':
        required_keys = ['model_path', 'vocab_path']
        for key in required_keys:
            if key not in config:
                errors.append(f"Missing required config key: {key}")
            elif not os.path.exists(config[key]):
                errors.append(f"Path does not exist: {config[key]}")
    
    elif model_name == 'hmer':
        required_keys = ['model_path', 'config_path']
        for key in required_keys:
            if key not in config:
                errors.append(f"Missing required config key: {key}")
            elif not os.path.exists(config[key]):
                errors.append(f"Path does not exist: {config[key]}")
    
    elif model_name in ['densenet', 'transformer']:
        if 'model_path' in config and not os.path.exists(config['model_path']):
            errors.append(f"Model path does not exist: {config['model_path']}")
    
    return len(errors) == 0, errors

def cleanup_gpu_memory():
    """Clean up GPU memory across different frameworks."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
    
    # Add cleanup for other frameworks if needed
    try:
        import gc
        gc.collect()
    except ImportError:
        pass

def estimate_memory_requirements(image_count: int, model_count: int = 1) -> Dict[str, float]:
    """Estimate memory requirements for benchmarking."""
    # Rough estimates based on typical usage
    base_memory_mb = 100  # Base Python + libraries
    per_image_mb = 5      # Average memory per image processing
    per_model_mb = 500    # Average model loading memory
    
    estimated = {
        'base_memory_mb': base_memory_mb,
        'model_memory_mb': per_model_mb * model_count,
        'processing_memory_mb': per_image_mb * min(image_count, 10),  # Assume batch processing
        'total_estimated_mb': base_memory_mb + (per_model_mb * model_count) + (per_image_mb * min(image_count, 10))
    }
    
    return estimated

def create_directory_structure(base_dir: str) -> Dict[str, str]:
    """Create standardized directory structure for benchmarking."""
    directories = {
        'base': base_dir,
        'results': os.path.join(base_dir, 'results'),
        'logs': os.path.join(base_dir, 'logs'),
        'models': os.path.join(base_dir, 'models'),
        'datasets': os.path.join(base_dir, 'datasets'),
        'configs': os.path.join(base_dir, 'configs'),
        'temp': os.path.join(base_dir, 'temp')
    }
    
    for dir_type, dir_path in directories.items():
        os.makedirs(dir_path, exist_ok=True)
        
        # Create README files for key directories
        if dir_type in ['results', 'models', 'datasets']:
            readme_path = os.path.join(dir_path, 'README.md')
            if not os.path.exists(readme_path):
                with open(readme_path, 'w') as f:
                    if dir_type == 'results':
                        f.write("# Benchmark Results\n\nThis directory contains benchmark results and reports.\n")
                    elif dir_type == 'models':
                        f.write("# Model Files\n\nPlace model weights and configuration files here.\n")
                    elif dir_type == 'datasets':
                        f.write("# Datasets\n\nPlace dataset images and ground truth files here.\n")
    
    return directories