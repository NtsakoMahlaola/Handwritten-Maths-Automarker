#!/usr/bin/env python3
"""
Main Benchmark Controller for LaTeX-OCR Models
Orchestrates comprehensive benchmarking across multiple models
"""

import argparse
import sys
import os
from pathlib import Path

from config_manager import ConfigManager
from benchmark_runner import BenchmarkRunner
from model_factory import ModelFactory
from utils import setup_logging

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive LaTeX-OCR Multi-Model Benchmarking Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --model pix2tex --profile quick_test
  python main.py --model latex-ocr --images-dir ./test_imgs --ground-truth ./gt.json
  python main.py --model all --profile full_benchmark
  python main.py --list-profiles
  python main.py --list-models
  python main.py --add-profile test:./imgs:./gt.json:./results
        """
    )
    
    # Model selection
    parser.add_argument('--model', type=str, default='pix2tex',
                       help='Model to benchmark (pix2tex, pix2text-mfr-onnx, TrOCRTunedModel,latex-finetuned, trocr-vision2seq-math, trocr-math, trocr-large-handwritten, latex-ocr,trocr, im2markup, hmer, densenet, transformer, all)')
    parser.add_argument('--list-models', action='store_true',
                       help='List all available models and their status')
    
    # Profile management
    parser.add_argument('--profile', type=str, 
                       help='Use a saved profile for benchmark settings')
    parser.add_argument('--list-profiles', action='store_true', 
                       help='List all available profiles')
    parser.add_argument('--add-profile', type=str, 
                       help='Add profile: NAME:IMAGES_DIR:GROUND_TRUTH:OUTPUT_DIR')
    
    # Direct path specification
    parser.add_argument('--images-dir', type=str, 
                       help='Directory containing test images')
    parser.add_argument('--ground-truth', type=str, 
                       help='JSON file with ground truth LaTeX')
    parser.add_argument('--output-dir', type=str, 
                       help='Directory to save results')
    
    # Configuration
    parser.add_argument('--config', type=str, default='benchmark_config.json',
                       help='Config file to use')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    # Benchmark options
    parser.add_argument('--compare-models', action='store_true',
                       help='Generate cross-model comparison report')
    parser.add_argument('--quick-mode', action='store_true',
                       help='Run in quick mode (reduced metrics)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    
    # Initialize configuration manager
    config_manager = ConfigManager(args.config)
    
    # Handle listing commands
    if args.list_profiles:
        config_manager.list_profiles()
        return 0
    
    if args.list_models:
        ModelFactory.list_available_models()
        return 0
    
    # Handle profile management
    if args.add_profile:
        parts = args.add_profile.split(':', 3)
        if len(parts) == 4:
            config_manager.add_profile(parts[0], parts[1], parts[2], parts[3])
            return 0
        else:
            print("Error: Use format --add-profile name:images_dir:ground_truth:output_dir")
            return 1
    
    # Determine benchmark settings
    try:
        images_dir, ground_truth, output_dir = config_manager.get_benchmark_settings(
            profile=args.profile,
            images_dir=args.images_dir,
            ground_truth=args.ground_truth,
            output_dir=args.output_dir
        )
    except Exception as e:
        print(f"❌ Configuration Error: {e}")
        return 1
    
    # Validate paths
    if not Path(images_dir).exists():
        print(f"❌ Error: Images directory not found: {images_dir}")
        return 1
    
    if not Path(ground_truth).exists():
        print(f"❌ Error: Ground truth file not found: {ground_truth}")
        return 1
    
    # Initialize benchmark runner
    benchmark_runner = BenchmarkRunner(
        quick_mode=args.quick_mode,
        verbose=args.verbose
    )
    
    # Run benchmarks
    try:
        if args.model.lower() == 'all':
            # Benchmark all available models
            results = benchmark_runner.run_all_models(
                images_dir=images_dir,
                ground_truth_file=ground_truth,
                output_dir=output_dir
            )
            
            if args.compare_models:
                benchmark_runner.generate_comparison_report(results, output_dir)
                
        else:
            # Benchmark single model
            results = benchmark_runner.run_single_model(
                model_name=args.model,
                images_dir=images_dir,
                ground_truth_file=ground_truth,
                output_dir=output_dir
            )
        
        print("\n✅ Benchmarking completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Benchmarking interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Benchmarking failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
