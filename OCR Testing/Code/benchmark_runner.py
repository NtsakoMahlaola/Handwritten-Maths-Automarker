"""
Benchmark Runner for LaTeX-OCR Models
Core benchmarking logic and orchestration
"""

import os
import json
import time
import psutil
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import torch
import numpy as np
import pandas as pd

from model_factory import ModelFactory, BaseLatexOCRModel
from metrics_calculator import MetricsCalculator
from performance_monitor import PerformanceMonitor
from results_manager import ResultsManager

logger = logging.getLogger(__name__)

class BenchmarkRunner:
    """Main benchmark orchestrator for LaTeX-OCR models."""
    
    def __init__(self, quick_mode: bool = False, verbose: bool = False):
        self.quick_mode = quick_mode
        self.verbose = verbose
        self.metrics_calculator = MetricsCalculator()
        self.performance_monitor = PerformanceMonitor()
        self.results_manager = ResultsManager()
        
    def load_ground_truth(self, ground_truth_file: str) -> Dict[str, str]:
        """Load ground truth LaTeX from JSON file."""
        if not os.path.exists(ground_truth_file):
            raise FileNotFoundError(f"Ground truth file not found: {ground_truth_file}")
        
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        
        logger.info(f"Loaded ground truth for {len(ground_truth)} images")
        return ground_truth
    
    def run_single_model(self, model_name: str, images_dir: str, 
                        ground_truth_file: str, output_dir: str) -> Dict[str, Any]:
        """Run benchmark on a single model."""
        logger.info(f"Starting benchmark for model: {model_name}")
        
        # Create model instance
        model = ModelFactory.create_model(model_name)
        if not model:
            raise ValueError(f"Failed to create model: {model_name}")
        
        # Initialize model
        if not model.initialize():
            raise RuntimeError(f"Failed to initialize model: {model_name}")
        
        try:
            # Load ground truth
            ground_truth = self.load_ground_truth(ground_truth_file)
            
            # Run benchmark
            results = self._benchmark_model(
                model=model,
                images_dir=images_dir,
                ground_truth=ground_truth,
                model_name=model_name
            )
            
            # Save results
            self.results_manager.save_model_results(
                results=results,
                model_name=model_name,
                output_dir=output_dir
            )
            
            return results
            
        finally:
            # Clean up model resources
            model.cleanup()
    
    def run_all_models(self, images_dir: str, ground_truth_file: str, 
                      output_dir: str) -> Dict[str, Any]:
        """Run benchmark on all available models."""
        logger.info("Starting benchmark for all available models")
        
        working_models = ModelFactory.get_working_models()
        if not working_models:
            raise RuntimeError("No working models found. Please install dependencies.")
        
        logger.info(f"Found {len(working_models)} working models: {working_models}")
        
        all_results = {}
        ground_truth = self.load_ground_truth(ground_truth_file)
        
        for model_name in working_models:
            logger.info(f"\n{'='*60}")
            logger.info(f"Benchmarking model: {model_name.upper()}")
            logger.info(f"{'='*60}")
            
            try:
                # Create and initialize model
                model = ModelFactory.create_model(model_name)
                if not model or not model.initialize():
                    logger.error(f"Failed to initialize {model_name}, skipping...")
                    continue
                
                # Run benchmark
                results = self._benchmark_model(
                    model=model,
                    images_dir=images_dir,
                    ground_truth=ground_truth,
                    model_name=model_name
                )
                
                all_results[model_name] = results
                
                # Save individual model results
                self.results_manager.save_model_results(
                    results=results,
                    model_name=model_name,
                    output_dir=output_dir
                )
                
                # Clean up
                model.cleanup()
                
            except Exception as e:
                logger.error(f"Error benchmarking {model_name}: {e}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
                continue
        
        return all_results
    
    def _benchmark_model(self, model: BaseLatexOCRModel, images_dir: str,
                        ground_truth: Dict[str, str], model_name: str) -> Dict[str, Any]:
        """Run comprehensive benchmark on a single model."""
        results = {
            'model_name': model_name,
            'benchmark_start': datetime.now().isoformat(),
            'predictions': [],
            'summary': {},
            'failed_images': []
        }
        
        total_images = len(ground_truth)
        processed_count = 0
        failed_count = 0
        
        print(f"\nBenchmarking {model_name} on {total_images} images...")
        print(f"Images directory: {images_dir}")
        print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        print("=" * 70)
        
        # Performance tracking
        total_inference_time = 0
        total_memory_used = 0
        
        for image_name, true_latex in ground_truth.items():
            image_path = os.path.join(images_dir, image_name)
            
            if not os.path.exists(image_path):
                logger.warning(f"Image {image_name} not found, skipping...")
                results['failed_images'].append({
                    'image_name': image_name,
                    'reason': 'File not found',
                    'ground_truth': true_latex
                })
                continue
            
            print(f"\nProcessing: {image_name}")
            if self.verbose:
                print(f"Ground truth: {true_latex}")
            
            # Predict with performance monitoring
            prediction_result = self._predict_with_monitoring(
                model=model,
                image_path=image_path,
                image_name=image_name
            )
            
            if prediction_result is None:
                failed_count += 1
                results['failed_images'].append({
                    'image_name': image_name,
                    'reason': 'Prediction failed',
                    'ground_truth': true_latex
                })
                continue
            
            prediction, inference_time, performance_stats = prediction_result
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_all_metrics(
                prediction=prediction,
                ground_truth=true_latex,
                quick_mode=self.quick_mode
            )
            
            # Store comprehensive results
            result_entry = {
                'image_name': image_name,
                'ground_truth': true_latex,
                'prediction': prediction,
                'inference_time_seconds': inference_time,
                'timestamp': datetime.now().isoformat(),
                **metrics,
                **performance_stats
            }
            
            results['predictions'].append(result_entry)
            
            # Update totals
            total_inference_time += inference_time
            total_memory_used += performance_stats.get('memory_used_mb', 0)
            processed_count += 1
            
            # Print immediate results
            print(f"Prediction: {prediction}")
            print(f"Exact match: {'✅' if metrics['exact_match'] else '❌'}")
            print(f"Time: {inference_time:.3f}s")
            print(f"Memory: {performance_stats.get('memory_used_mb', 0):.2f}MB")
            print(f"Char accuracy: {metrics['character_accuracy']:.3f}")
            
            if not self.quick_mode:
                print(f"Token accuracy: {metrics.get('token_accuracy', 0):.3f}")
        
        # Calculate summary statistics
        results['summary'] = self._calculate_summary_statistics(
            predictions=results['predictions'],
            total_images=total_images,
            processed_count=processed_count,
            failed_count=failed_count,
            total_inference_time=total_inference_time,
            total_memory_used=total_memory_used,
            model_name=model_name
        )
        
        results['benchmark_end'] = datetime.now().isoformat()
        
        # Print summary
        self._print_summary(results['summary'], model_name)
        
        return results
    
    def _predict_with_monitoring(self, model: BaseLatexOCRModel, 
                               image_path: str, image_name: str) -> Optional[Tuple[str, float, Dict]]:
        """Predict with comprehensive performance monitoring."""
        try:
            # Start monitoring
            monitor_data = self.performance_monitor.start_monitoring()
            
            # Make prediction
            start_time = time.time()
            prediction = model.predict(image_path)
            inference_time = time.time() - start_time
            
            # Stop monitoring and get stats
            performance_stats = self.performance_monitor.stop_monitoring(monitor_data)
            performance_stats['inference_time_seconds'] = inference_time
            
            if prediction is None:
                logger.error(f"Model returned None for {image_name}")
                return None
            
            return prediction, inference_time, performance_stats
            
        except Exception as e:
            logger.error(f"Error predicting {image_name}: {e}")
            return None
    
    def _calculate_summary_statistics(self, predictions: List[Dict], 
                                    total_images: int, processed_count: int,
                                    failed_count: int, total_inference_time: float,
                                    total_memory_used: float, model_name: str) -> Dict:
        """Calculate comprehensive summary statistics."""
        if not predictions:
            return {
                'model_name': model_name,
                'total_images': total_images,
                'processed_count': 0,
                'failed_count': failed_count,
                'success_rate': 0.0,
                'error': 'No successful predictions'
            }
        
        df = pd.DataFrame(predictions)
        
        summary = {
            'model_name': model_name,
            'total_images': total_images,
            'successfully_processed': processed_count,
            'failed_processing': failed_count,
            'success_rate': processed_count / total_images if total_images > 0 else 0,
            'total_processing_time_seconds': total_inference_time,
            'average_inference_time_seconds': total_inference_time / processed_count,
            'total_memory_used_mb': total_memory_used,
            'average_memory_used_mb': total_memory_used / processed_count,
            'exact_match_rate': df['exact_match'].mean(),
            'average_character_accuracy': df['character_accuracy'].mean(),
            'median_character_accuracy': df['character_accuracy'].median(),
            'std_character_accuracy': df['character_accuracy'].std(),
            'average_levenshtein_distance': df['levenshtein_distance'].mean(),
            'median_levenshtein_distance': df['levenshtein_distance'].median(),
            'timestamp': datetime.now().isoformat(),
            'device': df['device'].iloc[0] if 'device' in df.columns and not df.empty else 'Unknown'
        }
        
        # Add extended metrics if not in quick mode
        if not self.quick_mode and 'word_error_rate' in df.columns:
            summary.update({
                'average_word_error_rate': df['word_error_rate'].mean(),
                'median_word_error_rate': df['word_error_rate'].median(),
                'average_token_accuracy': df.get('token_accuracy', pd.Series([0])).mean(),
                'median_token_accuracy': df.get('token_accuracy', pd.Series([0])).median()
            })
        
        # Add GPU statistics if available
        gpu_columns = ['gpu_memory_allocated_mb', 'gpu_memory_cached_mb', 'gpu_utilization']
        if any(col in df.columns for col in gpu_columns):
            for col in gpu_columns:
                if col in df.columns:
                    summary[f'average_{col}'] = df[col].mean()
        
        return summary
    
    def _print_summary(self, summary: Dict, model_name: str):
        """Print comprehensive summary statistics."""
        print("\n" + "=" * 70)
        print(f"BENCHMARK SUMMARY - {model_name.upper()}")
        print("=" * 70)
        
        # Core metrics
        print(f"{'Total Images':<35}: {summary['total_images']}")
        print(f"{'Successfully Processed':<35}: {summary['successfully_processed']}")
        print(f"{'Failed Processing':<35}: {summary['failed_processing']}")
        print(f"{'Success Rate':<35}: {summary['success_rate']:.1%}")
        
        if summary['successfully_processed'] > 0:
            print(f"{'Exact Match Rate':<35}: {summary['exact_match_rate']:.1%}")
            print(f"{'Average Character Accuracy':<35}: {summary['average_character_accuracy']:.3f}")
            print(f"{'Median Character Accuracy':<35}: {summary['median_character_accuracy']:.3f}")
            print(f"{'Average Inference Time':<35}: {summary['average_inference_time_seconds']:.3f}s")
            print(f"{'Total Processing Time':<35}: {summary['total_processing_time_seconds']:.1f}s")
            print(f"{'Average Memory Usage':<35}: {summary['average_memory_used_mb']:.2f}MB")
            print(f"{'Device':<35}: {summary['device']}")
        
        print("=" * 70)
    
    def generate_comparison_report(self, all_results: Dict[str, Any], output_dir: str):
        """Generate cross-model comparison report."""
        logger.info("Generating cross-model comparison report...")
        
        self.results_manager.save_comparison_report(
            all_results=all_results,
            output_dir=output_dir
        )
        
        # Print comparison summary
        print("\n" + "=" * 80)
        print("CROSS-MODEL COMPARISON SUMMARY")
        print("=" * 80)
        
        comparison_data = []
        for model_name, results in all_results.items():
            summary = results.get('summary', {})
            if summary and summary.get('successfully_processed', 0) > 0:
                comparison_data.append({
                    'Model': model_name,
                    'Success Rate': f"{summary.get('success_rate', 0):.1%}",
                    'Exact Match': f"{summary.get('exact_match_rate', 0):.1%}",
                    'Char Accuracy': f"{summary.get('average_character_accuracy', 0):.3f}",
                    'Avg Time (s)': f"{summary.get('average_inference_time_seconds', 0):.3f}",
                    'Avg Memory (MB)': f"{summary.get('average_memory_used_mb', 0):.1f}"
                })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            print(df_comparison.to_string(index=False))
        else:
            print("No successful results to compare.")
        
        print("=" * 80)
