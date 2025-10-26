"""
Results Manager for LaTeX-OCR Benchmarking
Handles saving, formatting, and reporting of benchmark results
"""

import os
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class ResultsManager:
    """Comprehensive results management for LaTeX-OCR benchmarking."""
    
    def __init__(self):
        self.timestamp_format = "%Y%m%d_%H%M%S"
        
    def save_model_results(self, results: Dict[str, Any], model_name: str, output_dir: str):
        """Save comprehensive results for a single model."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime(self.timestamp_format)
        
        # Create model-specific subdirectory
        model_dir = os.path.join(output_dir, f"{model_name}_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info(f"Saving results for {model_name} to {model_dir}")
        
        # Save detailed predictions as CSV
        self._save_predictions_csv(results['predictions'], model_dir, model_name, timestamp)
        
        # Save summary as JSON
        self._save_summary_json(results['summary'], model_dir, model_name, timestamp)
        
        # Save human-readable comparison
        self._save_human_readable_report(results, model_dir, model_name, timestamp)
        
        # Save failed predictions log
        if results.get('failed_images'):
            self._save_failed_predictions(results['failed_images'], model_dir, model_name, timestamp)
        
        # Save complete results as JSON
        self._save_complete_results_json(results, model_dir, model_name, timestamp)
        
        logger.info(f"✅ Results saved for {model_name}")
        return model_dir
    
    def _save_predictions_csv(self, predictions: List[Dict], output_dir: str, 
                             model_name: str, timestamp: str):
        """Save detailed predictions to CSV."""
        if not predictions:
            logger.warning("No predictions to save to CSV")
            return
        
        csv_file = os.path.join(output_dir, f"{model_name}_detailed_predictions_{timestamp}.csv")
        
        try:
            df = pd.DataFrame(predictions)
            
            # Ensure consistent column ordering
            priority_columns = [
                'image_name', 'ground_truth', 'prediction', 'exact_match',
                'character_accuracy', 'inference_time_seconds', 'memory_used_mb'
            ]
            
            columns = []
            for col in priority_columns:
                if col in df.columns:
                    columns.append(col)
            
            # Add remaining columns
            for col in df.columns:
                if col not in columns:
                    columns.append(col)
            
            df = df[columns]
            df.to_csv(csv_file, index=False, encoding='utf-8')
            logger.info(f"Detailed predictions saved to: {csv_file}")
            
        except Exception as e:
            logger.error(f"Failed to save predictions CSV: {e}")
    
    def _save_summary_json(self, summary: Dict[str, Any], output_dir: str,
                          model_name: str, timestamp: str):
        """Save summary statistics to JSON."""
        json_file = os.path.join(output_dir, f"{model_name}_summary_{timestamp}.json")
        
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Summary saved to: {json_file}")
        except Exception as e:
            logger.error(f"Failed to save summary JSON: {e}")
    
    def _save_human_readable_report(self, results: Dict[str, Any], output_dir: str,
                                   model_name: str, timestamp: str):
        """Save human-readable benchmark report."""
        report_file = os.path.join(output_dir, f"{model_name}_report_{timestamp}.txt")
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                self._write_human_readable_report(f, results, model_name)
            logger.info(f"Human-readable report saved to: {report_file}")
        except Exception as e:
            logger.error(f"Failed to save human-readable report: {e}")
    
    def _write_human_readable_report(self, file, results: Dict[str, Any], model_name: str):
        """Write comprehensive human-readable report."""
        f = file
        summary = results.get('summary', {})
        predictions = results.get('predictions', [])
        failed_images = results.get('failed_images', [])
        
        # Header
        f.write("=" * 100 + "\n")
        f.write(f"LATEX-OCR BENCHMARK REPORT - {model_name.upper()}\n")
        f.write("=" * 100 + "\n\n")
        
        # Benchmark metadata
        f.write("BENCHMARK INFORMATION\n")
        f.write("-" * 50 + "\n")
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Benchmark Start: {results.get('benchmark_start', 'Unknown')}\n")
        f.write(f"Benchmark End: {results.get('benchmark_end', 'Unknown')}\n")
        f.write(f"Device Used: {summary.get('device', 'Unknown')}\n\n")
        
        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total Images: {summary.get('total_images', 0)}\n")
        f.write(f"Successfully Processed: {summary.get('successfully_processed', 0)}\n")
        f.write(f"Failed Processing: {summary.get('failed_processing', 0)}\n")
        f.write(f"Success Rate: {summary.get('success_rate', 0):.1%}\n\n")
        
        if summary.get('successfully_processed', 0) > 0:
            f.write("ACCURACY METRICS\n")
            f.write("-" * 50 + "\n")
            f.write(f"Exact Match Rate: {summary.get('exact_match_rate', 0):.1%}\n")
            f.write(f"Average Character Accuracy: {summary.get('average_character_accuracy', 0):.3f}\n")
            f.write(f"Median Character Accuracy: {summary.get('median_character_accuracy', 0):.3f}\n")
            f.write(f"Std Character Accuracy: {summary.get('std_character_accuracy', 0):.3f}\n")
            f.write(f"Average Levenshtein Distance: {summary.get('average_levenshtein_distance', 0):.1f}\n")
            
            if 'average_word_error_rate' in summary:
                f.write(f"Average Word Error Rate: {summary.get('average_word_error_rate', 0):.3f}\n")
                f.write(f"Average Token Accuracy: {summary.get('average_token_accuracy', 0):.3f}\n")
            
            f.write("\nPERFORMANCE METRICS\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total Processing Time: {summary.get('total_processing_time_seconds', 0):.1f}s\n")
            f.write(f"Average Inference Time: {summary.get('average_inference_time_seconds', 0):.3f}s\n")
            f.write(f"Average Memory Usage: {summary.get('average_memory_used_mb', 0):.2f}MB\n")
            
            # GPU metrics if available
            if 'average_gpu_memory_allocated_mb' in summary:
                f.write(f"Average GPU Memory: {summary.get('average_gpu_memory_allocated_mb', 0):.2f}MB\n")
                f.write(f"Average GPU Utilization: {summary.get('average_gpu_utilization_percent', 0):.1f}%\n")
        
        # Failed images
        if failed_images:
            f.write(f"\nFAILED IMAGES ({len(failed_images)})\n")
            f.write("-" * 50 + "\n")
            for fail in failed_images[:10]:  # Show first 10
                f.write(f"• {fail['image_name']}: {fail['reason']}\n")
            if len(failed_images) > 10:
                f.write(f"... and {len(failed_images) - 10} more\n")
        
        # Detailed predictions
        f.write(f"\nDETAILED PREDICTIONS ({len(predictions)})\n")
        f.write("=" * 100 + "\n\n")
        
        for i, result in enumerate(predictions, 1):
            f.write(f"IMAGE {i}: {result['image_name']}\n")
            f.write(f"Ground Truth: {result['ground_truth']}\n")
            f.write(f"Prediction:   {result['prediction']}\n")
            f.write(f"Exact Match:  {'✅ YES' if result['exact_match'] else '❌ NO'}\n")
            f.write(f"Char Accuracy: {result['character_accuracy']:.3f} | ")
            f.write(f"Time: {result['inference_time_seconds']:.3f}s | ")
            f.write(f"Memory: {result.get('memory_used_mb', 0):.2f}MB\n")
            
            if 'token_accuracy' in result:
                f.write(f"Token Accuracy: {result['token_accuracy']:.3f} | ")
            f.write(f"Levenshtein: {result['levenshtein_distance']}\n")
            
            f.write("-" * 100 + "\n\n")
    
    def _save_failed_predictions(self, failed_images: List[Dict], output_dir: str,
                                model_name: str, timestamp: str):
        """Save failed predictions log."""
        if not failed_images:
            return
        
        failed_file = os.path.join(output_dir, f"{model_name}_failed_{timestamp}.json")
        
        try:
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(failed_images, f, indent=2, ensure_ascii=False)
            logger.info(f"Failed predictions saved to: {failed_file}")
        except Exception as e:
            logger.error(f"Failed to save failed predictions: {e}")
    
    def _save_complete_results_json(self, results: Dict[str, Any], output_dir: str,
                                   model_name: str, timestamp: str):
        """Save complete results as JSON for later analysis."""
        json_file = os.path.join(output_dir, f"{model_name}_complete_results_{timestamp}.json")
        
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Complete results saved to: {json_file}")
        except Exception as e:
            logger.error(f"Failed to save complete results: {e}")
    
    def save_comparison_report(self, all_results: Dict[str, Any], output_dir: str):
        """Save cross-model comparison report."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime(self.timestamp_format)
        
        # Save comparison CSV
        self._save_comparison_csv(all_results, output_dir, timestamp)
        
        # Save comparison report
        self._save_comparison_text_report(all_results, output_dir, timestamp)
        
        # Save comparison JSON
        self._save_comparison_json(all_results, output_dir, timestamp)
    
    def _save_comparison_csv(self, all_results: Dict[str, Any], output_dir: str, timestamp: str):
        """Save model comparison as CSV."""
        csv_file = os.path.join(output_dir, f"model_comparison_{timestamp}.csv")
        
        try:
            comparison_data = []
            for model_name, results in all_results.items():
                summary = results.get('summary', {})
                if summary and summary.get('successfully_processed', 0) > 0:
                    row = {
                        'Model': model_name,
                        'Total Images': summary.get('total_images', 0),
                        'Success Rate': summary.get('success_rate', 0),
                        'Exact Match Rate': summary.get('exact_match_rate', 0),
                        'Avg Character Accuracy': summary.get('average_character_accuracy', 0),
                        'Median Character Accuracy': summary.get('median_character_accuracy', 0),
                        'Avg Levenshtein Distance': summary.get('average_levenshtein_distance', 0),
                        'Avg Inference Time (s)': summary.get('average_inference_time_seconds', 0),
                        'Avg Memory Usage (MB)': summary.get('average_memory_used_mb', 0),
                        'Device': summary.get('device', 'Unknown')
                    }
                    
                    # Add extended metrics if available
                    if 'average_word_error_rate' in summary:
                        row['Avg Word Error Rate'] = summary.get('average_word_error_rate', 0)
                        row['Avg Token Accuracy'] = summary.get('average_token_accuracy', 0)
                    
                    # Add GPU metrics if available
                    if 'average_gpu_memory_allocated_mb' in summary:
                        row['Avg GPU Memory (MB)'] = summary.get('average_gpu_memory_allocated_mb', 0)
                        row['Avg GPU Utilization (%)'] = summary.get('average_gpu_utilization_percent', 0)
                    
                    comparison_data.append(row)
            
            if comparison_data:
                df = pd.DataFrame(comparison_data)
                df.to_csv(csv_file, index=False)
                logger.info(f"Model comparison CSV saved to: {csv_file}")
            else:
                logger.warning("No successful results to save in comparison CSV")
                
        except Exception as e:
            logger.error(f"Failed to save comparison CSV: {e}")
    
    def _save_comparison_text_report(self, all_results: Dict[str, Any], output_dir: str, timestamp: str):
        """Save detailed comparison text report."""
        report_file = os.path.join(output_dir, f"model_comparison_report_{timestamp}.txt")
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 100 + "\n")
                f.write("LATEX-OCR MODELS COMPARISON REPORT\n")
                f.write("=" * 100 + "\n\n")
                
                f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Models Benchmarked: {len(all_results)}\n\n")
                
                # Model rankings
                self._write_model_rankings(f, all_results)
                
                # Detailed comparison
                self._write_detailed_model_comparison(f, all_results)
                
                # Performance analysis
                self._write_performance_analysis(f, all_results)
            
            logger.info(f"Model comparison report saved to: {report_file}")
        except Exception as e:
            logger.error(f"Failed to save comparison report: {e}")
    
    def _write_model_rankings(self, f, all_results: Dict[str, Any]):
        """Write model rankings section."""
        f.write("MODEL RANKINGS\n")
        f.write("-" * 50 + "\n")
        
        # Rank by exact match rate
        exact_match_rankings = []
        char_accuracy_rankings = []
        speed_rankings = []
        
        for model_name, results in all_results.items():
            summary = results.get('summary', {})
            if summary and summary.get('successfully_processed', 0) > 0:
                exact_match_rankings.append((
                    model_name, 
                    summary.get('exact_match_rate', 0)
                ))
                char_accuracy_rankings.append((
                    model_name,
                    summary.get('average_character_accuracy', 0)
                ))
                speed_rankings.append((
                    model_name,
                    summary.get('average_inference_time_seconds', float('inf'))
                ))
        
        # Sort rankings
        exact_match_rankings.sort(key=lambda x: x[1], reverse=True)
        char_accuracy_rankings.sort(key=lambda x: x[1], reverse=True)
        speed_rankings.sort(key=lambda x: x[1])  # Lower is better for speed
        
        f.write("By Exact Match Rate:\n")
        for i, (model, rate) in enumerate(exact_match_rankings, 1):
            f.write(f"  {i}. {model}: {rate:.1%}\n")
        
        f.write("\nBy Character Accuracy:\n")
        for i, (model, acc) in enumerate(char_accuracy_rankings, 1):
            f.write(f"  {i}. {model}: {acc:.3f}\n")
        
        f.write("\nBy Speed (Average Inference Time):\n")
        for i, (model, time) in enumerate(speed_rankings, 1):
            f.write(f"  {i}. {model}: {time:.3f}s\n")
        
        f.write("\n")
    
    def _write_detailed_model_comparison(self, f, all_results: Dict[str, Any]):
        """Write detailed model comparison section."""
        f.write("DETAILED MODEL COMPARISON\n")
        f.write("-" * 50 + "\n")
        
        for model_name, results in all_results.items():
            summary = results.get('summary', {})
            f.write(f"\n{model_name.upper()}\n")
            f.write("─" * 30 + "\n")
            
            if summary and summary.get('successfully_processed', 0) > 0:
                f.write(f"Success Rate: {summary.get('success_rate', 0):.1%}\n")
                f.write(f"Exact Match Rate: {summary.get('exact_match_rate', 0):.1%}\n")
                f.write(f"Character Accuracy: {summary.get('average_character_accuracy', 0):.3f} ± {summary.get('std_character_accuracy', 0):.3f}\n")
                f.write(f"Inference Time: {summary.get('average_inference_time_seconds', 0):.3f}s\n")
                f.write(f"Memory Usage: {summary.get('average_memory_used_mb', 0):.2f}MB\n")
                f.write(f"Device: {summary.get('device', 'Unknown')}\n")
                
                if 'average_word_error_rate' in summary:
                    f.write(f"Word Error Rate: {summary.get('average_word_error_rate', 0):.3f}\n")
                    f.write(f"Token Accuracy: {summary.get('average_token_accuracy', 0):.3f}\n")
            else:
                f.write("❌ No successful predictions\n")
        
        f.write("\n")
    
    def _write_performance_analysis(self, f, all_results: Dict[str, Any]):
        """Write performance analysis section."""
        f.write("PERFORMANCE ANALYSIS\n")
        f.write("-" * 50 + "\n")
        
        successful_models = []
        for model_name, results in all_results.items():
            summary = results.get('summary', {})
            if summary and summary.get('successfully_processed', 0) > 0:
                successful_models.append((model_name, summary))
        
        if not successful_models:
            f.write("No successful models to analyze.\n")
            return
        
        # Best accuracy
        best_exact_match = max(successful_models, key=lambda x: x[1].get('exact_match_rate', 0))
        best_char_acc = max(successful_models, key=lambda x: x[1].get('average_character_accuracy', 0))
        fastest = min(successful_models, key=lambda x: x[1].get('average_inference_time_seconds', float('inf')))
        most_efficient = min(successful_models, key=lambda x: x[1].get('average_memory_used_mb', float('inf')))
        
        f.write(f"🏆 Best Exact Match Rate: {best_exact_match[0]} ({best_exact_match[1].get('exact_match_rate', 0):.1%})\n")
        f.write(f"🏆 Best Character Accuracy: {best_char_acc[0]} ({best_char_acc[1].get('average_character_accuracy', 0):.3f})\n")
        f.write(f"⚡ Fastest Model: {fastest[0]} ({fastest[1].get('average_inference_time_seconds', 0):.3f}s)\n")
        f.write(f"💾 Most Memory Efficient: {most_efficient[0]} ({most_efficient[1].get('average_memory_used_mb', 0):.2f}MB)\n")
        
        # Overall recommendations
        f.write("\nRECOMMENDATIONS\n")
        f.write("-" * 30 + "\n")
        
        if best_exact_match[1].get('exact_match_rate', 0) > 0.8:
            f.write(f"• For highest accuracy: Use {best_exact_match[0]}\n")
        
        if fastest[1].get('average_inference_time_seconds', 0) < 1.0:
            f.write(f"• For real-time applications: Use {fastest[0]}\n")
        
        if most_efficient[1].get('average_memory_used_mb', 0) < 100:
            f.write(f"• For resource-constrained environments: Use {most_efficient[0]}\n")
        
        f.write("\n")
    
    def _save_comparison_json(self, all_results: Dict[str, Any], output_dir: str, timestamp: str):
        """Save complete comparison results as JSON."""
        json_file = os.path.join(output_dir, f"model_comparison_complete_{timestamp}.json")
        
        try:
            comparison_data = {
                'timestamp': datetime.now().isoformat(),
                'total_models': len(all_results),
                'models': {}
            }
            
            for model_name, results in all_results.items():
                comparison_data['models'][model_name] = {
                    'summary': results.get('summary', {}),
                    'prediction_count': len(results.get('predictions', [])),
                    'failed_count': len(results.get('failed_images', [])),
                    'benchmark_duration': results.get('benchmark_end', '') and results.get('benchmark_start', ''),
                }
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(comparison_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Complete comparison JSON saved to: {json_file}")
        except Exception as e:
            logger.error(f"Failed to save comparison JSON: {e}")
    
    def load_previous_results(self, results_file: str) -> Optional[Dict[str, Any]]:
        """Load previously saved results for analysis."""
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load previous results from {results_file}: {e}")
            return None
    
    def merge_results(self, *result_sets: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple result sets for combined analysis."""
        merged = {
            'models': {},
            'merged_timestamp': datetime.now().isoformat(),
            'source_count': len(result_sets)
        }
        
        for i, result_set in enumerate(result_sets):
            if 'models' in result_set:
                for model_name, model_data in result_set['models'].items():
                    key = f"{model_name}_set{i+1}" if model_name in merged['models'] else model_name
                    merged['models'][key] = model_data
        
        return merged
    
    def export_results_for_analysis(self, results: Dict[str, Any], output_file: str, format: str = 'csv'):
        """Export results in format suitable for external analysis (R, Python, etc.)."""
        try:
            if format.lower() == 'csv':
                self._export_csv_for_analysis(results, output_file)
            elif format.lower() == 'json':
                self._export_json_for_analysis(results, output_file)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Results exported for analysis to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
    
    def _export_csv_for_analysis(self, results: Dict[str, Any], output_file: str):
        """Export results as analysis-ready CSV."""
        analysis_data = []
        
        for model_name, model_results in results.get('models', {}).items():
            summary = model_results.get('summary', {})
            predictions = model_results.get('predictions', [])
            
            for pred in predictions:
                row = {
                    'model_name': model_name,
                    'image_name': pred.get('image_name', ''),
                    'ground_truth_length': len(pred.get('ground_truth', '')),
                    'prediction_length': len(pred.get('prediction', '')),
                    'exact_match': pred.get('exact_match', 0),
                    'character_accuracy': pred.get('character_accuracy', 0),
                    'levenshtein_distance': pred.get('levenshtein_distance', 0),
                    'inference_time_seconds': pred.get('inference_time_seconds', 0),
                    'memory_used_mb': pred.get('memory_used_mb', 0),
                    'device': pred.get('device', 'Unknown')
                }
                
                # Add extended metrics if available
                if 'word_error_rate' in pred:
                    row['word_error_rate'] = pred['word_error_rate']
                    row['token_accuracy'] = pred.get('token_accuracy', 0)
                
                analysis_data.append(row)
        
        if analysis_data:
            df = pd.DataFrame(analysis_data)
            df.to_csv(output_file, index=False)
        else:
            logger.warning("No data to export")
    
    def _export_json_for_analysis(self, results: Dict[str, Any], output_file: str):
        """Export results as analysis-ready JSON."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    def generate_summary_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary statistics across all models."""
        summary_stats = {
            'total_models_tested': len(results.get('models', {})),
            'total_images_processed': 0,
            'total_successful_predictions': 0,
            'total_failed_predictions': 0,
            'model_statistics': {},
            'cross_model_metrics': {},
            'generation_timestamp': datetime.now().isoformat()
        }
        
        all_accuracies = []
        all_inference_times = []
        all_memory_usage = []
        
        for model_name, model_results in results.get('models', {}).items():
            summary = model_results.get('summary', {})
            predictions = model_results.get('predictions', [])
            
            if summary:
                summary_stats['total_images_processed'] += summary.get('total_images', 0)
                summary_stats['total_successful_predictions'] += summary.get('successfully_processed', 0)
                summary_stats['total_failed_predictions'] += summary.get('failed_processing', 0)
                
                summary_stats['model_statistics'][model_name] = summary
                
                # Collect metrics for cross-model analysis
                if predictions:
                    model_accuracies = [p.get('character_accuracy', 0) for p in predictions]
                    model_times = [p.get('inference_time_seconds', 0) for p in predictions]
                    model_memory = [p.get('memory_used_mb', 0) for p in predictions]
                    
                    all_accuracies.extend(model_accuracies)
                    all_inference_times.extend(model_times)
                    all_memory_usage.extend(model_memory)
        
        # Calculate cross-model statistics
        if all_accuracies:
            summary_stats['cross_model_metrics'] = {
                'accuracy_stats': {
                    'mean': np.mean(all_accuracies),
                    'std': np.std(all_accuracies),
                    'min': np.min(all_accuracies),
                    'max': np.max(all_accuracies),
                    'median': np.median(all_accuracies)
                },
                'inference_time_stats': {
                    'mean': np.mean(all_inference_times),
                    'std': np.std(all_inference_times),
                    'min': np.min(all_inference_times),
                    'max': np.max(all_inference_times),
                    'median': np.median(all_inference_times)
                },
                'memory_usage_stats': {
                    'mean': np.mean(all_memory_usage),
                    'std': np.std(all_memory_usage),
                    'min': np.min(all_memory_usage),
                    'max': np.max(all_memory_usage),
                    'median': np.median(all_memory_usage)
                }
            }
        
        return summary_stats