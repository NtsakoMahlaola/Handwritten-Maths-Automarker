import json
import os
import glob
from statistics import mean, median
from typing import Dict, List, Any

def calculate_averages_from_file(file_path: str) -> Dict[str, Any]:
    """Calculate averages from a single benchmark result file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        predictions = data.get('predictions', [])
        
        if not predictions:
            return {
                "file_name": os.path.basename(file_path),
                "error": "No predictions found in file"
            }
        
        # Initialize lists for all metrics
        metrics = {
            'inference_time_seconds': [],
            'exact_match': [],
            'levenshtein_distance': [],
            'character_accuracy': [],
            'prediction_length': [],
            'ground_truth_length': [],
            'length_difference': [],
            'word_error_rate': [],
            'token_accuracy': [],
            'latex_command_accuracy': [],
            'structural_similarity': [],
            'semantic_similarity': [],
            'bleu_score': [],
            'jaccard_similarity': [],
            'total_time_seconds': [],
            'memory_used_mb': [],
            'final_memory_mb': [],
            'cpu_usage_before': [],
            'cpu_usage_after': [],
            'cpu_usage_delta': []
        }
        
        # Collect all metrics from predictions
        for pred in predictions:
            for metric in metrics.keys():
                if metric in pred:
                    metrics[metric].append(pred[metric])
        
        # Calculate averages
        averages = {
            "file_name": os.path.basename(file_path),
            "model_name": data.get('model_name', 'unknown'),
            "total_images": len(predictions),
            "successfully_processed": data.get('summary', {}).get('successfully_processed', 0),
            "failed_processing": data.get('summary', {}).get('failed_processing', 0),
            "success_rate": data.get('summary', {}).get('success_rate', 0),
            "averages": {}
        }
        
        # Calculate mean and median for each metric
        for metric, values in metrics.items():
            if values:
                averages["averages"][f"average_{metric}"] = mean(values)
                averages["averages"][f"median_{metric}"] = median(values)
                averages["averages"][f"min_{metric}"] = min(values)
                averages["averages"][f"max_{metric}"] = max(values)
            else:
                averages["averages"][f"average_{metric}"] = 0
                averages["averages"][f"median_{metric}"] = 0
                averages["averages"][f"min_{metric}"] = 0
                averages["averages"][f"max_{metric}"] = 0
        
        # Add summary metrics if available
        summary = data.get('summary', {})
        if summary:
            averages["summary_metrics"] = {
                "total_processing_time_seconds": summary.get('total_processing_time_seconds'),
                "average_inference_time_seconds": summary.get('average_inference_time_seconds'),
                "exact_match_rate": summary.get('exact_match_rate'),
                "average_character_accuracy": summary.get('average_character_accuracy'),
                "median_character_accuracy": summary.get('median_character_accuracy'),
                "average_levenshtein_distance": summary.get('average_levenshtein_distance'),
                "median_levenshtein_distance": summary.get('median_levenshtein_distance'),
                "average_word_error_rate": summary.get('average_word_error_rate'),
                "median_word_error_rate": summary.get('median_word_error_rate'),
                "average_token_accuracy": summary.get('average_token_accuracy'),
                "median_token_accuracy": summary.get('median_token_accuracy')
            }
        
        # Add system info from first prediction (assuming it's consistent)
        if predictions and 'system_info' in predictions[0]:
            averages["system_info"] = predictions[0]['system_info']
            averages["device"] = predictions[0].get('device', 'CPU')
        
        # Add benchmark timing info
        averages["benchmark_start"] = data.get('benchmark_start')
        averages["benchmark_end"] = data.get('benchmark_end')
        averages["timestamp"] = data.get('summary', {}).get('timestamp')
        
        return averages
        
    except Exception as e:
        return {
            "file_name": os.path.basename(file_path),
            "error": f"Error processing file: {str(e)}"
        }

def process_all_benchmark_files(input_folder: str = ".", output_file: str = "benchmark_averages.json") -> Dict[str, Any]:
    """Process all benchmark JSON files in a folder and save averages to a new file."""
    
    # Find all JSON files (adjust pattern if needed)
    json_files = glob.glob(os.path.join(input_folder, "*complete_results*.json"))
    
    if not json_files:
        print(f"No benchmark files found in {input_folder}")
        return {}
    
    print(f"Found {len(json_files)} benchmark files:")
    for file in json_files:
        print(f"  - {os.path.basename(file)}")
    
    # Process each file
    all_averages = {
        "consolidated_benchmark_averages": [],
        "total_files_processed": len(json_files),
        "processing_timestamp": None
    }
    
    for file_path in json_files:
        print(f"Processing {os.path.basename(file_path)}...")
        averages = calculate_averages_from_file(file_path)
        all_averages["consolidated_benchmark_averages"].append(averages)
    
    # Add overall statistics across all models
    all_averages["overall_statistics"] = calculate_overall_statistics(all_averages["consolidated_benchmark_averages"])
    all_averages["processing_timestamp"] = json_files[0]  # Use first file's timestamp as reference
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_averages, f, indent=2, ensure_ascii=False)
    
    print(f"\nAverages saved to {output_file}")
    print(f"Processed {len(json_files)} files successfully")
    
    return all_averages

def calculate_overall_statistics(all_averages: List[Dict]) -> Dict[str, Any]:
    """Calculate overall statistics across all models."""
    if not all_averages:
        return {}
    
    # Collect all average values for key metrics
    key_metrics = [
        'average_character_accuracy', 
        'average_levenshtein_distance',
        'average_inference_time_seconds',
        'average_exact_match'
    ]
    
    overall_stats = {}
    
    for metric in key_metrics:
        values = []
        for avg_data in all_averages:
            if 'averages' in avg_data and metric in avg_data['averages']:
                values.append(avg_data['averages'][metric])
        
        if values:
            overall_stats[f"overall_{metric}"] = mean(values)
            overall_stats[f"overall_median_{metric}"] = median(values)
            overall_stats[f"overall_min_{metric}"] = min(values)
            overall_stats[f"overall_max_{metric}"] = max(values)
    
    # Count successful models
    successful_models = sum(1 for avg_data in all_averages if 'error' not in avg_data)
    overall_stats["successful_models_processed"] = successful_models
    overall_stats["total_models"] = len(all_averages)
    
    return overall_stats

def main():
    """Main function to run the benchmark averaging script."""
    
    # You can modify these paths as needed
    input_folder = "."  # Current directory, change to your folder path
    output_file = "benchmark_averages.json"
    
    print("Starting benchmark file processing...")
    print(f"Input folder: {input_folder}")
    print(f"Output file: {output_file}")
    print("-" * 50)
    
    try:
        results = process_all_benchmark_files(input_folder, output_file)
        
        # Print summary
        print("\n" + "=" * 50)
        print("PROCESSING SUMMARY")
        print("=" * 50)
        
        successful = 0
        for result in results.get("consolidated_benchmark_averages", []):
            if 'error' not in result:
                successful += 1
                model_name = result.get('model_name', 'unknown')
                char_acc = result['averages'].get('average_character_accuracy', 0)
                inf_time = result['averages'].get('average_inference_time_seconds', 0)
                print(f"✓ {model_name}: Accuracy={char_acc:.3f}, Time={inf_time:.2f}s")
            else:
                print(f"✗ {result['file_name']}: {result['error']}")
        
        print(f"\nSuccessfully processed: {successful}/{len(results['consolidated_benchmark_averages'])} files")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()