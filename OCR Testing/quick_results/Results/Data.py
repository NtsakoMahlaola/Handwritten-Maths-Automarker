import json
import os

def generate_latex_tables(benchmark_file):
    """Generate three LaTeX tables from benchmark data"""
    
    with open(benchmark_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    models_data = data['consolidated_benchmark_averages']
    
    # Table 1: Accuracy Metrics (CER, WER, Levenshtein Distance)
    table1 = """\\begin{table}[htbp]
\\centering
\\caption{OCR Benchmark Results - Accuracy Metrics}
\\label{tab:ocr_benchmark_accuracy}
\\begin{tabular}{|l|c|c|c|c|}
\\hline
\\textbf{Model} & \\textbf{Exact Match (\\%)} & \\textbf{CER (\\%)} & \\textbf{WER (\\%)} & \\textbf{Levenshtein Distance} \\\\
\\hline
"""
    
    # Table 2: Similarity Metrics (BLEU, Structural, Semantic)
    table2 = """\\begin{table}[htbp]
\\centering
\\caption{OCR Benchmark Results - Similarity Metrics}
\\label{tab:ocr_benchmark_similarity}
\\begin{tabular}{|l|c|c|c|}
\\hline
\\textbf{Model} & \\textbf{BLEU Score} & \\textbf{Structural Similarity} & \\textbf{Semantic Similarity} \\\\
\\hline
"""
    
    # Table 3: Performance Metrics (Inference Time, Memory, Scalability)
    table3 = """\\begin{table}[htbp]
\\centering
\\caption{OCR Benchmark Results - Performance Metrics}
\\label{tab:ocr_benchmark_performance}
\\begin{tabular}{|l|c|c|c|}
\\hline
\\textbf{Model} & \\textbf{Inference Time (s)} & \\textbf{Memory (MB)} & \\textbf{Scalability Score} \\\\
\\hline
"""
    
    for model_data in models_data:
        model_name = model_data['model_name']
        averages = model_data['averages']
        summary = model_data.get('summary_metrics', {})
        
        # Extract metrics for Table 1
        exact_match_pct = summary.get('exact_match_rate', 0) * 100
        cer = (1 - averages.get('average_character_accuracy', 0)) * 100  # Character Error Rate
        wer = averages.get('average_word_error_rate', 0) * 100  # Word Error Rate
        levenshtein = averages.get('average_levenshtein_distance', 0)
        
        # Extract metrics for Table 2
        bleu = averages.get('average_bleu_score', 0)
        structural_sim = averages.get('average_structural_similarity', 0)
        semantic_sim = averages.get('average_semantic_similarity', 0)
        
        # Extract metrics for Table 3
        inference_time = averages.get('average_inference_time_seconds', 0)
        memory_used = averages.get('average_memory_used_mb', 0)
        
        # Calculate scalability score (made up based on speed and accuracy)
        # Higher is better: faster inference + better accuracy = better scalability
        speed_score = 1 / (inference_time + 0.1)  # Avoid division by zero
        accuracy_score = averages.get('average_character_accuracy', 0)
        scalability_score = min(10, (speed_score * 2 + accuracy_score * 8))  # Scale to 0-10
        
        # Format values for tables
        table1 += f"\\texttt{{{model_name}}} & {exact_match_pct:.1f} & {cer:.1f} & {wer:.1f} & {levenshtein:.1f} \\\\\\\\\n\\hline\n"
        
        table2 += f"\\texttt{{{model_name}}} & {bleu:.3f} & {structural_sim:.3f} & {semantic_sim:.3f} \\\\\\\\\n\\hline\n"
        
        table3 += f"\\texttt{{{model_name}}} & {inference_time:.2f} & {memory_used:.1f} & {scalability_score:.1f} \\\\\\\\\n\\hline\n"
    
    # Close tables
    table1 += "\\end{tabular}\n\\end{table}\n"
    table2 += "\\end{tabular}\n\\end{table}\n"
    table3 += "\\end{tabular}\n\\end{table}\n"
    
    return table1, table2, table3

def save_latex_tables_to_file(tables, output_file):
    """Save the generated LaTeX tables to a file"""
    
    table1, table2, table3 = tables
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("% LaTeX Tables Generated from OCR Benchmark Results\n")
        f.write("% ==================================================\n\n")
        
        f.write("% Table 1: Accuracy Metrics\n")
        f.write(table1)
        f.write("\n\n")
        
        f.write("% Table 2: Similarity Metrics\n")
        f.write(table2)
        f.write("\n\n")
        
        f.write("% Table 3: Performance Metrics\n")
        f.write(table3)
        
    print(f"LaTeX tables saved to {output_file}")

def main():
    # Configuration
    benchmark_file = "benchmark_averages.json"  # Your input file
    output_file = "benchmark_tables.tex"        # Output LaTeX file
    
    if not os.path.exists(benchmark_file):
        print(f"Error: Benchmark file '{benchmark_file}' not found!")
        return
    
    try:
        # Generate tables
        tables = generate_latex_tables(benchmark_file)
        
        # Save to file
        save_latex_tables_to_file(tables, output_file)
        
        # Also print to console for quick copy-paste
        print("\n" + "="*60)
        print("GENERATED LATEX TABLES")
        print("="*60)
        
        table1, table2, table3 = tables
        print("\nTABLE 1 - Accuracy Metrics:")
        print(table1)
        
        print("\nTABLE 2 - Similarity Metrics:")
        print(table2)
        
        print("\nTABLE 3 - Performance Metrics:")
        print(table3)
        
    except Exception as e:
        print(f"Error processing benchmark data: {str(e)}")

if __name__ == "__main__":
    main()