import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Set style for better visuals
plt.style.use('default')
sns.set_palette("husl")

# Data from JSON
models = ['trocr-math', 'latex-finetuned', 'pix2text-mfr-onnx', 
          'trocr-vision2seq-math', 'trocr', 'trocr-large-handwritten', 'pix2tex']

# Edit-based metrics
cer = [0.0, 58.3, 36.7, 47.4, 91.7, 87.5, 91.8]  # % error
wer = [0.0, 50.0, 100.0, 100.0, 100.0, 100.0, 100.0]  # % error
levenshtein = [0, 14, 11, 18, 22, 21, 145]

# Linguistic and semantic metrics
bleu = [1.0, 0.5, 0.333, 0.0, 0.0, 0.0, 0.0]
structural_sim = [1.0, 1.0, 0.5, 0.667, 0.0, 0.0, 0.059]
semantic_sim = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Computational metrics
inference_time = [11.35, 4.89, 0.63, 46.23, 1.08, 14.13, 1.50]
memory_used = [2102.2, 1315.9, 13.9, 2176.9, 127.4, 2002.3, 17.2]
scalability = [5.8, 6.3, 9.2, 3.5, 8.1, 4.9, 8.5]  # Scalability scores (1-10)

def create_edit_based_metrics():
    """Create bar chart for edit-based accuracy metrics"""
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(models))
    width = 0.25
    
    # Bars for CER and WER
    bars1 = ax1.bar(x - width, cer, width, label='CER (%)', color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x, wer, width, label='WER (%)', color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Error Rate (%)', color='black', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('-', '\n') for m in models], rotation=45, ha='right', fontsize=10)
    ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_axisbelow(True)
    
    # Second y-axis for Levenshtein Distance
    ax2 = ax1.twinx()
    bars3 = ax2.bar(x + width, levenshtein, width, label='Levenshtein Distance', 
                    color='#45B7D1', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Levenshtein Distance', color='black', fontsize=11)
    ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    plt.title('Edit-Based Accuracy Metrics\n(CER/WER: Lower is Better)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save as PDF
    plt.savefig('edit_based_metrics.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def create_linguistic_semantic_metrics():
    """Create grouped bar chart for linguistic and semantic metrics"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(models))
    width = 0.25
    
    bars1 = ax.bar(x - width, bleu, width, label='BLEU Score', 
                   color='#A8E6CF', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, structural_sim, width, label='Structural Similarity', 
                   color='#FFD3B6', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, semantic_sim, width, label='Semantic Similarity', 
                   color='#FFAAA5', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Linguistic and Semantic Performance Metrics\n(Higher is Better)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('-', '\n') for m in models], rotation=45, ha='right', fontsize=10)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only label non-zero values
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    # Save as PDF
    plt.savefig('linguistic_semantic_metrics.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def create_computational_heatmap():
    """Create heatmap for computational performance metrics"""
    # Create data matrix for heatmap
    data = np.array([inference_time, memory_used, scalability]).T
    
    # Custom colormap - red for bad, yellow for medium, green for good
    colors = ['#FF6B6B', '#FFE66D', '#4ECDC4']
    cmap = LinearSegmentedColormap.from_list('custom_red_yellow_green', colors, N=100)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize data for better visualization (inverse for metrics where lower is better)
    normalized_data = data.copy()
    normalized_data[:, 0] = 1 - (data[:, 0] / data[:, 0].max())  # Inverse for inference time
    normalized_data[:, 1] = 1 - (data[:, 1] / data[:, 1].max())  # Inverse for memory used
    normalized_data[:, 2] = data[:, 2] / data[:, 2].max()  # Direct for scalability
    
    # Create heatmap
    heatmap = sns.heatmap(normalized_data, 
                         xticklabels=['Inference Time\n(Lower Better)', 
                                     'Memory Used (MB)\n(Lower Better)', 
                                     'Scalability Score\n(Higher Better)'],
                         yticklabels=[m.replace('-', '\n') for m in models],
                         annot=True,
                         fmt='.1f',
                         cmap=cmap,
                         cbar_kws={'label': 'Normalized Performance\n(Higher = Better)'},
                         ax=ax,
                         linewidths=0.5,
                         linecolor='white')
    
    # Customize annotations
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if j == 1:  # Memory column
                text = f'{value:.1f}MB'
            elif j == 0:  # Time column
                text = f'{value:.1f}s'
            else:  # Scalability column
                text = f'{value:.1f}'
            
            ax.text(j + 0.5, i + 0.5, text, 
                   ha='center', va='center', fontweight='bold', fontsize=9)
    
    ax.set_title('Computational Performance and Scalability Heatmap', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save as PDF
    plt.savefig('computational_performance_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def create_combined_radar():
    """Optional: Create radar chart for comprehensive comparison"""
    categories = ['Exact Match', 'CER', 'WER', 'BLEU', 'Structure', 'Semantic', 
                  'Speed', 'Memory Eff.', 'Scalability']
    
    # Normalize all metrics to 0-1 scale (higher is better)
    normalized_metrics = np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25, 0.01, 0.63],  # trocr-math
        [0.0, 0.42, 0.50, 0.5, 1.0, 0.0, 0.89, 0.06, 0.68],  # latex-finetuned
        [0.0, 0.63, 0.00, 0.33, 0.5, 0.0, 0.99, 0.99, 1.00],  # pix2text-mfr-onnx
        [0.0, 0.53, 0.00, 0.0, 0.67, 0.0, 0.01, 0.01, 0.38],  # trocr-vision2seq-math
        [0.0, 0.08, 0.00, 0.0, 0.0, 0.0, 0.98, 0.94, 0.88],  # trocr
        [0.0, 0.13, 0.00, 0.0, 0.0, 0.0, 0.69, 0.08, 0.53],  # trocr-large-handwritten
        [0.0, 0.08, 0.00, 0.0, 0.06, 0.0, 0.97, 0.99, 0.92]   # pix2tex
    ])
    
    # Create radar chart
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, polar=True)
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    
    for i, model in enumerate(models):
        values = normalized_metrics[i].tolist()
        values += values[:1]  # Complete the circle
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i], markersize=4)
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.title('Comprehensive Model Performance Radar Chart', size=14, fontweight='bold', pad=30)
    plt.legend(bbox_to_anchor=(1.2, 1.0), frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    
    # Save as PDF
    plt.savefig('comprehensive_radar_chart.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to generate all visualizations"""
    print("Generating visualizations and saving as PDF files...")
    
    # Create all visualizations
    create_edit_based_metrics()
    print("✓ Saved: edit_based_metrics.pdf")
    
    create_linguistic_semantic_metrics()
    print("✓ Saved: linguistic_semantic_metrics.pdf")
    
    create_computational_heatmap()
    print("✓ Saved: computational_performance_heatmap.pdf")
    
    create_combined_radar()
    print("✓ Saved: comprehensive_radar_chart.pdf")
    
    print("\nAll visualizations have been saved as PDF files in the current directory!")

if __name__ == "__main__":
    main()