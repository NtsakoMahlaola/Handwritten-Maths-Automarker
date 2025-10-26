# README.md - Installation and Usage Instructions

# LaTeX-OCR Multi-Model Benchmarking System

A comprehensive benchmarking system for evaluating multiple LaTeX-OCR models on mathematical expression recognition tasks.

## Features

- **Multi-Model Support**: Benchmark multiple LaTeX-OCR models simultaneously
- **Comprehensive Metrics**: Exact match, character accuracy, BLEU score, and more
- **Performance Monitoring**: CPU, memory, and GPU usage tracking
- **Flexible Configuration**: Profile-based configuration management
- **Detailed Reporting**: CSV, JSON, and human-readable reports
- **Cross-Model Comparison**: Generate comparison reports across all models

## Supported Models

- **Pix2Tex** ✅ - Ready to use
- **LaTeX-OCR (lukas-blecher)** ✅ - Ready to use
- **Im2Markup (Harvard NLP)** 🚧 - Setup required
- **HMER models** 🚧 - Setup required
- **DenseNet + Attention** 🚧 - Implementation required
- **Transformer-based HME** 🚧 - Implementation required

## Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/latex-ocr-benchmark.git
cd latex-ocr-benchmark

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### GPU Support (Optional but Recommended)

```bash
# For CUDA support, install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install GPU monitoring
pip install GPUtil
```

### Development Installation

```bash
pip install -e ".[dev]"
pre-commit install
```

## Quick Start

### 1. Prepare Your Dataset

Create a directory structure:
```
my_dataset/
├── images/
│   ├── equation_001.png
│   ├── equation_002.png
│   └── ...
└── ground_truth.json
```

Ground truth format (`ground_truth.json`):
```json
{
    "equation_001.png": "x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}",
    "equation_002.png": "E = mc^2",
    ...
}
```

### 2. Run Benchmarks

```bash
# Benchmark a single model
python main.py --model pix2tex --images-dir my_dataset/images --ground-truth my_dataset/ground_truth.json

# Benchmark all available models
python main.py --model all --images-dir my_dataset/images --ground-truth my_dataset/ground_truth.json

# Use a predefined profile
python main.py --model all --profile quick_test

# Generate cross-model comparison
python main.py --model all --profile full_benchmark --compare-models
```

### 3. Check Results

Results are saved in the specified output directory with timestamps:
- `model_name_detailed_predictions_YYYYMMDD_HHMMSS.csv`
- `model_name_summary_YYYYMMDD_HHMMSS.json`
- `model_name_report_YYYYMMDD_HHMMSS.txt`
- `model_comparison_YYYYMMDD_HHMMSS.csv` (when using --compare-models)

## Configuration Management

### List Available Profiles
```bash
python main.py --list-profiles
```

### Add New Profile
```bash
python main.py --add-profile "my_dataset:./images:./ground_truth.json:./results"
```

### List Available Models
```bash
python main.py --list-models
```

## Advanced Usage

### Custom Configuration

Edit `benchmark_config.json` to customize:
- Model-specific settings
- Benchmark parameters
- Resource limits
- Output formats

### Extending with New Models

1. Create a new model class inheriting from `BaseLatexOCRModel`
2. Implement required methods: `initialize()`, `predict()`, `model_name`, `requires_packages`
3. Register the model in `ModelFactory`

Example:
```python
class MyCustomModel(BaseLatexOCRModel):
    @property
    def model_name(self) -> str:
        return "my_custom_model"
    
    @property
    def requires_packages(self) -> List[str]:
        return ["my_model_package"]
    
    def initialize(self) -> bool:
        # Initialize your model
        pass
    
    def predict(self, image_path: str) -> Optional[str]:
        # Make prediction
        pass

# Register the model
ModelFactory.register_model("my_custom_model", MyCustomModel)
```

## Model Setup Instructions

### Pix2Tex (Ready to Use)
```bash
pip install pix2tex
```

### LaTeX-OCR (Ready to Use)
```bash
pip install pix2tex  # Same package as Pix2Tex
```

### Im2Markup (Manual Setup Required)
1. Clone the repository: `git clone https://github.com/harvardnlp/im2markup`
2. Download pre-trained models
3. Update model paths in configuration

### HMER Models (Manual Setup Required)
1. Visit: https://github.com/ZZR0/awesome-hmer
2. Choose a specific HMER implementation
3. Download CROHME-trained models
4. Update configuration with model and config paths

## Performance Optimization

### GPU Usage
- Ensure CUDA-compatible PyTorch is installed
- Use `--model all` to benchmark GPU vs CPU performance
- Monitor GPU memory usage in results

### Memory Management
- Use `--quick-mode` for reduced memory usage
- Adjust `memory_limit_mb` in configuration
- Clean up GPU memory between model runs

### Large Datasets
- Process in batches for very large datasets
- Use profiles for different dataset sizes
- Monitor system resources during benchmarking

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install missing dependencies
   ```bash
   python main.py --list-models  # Check which models are available
   pip install <missing-package>
   ```

2. **CUDA Out of Memory**: Reduce batch size or use CPU
   ```bash
   export CUDA_VISIBLE_DEVICES=""  # Force CPU usage
   ```

3. **Permission Errors**: Check output directory permissions
   ```bash
   chmod 755 /path/to/output/directory
   ```

4. **Ground Truth Format**: Ensure JSON format is correct
   ```bash
   python -m json.tool ground_truth.json  # Validate JSON
   ```

### Debug Mode
```bash
python main.py --verbose --model pix2tex --profile quick_test
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Setup
```bash
git clone https://github.com/yourusername/latex-ocr-benchmark.git
cd latex-ocr-benchmark
pip install -e ".[dev]"
pre-commit install
pytest
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use this benchmarking system in your research, please cite:

```bibtex
@software{latex_ocr_benchmark,
    title={LaTeX-OCR Multi-Model Benchmarking System},
    author={Your Name},
    year={2024},
    url={https://github.com/yourusername/latex-ocr-benchmark}
}
```

## Acknowledgments

- Pix2Tex developers for the original LaTeX-OCR implementation
- Contributors to the LaTeX-OCR and HMER communities
- CROHME competition organizers for providing evaluation datasets
