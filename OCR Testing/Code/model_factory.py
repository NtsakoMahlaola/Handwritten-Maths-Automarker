"""
Model Factory for LaTeX-OCR Models
Handles instantiation and management of different OCR models
"""

import importlib
import sys
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BaseLatexOCRModel(ABC):
    """Abstract base class for LaTeX-OCR models."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.model = None
        self.is_initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the model. Returns True if successful."""
        pass
    
    @abstractmethod
    def predict(self, image_path: str) -> Optional[str]:
        """Predict LaTeX from image. Returns LaTeX string or None if failed."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass
    
    @property
    @abstractmethod
    def requires_packages(self) -> List[str]:
        """Return list of required packages."""
        pass
    
    def cleanup(self):
        """Clean up model resources."""
        if hasattr(self.model, 'cpu'):
            try:
                self.model.cpu()
            except:
                pass
        self.model = None
        self.is_initialized = False

class Pix2TexModel(BaseLatexOCRModel):
    """Pix2Tex LaTeX-OCR model wrapper."""
    
    @property
    def model_name(self) -> str:
        return "pix2tex"
    
    @property
    def requires_packages(self) -> List[str]:
        return ["pix2tex"]
    
    def initialize(self) -> bool:
        """Initialize Pix2Tex model."""
        try:
            from pix2tex.cli import LatexOCR
            self.model = LatexOCR()
            self.is_initialized = True
            logger.info("Pix2Tex model initialized successfully")
            return True
        except ImportError as e:
            logger.error(f"Failed to import pix2tex: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Pix2Tex: {e}")
            return False
    
    def predict(self, image_path: str) -> Optional[str]:
        """Predict LaTeX from image using Pix2Tex."""
        if not self.is_initialized:
            return None
        
        try:
            from PIL import Image
            img = Image.open(image_path).convert('RGB')
            return self.model(img)
        except Exception as e:
            logger.error(f"Pix2Tex prediction failed for {image_path}: {e}")
            return None
class TrOCRTunedModel(BaseLatexOCRModel):
    """TrOCR model fine-tuned by Ntsako12 for mathematical expressions."""
    
    @property
    def model_name(self) -> str:
        return "trocr-tuned"
    
    @property
    def requires_packages(self) -> List[str]:
        return ["transformers", "torch", "PIL", "requests"]
    
    def initialize(self) -> bool:
        """Initialize TrOCR Tuned model."""
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            # Use the specific tuned model
            model_variant = self.config.get('model_variant', 'Ntsako12/TrOCR_Tuned')
            
            self.processor = TrOCRProcessor.from_pretrained(model_variant)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_variant)
            
            self.is_initialized = True
            logger.info(f"TrOCR Tuned model initialized successfully with variant: {model_variant}")
            return True
        except ImportError as e:
            logger.error(f"Failed to import TrOCR Tuned dependencies: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize TrOCR Tuned: {e}")
            return False
    
    def predict(self, image_path: str) -> Optional[str]:
        """Predict mathematical expressions from image using TrOCR Tuned."""
        if not self.is_initialized:
            return None
        
        try:
            from PIL import Image
            
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            
            # Process image and generate prediction
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return generated_text
        except Exception as e:
            logger.error(f"TrOCR Tuned prediction failed for {image_path}: {e}")
            return None
    
class LatexOCRModel(BaseLatexOCRModel):
    """LaTeX-OCR (lukas-blecher) model wrapper."""
    
    @property
    def model_name(self) -> str:
        return "latex-ocr"
    
    @property
    def requires_packages(self) -> List[str]:
        return ["pix2tex", "torch", "torchvision", "transformers"]
    
    def initialize(self) -> bool:
        """Initialize LaTeX-OCR model."""
        try:
            from pix2tex import cli
            # Try to use the lukas-blecher version if available
            self.model = cli.LatexOCR()
            self.is_initialized = True
            logger.info("LaTeX-OCR model initialized successfully")
            return True
        except ImportError as e:
            logger.error(f"Failed to import latex-ocr dependencies: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize LaTeX-OCR: {e}")
            return False
    
    def predict(self, image_path: str) -> Optional[str]:
        """Predict LaTeX from image using LaTeX-OCR."""
        if not self.is_initialized:
            return None
        
        try:
            from PIL import Image
            img = Image.open(image_path).convert('RGB')
            return self.model(img)
        except Exception as e:
            logger.error(f"LaTeX-OCR prediction failed for {image_path}: {e}")
            return None

class TrOCRModel(BaseLatexOCRModel):
    """TrOCR (Transformer OCR) model wrapper for handwritten text recognition."""
    
    @property
    def model_name(self) -> str:
        return "trocr"
    
    @property
    def requires_packages(self) -> List[str]:
        return ["transformers", "torch", "PIL", "requests"]
    
    def initialize(self) -> bool:
        """Initialize TrOCR model."""
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            # Get model variant from config or use default
            model_variant = self.config.get('model_variant', 'microsoft/trocr-small-handwritten')
            
            self.processor = TrOCRProcessor.from_pretrained(model_variant)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_variant)
            
            self.is_initialized = True
            logger.info(f"TrOCR model initialized successfully with variant: {model_variant}")
            return True
        except ImportError as e:
            logger.error(f"Failed to import TrOCR dependencies: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize TrOCR: {e}")
            return False
    
    def predict(self, image_path: str) -> Optional[str]:
        """Predict text from image using TrOCR."""
        if not self.is_initialized:
            return None
        
        try:
            from PIL import Image
            
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            
            # Process image and generate prediction
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return generated_text
        except Exception as e:
            logger.error(f"TrOCR prediction failed for {image_path}: {e}")
            return None

class TrOCRMathModel(BaseLatexOCRModel):
    """TrOCR model fine-tuned for handwritten mathematical expressions."""
    
    @property
    def model_name(self) -> str:
        return "trocr-math"
    
    @property
    def requires_packages(self) -> List[str]:
        return ["transformers", "torch", "PIL", "requests"]
    
    def initialize(self) -> bool:
        """Initialize TrOCR Math model."""
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            # Get model variant from config or use default
            model_variant = self.config.get('model_variant', 'fhswf/TrOCR_Math_handwritten')
            
            self.processor = TrOCRProcessor.from_pretrained(model_variant)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_variant)
            
            self.is_initialized = True
            logger.info(f"TrOCR Math model initialized successfully with variant: {model_variant}")
            return True
        except ImportError as e:
            logger.error(f"Failed to import TrOCR Math dependencies: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize TrOCR Math: {e}")
            return False
    
    def predict(self, image_path: str) -> Optional[str]:
        """Predict mathematical expressions from image using TrOCR Math."""
        if not self.is_initialized:
            return None
        
        try:
            from PIL import Image
            
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            
            # Process image and generate prediction
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return generated_text
        except Exception as e:
            logger.error(f"TrOCR Math prediction failed for {image_path}: {e}")
            return None
        
class TrOCRMathModel2(BaseLatexOCRModel):
    """TrOCR model using microsoft/trocr-large-handwritten (large variant)."""
    
    @property
    def model_name(self) -> str:
        return "trocr-large-handwritten"
    
    @property
    def requires_packages(self) -> List[str]:
        return ["transformers", "torch", "PIL", "requests"]
    
    def initialize(self) -> bool:
        """Initialize the large TrOCR handwritten model."""
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            # Force use of the large handwritten variant
            model_variant = 'microsoft/trocr-large-handwritten'
            
            self.processor = TrOCRProcessor.from_pretrained(model_variant)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_variant)
            
            self.is_initialized = True
            logger.info(f"TrOCR Math model initialized successfully with variant: {model_variant}")
            return True
        except ImportError as e:
            logger.error(f"Failed to import TrOCR dependencies: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize TrOCR Math large handwritten: {e}")
            return False
    
    def predict(self, image_path: str) -> Optional[str]:
        """Predict text from image using TrOCR large handwritten."""
        if not self.is_initialized:
            return None
        
        try:
            from PIL import Image
            
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return generated_text
        except Exception as e:
            logger.error(f"TrOCR Math prediction failed for {image_path}: {e}")
            return None

class TrOCRMathModel3(BaseLatexOCRModel):
    """TrOCR model using bot343565/fine-tuned-TrOCR-Math with Vision2Seq architecture."""
    
    @property
    def model_name(self) -> str:
        return "trocr-vision2seq-math"
    
    @property
    def requires_packages(self) -> List[str]:
        return ["transformers", "torch", "PIL", "requests"]
    
    def initialize(self) -> bool:
        """Initialize Vision2Seq TrOCR Math model."""
        try:
            from transformers import AutoTokenizer, AutoModelForVision2Seq, pipeline
            
            # Get model variant from config or use default
            model_variant = self.config.get('model_variant', 'bot343565/fine-tuned-TrOCR-Math')
            
            # Option 1: Use pipeline (simpler)
            use_pipeline = self.config.get('use_pipeline', True)
            
            if use_pipeline:
                self.pipe = pipeline("image-to-text", model=model_variant)
                logger.info(f"Using pipeline for Vision2Seq TrOCR Math model: {model_variant}")
            else:
                # Option 2: Use components directly
                self.tokenizer = AutoTokenizer.from_pretrained(model_variant)
                self.model = AutoModelForVision2Seq.from_pretrained(model_variant)
                logger.info(f"Using direct components for Vision2Seq TrOCR Math model: {model_variant}")
            
            self.is_initialized = True
            logger.info(f"Vision2Seq TrOCR Math model initialized successfully with variant: {model_variant}")
            return True
        except ImportError as e:
            logger.error(f"Failed to import Vision2Seq TrOCR Math dependencies: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Vision2Seq TrOCR Math: {e}")
            return False
    
    def predict(self, image_path: str) -> Optional[str]:
        """Predict mathematical expressions from image using Vision2Seq TrOCR Math."""
        if not self.is_initialized:
            return None
        
        try:
            from PIL import Image
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Use pipeline if available, otherwise use components directly
            if hasattr(self, 'pipe'):
                # Use pipeline
                result = self.pipe(image)
                return result[0]['generated_text'] if result else None
            else:
                # Use components directly
                inputs = self.tokenizer(images=image, return_tensors="pt")
                generated_ids = self.model.generate(**inputs)
                generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return generated_text
                
        except Exception as e:
            logger.error(f"Vision2Seq TrOCR Math prediction failed for {image_path}: {e}")
            return None

class Pix2TextMFRModel(BaseLatexOCRModel):
    """Pix2Text MFR (Math Formula Recognition) model from breezedeus."""
    
    @property
    def model_name(self) -> str:
        return "pix2text-mfr"
    
    @property
    def requires_packages(self) -> List[str]:
        return ["transformers", "torch", "PIL", "requests"]
    
    def initialize(self) -> bool:
        """Initialize Pix2Text MFR model."""
        try:
            from transformers import AutoTokenizer, AutoModelForVision2Seq, pipeline
            
            # Get model variant from config or use default
            model_variant = self.config.get('model_variant', 'breezedeus/pix2text-mfr-1.5')
            
            # Option 1: Use pipeline (simpler)
            use_pipeline = self.config.get('use_pipeline', True)
            
            if use_pipeline:
                self.pipe = pipeline("image-to-text", model=model_variant)
                logger.info(f"Using pipeline for Pix2Text MFR model: {model_variant}")
            else:
                # Option 2: Use components directly
                self.tokenizer = AutoTokenizer.from_pretrained(model_variant)
                self.model = AutoModelForVision2Seq.from_pretrained(model_variant)
                logger.info(f"Using direct components for Pix2Text MFR model: {model_variant}")
            
            self.is_initialized = True
            logger.info(f"Pix2Text MFR model initialized successfully with variant: {model_variant}")
            return True
        except ImportError as e:
            logger.error(f"Failed to import Pix2Text MFR dependencies: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Pix2Text MFR: {e}")
            return False
    
    def predict(self, image_path: str) -> Optional[str]:
        """Predict mathematical expressions from image using Pix2Text MFR."""
        if not self.is_initialized:
            return None
        
        try:
            from PIL import Image
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Use pipeline if available, otherwise use components directly
            if hasattr(self, 'pipe'):
                # Use pipeline
                result = self.pipe(image)
                return result[0]['generated_text'] if result else None
            else:
                # Use components directly
                inputs = self.tokenizer(images=image, return_tensors="pt")
                generated_ids = self.model.generate(**inputs)
                generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return generated_text
                
        except Exception as e:
            logger.error(f"Pix2Text MFR prediction failed for {image_path}: {e}")
            return None

class Pix2TextMFRONNXModel(BaseLatexOCRModel):
    """Pix2Text MFR model optimized with ONNX runtime for faster inference."""
    
    @property
    def model_name(self) -> str:
        return "pix2text-mfr-onnx"
    
    @property
    def requires_packages(self) -> List[str]:
        return ["transformers", "optimum", "onnxruntime", "PIL", "torch"]
    
    def initialize(self) -> bool:
        """Initialize Pix2Text MFR ONNX model."""
        try:
            from transformers import TrOCRProcessor
            from optimum.onnxruntime import ORTModelForVision2Seq
            
            # Get model variant from config or use default
            model_variant = self.config.get('model_variant', 'breezedeus/pix2text-mfr')
            
            # Initialize processor and ONNX model
            self.processor = TrOCRProcessor.from_pretrained(model_variant)
            self.model = ORTModelForVision2Seq.from_pretrained(model_variant, use_cache=False)
            
            self.is_initialized = True
            logger.info(f"Pix2Text MFR ONNX model initialized successfully with variant: {model_variant}")
            return True
        except ImportError as e:
            logger.error(f"Failed to import Pix2Text MFR ONNX dependencies: {e}")
            logger.info("To use ONNX model, install: pip install optimum[onnxruntime]")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Pix2Text MFR ONNX: {e}")
            return False
    
    def predict(self, image_path: str) -> Optional[str]:
        """Predict mathematical expressions from image using Pix2Text MFR ONNX."""
        if not self.is_initialized:
            return None
        
        try:
            from PIL import Image
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Process image and generate prediction
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return generated_text
                
        except Exception as e:
            logger.error(f"Pix2Text MFR ONNX prediction failed for {image_path}: {e}")
            return None

class LatexFinetunedModel(BaseLatexOCRModel):
    """LaTeX fine-tuned model from tjoab/latex_finetuned."""
    
    @property
    def model_name(self) -> str:
        return "latex-finetuned"
    
    @property
    def requires_packages(self) -> List[str]:
        return ["transformers", "torch", "PIL", "requests"]
    
    def initialize(self) -> bool:
        """Initialize LaTeX fine-tuned model."""
        try:
            from transformers import AutoTokenizer, AutoModelForVision2Seq, pipeline
            
            # Get model variant from config or use default
            model_variant = self.config.get('model_variant', 'tjoab/latex_finetuned')
            
            # Option 1: Use pipeline (simpler)
            use_pipeline = self.config.get('use_pipeline', True)
            
            if use_pipeline:
                self.pipe = pipeline("image-to-text", model=model_variant)
                logger.info(f"Using pipeline for LaTeX fine-tuned model: {model_variant}")
            else:
                # Option 2: Use components directly
                self.tokenizer = AutoTokenizer.from_pretrained(model_variant)
                self.model = AutoModelForVision2Seq.from_pretrained(model_variant)
                logger.info(f"Using direct components for LaTeX fine-tuned model: {model_variant}")
            
            self.is_initialized = True
            logger.info(f"LaTeX fine-tuned model initialized successfully with variant: {model_variant}")
            return True
        except ImportError as e:
            logger.error(f"Failed to import LaTeX fine-tuned model dependencies: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize LaTeX fine-tuned model: {e}")
            return False
    
    def predict(self, image_path: str) -> Optional[str]:
        """Predict LaTeX from image using the fine-tuned model."""
        if not self.is_initialized:
            return None
        
        try:
            from PIL import Image
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Use pipeline if available, otherwise use components directly
            if hasattr(self, 'pipe'):
                # Use pipeline
                result = self.pipe(image)
                return result[0]['generated_text'] if result else None
            else:
                # Use components directly
                inputs = self.tokenizer(images=image, return_tensors="pt")
                generated_ids = self.model.generate(**inputs)
                generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return generated_text
                
        except Exception as e:
            logger.error(f"LaTeX fine-tuned model prediction failed for {image_path}: {e}")
            return None


class Im2MarkupModel(BaseLatexOCRModel):
    """Im2Markup (Harvard NLP) model wrapper."""
    
    @property
    def model_name(self) -> str:
        return "im2markup"
    
    @property
    def requires_packages(self) -> List[str]:
        return ["torch", "torchvision", "numpy", "PIL"]
    
    def initialize(self) -> bool:
        """Initialize Im2Markup model."""
        try:
            # This is a placeholder implementation
            # The actual implementation would require the Im2Markup repository
            logger.warning("Im2Markup model is not yet implemented")
            logger.info("To use Im2Markup:")
            logger.info("1. Clone: git clone https://github.com/harvardnlp/im2markup")
            logger.info("2. Download pre-trained models")
            logger.info("3. Configure model_path and vocab_path in config")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Im2Markup: {e}")
            return False
    
    def predict(self, image_path: str) -> Optional[str]:
        """Predict LaTeX from image using Im2Markup."""
        # Placeholder implementation
        logger.warning("Im2Markup prediction not implemented")
        return None

class HMERModel(BaseLatexOCRModel):
    """HMER (Handwritten Math Expression Recognition) model wrapper."""
    
    @property
    def model_name(self) -> str:
        return "hmer"
    
    @property
    def requires_packages(self) -> List[str]:
        return ["torch", "torchvision", "opencv-python", "yaml"]
    
    def initialize(self) -> bool:
        """Initialize HMER model."""
        try:
            # Placeholder implementation
            logger.warning("HMER model is not yet implemented")
            logger.info("To use HMER models:")
            logger.info("1. Visit: https://github.com/ZZR0/awesome-hmer")
            logger.info("2. Choose and clone a specific HMER repository")
            logger.info("3. Download CROHME-trained models")
            logger.info("4. Configure model_path and config_path")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize HMER: {e}")
            return False
    
    def predict(self, image_path: str) -> Optional[str]:
        """Predict LaTeX from image using HMER."""
        # Placeholder implementation
        logger.warning("HMER prediction not implemented")
        return None

class DenseNetModel(BaseLatexOCRModel):
    """DenseNet + Attention model wrapper."""
    
    @property
    def model_name(self) -> str:
        return "densenet"
    
    @property
    def requires_packages(self) -> List[str]:
        return ["torch", "torchvision", "numpy"]
    
    def initialize(self) -> bool:
        """Initialize DenseNet model."""
        try:
            # Placeholder implementation
            logger.warning("DenseNet model is not yet implemented")
            logger.info("To use DenseNet + Attention:")
            logger.info("1. Implement CNN-Seq2Seq architecture")
            logger.info("2. Train on CROHME dataset or use pre-trained weights")
            logger.info("3. Configure model_path and attention_type")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize DenseNet: {e}")
            return False
    
    def predict(self, image_path: str) -> Optional[str]:
        """Predict LaTeX from image using DenseNet."""
        # Placeholder implementation
        logger.warning("DenseNet prediction not implemented")
        return None

class TransformerModel(BaseLatexOCRModel):
    """Transformer-based HME model wrapper."""
    
    @property
    def model_name(self) -> str:
        return "transformer"
    
    @property
    def requires_packages(self) -> List[str]:
        return ["torch", "transformers", "numpy"]
    
    def initialize(self) -> bool:
        """Initialize Transformer model."""
        try:
            # Placeholder implementation
            logger.warning("Transformer model is not yet implemented")
            logger.info("To use Transformer-based HME:")
            logger.info("1. Implement encoder-decoder Transformer")
            logger.info("2. Train on CROHME dataset")
            logger.info("3. Configure model_path, num_heads, num_layers")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Transformer: {e}")
            return False
    
    def predict(self, image_path: str) -> Optional[str]:
        """Predict LaTeX from image using Transformer."""
        # Placeholder implementation  
        logger.warning("Transformer prediction not implemented")
        return None

class ModelFactory:
    """Factory class for creating and managing LaTeX-OCR models."""
    
    _models = {
        "pix2tex": Pix2TexModel,
        "latex-ocr": LatexOCRModel,
        "trocr": TrOCRModel,
        "trocr-math": TrOCRMathModel,
        "trocr-tuned": TrOCRTunedModel,
        "trocr-large-handwritten": TrOCRMathModel2,
        "trocr-vision2seq-math": TrOCRMathModel3,
        "pix2text-mfr": Pix2TextMFRModel,
        "pix2text-mfr-onnx": Pix2TextMFRONNXModel,
        "latex-finetuned": LatexFinetunedModel,
        "im2markup": Im2MarkupModel,
        "hmer": HMERModel,
        "densenet": DenseNetModel,
        "transformer": TransformerModel
    }
    
    @classmethod
    def create_model(cls, model_name: str, config: Dict = None) -> Optional[BaseLatexOCRModel]:
        """Create a model instance by name."""
        if model_name not in cls._models:
            logger.error(f"Unknown model: {model_name}")
            return None
        
        try:
            model_class = cls._models[model_name]
            return model_class(config)
        except Exception as e:
            logger.error(f"Failed to create model {model_name}: {e}")
            return None
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available model names."""
        return list(cls._models.keys())
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[Dict]:
        """Get information about a specific model."""
        if model_name not in cls._models:
            return None
        
        model_class = cls._models[model_name]
        # Create temporary instance to get info
        temp_instance = model_class()
        
        return {
            "name": temp_instance.model_name,
            "requires_packages": temp_instance.requires_packages,
            "class": model_class.__name__,
            "available": cls._check_model_availability(model_name)
        }
    
    @classmethod
    def _check_model_availability(cls, model_name: str) -> bool:
        """Check if a model's dependencies are available."""
        if model_name not in cls._models:
            return False
        
        model_class = cls._models[model_name]
        temp_instance = model_class()
        
        # Check if required packages are available
        for package in temp_instance.requires_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                return False
        
        return True
    
    @classmethod
    def list_available_models(cls):
        """Print detailed information about all models."""
        print("\n" + "=" * 80)
        print("AVAILABLE LATEX-OCR MODELS")
        print("=" * 80)
        
        for model_name in cls.get_available_models():
            info = cls.get_model_info(model_name)
            status = "✅ Available" if info["available"] else "❌ Dependencies Missing"
            
            print(f"\n🔤 {model_name.upper()}")
            print(f"   Status: {status}")
            print(f"   Required packages: {', '.join(info['requires_packages'])}")
            
            if not info["available"]:
                print(f"   💡 Install missing dependencies to enable this model")
        
        # Show installation instructions
        print(f"\n📦 INSTALLATION INSTRUCTIONS")
        print(f"   pix2tex:     pip install pix2tex")
        print(f"   latex-ocr:   pip install pix2tex (same package)")
        print(f"   trocr:       pip install transformers torch")
        print(f"   trocr-math:  pip install transformers torch")
        print(f"   trocr-vision2seq-math: pip install transformers torch")
        print(f"   pix2text-mfr: pip install transformers torch")
        print(f"   pix2text-mfr-onnx: pip install transformers optimum[onnxruntime] torch")
        print(f"   latex-finetuned: pip install transformers torch")
        print(f"   im2markup:   Manual setup required (see GitHub)")
        print(f"   hmer:        Manual setup required (see awesome-hmer)")
        print(f"   densenet:    Implementation required")
        print(f"   transformer: Implementation required")
    
    @classmethod
    def register_model(cls, name: str, model_class: type):
        """Register a new model class."""
        if not issubclass(model_class, BaseLatexOCRModel):
            raise ValueError("Model class must inherit from BaseLatexOCRModel")
        
        cls._models[name] = model_class
        logger.info(f"Registered new model: {name}")
    
    @classmethod
    def get_working_models(cls) -> List[str]:
        """Get list of models that have their dependencies available."""
        working_models = []
        for model_name in cls.get_available_models():
            if cls._check_model_availability(model_name):
                working_models.append(model_name)
        return working_models