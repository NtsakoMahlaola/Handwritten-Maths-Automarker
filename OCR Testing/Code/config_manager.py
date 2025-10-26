"""
Configuration Manager for LaTeX-OCR Benchmarking
Handles all configuration loading, saving, and profile management
"""

import json
import os
from typing import Dict, Optional, Tuple
from pathlib import Path

class ConfigManager:
    def __init__(self, config_file: str = "benchmark_config.json"):
        self.config_file = config_file
        # base directory for resolving relative paths in the config file
        self.base_dir = Path(self.config_file).resolve().parent
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Load configuration from JSON file or create default."""
        default_config = {
            "default_images_dir": "test_images",
            "default_ground_truth": "ground_truth.json", 
            "default_output_dir": "benchmark_results",
            "model_configs": {
                "pix2tex": {
                    "enabled": True,
                    "config": {}
                },
                "latex-ocr": {
                    "enabled": True,
                    "config": {
                        "checkpoint": "latest",
                        "device": "auto"
                    }
                },
                "im2markup": {
                    "enabled": False,
                    "config": {
                        "model_path": "models/im2markup",
                        "vocab_path": "models/im2markup/vocab.pkl"
                    }
                },
                "hmer": {
                    "enabled": False,
                    "config": {
                        "model_path": "models/hmer",
                        "config_path": "models/hmer/config.yaml"
                    }
                },
                "densenet": {
                    "enabled": False,
                    "config": {
                        "model_path": "models/densenet",
                        "attention_type": "coverage"
                    }
                },
                "transformer": {
                    "enabled": False,
                    "config": {
                        "model_path": "models/transformer",
                        "num_heads": 8,
                        "num_layers": 6
                    }
                }
            },
            "benchmark_settings": {
                "max_image_size": [1024, 1024],
                "timeout_seconds": 30,
                "memory_limit_mb": 2048,
                "detailed_metrics": True,
                "save_predictions": True
            },
            "saved_profiles": {
                "quick_test": {
                    "images_dir": r"C:\Users\Naseeka\Desktop\Thesis work\OCR Testing\Handwritten_equations_images",
                    "ground_truth": r"C:\Users\Naseeka\Desktop\Thesis work\OCR Testing\output.json",
                    "output_dir": r"C:\Users\Naseeka\Desktop\Thesis work\OCR Testing\quick_results",
                    "description": "Small dataset for quick testing"
                },
                "full_benchmark": {
                    "images_dir": "full_test_images", 
                    "ground_truth": "full_ground_truth.json",
                    "output_dir": "full_results",
                    "description": "Complete benchmark dataset"
                },
                "crohme_2016": {
                    "images_dir": r"C:\Users\Naseeka\Desktop\Thesis work\OCR Testing\Handwritten_equations_images",
                    "ground_truth": r"C:\Users\Naseeka\Desktop\Thesis work\OCR Testing\output.json",
                    "output_dir": r"C:\Users\Naseeka\Desktop\Thesis work\OCR Testing\quick_results",
                    "description": "CROHME 2016 Competition dataset"
                },
                "crohme_2019": {
                    "images_dir": "datasets/crohme2019/test_images",
                    "ground_truth": "datasets/crohme2019/ground_truth.json", 
                    "output_dir": "results/crohme2019",
                    "description": "CROHME 2019 Competition dataset"
                }
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                # Merge with default config to ensure all keys exist
                return self._merge_configs(default_config, loaded_config)
            except json.JSONDecodeError as e:
                print(f"Warning: {self.config_file} is corrupted ({e}). Using default config.")
                return default_config
        else:
            # Create default config file
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            print(f"Created default configuration file: {self.config_file}")
            return default_config
    
    def _merge_configs(self, default: Dict, loaded: Dict) -> Dict:
        """Recursively merge loaded config with default config."""
        result = default.copy()
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def _resolve_path(self, path_str: Optional[str]) -> Optional[str]:
        """Resolve a path string:
         - expanduser (~)
         - if relative, interpret relative to the config file directory (self.base_dir)
         - return absolute resolved path as string
        """
        if path_str is None or path_str == "":
            return path_str
        try:
            expanded = os.path.expanduser(path_str)
            p = Path(expanded)
            if not p.is_absolute():
                p = (self.base_dir / p)
            # Try to resolve; if it fails, return the joined path
            try:
                return str(p.resolve())
            except Exception:
                return str(p)
        except Exception:
            return path_str

    def save_config(self):
        """Save current configuration to file."""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        print(f"Configuration saved to: {self.config_file}")
    
    def add_profile(self, profile_name: str, images_dir: str, 
                   ground_truth: str, output_dir: str, description: str = ""):
        """Add a new profile to the configuration."""
        self.config["saved_profiles"][profile_name] = {
            "images_dir": images_dir,
            "ground_truth": ground_truth, 
            "output_dir": output_dir,
            "description": description or f"Profile {profile_name}"
        }
        self.save_config()
        print(f"Profile '{profile_name}' added successfully!")
    
    def remove_profile(self, profile_name: str) -> bool:
        """Remove a profile from configuration."""
        if profile_name in self.config["saved_profiles"]:
            del self.config["saved_profiles"][profile_name]
            self.save_config()
            print(f"Profile '{profile_name}' removed successfully!")
            return True
        else:
            print(f"Profile '{profile_name}' not found!")
            return False
    
    def list_profiles(self):
        """List all available profiles with descriptions."""
        print("\n" + "=" * 80)
        print("AVAILABLE BENCHMARK PROFILES")
        print("=" * 80)
        
        profiles = self.config.get("saved_profiles", {})
        if not profiles:
            print("No saved profiles found.")
            return
        
        for profile_name, settings in profiles.items():
            print(f"\n📁 {profile_name.upper()}")
            print(f"   Description: {settings.get('description', 'No description')}")
            # Resolve displayed paths
            images_path = self._resolve_path(settings.get('images_dir', ''))
            gt_path = self._resolve_path(settings.get('ground_truth', ''))
            out_path = self._resolve_path(settings.get('output_dir', ''))
            print(f"   Images:      {images_path}")
            print(f"   Ground Truth: {gt_path}")
            print(f"   Output:      {out_path}")
            
            # Check if paths exist using resolved paths
            status_icons = []
            if images_path and Path(images_path).exists():
                status_icons.append("📂✅")
            else:
                status_icons.append("📂❌")
                
            if gt_path and Path(gt_path).exists():
                status_icons.append("📄✅")
            else:
                status_icons.append("📄❌")
                
            print(f"   Status:      {' '.join(status_icons)}")
        
        print(f"\n📋 Default Settings:")
        # Show defaults resolved relative to config file too
        print(f"   Images:      {self._resolve_path(self.config.get('default_images_dir', 'test_images'))}")
        print(f"   Ground Truth: {self._resolve_path(self.config.get('default_ground_truth', 'ground_truth.json'))}")
        print(f"   Output:      {self._resolve_path(self.config.get('default_output_dir', 'benchmark_results'))}")
    
    def get_profile_settings(self, profile_name: str) -> Optional[Dict]:
        """Get settings for a specific profile."""
        return self.config.get("saved_profiles", {}).get(profile_name)
    
    def get_benchmark_settings(self, profile: Optional[str] = None, 
                              images_dir: Optional[str] = None,
                              ground_truth: Optional[str] = None,
                              output_dir: Optional[str] = None) -> Tuple[str, str, str]:
        """Get benchmark settings from profile or command line arguments."""
        if profile:
            profile_settings = self.get_profile_settings(profile)
            if not profile_settings:
                raise ValueError(f"Profile '{profile}' not found!")
            
            # Use provided CLI args or profile values, then resolve paths
            final_images_dir = images_dir or profile_settings.get('images_dir', '')
            final_ground_truth = ground_truth or profile_settings.get('ground_truth', '')
            final_output_dir = output_dir or profile_settings.get('output_dir', '')
            
            print(f"Using profile: {profile}")
            if profile_settings.get('description'):
                print(f"Description: {profile_settings['description']}")
        else:
            # Use command line arguments or defaults
            final_images_dir = images_dir or self.config.get('default_images_dir', 'test_images')
            final_ground_truth = ground_truth or self.config.get('default_ground_truth', 'ground_truth.json')
            final_output_dir = output_dir or self.config.get('default_output_dir', 'benchmark_results')
        
        # Resolve to absolute paths (expand ~ and make relative->absolute relative to config file)
        final_images_dir = self._resolve_path(final_images_dir)
        final_ground_truth = self._resolve_path(final_ground_truth)
        final_output_dir = self._resolve_path(final_output_dir)
        
        return final_images_dir, final_ground_truth, final_output_dir
    
    def get_model_config(self, model_name: str) -> Dict:
        """Get configuration for a specific model."""
        model_configs = self.config.get("model_configs", {})
        return model_configs.get(model_name, {})
    
    def is_model_enabled(self, model_name: str) -> bool:
        """Check if a model is enabled in configuration."""
        model_config = self.get_model_config(model_name)
        return model_config.get("enabled", False)
    
    def get_benchmark_config(self) -> Dict:
        """Get general benchmark settings."""
        return self.config.get("benchmark_settings", {})
    
    def update_model_config(self, model_name: str, config_updates: Dict):
        """Update configuration for a specific model."""
        if "model_configs" not in self.config:
            self.config["model_configs"] = {}
        
        if model_name not in self.config["model_configs"]:
            self.config["model_configs"][model_name] = {"enabled": False, "config": {}}
        
        # Update the model configuration
        self.config["model_configs"][model_name]["config"].update(config_updates)
        self.save_config()
        print(f"Configuration updated for model: {model_name}")
    
    def enable_model(self, model_name: str):
        """Enable a model in the configuration."""
        if "model_configs" not in self.config:
            self.config["model_configs"] = {}
        
        if model_name not in self.config["model_configs"]:
            self.config["model_configs"][model_name] = {"enabled": True, "config": {}}
        else:
            self.config["model_configs"][model_name]["enabled"] = True
        
        self.save_config()
        print(f"Model '{model_name}' enabled.")
    
    def disable_model(self, model_name: str):
        """Disable a model in the configuration."""
        if model_name in self.config.get("model_configs", {}):
            self.config["model_configs"][model_name]["enabled"] = False
            self.save_config()
            print(f"Model '{model_name}' disabled.")
        else:
            print(f"Model '{model_name}' not found in configuration.")