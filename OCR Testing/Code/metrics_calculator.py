"""
Metrics Calculator for LaTeX-OCR Benchmarking
Handles all accuracy and similarity metric calculations
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
import Levenshtein
import numpy as np

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Comprehensive metrics calculator for LaTeX-OCR evaluation."""
    
    def __init__(self):
        self.latex_commands = self._load_common_latex_commands()
    
    def _load_common_latex_commands(self) -> set:
        """Load common LaTeX commands for enhanced tokenization."""
        return {
            'frac', 'sqrt', 'sum', 'int', 'prod', 'lim', 'sin', 'cos', 'tan',
            'log', 'ln', 'exp', 'alpha', 'beta', 'gamma', 'delta', 'epsilon',
            'theta', 'lambda', 'mu', 'pi', 'sigma', 'phi', 'psi', 'omega',
            'infty', 'partial', 'nabla', 'cdot', 'times', 'div', 'pm', 'mp',
            'leq', 'geq', 'neq', 'approx', 'equiv', 'sim', 'propto',
            'subset', 'supset', 'subseteq', 'supseteq', 'in', 'notin',
            'cup', 'cap', 'emptyset', 'mathbb', 'mathcal', 'mathrm',
            'left', 'right', 'big', 'Big', 'bigg', 'Bigg'
        }
    
    def calculate_all_metrics(self, prediction: str, ground_truth: str, 
                             quick_mode: bool = False) -> Dict[str, float]:
        """Calculate comprehensive accuracy metrics."""
        if prediction is None or ground_truth is None:
            return self._get_failed_metrics(ground_truth)
        
        # Clean strings for comparison
        pred_clean = self._clean_latex_string(prediction)
        truth_clean = self._clean_latex_string(ground_truth)
        
        # Basic metrics (always calculated)
        metrics = {
            'exact_match': self._calculate_exact_match(pred_clean, truth_clean),
            'levenshtein_distance': self._calculate_levenshtein_distance(pred_clean, truth_clean),
            'character_accuracy': self._calculate_character_accuracy(pred_clean, truth_clean),
            'prediction_length': len(pred_clean),
            'ground_truth_length': len(truth_clean),
            'length_difference': abs(len(pred_clean) - len(truth_clean)),
            'prediction_clean': pred_clean,
            'ground_truth_clean': truth_clean
        }
        
        # Extended metrics (skipped in quick mode)
        if not quick_mode:
            metrics.update({
                'word_error_rate': self._calculate_word_error_rate(pred_clean, truth_clean),
                'token_accuracy': self._calculate_token_accuracy(pred_clean, truth_clean),
                'latex_command_accuracy': self._calculate_latex_command_accuracy(pred_clean, truth_clean),
                'structural_similarity': self._calculate_structural_similarity(pred_clean, truth_clean),
                'semantic_similarity': self._calculate_semantic_similarity(pred_clean, truth_clean),
                'bleu_score': self._calculate_bleu_score(pred_clean, truth_clean),
                'jaccard_similarity': self._calculate_jaccard_similarity(pred_clean, truth_clean)
            })
        
        return metrics
    
    def _get_failed_metrics(self, ground_truth: Optional[str]) -> Dict[str, float]:
        """Return metrics for failed predictions."""
        return {
            'exact_match': 0,
            'levenshtein_distance': float('inf'),
            'word_error_rate': 1.0,
            'character_accuracy': 0,
            'token_accuracy': 0,
            'latex_command_accuracy': 0,
            'structural_similarity': 0,
            'semantic_similarity': 0,
            'bleu_score': 0,
            'jaccard_similarity': 0,
            'prediction_length': 0,
            'ground_truth_length': len(ground_truth) if ground_truth else 0,
            'length_difference': len(ground_truth) if ground_truth else 0,
            'prediction_clean': '',
            'ground_truth_clean': ground_truth or ''
        }
    
    def _clean_latex_string(self, latex_str: str) -> str:
        """Clean and normalize LaTeX string for comparison."""
        if not latex_str:
            return ""
        
        # Remove excessive whitespace
        cleaned = ' '.join(latex_str.split())
        
        # Normalize common LaTeX variations
        cleaned = re.sub(r'\\left\s*([(){}[\]|])', r'\1', cleaned)
        cleaned = re.sub(r'\\right\s*([(){}[\]|])', r'\1', cleaned)
        cleaned = re.sub(r'\s*{\s*', '{', cleaned)
        cleaned = re.sub(r'\s*}\s*', '}', cleaned)
        cleaned = re.sub(r'\s*_\s*', '_', cleaned)
        cleaned = re.sub(r'\s*\^\s*', '^', cleaned)
        
        return cleaned.strip()
    
    def _calculate_exact_match(self, pred: str, truth: str) -> int:
        """Calculate exact string match."""
        return 1 if pred == truth else 0
    
    def _calculate_levenshtein_distance(self, pred: str, truth: str) -> int:
        """Calculate Levenshtein (edit) distance."""
        return Levenshtein.distance(pred, truth)
    
    def _calculate_character_accuracy(self, pred: str, truth: str) -> float:
        """Calculate character-level accuracy."""
        if not truth:
            return 1.0 if not pred else 0.0
        
        max_len = max(len(pred), len(truth))
        if max_len == 0:
            return 1.0
        
        lev_distance = self._calculate_levenshtein_distance(pred, truth)
        return max(0, 1 - (lev_distance / max_len))
    
    def _calculate_word_error_rate(self, pred: str, truth: str) -> float:
        """Calculate Word Error Rate (WER)."""
        pred_tokens = pred.split()
        truth_tokens = truth.split()
        
        if not truth_tokens:
            return 0.0 if not pred_tokens else 1.0
        
        # Use Levenshtein distance on token sequences
        wer = Levenshtein.distance(pred_tokens, truth_tokens) / len(truth_tokens)
        return min(1.0, wer)  # Cap at 1.0
    
    def _calculate_token_accuracy(self, pred: str, truth: str) -> float:
        """Calculate token-based accuracy."""
        pred_tokens = pred.split()
        truth_tokens = truth.split()
        
        if not truth_tokens:
            return 1.0 if not pred_tokens else 0.0
        
        # Calculate overlap
        correct_tokens = sum(1 for p, t in zip(pred_tokens, truth_tokens) if p == t)
        return correct_tokens / len(truth_tokens)
    
    def _tokenize_latex(self, latex_str: str) -> List[str]:
        """Tokenize LaTeX string into meaningful components."""
        tokens = []
        i = 0
        
        while i < len(latex_str):
            if latex_str[i] == '\\':
                # LaTeX command
                j = i + 1
                while j < len(latex_str) and latex_str[j].isalpha():
                    j += 1
                tokens.append(latex_str[i:j])
                i = j
            elif latex_str[i] in '{}[]()_^':
                # Special characters
                tokens.append(latex_str[i])
                i += 1
            elif latex_str[i].isspace():
                # Skip whitespace
                i += 1
            else:
                # Regular character or number
                j = i
                while j < len(latex_str) and not latex_str[j].isspace() and latex_str[j] not in '{}[]()_^\\':
                    j += 1
                if i < j:
                    tokens.append(latex_str[i:j])
                i = j
        
        return tokens
    
    def _calculate_latex_command_accuracy(self, pred: str, truth: str) -> float:
        """Calculate accuracy of LaTeX commands specifically."""
        pred_tokens = self._tokenize_latex(pred)
        truth_tokens = self._tokenize_latex(truth)
        
        pred_commands = [t for t in pred_tokens if t.startswith('\\')]
        truth_commands = [t for t in truth_tokens if t.startswith('\\')]
        
        if not truth_commands:
            return 1.0 if not pred_commands else 0.0
        
        correct_commands = sum(1 for p, t in zip(pred_commands, truth_commands) if p == t)
        return correct_commands / len(truth_commands)
    
    def _calculate_structural_similarity(self, pred: str, truth: str) -> float:
        """Calculate structural similarity based on brackets and grouping."""
        pred_structure = re.sub(r'[^{}[\]()]', '', pred)
        truth_structure = re.sub(r'[^{}[\]()]', '', truth)
        
        if not truth_structure:
            return 1.0 if not pred_structure else 0.0
        
        max_len = max(len(pred_structure), len(truth_structure))
        if max_len == 0:
            return 1.0
        
        lev_dist = Levenshtein.distance(pred_structure, truth_structure)
        return max(0, 1 - (lev_dist / max_len))
    
    def _calculate_semantic_similarity(self, pred: str, truth: str) -> float:
        """Calculate semantic similarity (simplified approach)."""
        # Extract mathematical operators and functions
        pred_math = re.findall(r'\\[a-zA-Z]+|[+\-*/=<>≤≥≠∞∑∏∫]', pred)
        truth_math = re.findall(r'\\[a-zA-Z]+|[+\-*/=<>≤≥≠∞∑∏∫]', truth)
        
        if not truth_math:
            return 1.0 if not pred_math else 0.0
        
        # Calculate Jaccard similarity of mathematical elements
        pred_set = set(pred_math)
        truth_set = set(truth_math)
        
        intersection = len(pred_set.intersection(truth_set))
        union = len(pred_set.union(truth_set))
        
        return intersection / union if union > 0 else 1.0
    
    def _calculate_bleu_score(self, pred: str, truth: str) -> float:
        """Calculate BLEU score (simplified 1-gram version)."""
        pred_tokens = pred.split()
        truth_tokens = truth.split()
        
        if not truth_tokens:
            return 1.0 if not pred_tokens else 0.0
        
        if not pred_tokens:
            return 0.0
        
        # Count matching tokens
        pred_counts = {}
        truth_counts = {}
        
        for token in pred_tokens:
            pred_counts[token] = pred_counts.get(token, 0) + 1
        
        for token in truth_tokens:
            truth_counts[token] = truth_counts.get(token, 0) + 1
        
        # Calculate precision
        matches = 0
        for token, count in pred_counts.items():
            matches += min(count, truth_counts.get(token, 0))
        
        precision = matches / len(pred_tokens) if pred_tokens else 0
        
        # Brevity penalty
        bp = min(1, len(pred_tokens) / len(truth_tokens)) if truth_tokens else 0
        
        return bp * precision
    
    def _calculate_jaccard_similarity(self, pred: str, truth: str) -> float:
        """Calculate Jaccard similarity of character sets."""
        pred_set = set(pred.replace(' ', ''))
        truth_set = set(truth.replace(' ', ''))
        
        if not truth_set:
            return 1.0 if not pred_set else 0.0
        
        intersection = len(pred_set.intersection(truth_set))
        union = len(pred_set.union(truth_set))
        
        return intersection / union if union > 0 else 1.0
    
    def calculate_aggregate_score(self, metrics: Dict[str, float], 
                                 weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted aggregate score from multiple metrics."""
        if weights is None:
            # Default weights
            weights = {
                'exact_match': 0.3,
                'character_accuracy': 0.25,
                'token_accuracy': 0.2,
                'latex_command_accuracy': 0.15,
                'structural_similarity': 0.1
            }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics and metrics[metric] != float('inf'):
                score += metrics[metric] * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def get_metric_descriptions(self) -> Dict[str, str]:
        """Return descriptions of all available metrics."""
        return {
            'exact_match': 'Binary indicator of perfect string match',
            'levenshtein_distance': 'Edit distance between prediction and ground truth',
            'character_accuracy': 'Character-level accuracy (1 - normalized edit distance)',
            'word_error_rate': 'Word Error Rate based on token sequences',
            'token_accuracy': 'Fraction of tokens that match in correct positions',
            'latex_command_accuracy': 'Accuracy of LaTeX commands specifically',
            'structural_similarity': 'Similarity of bracket/grouping structures',
            'semantic_similarity': 'Similarity of mathematical operators and functions',
            'bleu_score': 'BLEU score (simplified 1-gram version)',
            'jaccard_similarity': 'Jaccard similarity of character sets',
            'prediction_length': 'Length of predicted string',
            'ground_truth_length': 'Length of ground truth string',
            'length_difference': 'Absolute difference in string lengths'
        }
            