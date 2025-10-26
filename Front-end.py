import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk, ImageOps, ImageEnhance
import customtkinter as ctk
import os
import cv2
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json
import webbrowser
import re
import random
import sympy
import subprocess
import time
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor

# Add power operator transformation and implicit multiplication
transformations = standard_transformations + (convert_xor, implicit_multiplication_application)

# ==================== SYMPY MARKING FUNCTIONS ====================

def clean_step(step_str):
    """Remove natural language and extract mathematical content"""
    # Remove common natural language prefixes
    prefixes_to_remove = [
        r"^f'\(x\)\s*=\s*",
        r"^Set\s+f'\(x\)\s*=\s*0:\s*",
        r"^Set\s+",
        r"^Using\s+.*?:\s*",
        r"^Reference angle is\s+",
        r"^I think\s+.*?(?:because|so)\s+.*?(?:and|$)",
        r"^Reject\s+.*?(?:as|because|so)\s+.*?,\s*(?:so\s+)?",
        r"^limit as x->\d+\s+of\s+",
        r"^limit\s+of\s+",
    ]
    
    cleaned = step_str
    for pattern in prefixes_to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()
    
    # Remove trailing natural language
    suffixes_patterns = [
        r'\s+as\s+log\s+domain.*$',
        r'\s+because.*$',
        r'\s+so\s+.*$',
    ]
    
    for pattern in suffixes_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()
    
    return cleaned

def extract_solutions_from_text(step_str):
    """Extract solution values from natural language statements"""
    # Patterns for "x = 2, x = 3" or "x = 2 or x = 3" or "x=2, x=3"
    solution_patterns = [
        r'x\s*=\s*(-?[\d.]+)(?:\s*,\s*|\s+or\s+|\s+and\s+)x\s*=\s*(-?[\d.]+)',
        r'x\s*=\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)',
    ]
    
    for pattern in solution_patterns:
        match = re.search(pattern, step_str, re.IGNORECASE)
        if match:
            # Return as a set-like structure
            val1, val2 = match.groups()
            return f"{{{val1}, {val2}}}"
    
    # Single solution pattern
    single_match = re.search(r'x\s*=\s*(-?[\d.]+)$', step_str.strip(), re.IGNORECASE)
    if single_match:
        return f"{{{single_match.group(1)}}}"
    
    return None

def parse_equation_or_expression(step_str, local_dict):
    """Parse equations or expressions with improved handling"""
    original_step = step_str
    step_str = clean_step(step_str)
    
    # Check if this is a solution statement (not an equation to verify)
    solution_set = extract_solutions_from_text(original_step)
    if solution_set:
        return {"type": "solution_set", "value": solution_set, "raw": original_step}
    
    # Handle "Undefined" as a special case
    if step_str.lower().strip() == "undefined":
        return {"type": "undefined", "raw": original_step}
    
    # Try to parse as equation
    if '=' in step_str:
        parts = step_str.split('=')
        if len(parts) == 2:
            try:
                left_str, right_str = parts[0].strip(), parts[1].strip()
                left_expr = parse_expr(left_str, local_dict=local_dict, transformations=transformations)
                right_expr = parse_expr(right_str, local_dict=local_dict, transformations=transformations)
                return sympy.Eq(left_expr, right_expr)
            except Exception as e:
                pass
    
    # Try to parse as expression
    try:
        expr = parse_expr(step_str, local_dict=local_dict, transformations=transformations)
        return expr
    except Exception as e:
        pass
    
    return None

def is_equivalent_math(obj1, obj2, local_dict, tolerance=1e-6):
    """Check mathematical equivalence with improved handling"""
    
    # Handle special cases
    if isinstance(obj1, dict) or isinstance(obj2, dict):
        # Solution sets or undefined values - check string equality
        if isinstance(obj1, dict) and isinstance(obj2, dict):
            if obj1.get("type") == "solution_set" and obj2.get("type") == "solution_set":
                # Compare solution sets
                return obj1.get("value") == obj2.get("value")
            if obj1.get("type") == "undefined" or obj2.get("type") == "undefined":
                return obj1.get("type") == obj2.get("type")
        return False
    
    try:
        # Handle equation objects
        if isinstance(obj1, sympy.Eq) and isinstance(obj2, sympy.Eq):
            # Check if both equations are equivalent
            diff_lhs = sympy.simplify(obj1.lhs - obj2.lhs)
            diff_rhs = sympy.simplify(obj1.rhs - obj2.rhs)
            if diff_lhs == 0 and diff_rhs == 0:
                return True
            
            # Check if rearranged (lhs1 - rhs1 = lhs2 - rhs2)
            diff1 = sympy.simplify(obj1.lhs - obj1.rhs)
            diff2 = sympy.simplify(obj2.lhs - obj2.rhs)
            if sympy.simplify(diff1 - diff2) == 0:
                return True
        
        # Handle expression objects
        elif not isinstance(obj1, sympy.Eq) and not isinstance(obj2, sympy.Eq):
            # Direct simplification
            diff = sympy.simplify(obj1 - obj2)
            if diff == 0:
                return True
            
            # Try expansion
            diff_expanded = sympy.expand(obj1 - obj2)
            if diff_expanded == 0:
                return True
            
            # Try trigsimp for trig functions
            if any(func in str(obj1) or func in str(obj2) for func in ['sin', 'cos', 'tan', 'sec', 'csc', 'cot']):
                diff_trig = sympy.trigsimp(obj1 - obj2)
                if diff_trig == 0:
                    return True
            
            # Numerical verification
            symbols = list(obj1.free_symbols.union(obj2.free_symbols))
            if symbols:
                all_match = True
                for _ in range(5):
                    # Generate test values avoiding problematic points
                    values = {s: random.uniform(0.5, 5) for s in symbols}
                    try:
                        val1 = complex(obj1.subs(values))
                        val2 = complex(obj2.subs(values))
                        if abs(val1 - val2) > tolerance:
                            all_match = False
                            break
                    except:
                        continue
                if all_match:
                    return True
        
        return False
    except Exception as e:
        return False

def extract_base_function(question_text):
    """Extract the base mathematical function from question text"""
    patterns = [
        r'f\(x\)\s*=\s*([^.]+)',
        r'of\s+f\(x\)\s*=\s*([^.]+)',
        r'of\s+([^.]+?)\s+(?:with|as|$)',
        r':\s+([^?]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, question_text)
        if match:
            expr = match.group(1).strip()
            # Remove trailing punctuation and words
            expr = re.sub(r'\s+with\s+respect.*$', '', expr, flags=re.IGNORECASE)
            expr = re.sub(r'\s+as\s+x\s+approaches.*$', '', expr, flags=re.IGNORECASE)
            expr = re.sub(r'[.,;]$', '', expr)
            return expr.strip()
    
    return question_text

def enhanced_verify_steps(derivation, initial_math_obj, expected_final_answer, local_dict):
    """Verify derivation steps with proper step-by-step comparison"""
    feedback = []
    
    # Parse the expected final answer
    expected_final_parsed = parse_equation_or_expression(expected_final_answer, local_dict)
    
    if not derivation:
        return [{"status": "Error", "comment": "No derivation steps provided"}]
    
    # Check first step against initial math object
    first_step_parsed = parse_equation_or_expression(derivation[0], local_dict)
    if first_step_parsed is None:
        feedback.append({
            "step": 1,
            "student_work": derivation[0],
            "status": "Unparseable",
            "comment": "Could not parse as valid mathematical expression"
        })
    elif is_equivalent_math(initial_math_obj, first_step_parsed, local_dict):
        feedback.append({
            "step": 1,
            "student_work": derivation[0],
            "status": "Correct",
            "comment": "Valid starting point"
        })
    else:
        feedback.append({
            "step": 1,
            "student_work": derivation[0],
            "status": "Incorrect",
            "comment": "Does not match the problem statement"
        })
    
    # Check subsequent steps
    for i in range(1, len(derivation)):
        try:
            prev_parsed = parse_equation_or_expression(derivation[i-1], local_dict)
            current_parsed = parse_equation_or_expression(derivation[i], local_dict)
            
            if current_parsed is None:
                feedback.append({
                    "step": i+1,
                    "student_work": derivation[i],
                    "status": "Unparseable",
                    "comment": "Could not parse as valid mathematical expression"
                })
                continue
            
            if prev_parsed is None:
                # If previous step was unparseable, we can't verify this step
                feedback.append({
                    "step": i+1,
                    "student_work": derivation[i],
                    "status": "Unknown",
                    "comment": "Cannot verify due to previous unparseable step"
                })
            elif is_equivalent_math(prev_parsed, current_parsed, local_dict):
                feedback.append({
                    "step": i+1,
                    "student_work": derivation[i],
                    "status": "Correct",
                    "comment": "Valid mathematical transformation"
                })
            else:
                feedback.append({
                    "step": i+1,
                    "student_work": derivation[i],
                    "status": "Incorrect",
                    "comment": "Mathematical error or invalid transformation"
                })
                
        except Exception as e:
            feedback.append({
                "step": i+1,
                "student_work": derivation[i],
                "status": "Error",
                "comment": f"Evaluation error: {str(e)[:100]}"
            })
    
    # FINAL ANSWER VALIDATION
    if derivation:
        last_step = derivation[-1]
        last_parsed = parse_equation_or_expression(last_step, local_dict)
        
        final_correct = False
        if last_parsed and expected_final_parsed:
            final_correct = is_equivalent_math(last_parsed, expected_final_parsed, local_dict)
        
        # Add final answer validation
        feedback.append({
            "step": "Final Answer",
            "student_work": last_step,
            "status": "Correct" if final_correct else "Incorrect",
            "comment": "Final answer matches expected result" if final_correct else "Final answer does not match expected result"
        })
    
    return feedback

def latex_to_readable(latex_str):
    """Convert LaTeX to readable mathematical text"""
    # Basic LaTeX to readable conversions
    conversions = [
        (r'\\frac{(.*?)}{(.*?)}', r'(\1)/(\2)'),
        (r'\\sqrt{(.*?)}', r'sqrt(\1)'),
        (r'\\sin', r'sin'),
        (r'\\cos', r'cos'),
        (r'\\tan', r'tan'),
        (r'\\log', r'log'),
        (r'\\ln', r'ln'),
        (r'\\pi', r'pi'),
        (r'\\infty', r'infinity'),
        (r'\\left\(', r'('),
        (r'\\right\)', r')'),
        (r'\\cdot', r'*'),
        (r'\\times', r'*'),
        (r'\\div', r'/'),
        (r'\\^', r'^'),
        (r'\{', r'('),
        (r'\}', r')'),
        (r'\\ ', r' '),  # Convert LaTeX spaces to normal spaces
        (r'\s+', r' '),  # Normalize multiple spaces
    ]
    
    readable = latex_str
    for pattern, replacement in conversions:
        readable = re.sub(pattern, replacement, readable)
    
    return readable.strip()

def mark_calculus_enhanced(question, student_derivation):
    """Enhanced calculus marking"""
    local_symbols = {var: sympy.Symbol(var) for var in question.variables}
    
    try:
        # Extract base function
        base_func_str = extract_base_function(question.question)
        base_function = parse_expr(base_func_str, local_dict=local_symbols, transformations=transformations)
        
        return enhanced_verify_steps(
            student_derivation, 
            base_function, 
            question.final_answer,
            local_symbols
        )
        
    except Exception as e:
        return [{"status": "Error", "comment": f"Failed to parse problem: {str(e)[:100]}"}]

def mark_algebra_enhanced(question, student_derivation):
    """Enhanced algebra marking"""
    local_symbols = {var: sympy.Symbol(var) for var in question.variables}
    # Add common constants
    if 'pi' in question.variables:
        local_symbols['pi'] = sympy.pi
    if 'k' in question.variables:
        local_symbols['k'] = sympy.Symbol('k', integer=True)
    
    try:
        # Extract mathematical content
        math_content = extract_base_function(question.question)
        
        # Parse the initial mathematical object
        if '=' in math_content:
            parts = math_content.split('=')
            if len(parts) == 2:
                left_expr = parse_expr(parts[0].strip(), local_dict=local_symbols, transformations=transformations)
                right_expr = parse_expr(parts[1].strip(), local_dict=local_symbols, transformations=transformations)
                initial_obj = sympy.Eq(left_expr, right_expr)
        else:
            # Expression problem
            initial_obj = parse_expr(math_content.strip(), local_dict=local_symbols, transformations=transformations)
        
        return enhanced_verify_steps(
            student_derivation, 
            initial_obj, 
            question.final_answer,
            local_symbols
        )
        
    except Exception as e:
        return [{"status": "Error", "comment": f"Failed to parse problem: {str(e)[:100]}"}]

def mark_trigonometry_enhanced(question, student_derivation):
    """Enhanced trigonometry marking"""
    local_symbols = {var: sympy.Symbol(var) for var in question.variables}
    local_symbols['pi'] = sympy.pi
    if 'k' in question.variables:
        local_symbols['k'] = sympy.Symbol('k', integer=True)
    
    return mark_algebra_enhanced(question, student_derivation)

def main_marker_enhanced(question_data, student_derivation):
    """Main dispatcher using enhanced marking functions"""
    field = question_data.field_of_maths
    
    if field == "Calculus":
        return mark_calculus_enhanced(question_data, student_derivation)
    elif field == "Algebra":
        return mark_algebra_enhanced(question_data, student_derivation)
    elif field == "Trigonometry":
        return mark_trigonometry_enhanced(question_data, student_derivation)
    else:
        return [{"status": "Error", "comment": f"No marking function available for '{field}'."}]

# ==================== SHARED CLASSES ====================

class ImagePreprocessor:
    """Wrapper for the preprocessing pipeline"""
    
    @staticmethod
    def invert_if_necessary(img):
        """Ensure the image has white background and black text."""
        mean_val = np.mean(img)
        if mean_val < 128:
            return cv2.bitwise_not(img)
        return img
    
    @staticmethod
    def adaptive_denoise_and_binarize(img_gray, median_k=3, block_size=35, C=12):
        """Adaptive thresholding for binarization"""
        if median_k > 1:
            med = cv2.medianBlur(img_gray, median_k)
        else:
            med = img_gray
        binary = cv2.adaptiveThreshold(med, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, block_size, C)
        return binary
    
    @staticmethod
    def preprocess_image(pil_image, params=None):
        """Main preprocessing pipeline for PIL images"""
        if params is None:
            params = {
                'median_k': 3,
                'block_size': 35,
                'C': 12,
                'open_kernel': (3, 3),
                'close_kernel': (3, 3),
                'open_iters': 1,
                'close_iters': 1,
                'min_area': 200,
                'dilate_kernel': (2, 2),
                'dilate_iters': 1,
                'deskew': False
            }
        
        # Convert PIL to OpenCV
        img_array = np.array(pil_image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array
        
        # Apply preprocessing pipeline
        binary = ImagePreprocessor.adaptive_denoise_and_binarize(
            img_gray, 
            median_k=params['median_k'],
            block_size=params['block_size'], 
            C=params['C']
        )
        
        # Ensure white background, black text
        final = ImagePreprocessor.invert_if_necessary(binary)
        
        # Convert back to PIL Image
        return Image.fromarray(final)

class LatexOCRConverter:
    """Placeholder for LaTeX OCR conversion"""
    
    @staticmethod
    def convert_to_latex(pil_image):
        """Placeholder for OCR model integration."""
        placeholder_latex = [
            "\\frac{d}{dx}\\left(\\int_{0}^{x} f(t)\\,dt\\right) = f(x)",
            "E = mc^2",
            "\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}",
            "x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}"
        ]
        
        import time
        time.sleep(1)
        import random
        return random.choice(placeholder_latex)

class TeacherQuestion:
    """Class to store teacher's question information"""
    
    def __init__(self):
        self.id = 1
        self.field_of_maths = ""
        self.topic = ""
        self.question = ""
        self.variables = []
        self.final_answer = ""
        self.student_derivations = {}  # {student_id: {
                                       #   'image': PIL.Image,           # Original uploaded image
                                       #   'original': PIL.Image,        # Copy of original
                                       #   'name': str,                  # Student identifier
                                       #   'latex': str,                 # OCR output from image
                                       #   'readable_text': str,         # Converted from LaTeX to normal math text
                                       #   'sympy_analysis': list,       # Results from SymPy marking
                                       #   'llm_marking': dict           # Results from LLM analysis
                                       # }}

# ==================== BASE PAGE CLASS ====================

class BasePage(ctk.CTkFrame):
    """Base class for all pages with common functionality"""
    
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.setup_page()
    
    def setup_page(self):
        """Override this method in child classes"""
        pass

# ==================== PAGE 1: INTRODUCTION ====================
class IntroductionPage(BasePage):
    def setup_page(self):
        # Configure main grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Main scrollable frame
        main_scroll = ctk.CTkScrollableFrame(self, fg_color="white")
        main_scroll.grid(row=0, column=0, sticky="nsew")
        main_scroll.grid_columnconfigure(0, weight=1)
        
        # Header
        header_frame = ctk.CTkFrame(main_scroll, fg_color="#2563eb", height=100)
        header_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        header_frame.grid_propagate(False)
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="MathAutoMark - Intelligent Mathematics Grading",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="white"
        )
        title_label.pack(expand=True, pady=(10, 0))
        
        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="Automated marking of mathematical equations and derivations",
            font=ctk.CTkFont(size=14),
            text_color="#dbeafe"
        )
        subtitle_label.pack(expand=True, pady=(0, 10))
        
        # Hero Section
        hero_frame = ctk.CTkFrame(main_scroll, fg_color="#f8fafc", corner_radius=15)
        hero_frame.grid(row=1, column=0, sticky="ew", padx=50, pady=30)
        hero_frame.grid_columnconfigure(0, weight=1)
        
        hero_title = ctk.CTkLabel(
            hero_frame,
            text="Become a Pro in Automated Math Grading",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#1e293b"
        )
        hero_title.grid(row=0, column=0, pady=(20, 10))
        
        hero_text = ctk.CTkLabel(
            hero_frame,
            text="Grade your way to efficient teaching with our Intelligent Math Assessment System\nand build professional grading workflows to take your teaching to the next level.",
            font=ctk.CTkFont(size=14),
            text_color="#64748b",
            justify="center"
        )
        hero_text.grid(row=1, column=0, pady=(0, 20))
        
        # Features Title
        features_title = ctk.CTkLabel(
            main_scroll,
            text="Platform Features",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#1e293b"
        )
        features_title.grid(row=2, column=0, pady=(20, 15))
        
        # Features Grid - SIMPLE AND CLEAN
        features_container = ctk.CTkFrame(main_scroll, fg_color="transparent")
        features_container.grid(row=3, column=0, sticky="ew", padx=50, pady=10)
        features_container.grid_columnconfigure(0, weight=1)
        features_container.grid_columnconfigure(1, weight=1)
        
        # Define features
        features = [
            ("📸 Upload Student Work", "Automatic image preprocessing"),
            ("🔍 OCR Conversion", "Convert to LaTeX format"), 
            ("📝 Readable Text", "Transform LaTeX to text"),
            ("✅ SymPy Verification", "Mathematical correctness"),
            ("🤖 LLM Analysis", "Intelligent feedback"),
            ("⚡ Batch Processing", "Multiple students")
        ]
        
        # Create features in two columns
        for i, (title, desc) in enumerate(features):
            col = i % 2
            row = i // 2
            
            feature_frame = ctk.CTkFrame(features_container, fg_color="transparent")
            feature_frame.grid(row=row, column=col, sticky="w", padx=20, pady=12)
            
            feature_label = ctk.CTkLabel(
                feature_frame,
                text=title,
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color="#374151",
                anchor="w"
            )
            feature_label.grid(row=0, column=0, sticky="w")
            
            desc_label = ctk.CTkLabel(
                feature_frame,
                text=desc,
                font=ctk.CTkFont(size=12),
                text_color="#64748b",
                anchor="w"
            )
            desc_label.grid(row=1, column=0, sticky="w", pady=(2, 0))
        
        # CTA Button
        start_btn = ctk.CTkButton(
            main_scroll,
            text="🎓 Start Grading Now", 
            command=lambda: self.controller.show_page("TeacherInputPage"),
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#7e22ce",
            hover_color="#6b21a8",
            height=50,
            width=250,
            corner_radius=10
        )
        start_btn.grid(row=4, column=0, pady=30)
        
        # Footer
        footer = ctk.CTkLabel(
            main_scroll,
            text="Built with ❤️ for educators worldwide",
            font=ctk.CTkFont(size=12),
            text_color="#94a3b8"
        )
        footer.grid(row=5, column=0, pady=20)

# ==================== PAGE 2: TEACHER INPUT ====================

class TeacherInputPage(BasePage):
    def setup_page(self):
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Main container
        main_container = ctk.CTkFrame(self, fg_color="white")
        main_container.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        main_container.grid_columnconfigure(0, weight=1)
        main_container.grid_rowconfigure(0, weight=1)
        
        # Create scrollable frame
        scrollable_frame = ctk.CTkScrollableFrame(main_container, fg_color="white")
        scrollable_frame.grid(row=0, column=0, sticky="nsew")
        scrollable_frame.grid_columnconfigure(0, weight=1)
        
        # Header with back button
        header_frame = ctk.CTkFrame(scrollable_frame, fg_color="#2563eb", height=100)
        header_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        header_frame.grid_propagate(False)
        header_frame.grid_columnconfigure(1, weight=1)
        
        back_btn = ctk.CTkButton(
            header_frame,
            text="← Back to Home",
            command=lambda: self.controller.show_page("IntroductionPage"),
            font=ctk.CTkFont(size=14),
            fg_color="transparent",
            text_color="white",
            hover_color="#1d4ed8",
            width=120
        )
        back_btn.grid(row=0, column=0, padx=20, pady=20, sticky="w")
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="Step 1: Define Your Mathematical Question",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="white"
        )
        title_label.grid(row=0, column=1, padx=20, pady=20)
        
        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="Enter the question details that students will be solving",
            font=ctk.CTkFont(size=14),
            text_color="#dbeafe"
        )
        subtitle_label.grid(row=1, column=1, padx=20, pady=(0, 20), sticky="w")
        
        # Form content
        form_frame = ctk.CTkFrame(scrollable_frame, fg_color="#f8fafc")
        form_frame.grid(row=1, column=0, sticky="nsew", padx=50, pady=30)
        form_frame.grid_columnconfigure(0, weight=1)
        
        # Form title
        form_title = ctk.CTkLabel(
            form_frame,
            text="Question Details",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#1e293b"
        )
        form_title.pack(pady=(30, 20))
        
        # Form fields
        fields = [
            ("Field of Mathematics:", "field_of_maths", "e.g., Calculus, Algebra, Statistics"),
            ("Topic:", "topic", "e.g., Differentiation, Integration, Linear Equations"),
            ("Question:", "question", "Enter the mathematical question"),
            ("Variables (comma-separated):", "variables", "e.g., x, y, t"),
            ("Final Answer (mathematical expression):", "final_answer", "e.g., 3*x**2*cos(x) - x**3*sin(x)")
        ]
        
        self.form_entries = {}
        
        for i, (label_text, field_name, placeholder) in enumerate(fields):
            # Label
            label = ctk.CTkLabel(
                form_frame,
                text=label_text,
                font=ctk.CTkFont(size=16, weight="bold"),
                text_color="#374151"
            )
            label.pack(pady=(20, 5), anchor="w", padx=50)
            
            # Entry field
            if field_name == "question":
                # Use textbox for question
                entry = ctk.CTkTextbox(
                    form_frame,
                    height=100,
                    font=ctk.CTkFont(size=14),
                    fg_color="white",
                    border_color="#d1d5db",
                    border_width=2,
                    corner_radius=8
                )
                entry.pack(fill="x", padx=50, pady=5)
                entry.insert("1.0", placeholder)
            else:
                # Use regular entry
                entry = ctk.CTkEntry(
                    form_frame,
                    placeholder_text=placeholder,
                    font=ctk.CTkFont(size=14),
                    height=45,
                    fg_color="white",
                    border_color="#d1d5db",
                    border_width=2,
                    corner_radius=8
                )
                entry.pack(fill="x", padx=50, pady=5)
            
            self.form_entries[field_name] = entry
        
        # Button frame - FIXED POSITION
        button_frame = ctk.CTkFrame(form_frame, fg_color="transparent")
        button_frame.pack(pady=40, fill="x", padx=50)
        
        # Next Step Button - LARGE AND PROMINENT
        next_btn = ctk.CTkButton(
            button_frame,
            text="✓ NEXT STEP: UPLOAD STUDENT IMAGES →",
            command=self.save_and_continue,
            font=ctk.CTkFont(size=18, weight="bold"),
            fg_color="#059669",
            hover_color="#047857",
            height=60,
            width=400,
            corner_radius=15
        )
        next_btn.pack(pady=20)
        
        # Help text
        help_label = ctk.CTkLabel(
            button_frame,
            text="Click above to save your question and proceed to student image uploads",
            font=ctk.CTkFont(size=12),
            text_color="#64748b"
        )
        help_label.pack(pady=(0, 10))
        
        # Add some bottom padding to ensure button is visible
        bottom_padding = ctk.CTkFrame(form_frame, fg_color="transparent", height=20)
        bottom_padding.pack(fill="x")
    
    def save_and_continue(self):
        """Save question and move to next page"""
        try:
            # Get form data
            question_data = TeacherQuestion()
            question_data.field_of_maths = self.form_entries['field_of_maths'].get()
            question_data.topic = self.form_entries['topic'].get()
            
            # Get question from textbox
            question_data.question = self.form_entries['question'].get("1.0", "end-1c")
            
            # Parse variables
            variables_text = self.form_entries['variables'].get()
            question_data.variables = [v.strip() for v in variables_text.split(',') if v.strip()]
            
            question_data.final_answer = self.form_entries['final_answer'].get()
            
            # Validate required fields
            if not all([question_data.field_of_maths, 
                       question_data.topic, 
                       question_data.question.strip(),
                       question_data.final_answer]):
                messagebox.showerror("Error", "Please fill in all required fields")
                return
            
            if question_data.question.strip() == "Enter the mathematical question":
                messagebox.showerror("Error", "Please enter your actual question text")
                return
            
            # Save to controller and proceed
            self.controller.current_question = question_data
            messagebox.showinfo("Success", "Question saved successfully! Now you can upload student derivations.")
            self.controller.show_page("ImageUploadPage")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save question: {str(e)}")

# ==================== PAGE 3: IMAGE UPLOAD & OCR CONVERSION ====================

# ==================== PAGE 3: IMAGE UPLOAD & OCR CONVERSION ====================

class ImageUploadPage(BasePage):
    def setup_page(self):
        # Initialize
        self.image_files = []
        self.current_folder = None
        self.ocr_results = {}
        
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Main content area
        self.main_frame = ctk.CTkFrame(self, fg_color="white")
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)
        
        self.show_folder_selection_view()
    
    def show_folder_selection_view(self):
        """Show folder selection view"""
        self.clear_main_frame()
        
        folder_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        folder_frame.grid(row=0, column=0, sticky="nsew")
        folder_frame.grid_columnconfigure(0, weight=1)
        folder_frame.grid_rowconfigure(1, weight=1)
        
        # Title
        title_label = ctk.CTkLabel(
            folder_frame,
            text="📁 Select Folder with Student Images",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#1e293b"
        )
        title_label.grid(row=0, column=0, pady=(50, 20))
        
        # Select folder button
        select_btn = ctk.CTkButton(
            folder_frame,
            text="Choose Images Folder",
            command=self.select_images_folder,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#2563eb",
            hover_color="#1d4ed8",
            height=50,
            width=250,
            corner_radius=10
        )
        select_btn.grid(row=1, column=0, pady=20)
        
        # Instructions
        instruction_label = ctk.CTkLabel(
            folder_frame,
            text="Select a folder containing images of student mathematical work\nSupported formats: PNG, JPG, JPEG, BMP, TIFF, WEBP",
            font=ctk.CTkFont(size=14),
            text_color="#64748b",
            justify="center"
        )
        instruction_label.grid(row=2, column=0, pady=10)
    
    def select_images_folder(self):
        """Select folder containing student images"""
        folder_path = filedialog.askdirectory(title="Select Folder with Student Images")
        if not folder_path:
            return
        
        try:
            self.current_folder = Path(folder_path)
            
            # Find all image files - FIXED: Use set to avoid duplicates
            image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
            self.image_files = []
            
            # Use rglob with case-insensitive matching
            for img_path in self.current_folder.rglob('*'):
                if img_path.suffix.lower() in image_extensions:
                    self.image_files.append(img_path)
            
            # Remove duplicates by filename (case-insensitive)
            seen = set()
            unique_files = []
            for img_path in self.image_files:
                if img_path.name.lower() not in seen:
                    seen.add(img_path.name.lower())
                    unique_files.append(img_path)
            
            self.image_files = unique_files
            
            if not self.image_files:
                messagebox.showwarning("Warning", "No image files found in selected folder")
                return
            
            # Show image gallery view
            self.show_image_gallery()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load folder: {str(e)}")
    
    def show_image_gallery(self):
        """Show gallery of all images with OCR conversion button"""
        self.clear_main_frame()
        
        # Header
        header_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=10)
        header_frame.grid_columnconfigure(1, weight=1)
        
        folder_label = ctk.CTkLabel(
            header_frame,
            text=f"Folder: {self.current_folder.name}",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#374151"
        )
        folder_label.grid(row=0, column=0, sticky="w")
        
        count_label = ctk.CTkLabel(
            header_frame,
            text=f"Images: {len(self.image_files)}",
            font=ctk.CTkFont(size=14),
            text_color="#64748b"
        )
        count_label.grid(row=0, column=1, sticky="e")
        
        # OCR Conversion Button
        ocr_btn = ctk.CTkButton(
            header_frame,
            text="🚀 Convert All Images with Gemini OCR",
            command=self.process_all_with_gemini,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#dc2626",
            hover_color="#b91c1c",
            height=50,
            width=300,
            corner_radius=10
        )
        ocr_btn.grid(row=1, column=0, columnspan=2, pady=20)
        
        # Images gallery
        gallery_frame = ctk.CTkScrollableFrame(self.main_frame, fg_color="transparent")
        gallery_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        gallery_frame.grid_columnconfigure(0, weight=1)
        
        # Create 2-column grid for images
        for i, img_path in enumerate(self.image_files):
            row = i // 2
            col = i % 2
            
            # Image card
            card = ctk.CTkFrame(gallery_frame, corner_radius=10)
            card.grid(row=row, column=col, sticky="nsew", padx=10, pady=10)
            card.grid_columnconfigure(0, weight=1)
            
            # Filename
            name_label = ctk.CTkLabel(
                card,
                text=img_path.name,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="#374151"
            )
            name_label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
            
            # Image preview
            try:
                image = Image.open(img_path)
                # Resize for preview
                image.thumbnail((200, 150), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                
                img_label = ctk.CTkLabel(card, image=photo, text="")
                img_label.image = photo  # Keep reference
                img_label.grid(row=1, column=0, padx=10, pady=5)
                
            except Exception as e:
                error_label = ctk.CTkLabel(
                    card,
                    text=f"Failed to load image\n{str(e)}",
                    text_color="#ef4444",
                    font=ctk.CTkFont(size=10)
                )
                error_label.grid(row=1, column=0, padx=10, pady=5)
            
            # Status (will be updated after OCR)
            status_label = ctk.CTkLabel(
                card,
                text="⏳ Ready for OCR",
                font=ctk.CTkFont(size=11),
                text_color="#6b7280"
            )
            status_label.grid(row=2, column=0, padx=10, pady=(5, 10))
            
            # Store reference for updating status later
            if not hasattr(self, 'image_status_labels'):
                self.image_status_labels = {}
            self.image_status_labels[img_path.name] = status_label
        
        # Configure grid columns for 2-column layout
        gallery_frame.grid_columnconfigure(0, weight=1)
        gallery_frame.grid_columnconfigure(1, weight=1)
    
    def clear_main_frame(self):
        """Clear the main frame content"""
        for widget in self.main_frame.winfo_children():
            widget.destroy()
    
    def configure_gemini(self):
        """Configure Gemini API"""
        try:
            import google.generativeai as genai
            genai.configure(api_key="AIzaSyDk-vLW9ZMKuniHFKIYbsFwCvb3YZoUHg4")
            model = genai.GenerativeModel("gemini-2.5-flash")
            return model
        except Exception as e:
            messagebox.showerror("Error", f"Failed to configure Gemini API: {str(e)}")
            return None
    
    def process_all_with_gemini(self):
        """Process ALL images with Gemini OCR and show results"""
        if not self.image_files:
            messagebox.showwarning("Warning", "No images to process")
            return
        
        # Configure Gemini
        model = self.configure_gemini()
        if not model:
            return
        
        # Create progress window
        progress_window = ctk.CTkToplevel(self)
        progress_window.title("Gemini OCR Processing")
        progress_window.geometry("500x300")
        progress_window.transient(self)
        progress_window.grab_set()
        
        progress_label = ctk.CTkLabel(
            progress_window, 
            text="Processing images with Gemini OCR...",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        progress_label.pack(pady=20)
        
        progress_bar = ctk.CTkProgressBar(progress_window, width=400)
        progress_bar.pack(pady=10)
        progress_bar.set(0)
        
        status_label = ctk.CTkLabel(progress_window, text="Starting...")
        status_label.pack(pady=10)
        
        # Log area
        log_text = ctk.CTkTextbox(progress_window, height=150, width=450)
        log_text.pack(pady=10, padx=20, fill="both", expand=True)
        log_text.insert("1.0", "Gemini OCR Processing Log:\n" + "="*40 + "\n")
        
        self.ocr_results = {}
        
        def add_log(message):
            log_text.insert("end", f"{message}\n")
            log_text.see("end")
            progress_window.update()
        
        # Process each image
        try:
            for i, img_path in enumerate(self.image_files):
                current_progress = i / len(self.image_files)
                progress_bar.set(current_progress)
                
                status_text = f"Processing {i+1}/{len(self.image_files)}: {img_path.name}"
                status_label.configure(text=status_text)
                add_log(f"🔄 {status_text}")
                
                # Update status in gallery
                if hasattr(self, 'image_status_labels') and img_path.name in self.image_status_labels:
                    self.image_status_labels[img_path.name].configure(
                        text="🔄 Processing...",
                        text_color="#f59e0b"
                    )
                
                progress_window.update()
                
                try:
                    # Read image data
                    with open(img_path, "rb") as f:
                        image_data = f.read()
                    
                    prompt = (
                        "Extract the mathematical equation or expression from this image "
                        "and convert it to Python Sympy-compatible format. "
                        "Use syntax like Eq(sin(x) + cos(x), 0), ** for powers, "
                        "and avoid LaTeX or Markdown. Output only the expression."
                    )
                    
                    # FIXED: Correct Gemini API content structure
                    response = model.generate_content([
                        prompt,
                        {"mime_type": "image/jpeg", "data": image_data}
                    ])
                    
                    sympy_expr = response.text.strip()
                    self.ocr_results[img_path.name] = sympy_expr
                    
                    # Update status in gallery
                    if hasattr(self, 'image_status_labels') and img_path.name in self.image_status_labels:
                        self.image_status_labels[img_path.name].configure(
                            text="✅ OCR Complete",
                            text_color="#059669"
                        )
                    
                    add_log(f"✅ Success: {sympy_expr[:50]}...")
                    
                except Exception as e:
                    error_msg = f"Error processing {img_path.name}: {str(e)}"
                    self.ocr_results[img_path.name] = f"Error: {str(e)}"
                    
                    # Update status in gallery
                    if hasattr(self, 'image_status_labels') and img_path.name in self.image_status_labels:
                        self.image_status_labels[img_path.name].configure(
                            text="❌ OCR Failed",
                            text_color="#dc2626"
                        )
                    
                    add_log(f"❌ {error_msg}")
                
                progress_window.update()
            
            # Final update
            progress_bar.set(1.0)
            status_label.configure(text="Processing completed!")
            add_log("🎉 All images processed successfully!")
            
            # Store results in student data structure
            self.save_ocr_results_to_students()
            
            # Show completion message
            messagebox.showinfo(
                "OCR Conversion Complete", 
                f"Successfully processed {len(self.image_files)} images!\n"
                f"Ready to proceed to grading."
            )
            
            progress_window.destroy()
            
            # Show results summary
            self.show_ocr_results_summary()
            
        except Exception as e:
            add_log(f"❌ Fatal error: {str(e)}")
            messagebox.showerror("Error", f"OCR processing failed: {str(e)}")
    
    def save_ocr_results_to_students(self):
        """Save OCR results to student data structure"""
        question = self.controller.current_question
        question.student_derivations.clear()  # Clear any existing data
        
        for i, (filename, ocr_result) in enumerate(self.ocr_results.items()):
            student_id = f"A{i+1:03d}"
            
            question.student_derivations[student_id] = {
                'name': student_id,
                'latex': ocr_result,
                'readable_text': latex_to_readable(ocr_result),
                'image_path': str(self.current_folder / filename)
            }
    
    def show_ocr_results_summary(self):
        """Show summary of OCR results"""
        self.clear_main_frame()
        
        # Header
        header_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=10)
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="✅ OCR Conversion Complete",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#059669"
        )
        title_label.pack(pady=10)
        
        summary_label = ctk.CTkLabel(
            header_frame,
            text=f"Processed {len(self.ocr_results)} images. Ready for grading!",
            font=ctk.CTkFont(size=16),
            text_color="#374151"
        )
        summary_label.pack(pady=5)
        
        # Results scrollable area
        results_frame = ctk.CTkScrollableFrame(self.main_frame, fg_color="transparent")
        results_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)
        results_frame.grid_columnconfigure(0, weight=1)
        
        # Show each result
        for i, (filename, ocr_result) in enumerate(self.ocr_results.items()):
            student_id = f"A{i+1:03d}"
            
            # Result card
            card = ctk.CTkFrame(results_frame, corner_radius=10)
            card.grid(row=i, column=0, sticky="ew", pady=5, padx=10)
            card.grid_columnconfigure(1, weight=1)
            
            # Student ID
            id_label = ctk.CTkLabel(
                card,
                text=student_id,
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color="#2563eb",
                width=80
            )
            id_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
            
            # Filename
            file_label = ctk.CTkLabel(
                card,
                text=filename,
                font=ctk.CTkFont(size=12),
                text_color="#64748b"
            )
            file_label.grid(row=0, column=1, padx=10, pady=10, sticky="w")
            
            # OCR Result
            result_text = ctk.CTkTextbox(
                card,
                height=60,
                font=ctk.CTkFont(size=11, family="Courier"),
                wrap="word"
            )
            result_text.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
            result_text.insert("1.0", ocr_result)
            result_text.configure(state="disabled")
        
        # Continue button
        continue_btn = ctk.CTkButton(
            self.main_frame,
            text="➡️ Continue to Grading",
            command=self.continue_to_grading,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#059669",
            hover_color="#047857",
            height=50,
            width=250,
            corner_radius=10
        )
        continue_btn.grid(row=2, column=0, pady=30)
    
    def continue_to_grading(self):
        """Continue to grading page"""
        question = self.controller.current_question
        
        if not question.student_derivations:
            messagebox.showwarning("Warning", "No student derivations processed")
            return
        
        # Check if we have OCR results
        valid_students = 0
        for student_id, data in question.student_derivations.items():
            if data.get('latex') and not data['latex'].startswith('Error:'):
                valid_students += 1
        
        if valid_students == 0:
            messagebox.showwarning("Warning", "No valid OCR results found. Please process images first.")
            return
        
        self.controller.show_page("GradingPage")

# ==================== SYMPY FINAL ANSWER MARKING FUNCTION ====================

def extract_final_answer_from_steps(student_work):
    """Extract only the final answer from comma-delimited steps"""
    if not student_work:
        return student_work
    
    # Split by commas to get individual steps
    steps = [step.strip() for step in student_work.split(',')]
    
    # Take the last step as the final answer
    if steps:
        final_answer = steps[-1]
        
        # Clean up common prefixes that might be in the final step
        prefixes_to_remove = [
            'Final Answer:',
            'Answer:',
            'Solution:',
            'Therefore:',
            'So:',
            'Thus:',
            'Hence:'
        ]
        
        for prefix in prefixes_to_remove:
            if final_answer.lower().startswith(prefix.lower()):
                final_answer = final_answer[len(prefix):].strip()
        
        # IMPROVED CLEANING: Remove any trailing punctuation, commas, or unmatched parentheses
        final_answer = re.sub(r'[.,;]$', '', final_answer).strip()
        
        # NEW: Remove trailing unmatched parentheses and fix common syntax issues
        # Remove trailing commas
        final_answer = re.sub(r',\s*$', '', final_answer)
        
        # Fix common OCR issues with parentheses
        # Remove trailing unmatched closing parentheses
        while final_answer.count('(') < final_answer.count(')') and final_answer.endswith(')'):
            final_answer = final_answer[:-1].strip()
        
        # Remove leading unmatched opening parentheses  
        while final_answer.count(')') < final_answer.count('(') and final_answer.startswith('('):
            final_answer = final_answer[1:].strip()
        
        # Fix double parentheses
        final_answer = re.sub(r'\(\(', '(', final_answer)
        final_answer = re.sub(r'\)\)', ')', final_answer)
        
        # Remove extra spaces around operators
        final_answer = re.sub(r'\s*\*\s*', '*', final_answer)
        final_answer = re.sub(r'\s*\+\s*', '+', final_answer)
        final_answer = re.sub(r'\s*-\s*', '-', final_answer)
        final_answer = re.sub(r'\s*/\s*', '/', final_answer)
        
        return final_answer
    
    return student_work

def mark_final_answer_only(question, student_work):
    """Mark only the final answer using SymPy without step-by-step verification"""
    local_symbols = {var: sympy.Symbol(var) for var in question.variables}
    
    try:
        # Extract just the final answer from the student work
        final_answer_text = extract_final_answer_from_steps(student_work)
        
        # Parse the expected final answer
        expected_parsed = parse_equation_or_expression(question.final_answer, local_symbols)
        
        # Parse the student's final answer
        student_parsed = parse_equation_or_expression(final_answer_text, local_symbols)
        
        if student_parsed is None:
            return [{
                "step": "Final Answer",
                "student_work": final_answer_text,
                "status": "Unparseable",
                "comment": f"Could not parse student's final answer as valid mathematical expression. Extracted: '{final_answer_text}'"
            }]
        
        # Check if the final answer matches
        if is_equivalent_math(student_parsed, expected_parsed, local_symbols):
            return [{
                "step": "Final Answer", 
                "student_work": final_answer_text,
                "status": "Correct",
                "comment": "Final answer matches expected result"
            }]
        else:
            return [{
                "step": "Final Answer",
                "student_work": final_answer_text,
                "status": "Incorrect", 
                "comment": f"Final answer does not match expected result. Student: {final_answer_text}, Expected: {question.final_answer}"
            }]
            
    except Exception as e:
        return [{
            "step": "Final Answer",
            "student_work": student_work,
            "status": "Error",
            "comment": f"Evaluation error: {str(e)[:100]}"
        }]
# ==================== PAGE 5: SYMPY MARKING ====================

class SymPyMarkingPage(BasePage):
    def setup_page(self):
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Header
        header_frame = ctk.CTkFrame(self, height=80)
        header_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        header_frame.grid_propagate(False)
        header_frame.grid_columnconfigure(1, weight=1)
        
        back_btn = ctk.CTkButton(
            header_frame,
            text="← Back to OCR",
            command=lambda: self.controller.show_page("ImageUploadPage"),
            font=ctk.CTkFont(size=14),
            fg_color="transparent",
            text_color="#2563eb",
            hover_color="#f1f5f9",
            width=120
        )
        back_btn.grid(row=0, column=0, padx=20, pady=20, sticky="w")
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="Step 4: SymPy Final Answer Marking",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#1e293b"
        )
        title_label.grid(row=0, column=1, padx=20, pady=20)
        
        # Content
        content_frame = ctk.CTkFrame(self, fg_color="white")
        content_frame.grid(row=1, column=0, sticky="nsew", padx=50, pady=30)
        content_frame.grid_columnconfigure(0, weight=1)
        
        # Mark all button
        mark_btn = ctk.CTkButton(
            content_frame,
            text="🧮 Check Final Answers with SymPy",
            command=self.mark_all_students,
            font=ctk.CTkFont(size=18, weight="bold"),
            fg_color="#7c3aed",
            hover_color="#6d28d9",
            height=60,
            width=350,
            corner_radius=15
        )
        mark_btn.grid(row=0, column=0, pady=50)
        
        # Results area
        self.results_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        self.results_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)
        self.results_frame.grid_columnconfigure(0, weight=1)
        
        # Next step button
        next_btn = ctk.CTkButton(
            content_frame,
            text="✓ NEXT STEP: LLM MARKING →",
            command=lambda: self.controller.show_page("LLMMarkingPage"),
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#7e22ce",
            hover_color="#6b21a8",
            height=50,
            width=350,
            corner_radius=12
        )
        next_btn.grid(row=2, column=0, pady=20)
        
        content_frame.grid_rowconfigure(1, weight=1)
    
    def mark_all_students(self):
        """Mark only final answers for all students using SymPy"""
        question = self.controller.current_question
        if not question.student_derivations:
            messagebox.showwarning("Warning", "No student derivations to mark")
            return
        
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Create scrollable results area
        scrollable = ctk.CTkScrollableFrame(self.results_frame)
        scrollable.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        scrollable.grid_columnconfigure(0, weight=1)
        
        # Process each student
        for i, (student_id, data) in enumerate(question.student_derivations.items()):
            try:
                # Use readable_text for SymPy analysis
                readable_text = data.get('readable_text', '')
                if not readable_text:
                    continue
                
                # Mark only the final answer using SymPy
                marking_results = mark_final_answer_only(question, readable_text)
                
                # Store results for LLM stage
                data['sympy_analysis'] = marking_results
                
                # Create result card
                card = ctk.CTkFrame(scrollable, corner_radius=10)
                card.grid(row=i, column=0, sticky="ew", pady=10, padx=10)
                card.grid_columnconfigure(0, weight=1)
                
                # Student ID header
                student_header = ctk.CTkFrame(card, fg_color="#1e293b", corner_radius=8)
                student_header.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
                student_header.grid_columnconfigure(1, weight=1)
                
                student_label = ctk.CTkLabel(
                    student_header,
                    text=f"Student {student_id}",
                    font=ctk.CTkFont(size=16, weight="bold"),
                    text_color="white"
                )
                student_label.grid(row=0, column=0, padx=15, pady=10, sticky="w")
                
                # Extract final answer status
                final_status = "Unknown"
                if marking_results:
                    final_status = marking_results[0].get('status', 'Unknown')
                
                # Score based on final answer
                score = 100 if final_status == "Correct" else 0
                
                score_label = ctk.CTkLabel(
                    student_header,
                    text=f"Score: {score:.1f}%",
                    font=ctk.CTkFont(size=14, weight="bold"),
                    text_color="#10b981" if score == 100 else "#ef4444"
                )
                score_label.grid(row=0, column=1, padx=15, pady=10, sticky="e")
                
                # Display final answer results
                for j, result in enumerate(marking_results):
                    step_frame = ctk.CTkFrame(card, fg_color="#f8fafc")
                    step_frame.grid(row=j+1, column=0, sticky="ew", padx=5, pady=2)
                    step_frame.grid_columnconfigure(1, weight=1)
                    
                    # Status indicator
                    status_color = {
                        "Correct": "#10b981",
                        "Incorrect": "#ef4444", 
                        "Unparseable": "#f59e0b",
                        "Unknown": "#6b7280",
                        "Error": "#dc2626"
                    }.get(result.get('status', 'Unknown'), "#6b7280")
                    
                    status_label = ctk.CTkLabel(
                        step_frame,
                        text=result.get('status', 'Unknown'),
                        font=ctk.CTkFont(size=12, weight="bold"),
                        text_color="white",
                        fg_color=status_color,
                        corner_radius=6,
                        width=100
                    )
                    status_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
                    
                    # Final answer info
                    final_answer_info = f"Final Answer: {result.get('student_work', '')}"
                    final_text = ctk.CTkLabel(
                        step_frame,
                        text=final_answer_info[:100] + "..." if len(final_answer_info) > 100 else final_answer_info,
                        font=ctk.CTkFont(size=12),
                        text_color="#374151",
                        wraplength=600,
                        justify="left"
                    )
                    final_text.grid(row=0, column=1, padx=10, pady=5, sticky="w")
                    
                    # Comment (if any)
                    comment = result.get('comment', '')
                    if comment:
                        comment_label = ctk.CTkLabel(
                            step_frame,
                            text=f"Comment: {comment}",
                            font=ctk.CTkFont(size=11),
                            text_color="#6b7280",
                            wraplength=600,
                            justify="left"
                        )
                        comment_label.grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 5), sticky="w")
                
                # Display expected answer for comparison
                expected_frame = ctk.CTkFrame(card, fg_color="#eff6ff")
                expected_frame.grid(row=len(marking_results) + 1, column=0, sticky="ew", padx=5, pady=2)
                
                expected_label = ctk.CTkLabel(
                    expected_frame,
                    text=f"Expected Answer: {question.final_answer}",
                    font=ctk.CTkFont(size=11, weight="bold"),
                    text_color="#1e40af"
                )
                expected_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
                
            except Exception as e:
                print(f"Error marking {student_id}: {e}")
                
                # Create error card
                card = ctk.CTkFrame(scrollable, corner_radius=10)
                card.grid(row=i, column=0, sticky="ew", pady=10, padx=10)
                
                error_header = ctk.CTkFrame(card, fg_color="#dc2626", corner_radius=8)
                error_header.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
                
                error_label = ctk.CTkLabel(
                    error_header,
                    text=f"Student {student_id} - Error",
                    font=ctk.CTkFont(size=16, weight="bold"),
                    text_color="white"
                )
                error_label.grid(row=0, column=0, padx=15, pady=10, sticky="w")
                
                error_msg = ctk.CTkLabel(
                    card,
                    text=f"Error: {str(e)}",
                    font=ctk.CTkFont(size=12),
                    text_color="#dc2626",
                    wraplength=600
                )
                error_msg.grid(row=1, column=0, padx=15, pady=10, sticky="w")

## ==================== PAGE 6: LLM MARKING ====================

class LLMMarkingPage(BasePage):
    def setup_page(self):
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Header
        header_frame = ctk.CTkFrame(self, height=80)
        header_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        header_frame.grid_propagate(False)
        header_frame.grid_columnconfigure(1, weight=1)
        
        back_btn = ctk.CTkButton(
            header_frame,
            text="← Back to SymPy",
            command=lambda: self.controller.show_page("SymPyMarkingPage"),
            font=ctk.CTkFont(size=14),
            fg_color="transparent",
            text_color="#2563eb",
            hover_color="#f1f5f9",
            width=120
        )
        back_btn.grid(row=0, column=0, padx=20, pady=20, sticky="w")
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="Step 5: LLM Intelligent Marking",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#1e293b"
        )
        title_label.grid(row=0, column=1, padx=20, pady=20)
        
        # Content - REARRANGED LAYOUT
        content_frame = ctk.CTkFrame(self, fg_color="white")
        content_frame.grid(row=1, column=0, sticky="nsew", padx=30, pady=20)  # Reduced padding
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_rowconfigure(3, weight=1)  # Results frame gets the space
        
        # Model configuration - MOVED UP
        config_frame = ctk.CTkFrame(content_frame, fg_color="#f8fafc")
        config_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=10)  # Reduced padding
        config_frame.grid_columnconfigure(1, weight=1)
        
        model_label = ctk.CTkLabel(
            config_frame,
            text="Ollama Model:",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#374151"
        )
        model_label.grid(row=0, column=0, padx=15, pady=8, sticky="w")  # Reduced padding
        
        self.model_entry = ctk.CTkEntry(
            config_frame,
            placeholder_text="Mathstral_Marker",
            font=ctk.CTkFont(size=14),
            height=35
        )
        self.model_entry.grid(row=0, column=1, padx=15, pady=8, sticky="ew")  # Reduced padding
        self.model_entry.insert(0, "Mathstral_Marker")
        
        # Mark all button - MOVED UP
        mark_btn = ctk.CTkButton(
            content_frame,
            text="🤖 Mark All Students with LLM",
            command=self.mark_all_students,
            font=ctk.CTkFont(size=18, weight="bold"),
            fg_color="#7e22ce",
            hover_color="#6b21a8",
            height=60,
            width=350,
            corner_radius=15
        )
        mark_btn.grid(row=1, column=0, pady=15)  # Reduced padding
        
        # Progress and logs area - MOVED UP and COMPACT
        self.progress_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        self.progress_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=10)  # Reduced padding
        self.progress_frame.grid_columnconfigure(0, weight=1)
        
        # Progress label - COMPACT
        self.progress_label = ctk.CTkLabel(
            self.progress_frame,
            text="Ready to start LLM marking...",
            font=ctk.CTkFont(size=12),  # Smaller font
            text_color="#64748b"
        )
        self.progress_label.grid(row=0, column=0, sticky="w", pady=(0, 5))  # Reduced padding
        
        # Progress bar - COMPACT
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame, height=8)  # Smaller height
        self.progress_bar.grid(row=1, column=0, sticky="ew", pady=(0, 8))  # Reduced padding
        self.progress_bar.set(0)
        
        # Logs textbox - COMPACT but still readable
        self.logs_textbox = ctk.CTkTextbox(
            self.progress_frame,
            height=120,  # Reduced from 250 to 120 (more compact)
            font=ctk.CTkFont(size=11, family="Consolas"),  # Smaller font
            fg_color="#1e293b",
            text_color="#e2e8f0"
        )
        self.logs_textbox.grid(row=2, column=0, sticky="nsew", pady=(0, 5))  # Reduced padding
        self.logs_textbox.insert("1.0", "LLM Marking Logs:\n" + "="*40 + "\n")
        self.logs_textbox.configure(state="disabled")
        
        # Results area - NOW TAKES MOST SPACE
        self.results_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        self.results_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=10)
        self.results_frame.grid_columnconfigure(0, weight=1)
        self.results_frame.grid_rowconfigure(0, weight=1)
        
        # Configure row weights to push results down and give it space
        content_frame.grid_rowconfigure(0, weight=0)  # Config frame - minimal space
        content_frame.grid_rowconfigure(1, weight=0)  # Mark button - minimal space  
        content_frame.grid_rowconfigure(2, weight=0)  # Progress - minimal space
        content_frame.grid_rowconfigure(3, weight=1)  # Results - takes remaining space


    def add_log(self, message: str):
        """Add a message to the logs with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.logs_textbox.configure(state="normal")
        self.logs_textbox.insert("end", log_entry)
        self.logs_textbox.see("end")
        self.logs_textbox.configure(state="disabled")
    
    def update_progress(self, current: int, total: int, message: str = ""):
        """Update progress bar and label"""
        progress = current / total if total > 0 else 0
        self.progress_bar.set(progress)
        
        if message:
            self.progress_label.configure(text=message)
        else:
            self.progress_label.configure(text=f"Processing {current}/{total} students...")
        
        # Force UI update
        self.update_idletasks()
    
    def run_ollama_with_logs(self, prompt: str, student_id: str, timeout: int = 120) -> str:
        """Run Ollama model with logging"""
        model_name = self.model_entry.get().strip() or "Mathstral_Marker"
        
        self.add_log(f"🔄 Invoking Ollama model '{model_name}' for student {student_id}...")
        
        try:
            # Start the process
            self.add_log(f"📤 Sending prompt to Ollama (timeout: {timeout}s)...")
            
            start_time = time.time()
            proc = subprocess.run(
                ["ollama", "run", model_name],
                input=prompt,
                text=True,
                capture_output=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace"
            )
            end_time = time.time()
            duration = end_time - start_time
            
            self.add_log(f"✅ Ollama response received in {duration:.1f}s")
            
            if proc.returncode != 0:
                error_msg = f"❌ Ollama failed with return code {proc.returncode}: {proc.stderr.strip()}"
                self.add_log(error_msg)
                return f"[ERROR] ollama failed: {proc.stderr.strip()}"
            
            self.add_log(f"📥 Received {len(proc.stdout)} characters from Ollama")
            return proc.stdout.strip()
            
        except FileNotFoundError:
            error_msg = "❌ ERROR: Ollama executable not found. Please install Ollama."
            self.add_log(error_msg)
            return "[ERROR] ollama executable not found."
        except subprocess.TimeoutExpired:
            error_msg = f"❌ ERROR: Ollama call timed out after {timeout} seconds"
            self.add_log(error_msg)
            return "[ERROR] ollama call timed out."
        except Exception as e:
            error_msg = f"❌ ERROR: Unexpected error: {str(e)}"
            self.add_log(error_msg)
            return f"[ERROR] {str(e)}"
    
    def extract_json_from_response(self, response: str):
        """Extract JSON from model response"""
        self.add_log("🔍 Extracting JSON from response...")
        
        cleaned = response.strip()
        
        # Remove markdown code blocks
        cleaned = re.sub(r'```json\s*', '', cleaned)
        cleaned = re.sub(r'```\s*', '', cleaned)
        
        # Try to find JSON object
        start_idx = cleaned.find('{')
        end_idx = cleaned.rfind('}')
        
        if start_idx == -1 or end_idx == -1:
            self.add_log("❌ No JSON object found in response")
            return {"error": "No JSON object found", "raw_response": response}
        
        json_str = cleaned[start_idx:end_idx+1]
        self.add_log(f"📄 Found JSON string ({len(json_str)} characters)")
        
        try:
            parsed = json.loads(json_str)
            self.add_log("✅ JSON parsed successfully")
            return parsed
        except json.JSONDecodeError as e:
            self.add_log(f"❌ JSON parse error: {str(e)}")
            return {
                "error": f"JSON parse error: {str(e)}",
                "json_attempt": json_str,
                "raw_response": response
            }
    
    def create_llm_prompt(self, question, student_id, readable_text, sympy_results):
        """Create LLM prompt for marking using readable text"""
        prompt = f"""BEGIN MATHS AUTOMARKING TASK. You are a math teacher grader. Analyze the student's derivation step by step, identify any mistakes, provide a final mark (0.0-5.0), list all mistakes made, give constructive feedback, and indicate whether you agree with SymPy's assessment.

QUESTION CONTEXT:
Field: {question.field_of_maths}
Topic: {question.topic}
Question: {question.question}
Expected Answer: {question.final_answer}

STUDENT WORK (Student {student_id}):
{readable_text}

SYMPY GRADING RESULTS:
{json.dumps(sympy_results, indent=2)}

YOUR TASK: Provide grading analysis in JSON format with fields: final_mark, mistakes_made, feedback, agrees_with_sympy, reasoning"""
        return prompt
    
    def mark_all_students(self):
        """Mark all student derivations using LLM with live progress"""
        question = self.controller.current_question
        if not question.student_derivations:
            messagebox.showwarning("Warning", "No student derivations to mark")
            return
        
        # Clear previous results and show progress area
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Reset progress
        self.progress_bar.set(0)
        self.progress_label.configure(text="Starting LLM marking...")
        self.logs_textbox.configure(state="normal")
        self.logs_textbox.delete("1.0", "end")
        self.logs_textbox.insert("1.0", "LLM Marking Logs:\n" + "="*40 + "\n")
        self.logs_textbox.configure(state="disabled")
        
        self.add_log("🚀 Starting LLM marking process...")
        self.add_log(f"📊 Processing {len(question.student_derivations)} students")
        
        # Create scrollable results area - USE AVAILABLE SPACE
        scrollable = ctk.CTkScrollableFrame(self.results_frame, fg_color="#f8fafc")  # Light grey background
        scrollable.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        scrollable.grid_columnconfigure(0, weight=1)
        
        # Process each student
        students = list(question.student_derivations.items())
        for i, (student_id, data) in enumerate(students):
            try:
                self.update_progress(i + 1, len(students), f"Processing student {student_id}...")
                self.add_log(f"🎯 Processing student {student_id} ({i+1}/{len(students)})")
                
                # Use readable_text and sympy_analysis
                readable_text = data.get('readable_text', '')
                sympy_results = data.get('sympy_analysis', [])
                
                if not readable_text:
                    self.add_log(f"⚠️  Skipping student {student_id}: No readable text")
                    continue
                
                self.add_log(f"📝 Student work length: {len(readable_text)} characters")
                self.add_log(f"🔢 SymPy results: {len(sympy_results)} items")
                
                # Create prompt for LLM using readable text
                prompt = self.create_llm_prompt(question, student_id, readable_text, sympy_results)
                self.add_log(f"📋 Created prompt ({len(prompt)} characters)")
                
                # Send to Ollama with logging
                response = self.run_ollama_with_logs(prompt, student_id)
                llm_result = self.extract_json_from_response(response)
                
                # Store LLM result
                data['llm_marking'] = llm_result
                self.add_log(f"💾 Stored LLM results for student {student_id}")
                
                # Create result card with MORE SPACE for feedback
                card = ctk.CTkFrame(scrollable, corner_radius=10, fg_color="white")
                card.grid(row=i, column=0, sticky="ew", pady=8, padx=8)
                card.grid_columnconfigure(0, weight=1)
                
                # Student ID header
                student_header = ctk.CTkFrame(card, fg_color="#1e293b", corner_radius=8)
                student_header.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
                student_header.grid_columnconfigure(1, weight=1)
                
                student_label = ctk.CTkLabel(
                    student_header,
                    text=f"Student {student_id} - LLM Analysis",
                    font=ctk.CTkFont(size=16, weight="bold"),
                    text_color="white"
                )
                student_label.grid(row=0, column=0, padx=15, pady=10, sticky="w")
                
                # LLM Results
                if 'error' in llm_result:
                    # Error case - MORE SPACE FOR ERROR MESSAGES
                    error_frame = ctk.CTkFrame(card, fg_color="#fef2f2")
                    error_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=2)
                    error_frame.grid_columnconfigure(0, weight=1)
                    
                    error_label = ctk.CTkLabel(
                        error_frame,
                        text=f"Error: {llm_result['error']}",
                        font=ctk.CTkFont(size=12),
                        text_color="#dc2626",
                        wraplength=600,  # Allow wrapping
                        justify="left"
                    )
                    error_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
                    
                    # Show raw response if available
                    if 'raw_response' in llm_result:
                        raw_text = ctk.CTkTextbox(
                            error_frame,
                            height=100,  # Space for raw response
                            font=ctk.CTkFont(size=10, family="Courier"),
                            fg_color="#1e293b",
                            text_color="#e2e8f0"
                        )
                        raw_text.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
                        raw_text.insert("1.0", f"Raw Response:\n{llm_result['raw_response']}")
                        raw_text.configure(state="disabled")
                    
                else:
                    # Successful analysis
                    # Final mark
                    mark_frame = ctk.CTkFrame(card, fg_color="#f0fdf4")
                    mark_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=2)
                    mark_frame.grid_columnconfigure(1, weight=1)
                    
                    mark_label = ctk.CTkLabel(
                        mark_frame,
                        text="Final Mark:",
                        font=ctk.CTkFont(size=12, weight="bold"),
                        text_color="#374151"
                    )
                    mark_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
                    
                    mark_value = llm_result.get('final_mark', 'N/A')
                    if mark_value == 'N/A' or not isinstance(mark_value, (int, float)):
                        mark_value = 0.0
                    
                    mark_display = ctk.CTkLabel(
                        mark_frame,
                        text=f"{mark_value}/5.0",
                        font=ctk.CTkFont(size=14, weight="bold"),
                        text_color="#059669"
                    )
                    mark_display.grid(row=0, column=1, padx=10, pady=5, sticky="e")
                    
                    # Agrees with SymPy
                    agrees_frame = ctk.CTkFrame(card, fg_color="#f8fafc")
                    agrees_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=2)
                    
                    agrees_value = llm_result.get('agrees_with_sympy', False)
                    agrees_color = "#059669" if agrees_value else "#dc2626"
                    agrees_text = "Yes" if agrees_value else "No"
                    
                    agrees_label = ctk.CTkLabel(
                        agrees_frame,
                        text=f"Agrees with SymPy: {agrees_text}",
                        font=ctk.CTkFont(size=12),
                        text_color=agrees_color
                    )
                    agrees_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
                    
                    # Mistakes - handle empty or missing lists
                    mistakes_list = llm_result.get('mistakes_made', [])
                    if not mistakes_list or mistakes_list == ["Field missing from LLM response"]:
                        # Show no mistakes if list is empty or contains only the default message
                        mistakes_frame = ctk.CTkFrame(card, fg_color="#f0fdf4")
                        mistakes_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=2)
                        
                        mistakes_label = ctk.CTkLabel(
                            mistakes_frame,
                            text="✅ No mistakes identified",
                            font=ctk.CTkFont(size=12, weight="bold"),
                            text_color="#059669"
                        )
                        mistakes_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
                    else:
                        # Show mistakes if they exist
                        mistakes_frame = ctk.CTkFrame(card, fg_color="#fffbeb")
                        mistakes_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=2)
                        mistakes_frame.grid_columnconfigure(0, weight=1)
                        
                        mistakes_label = ctk.CTkLabel(
                            mistakes_frame,
                            text="Mistakes Identified:",
                            font=ctk.CTkFont(size=12, weight="bold"),
                            text_color="#92400e"
                        )
                        mistakes_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
                        
                        mistakes_text = ", ".join(str(m) for m in mistakes_list)
                        mistakes_display = ctk.CTkTextbox(
                            mistakes_frame,
                            height=80,
                            font=ctk.CTkFont(size=11),
                            fg_color="#fffbeb",
                            text_color="#92400e",
                            wrap="word"
                        )
                        mistakes_display.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 5))
                        mistakes_display.insert("1.0", mistakes_text)
                        mistakes_display.configure(state="disabled")
                    
                    # Feedback - always show this section
                    feedback_text = llm_result.get('feedback', 'No feedback provided by LLM')
                    feedback_frame = ctk.CTkFrame(card, fg_color="#eff6ff")
                    feedback_frame.grid(row=4, column=0, sticky="ew", padx=5, pady=2)
                    feedback_frame.grid_columnconfigure(0, weight=1)
                    
                    feedback_label = ctk.CTkLabel(
                        feedback_frame,
                        text="LLM Feedback:",
                        font=ctk.CTkFont(size=12, weight="bold"),
                        text_color="#1e40af"
                    )
                    feedback_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
                    
                    feedback_display = ctk.CTkTextbox(
                        feedback_frame,
                        height=150,
                        font=ctk.CTkFont(size=11),
                        fg_color="#eff6ff",
                        text_color="#1e40af",
                        wrap="word"
                    )
                    feedback_display.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 5))
                    feedback_display.insert("1.0", feedback_text)
                    feedback_display.configure(state="disabled")
                    
                    # Reasoning - show if available
                    if 'reasoning' in llm_result and llm_result['reasoning']:
                        reasoning_frame = ctk.CTkFrame(card, fg_color="#f3f4f6")
                        reasoning_frame.grid(row=5, column=0, sticky="ew", padx=5, pady=2)
                        reasoning_frame.grid_columnconfigure(0, weight=1)
                        
                        reasoning_label = ctk.CTkLabel(
                            reasoning_frame,
                            text="LLM Reasoning:",
                            font=ctk.CTkFont(size=12, weight="bold"),
                            text_color="#374151"
                        )
                        reasoning_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
                        
                        reasoning_text = llm_result['reasoning']
                        reasoning_display = ctk.CTkTextbox(
                            reasoning_frame,
                            height=80,
                            font=ctk.CTkFont(size=11),
                            fg_color="#f3f4f6",
                            text_color="#374151",
                            wrap="word"
                        )
                        reasoning_display.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 5))
                        reasoning_display.insert("1.0", reasoning_text)
                        reasoning_display.configure(state="disabled")
                
                self.add_log(f"✅ Completed processing student {student_id}")
                
            except Exception as e:
                error_msg = f"❌ Error processing {student_id} with LLM: {str(e)}"
                self.add_log(error_msg)
                print(error_msg)
        
        # Final update
        self.update_progress(len(students), len(students), "LLM marking completed!")
        self.add_log("🎉 LLM marking process completed!")
        self.add_log("="*50)
# ==================== MAIN APPLICATION ====================

class MathAutoMarkApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.title("MathAutoMark - Intelligent Mathematics Grading")
        self.geometry("1200x800")
        self.minsize(1000, 700)
        
        # Set theme
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        
        # Initialize current question
        self.current_question = TeacherQuestion()
        
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Create container for pages
        self.container = ctk.CTkFrame(self, corner_radius=0)
        self.container.grid(row=0, column=0, sticky="nsew")
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        
        # Initialize pages
        self.pages = {}
        self.create_pages()
        
        # Show initial page
        self.show_page("IntroductionPage")
    
    def create_pages(self):
        """Create all application pages"""
        pages = [
            ("IntroductionPage", IntroductionPage),
            ("TeacherInputPage", TeacherInputPage),
            ("ImageUploadPage", ImageUploadPage),
            ("GradingPage", SymPyMarkingPage),
            ("LLMMarkingPage", LLMMarkingPage)
        ]
        
        for page_name, PageClass in pages:
            page = PageClass(self.container, self)
            self.pages[page_name] = page
            page.grid(row=0, column=0, sticky="nsew")
    
    def show_page(self, page_name):
        """Show the specified page"""
        page = self.pages[page_name]
        page.tkraise()
# ==================== APPLICATION START ====================

if __name__ == "__main__":
    app = MathAutoMarkApp()
    app.mainloop()