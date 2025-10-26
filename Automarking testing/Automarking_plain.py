import sympy
import json
import re
import random
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor

# Add power operator transformation and implicit multiplication
transformations = standard_transformations + (convert_xor, implicit_multiplication_application)

# ==============================================================================
#  TASK 1: THE TESTING DATASET
# ==============================================================================
import sympy
import json
import re
import random
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor

# Add power operator transformation and implicit multiplication
transformations = standard_transformations + (convert_xor, implicit_multiplication_application)

# ==============================================================================
#  TASK 1: THE TESTING DATASET (unchanged)
# ==============================================================================
questions_db = [
  {
    "id": 1,
    "field_of_maths": "Calculus",
    "topic": "Differentiation",
    "question": "Differentiate f(x) = x^3 * cos(x) with respect to x.",
    "variables": ["x"],
    "final_answer": "3*x**2*cos(x) - x**3*sin(x)",
    "student_answers": [
      {"student_id": "A", "type": "Fully Correct", "derivation": ["x^3 * cos(x)", "3*x**2*cos(x) + x**3*(-sin(x))", "3*x**2*cos(x) - x**3*sin(x)"]},
      {"student_id": "B", "type": "Partially Correct (Sign Error)", "derivation": ["x^3 * cos(x)", "3*x**2*cos(x) + x**3*(sin(x))", "3*x**2*cos(x) + x**3*sin(x)"]},
      {"student_id": "C", "type": "Incorrect Method (Forgot Product Rule)", "derivation": ["x^3 * cos(x)", "(3*x**2) * (-sin(x))", "-3*x**2*sin(x)"]},
      {"student_id": "D", "type": "Correct Final Answer, Invalid Step", "derivation": ["x^3 * cos(x)", "3*x**2*cos(x) + x**3*sin(x)", "3*x**2*cos(x) - x**3*sin(x)"]}
    ]
  },
  {
    "id": 2,
    "field_of_maths": "Algebra",
    "topic": "Quadratic Equations",
    "question": "Solve for x: x**2 - 5*x + 6 = 0",
    "variables": ["x"],
    "final_answer": "{2.0, 3.0}",
    "student_answers": [
      {"student_id": "A", "type": "Fully Correct", "derivation": ["x**2 - 5*x + 6 = 0", "(x - 2)*(x - 3) = 0", "x = 2, x = 3"]},
      {"student_id": "B", "type": "Incorrect (Factoring Error)", "derivation": ["x**2 - 5*x + 6 = 0", "(x + 2)*(x - 3) = 0", "x = -2, x = 3"]},
      {"student_id": "C", "type": "Correct Final Answer, Wrong Method (Guess)", "derivation": ["x**2 - 5*x + 6 = 0", "I think the answers are 2 and 3", "x=2, x=3"]},
      {"student_id": "D", "type": "Partially Correct (Forgot one solution)", "derivation": ["x**2 - 5*x + 6 = 0", "(x - 2)*(x - 3) = 0", "x = 2"]}
    ]
  },
  {
    "id": 3,
    "field_of_maths": "Trigonometry",
    "topic": "Identities",
    "question": "Prove the identity: tan(x) + cot(x) = sec(x)*csc(x)",
    "variables": ["x"],
    "final_answer": "sec(x)*csc(x)",
    "student_answers": [
      {"student_id": "A", "type": "Fully Correct", "derivation": ["tan(x) + cot(x)", "sin(x)/cos(x) + cos(x)/sin(x)", "(sin(x)**2 + cos(x)**2) / (cos(x)*sin(x))", "1 / (cos(x)*sin(x))", "sec(x)*csc(x)"]},
      {"student_id": "B", "type": "Partially Correct (Identity Error)", "derivation": ["tan(x) + cot(x)", "sin(x)/cos(x) + cos(x)/sin(x)", "(sin(x)**2 + cos(x)**2) / (cos(x)*sin(x))", "0 / (cos(x)*sin(x))"]},
      {"student_id": "C", "type": "Incorrect (Conceptual Error)", "derivation": ["tan(x) + cot(x)", "sin(x)/cos(x) + sin(x)/cos(x)", "2*tan(x)"]},
      {"student_id": "D", "type": "Correct Final Answer, Invalid Step", "derivation": ["tan(x) + cot(x)", "1 / cos(x) + 1 / sin(x)", "sec(x)*csc(x)"]}
    ]
  },
  {
    "id": 4,
    "field_of_maths": "Calculus",
    "topic": "Integration",
    "question": "Find the indefinite integral of f(x) = 4*x**3 + 3*x**2.",
    "variables": ["x", "C"],
    "final_answer": "x**4 + x**3 + C",
    "student_answers": [
        {"student_id": "A", "type": "Fully Correct", "derivation": ["4*x**3 + 3*x**2", "x**4 + x**3 + C"]},
        {"student_id": "B", "type": "Partially Correct (Forgot Constant of Integration)", "derivation": ["4*x**3 + 3*x**2", "x**4 + x**3"]},
        {"student_id": "C", "type": "Incorrect (Confused with differentiation)", "derivation": ["4*x**3 + 3*x**2", "12*x**2 + 6*x"]},
        {"student_id": "D", "type": "Correct Final Answer, No work", "derivation": ["4*x**3 + 3*x**2", "x**4 + x**3 + C"]}
    ]
  },
  {
    "id": 5,
    "field_of_maths": "Algebra",
    "topic": "Simplification",
    "question": "Simplify the expression: (x**2 - 9) / (x + 3)",
    "variables": ["x"],
    "final_answer": "x - 3",
    "student_answers": [
        {"student_id": "A", "type": "Fully Correct", "derivation": ["(x**2 - 9) / (x + 3)", "(x - 3)*(x + 3) / (x + 3)", "x - 3"]},
        {"student_id": "B", "type": "Incorrect (Invalid Cancellation)", "derivation": ["(x**2 - 9) / (x + 3)", "(x**2 / x) - (9 / 3)", "x - 3"]},
        {"student_id": "C", "type": "Incorrect (Factoring Error)", "derivation": ["(x**2 - 9) / (x + 3)", "(x - 3)*(x - 3) / (x + 3)"]},
        {"student_id": "D", "type": "Partially Correct (Didn't fully simplify)", "derivation": ["(x**2 - 9) / (x + 3)", "(x-3)*(x+3)/(x+3)"]}
    ]
  },
  {
      "id": 6,
      "field_of_maths": "Calculus",
      "topic": "Limits",
      "question": "Evaluate the limit of (x**2 - 4)/(x - 2) as x approaches 2.",
      "variables": ["x"],
      "final_answer": "4",
      "student_answers": [
        {"student_id": "A", "type": "Fully Correct", "derivation": ["(x**2 - 4)/(x - 2)", "limit as x->2 of (x-2)(x+2)/(x-2)", "limit as x->2 of (x+2)", "4"]},
        {"student_id": "B", "type": "Incorrect (Direct Substitution)", "derivation": ["(x**2 - 4)/(x - 2)", "(2**2 - 4)/(2-2)", "0/0", "Undefined"]},
        {"student_id": "C", "type": "Correct Final Answer, Wrong Method (L'Hopital's Rule)", "derivation": ["(x**2 - 4)/(x - 2)", "Using L'Hopital's Rule: limit of (2*x)/(1) as x->2", "4"]},
        {"student_id": "D", "type": "Partially Correct (Algebra mistake)", "derivation": ["(x**2 - 4)/(x - 2)", "limit as x->2 of (x-2)(x-2)/(x-2)", "limit as x->2 of (x-2)", "0"]}
    ]
  },
  {
      "id": 7,
      "field_of_maths": "Algebra",
      "topic": "Logarithms",
      "question": "Solve for x: log(x, 2) + log(x-2, 2) = 3",
      "variables": ["x"],
      "final_answer": "4",
      "student_answers": [
          {"student_id": "A", "type": "Fully Correct", "derivation": ["log(x, 2) + log(x-2, 2) = 3", "log(x*(x-2), 2) = 3", "x*(x-2) = 2**3", "x**2 - 2*x = 8", "x**2 - 2*x - 8 = 0", "(x-4)*(x+2) = 0", "x=4"]},
          {"student_id": "B", "type": "Partially Correct (Forgot to check domain)", "derivation": ["log(x, 2) + log(x-2, 2) = 3", "log(x*(x-2), 2) = 3", "x**2 - 2*x - 8 = 0", "(x-4)*(x+2) = 0", "x=4, x=-2"]},
          {"student_id": "C", "type": "Incorrect (Log Rule Error)", "derivation": ["log(x, 2) + log(x-2, 2) = 3", "log(x + (x-2), 2) = 3", "log(2*x - 2, 2) = 3", "2*x - 2 = 8", "x=5"]},
          {"student_id": "D", "type": "Incorrect (Exponent Error)", "derivation": ["log(x, 2) + log(x-2, 2) = 3", "log(x*(x-2), 2) = 3", "x*(x-2) = 3**2", "x**2 - 2*x - 9 = 0"]}
      ]
  },
  {
      "id": 8,
      "field_of_maths": "Trigonometry",
      "topic": "Solving Equations",
      "question": "Find the general solution for 2*sin(x) - 1 = 0.",
      "variables": ["x", "k", "pi"],
      "final_answer": "x = pi/6 + 2*k*pi or x = 5*pi/6 + 2*k*pi",
      "student_answers": [
          {"student_id": "A", "type": "Fully Correct", "derivation": ["2*sin(x) - 1 = 0", "sin(x) = 1/2", "Reference angle is pi/6", "x = pi/6 + 2*k*pi and x = pi - pi/6 + 2*k*pi", "x = 5*pi/6 + 2*k*pi"]},
          {"student_id": "B", "type": "Partially Correct (Forgot general solution)", "derivation": ["2*sin(x) - 1 = 0", "sin(x) = 1/2", "x = pi/6 and x = 5*pi/6"]},
          {"student_id": "C", "type": "Partially Correct (Forgot second quadrant solution)", "derivation": ["2*sin(x) - 1 = 0", "sin(x) = 1/2", "x = pi/6 + 2*k*pi"]},
          {"student_id": "D", "type": "Incorrect (Unit circle error)", "derivation": ["2*sin(x) - 1 = 0", "sin(x) = 1/2", "x = pi/3 + 2*k*pi and x = 2*pi/3 + 2*k*pi"]}
      ]
  },
  {
    "id": 9,
    "field_of_maths": "Algebra",
    "topic": "Financial Maths",
    "question": "R1500 is invested at 8% p.a. compounded monthly. What is the value after 2 years? Formula: A = P*(1 + i)**n",
    "variables": ["A", "P", "i", "n"],
    "final_answer": "1759.39",
    "student_answers": [
        {"student_id": "A", "type": "Fully Correct", "derivation": ["R1500 at 8% p.a. compounded monthly for 2 years", "P = 1500, i = 0.08/12, n = 2*12", "A = 1500 * (1 + 0.08/12)**24", "1759.39"]},
        {"student_id": "B", "type": "Incorrect (Wrong compounding periods)", "derivation": ["R1500 at 8% p.a. compounded monthly for 2 years", "P = 1500, i = 0.08/12, n = 2", "A = 1500 * (1 + 0.08/12)**2", "1520.08"]},
        {"student_id": "C", "type": "Incorrect (Did not divide interest rate)", "derivation": ["R1500 at 8% p.a. compounded monthly for 2 years", "P = 1500, i = 0.08, n = 24", "A = 1500 * (1 + 0.08)**24", "9511.92"]},
        {"student_id": "D", "type": "Partially Correct (Calculation error)", "derivation": ["R1500 at 8% p.a. compounded monthly for 2 years", "P = 1500, i = 0.08/12, n = 24", "A = 1500 * (1.00666)**24", "1755.50"]}
    ]
  },
  {
      "id": 10,
      "field_of_maths": "Calculus",
      "topic": "Stationary Points",
      "question": "Find the x-coordinates of the stationary points of f(x) = 2*x**3 - 3*x**2 - 12*x + 1.",
      "variables": ["x"],
      "final_answer": "x = -1 or x = 2",
      "student_answers": [
          {"student_id": "A", "type": "Fully Correct", "derivation": ["f(x) = 2*x**3 - 3*x**2 - 12*x + 1", "f'(x) = 6*x**2 - 6*x - 12", "Set f'(x) = 0: 6*x**2 - 6*x - 12 = 0", "x**2 - x - 2 = 0", "(x-2)*(x+1) = 0", "x = 2, x = -1"]},
          {"student_id": "B", "type": "Partially Correct (Differentiation Error)", "derivation": ["f(x) = 2*x**3 - 3*x**2 - 12*x + 1", "f'(x) = 6*x**2 - 6*x - 12x", "f'(x) = 6*x**2 - 18*x = 0", "6x(x-3) = 0", "x=0, x=3"]},
          {"student_id": "C", "type": "Partially Correct (Factoring Error)", "derivation": ["f(x) = 2*x**3 - 3*x**2 - 12*x + 1", "f'(x) = 6*x**2 - 6*x - 12", "x**2 - x - 2 = 0", "(x-1)*(x+2) = 0", "x=1, x=-2"]},
          {"student_id": "D", "type": "Incorrect Method (Set f(x)=0)", "derivation": ["f(x) = 2*x**3 - 3*x**2 - 12*x + 1", "2*x**3 - 3*x**2 - 12*x + 1 = 0"]}
      ]
  }
]

# ==============================================================================
#  ENHANCED PARSING AND VERIFICATION FUNCTIONS - FIXED VERSION
# ==============================================================================

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
        r"^Integral\(.*?\)\s*dx",
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
    
    return feedback

# ==============================================================================
#  ENHANCED MARKING FUNCTIONS - UPDATED
# ==============================================================================

def mark_calculus_enhanced(question, student):
    """Enhanced calculus marking"""
    local_symbols = {var: sympy.Symbol(var) for var in question['variables']}
    x = local_symbols.get('x')
    
    problem_str = question['question']
    
    try:
        # Extract base function
        base_func_str = extract_base_function(problem_str)
        base_function = parse_expr(base_func_str, local_dict=local_symbols, transformations=transformations)
        
        return enhanced_verify_steps(
            student['derivation'], 
            base_function, 
            question['final_answer'],
            local_symbols
        )
        
    except Exception as e:
        return [{"status": "Error", "comment": f"Failed to parse problem: {str(e)[:100]}"}]

def mark_algebra_enhanced(question, student):
    """Enhanced algebra marking"""
    local_symbols = {var: sympy.Symbol(var) for var in question['variables']}
    # Add common constants
    if 'pi' in question['variables']:
        local_symbols['pi'] = sympy.pi
    if 'k' in question['variables']:
        local_symbols['k'] = sympy.Symbol('k', integer=True)
    
    problem_str = question['question']
    
    try:
        # Extract mathematical content
        math_content = extract_base_function(problem_str)
        
        # Check if it's an equation or expression
        if '=' in math_content and 'Solve' not in problem_str:
            # Parse as equation
            parts = math_content.split('=')
            if len(parts) == 2:
                left_expr = parse_expr(parts[0].strip(), local_dict=local_symbols, transformations=transformations)
                right_expr = parse_expr(parts[1].strip(), local_dict=local_symbols, transformations=transformations)
                initial_obj = sympy.Eq(left_expr, right_expr)
        elif 'Solve' in problem_str or 'solve' in problem_str:
            # For solve problems, parse the equation
            if '=' in math_content:
                parts = math_content.split('=')
                left_expr = parse_expr(parts[0].strip(), local_dict=local_symbols, transformations=transformations)
                right_expr = parse_expr(parts[1].strip(), local_dict=local_symbols, transformations=transformations)
                initial_obj = sympy.Eq(left_expr, right_expr)
            else:
                # Expression = 0
                expr = parse_expr(math_content.strip(), local_dict=local_symbols, transformations=transformations)
                initial_obj = sympy.Eq(expr, 0)
        else:
            # Expression problem
            initial_obj = parse_expr(math_content.strip(), local_dict=local_symbols, transformations=transformations)
        
        return enhanced_verify_steps(
            student['derivation'], 
            initial_obj, 
            question['final_answer'],
            local_symbols
        )
        
    except Exception as e:
        return [{"status": "Error", "comment": f"Failed to parse problem: {str(e)[:100]}"}]

def mark_trigonometry_enhanced(question, student):
    """Enhanced trigonometry marking"""
    local_symbols = {var: sympy.Symbol(var) for var in question['variables']}
    local_symbols['pi'] = sympy.pi
    if 'k' in question['variables']:
        local_symbols['k'] = sympy.Symbol('k', integer=True)
    
    return mark_algebra_enhanced(question, student)

def main_marker_enhanced(question_data, student_answer):
    """Main dispatcher using enhanced marking functions"""
    field = question_data['field_of_maths']
    
    if field == "Calculus":
        return mark_calculus_enhanced(question_data, student_answer)
    elif field == "Algebra":
        return mark_algebra_enhanced(question_data, student_answer)
    elif field == "Trigonometry":
        return mark_trigonometry_enhanced(question_data, student_answer)
    else:
        return [{"status": "Error", "comment": f"No marking function available for '{field}'."}]

# ==============================================================================
#  OUTPUT LOGGING - SAVE AS JSON FILE
# ==============================================================================
if __name__ == "__main__":
    full_log = {}

    for question in questions_db:
        q_id = question['id']
        full_log[f"Question_{q_id}"] = {
            "metadata": {
                "field": question['field_of_maths'],
                "topic": question['topic'],
                "question_text": question['question'],
                "expected_answer": question['final_answer']
            },
            "student_verdicts": []
        }
        
        for student in question['student_answers']:
            # Get the verdict from the enhanced automarker
            verdict = main_marker_enhanced(question, student)
            
            # Append the detailed log for this student
            full_log[f"Question_{q_id}"]["student_verdicts"].append({
                "student_id": student['student_id'],
                "expected_type": student['type'],
                "submitted_derivation": student['derivation'],
                "sympy_grading": verdict
            })

    # Save the entire log as a JSON file
    output_filename = "automarking_results_fixed.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(full_log, f, indent=2, ensure_ascii=False)
    
    print(f"Fixed automarking completed! Results saved to {output_filename}")
    
    # Also print a summary to console
    print("\n=== FIXED SYSTEM SUMMARY ===")
    total_students = 0
    successful_parses = 0
    total_steps = 0
    correct_steps = 0
    correct_final_answers = 0
    
    for q_key, q_data in full_log.items():
        print(f"\n{q_key}: {q_data['metadata']['field']} - {q_data['metadata']['topic']}")
        for student in q_data['student_verdicts']:
            total_students += 1
            student_steps = len(student['sympy_grading'])
            
            # Count steps with valid parsing
            if student_steps > 0 and student['sympy_grading'][0].get('status') not in ['Error', 'Unparseable']:
                successful_parses += 1
            
            # Count correct steps (excluding final answer validation)
            correct_count = sum(1 for step in student['sympy_grading'] 
                              if step.get('status') == 'Correct' and step.get('step') != 'Final Answer')
            total_steps += student_steps - 1  # Exclude final answer step from step count
            
            # Check final answer correctness
            final_answer_step = next((step for step in student['sympy_grading'] 
                                   if step.get('step') == 'Final Answer' and step.get('status') == 'Correct'), None)
            if final_answer_step:
                correct_final_answers += 1
            
            correct_steps += correct_count
            
            print(f"  Student {student['student_id']}: {correct_count}/{student_steps-1} correct steps, Final Answer: {'✓' if final_answer_step else '✗'}")

    success_rate = (successful_parses / total_students) * 100 if total_students > 0 else 0
    step_accuracy = (correct_steps / total_steps) * 100 if total_steps > 0 else 0
    final_answer_accuracy = (correct_final_answers / total_students) * 100 if total_students > 0 else 0
    
    print(f"\nOverall Results:")
    print(f"  Student success rate: {success_rate:.1f}% ({successful_parses}/{total_students} students parsed successfully)")
    print(f"  Step accuracy: {step_accuracy:.1f}% ({correct_steps}/{total_steps} steps correct)")
    print(f"  Final answer accuracy: {final_answer_accuracy:.1f}% ({correct_final_answers}/{total_students} correct final answers)")

# ==============================================================================
#  ENHANCED PARSING AND VERIFICATION FUNCTIONS - FIXED VERSION
# ==============================================================================

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
        # Remove the problematic Integral pattern that was breaking parsing
        # r"^Integral\(.*?\)\s*dx",  # COMMENT THIS LINE OUT
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

def parse_integral_notation(step_str, local_dict):
    """Parse integral notation like 'Integral(4*x**3 + 3*x**2) dx'"""
    # Pattern for Integral(expression) dx
    integral_pattern = r'Integral\s*\(\s*([^)]+)\s*\)\s*dx'
    match = re.search(integral_pattern, step_str, re.IGNORECASE)
    
    if match:
        integrand_str = match.group(1).strip()
        try:
            # Parse the integrand expression
            integrand = parse_expr(integrand_str, local_dict=local_dict, transformations=transformations)
            # Create the integral object
            x = local_dict.get('x', sympy.Symbol('x'))
            integral_obj = sympy.Integral(integrand, x)
            return integral_obj
        except Exception as e:
            return None
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
    
    # Try to parse integral notation FIRST (before regular parsing)
    integral_obj = parse_integral_notation(original_step, local_dict)
    if integral_obj is not None:
        return integral_obj
    
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
    
    # Handle integral objects
    if isinstance(obj1, sympy.Integral) or isinstance(obj2, sympy.Integral):
        if isinstance(obj1, sympy.Integral) and isinstance(obj2, sympy.Integral):
            # Compare integrands
            return is_equivalent_math(obj1.args[0], obj2.args[0], local_dict, tolerance)
        elif isinstance(obj1, sympy.Integral) and not isinstance(obj2, sympy.Integral):
            # Compare integral with its evaluated result
            try:
                evaluated = obj1.doit()
                return is_equivalent_math(evaluated, obj2, local_dict, tolerance)
            except:
                return False
        elif not isinstance(obj1, sympy.Integral) and isinstance(obj2, sympy.Integral):
            try:
                evaluated = obj2.doit()
                return is_equivalent_math(obj1, evaluated, local_dict, tolerance)
            except:
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

def extract_solutions_from_text(step_str):
    """Extract solution values from natural language statements"""
    # Patterns for "x = 2, x = 3" or "x = 2 or x = 3" or "x=2, x=3"
    solution_patterns = [
        r'x\s*=\s*(-?[\d.]+)(?:\s*,\s*|\s+or\s+|\s+and\s+)x\s*=\s*(-?[\d.]+)',
        r'x\s*=\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)',
        r'x\s*=\s*(-?[\d.]+)\s+or\s+x\s*=\s*(-?[\d.]+)',
        r'x\s*=\s*(-?[\d.]+)\s+and\s+x\s*=\s*(-?[\d.]+)',
    ]
    
    for pattern in solution_patterns:
        match = re.search(pattern, step_str, re.IGNORECASE)
        if match:
            # Return as a set-like structure
            val1, val2 = match.groups()
            return f"{{{val1}, {val2}}}"
    
    # Single solution pattern
    single_match = re.search(r'x\s*=\s*(-?[\d.]+)', step_str.strip(), re.IGNORECASE)
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
    """Check mathematical equivalence with improved solution set handling"""
    
    # Handle solution sets
    if isinstance(obj1, dict) and obj1.get("type") == "solution_set":
        if isinstance(obj2, dict) and obj2.get("type") == "solution_set":
            return obj1.get("value") == obj2.get("value")
        # Compare solution set string with expected final answer
        expected_solution = parse_equation_or_expression(obj2, local_dict)
        if expected_solution and isinstance(expected_solution, dict) and expected_solution.get("type") == "solution_set":
            return obj1.get("value") == expected_solution.get("value")
    
    # ... rest of your existing logic
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

# ...existing code...
def enhanced_verify_steps(derivation, initial_math_obj, expected_final_answer=None, local_dict=None):
    """Verify derivation steps with proper step-by-step comparison (compatible signature)."""
    feedback = []
    if local_dict is None:
        local_dict = {}

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

    # If an expected_final_answer was provided, optionally include final check here
    if expected_final_answer is not None and derivation:
        last_step = derivation[-1]
        expected_parsed = parse_equation_or_expression(expected_final_answer, local_dict)
        last_parsed = parse_equation_or_expression(last_step, local_dict)
        final_correct = False
        if expected_parsed is not None and last_parsed is not None:
            try:
                final_correct = is_equivalent_math(last_parsed, expected_parsed, local_dict)
            except:
                final_correct = False
        feedback.append({
            "step": "Final Answer",
            "student_work": last_step,
            "status": "Correct" if final_correct else "Incorrect",
            "comment": "Final answer matches expected result" if final_correct else "Final answer does not match expected result"
        })

    return feedback

# ==============================================================================
#  ENHANCED MARKING FUNCTIONS
# ==============================================================================

def mark_calculus_enhanced(question, student):
    """Enhanced calculus marking with proper differentiation"""
    local_symbols = {var: sympy.Symbol(var) for var in question['variables']}
    x = local_symbols.get('x')
    
    problem_str = question['question']
    
    try:
        # Extract base function and differentiate it
        base_func_str = extract_base_function(problem_str)
        base_function = parse_expr(base_func_str, local_dict=local_symbols, transformations=transformations)
        
        # For differentiation problems, the true first step is the derivative
        if "differentiate" in problem_str.lower() or "f'" in problem_str:
            true_first_step = sympy.diff(base_function, x)
        else:
            true_first_step = base_function
            
        return enhanced_verify_steps(
            student['derivation'], 
            true_first_step, 
            question['final_answer'],
            local_symbols
        )
        
    except Exception as e:
        return [{"status": "Error", "comment": f"Failed to parse problem: {str(e)[:100]}"}]

def mark_algebra_enhanced(question, student):
    """Enhanced algebra marking"""
    local_symbols = {var: sympy.Symbol(var) for var in question['variables']}
    # Add common constants
    if 'pi' in question['variables']:
        local_symbols['pi'] = sympy.pi
    if 'k' in question['variables']:
        local_symbols['k'] = sympy.Symbol('k', integer=True)
    
    problem_str = question['question']
    
    try:
        # Extract mathematical content
        math_content = extract_base_function(problem_str)
        
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
            student['derivation'], 
            initial_obj, 
            question['final_answer'],
            local_symbols
        )
        
    except Exception as e:
        return [{"status": "Error", "comment": f"Failed to parse problem: {str(e)[:100]}"}]

def mark_trigonometry_enhanced(question, student):
    """Enhanced trigonometry marking"""
    local_symbols = {var: sympy.Symbol(var) for var in question['variables']}
    local_symbols['pi'] = sympy.pi
    if 'k' in question['variables']:
        local_symbols['k'] = sympy.Symbol('k', integer=True)
    
    return mark_algebra_enhanced(question, student)


def main_marker_enhanced(question_data, student_answer):
    """Main dispatcher using enhanced marking functions"""
    field = question_data['field_of_maths']
    
    if field == "Calculus":
        step_feedback = mark_calculus_enhanced(question_data, student_answer)
    elif field == "Algebra":
        step_feedback = mark_algebra_enhanced(question_data, student_answer)
    elif field == "Trigonometry":
        step_feedback = mark_trigonometry_enhanced(question_data, student_answer)
    else:
        return [{"status": "Error", "comment": f"No marking function available for '{field}'."}]
    
    # Add final answer validation
    if student_answer['derivation']:
        last_step = student_answer['derivation'][-1]
        local_symbols = {var: sympy.Symbol(var) for var in question_data['variables']}
        
        # Parse the expected final answer and student's last step
        expected_final_parsed = parse_equation_or_expression(question_data['final_answer'], local_symbols)
        last_step_parsed = parse_equation_or_expression(last_step, local_symbols)
        
        final_correct = False
        # Check if both parsed successfully and are equivalent
        if last_step_parsed is not None and expected_final_parsed is not None:
            try:
                final_correct = is_equivalent_math(last_step_parsed, expected_final_parsed, local_symbols)
            except:
                final_correct = False
        
        # Add final answer check to feedback
        step_feedback.append({
            "step": "Final Answer",
            "student_work": last_step,
            "status": "Correct" if final_correct else "Incorrect",
            "comment": "Final answer matches expected result" if final_correct else "Final answer does not match expected result"
        })
    
    return step_feedback

# ==============================================================================
#  OUTPUT LOGGING - SAVE AS JSON FILE
# ==============================================================================
if __name__ == "__main__":
    full_log = {}

    for question in questions_db:
        q_id = question['id']
        full_log[f"Question_{q_id}"] = {
            "metadata": {
                "field": question['field_of_maths'],
                "topic": question['topic'],
                "question_text": question['question']
            },
            "student_verdicts": []
        }
        
        for student in question['student_answers']:
            # Get the verdict from the enhanced automarker
            verdict = main_marker_enhanced(question, student)
            
            # Append the detailed log for this student
            full_log[f"Question_{q_id}"]["student_verdicts"].append({
                "student_id": student['student_id'],
                "expected_type": student['type'],
                "submitted_derivation": student['derivation'],
                "sympy_grading": verdict
            })

    # Save the entire log as a JSON file
    output_filename = "automarking_results_fixed6.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(full_log, f, indent=2, ensure_ascii=False)
    
    print(f"Fixed automarking completed! Results saved to {output_filename}")
    
    # Also print a summary to console
    print("\n=== FIXED SYSTEM SUMMARY ===")
    total_students = 0
    successful_parses = 0
    total_steps = 0
    correct_steps = 0
    
    
    for q_key, q_data in full_log.items():
        print(f"\n{q_key}: {q_data['metadata']['field']} - {q_data['metadata']['topic']}")
        for student in q_data['student_verdicts']:
            total_students += 1
            student_steps = len(student['sympy_grading'])
            if student_steps > 0 and student['sympy_grading'][0].get('status') not in ['Error', 'Unparseable']:
                successful_parses += 1
            
            correct_count = sum(1 for step in student['sympy_grading'] 
                              if step.get('status') in ['Correct', 'Assumed Correct'])
            total_steps += student_steps
            correct_steps += correct_count
            
            print(f"  Student {student['student_id']}: {correct_count}/{student_steps} correct steps")
    
    success_rate = (successful_parses / total_students) * 100 if total_students > 0 else 0
    step_accuracy = (correct_steps / total_steps) * 100 if total_steps > 0 else 0
    
    print(f"\nOverall Results:")
    print(f"  Student success rate: {success_rate:.1f}% ({successful_parses}/{total_students} students parsed successfully)")
    print(f"  Step accuracy: {step_accuracy:.1f}% ({correct_steps}/{total_steps} steps correct)")