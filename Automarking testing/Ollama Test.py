#!/usr/bin/env python3
"""
llm_automarker_improved.py

Improved version with better JSON parsing and simplified output.
Uses Ollama (Mistral) to mark student answers and provide feedback.

"""

import json
import subprocess
import time
import re
from typing import Dict, List, Any, Optional
from pathlib import Path

# Configuration
OLLAMA_MODEL = "mistral"  # Change to your preferred model
INPUT_FILE = "automarking_results_final_only.json"  # Output from the previous script
OUTPUT_FILE = "llm_automarking_results_improved.json"
USE_OLLAMA = True

# ---------- Ollama (Mistral) wrapper ----------
# ...existing code...
def run_ollama(prompt: str, model: str = OLLAMA_MODEL, timeout: int = 120) -> str:
    """
    Runs `ollama run <model>` sending prompt on stdin. Returns stdout text.
    """
    try:
        proc = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            encoding="utf-8",    # force utf-8 decoding
            errors="replace"     # replace undecodable bytes instead of raising
        )
        if proc.returncode != 0:
            # stderr already safely decoded
            return f"[ERROR] ollama failed: {proc.stderr.strip()}"
        return proc.stdout.strip()
    except FileNotFoundError:
        return "[ERROR] ollama executable not found. Install ollama or set USE_OLLAMA=False."
    except subprocess.TimeoutExpired:
        return "[ERROR] ollama call timed out."
# ...existing code...

def build_ollama_prompt(question_data: Dict, student_data: Dict, sympy_verdict: List) -> str:
    """
    Build a simplified prompt for Mistral to grade the student's work.
    """
    question_text = question_data["metadata"]["question_text"]
    expected_answer = question_data["metadata"]["expected_answer"]
    field = question_data["metadata"]["field"]
    topic = question_data["metadata"]["topic"]
    
    student_id = student_data["student_id"]
    derivation = student_data["submitted_derivation"]
    
    # Get SymPy final verdict
    sympy_final_status = "Unknown"
    for step in sympy_verdict:
        if step.get("step") == "Final Answer":
            sympy_final_status = step.get("status", "Unknown")
            break
    
    prompt_lines = [
        "You are a mathematics teacher grading student work. Grade this solution out of 5.",
        "",
        "QUESTION:",
        question_text,
        f"Expected: {expected_answer}",
        "",
        "STUDENT'S WORK:",
    ]
    
    # Add student's derivation steps
    for i, step in enumerate(derivation, 1):
        prompt_lines.append(f"Step {i}: {step}")
    
    prompt_lines.extend([
        "",
        f"SymPy automated check: {sympy_final_status}",
        "",
        "GRADING:",
        "- 1 point: Correct setup",
        "- 1 point: Method selection", 
        "- 1 point: Mathematical accuracy",
        "- 1 point: Logical progression",
        "- 1 point: Final answer",
        "",
        "RETURN JSON ONLY with this exact format:",
        "{",
        '  "final_mark": <number 0-5>,',
        '  "mistakes_made": ["mistake1", "mistake2"],',
        '  "agrees_with_sympy": true/false',
        "}",
        "",
        "Important: Return ONLY valid JSON, no other text."
    ])
    
    return "\n".join(prompt_lines)

def extract_json_from_llm_response(response: str) -> Dict[str, Any]:
    """
    Extract JSON from LLM response with robust parsing.
    """
    # Clean the response - remove markdown code blocks and extra backslashes
    cleaned = response.strip()
    
    # Remove markdown code blocks
    cleaned = re.sub(r'```json\s*', '', cleaned)
    cleaned = re.sub(r'```\s*', '', cleaned)
    
    # Fix common escaping issues
    cleaned = cleaned.replace('\\"', '"').replace('\\n', ' ').replace('\\t', ' ')
    
    # Remove extra backslashes that aren't part of valid JSON escapes
    cleaned = re.sub(r'\\([^"\\/bfnrtu])', r'\1', cleaned)
    
    # Try to find JSON object
    start_idx = cleaned.find('{')
    end_idx = cleaned.rfind('}')
    
    if start_idx == -1 or end_idx == -1:
        return {"error": "No JSON object found", "raw_response_preview": response[:200]}
    
    json_str = cleaned[start_idx:end_idx+1]
    
    # Validate and fix common JSON issues
    json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
    json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
    
    try:
        parsed = json.loads(json_str)
        
        # Validate required fields
        required_fields = ["final_mark", "mistakes_made", "agrees_with_sympy"]
        if not all(field in parsed for field in required_fields):
            return {
                "error": f"Missing required fields. Found: {list(parsed.keys())}",
                "partial_data": parsed,
                "raw_response_preview": response[:200]
            }
        
        # Ensure final_mark is numeric
        if isinstance(parsed["final_mark"], str):
            try:
                parsed["final_mark"] = float(parsed["final_mark"])
            except ValueError:
                return {
                    "error": f"final_mark must be numeric, got: {parsed['final_mark']}",
                    "partial_data": parsed,
                    "raw_response_preview": response[:200]
                }
        
        return parsed
        
    except json.JSONDecodeError as e:
        # Try one more cleanup pass
        json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)  # Remove control characters
        try:
            parsed = json.loads(json_str)
            return parsed
        except json.JSONDecodeError:
            return {
                "error": f"JSON parse error: {str(e)}",
                "json_attempt": json_str,
                "raw_response_preview": response[:200]
            }

# ---------- LLM Grading Pipeline ----------
def grade_with_llm(input_path: str = INPUT_FILE, output_path: str = OUTPUT_FILE, use_ollama: bool = USE_OLLAMA):
    """
    Main grading function that uses LLM to mark student answers.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load previous SymPy results
    with input_file.open("r", encoding="utf-8") as f:
        sympy_results = json.load(f)

    # Prepare results structure
    llm_results = {
        "metadata": {
            "grading_model": OLLAMA_MODEL if use_ollama else "None",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_questions": len(sympy_results),
            "input_file": input_path
        },
        "questions": {}
    }

    start_time = time.time()
    total_students = 0
    processed_students = 0
    successful_parses = 0

    # Process each question
    for question_key, question_data in sympy_results.items():
        print(f"\n{'='*60}")
        print(f"PROCESSING: {question_key}")
        print(f"Field: {question_data['metadata']['field']} - {question_data['metadata']['topic']}")
        print(f"Question: {question_data['metadata']['question_text']}")
        print(f"Expected Answer: {question_data['metadata']['expected_answer']}")
        print(f"{'='*60}")
        
        llm_results["questions"][question_key] = {
            "metadata": question_data["metadata"],
            "student_verdicts": []
        }

        # Process each student's answer
        for student_data in question_data["student_verdicts"]:
            total_students += 1
            student_id = student_data["student_id"]
            sympy_grading = student_data["sympy_grading"]
            
            print(f"\n--- Student {student_id} ---")
            print("Student's work:")
            for i, step in enumerate(student_data["submitted_derivation"], 1):
                print(f"  Step {i}: {step}")
            
            # Get SymPy final verdict for display
            sympy_final = "Unknown"
            for step in sympy_grading:
                if step.get("step") == "Final Answer":
                    sympy_final = f"{step.get('status', 'Unknown')} - {step.get('comment', '')}"
                    break
            print(f"SymPy verdict: {sympy_final}")
            
            # Prepare student result entry
            student_result = {
                "student_id": student_id,
                "submitted_derivation": student_data["submitted_derivation"],
                "sympy_grading": sympy_grading,
                "llm_grading": {}
            }

            if use_ollama:
                try:
                    # Build prompt and get LLM response
                    prompt = build_ollama_prompt(question_data, student_data, sympy_grading)
                    print("Calling LLM...")
                    llm_response = run_ollama(prompt)
                    
                    # Parse LLM response
                    llm_grade = extract_json_from_llm_response(llm_response)
                    
                    # Only store the parsed grade, not raw prompt/response
                    student_result["llm_grading"] = llm_grade
                    
                    # Display LLM results in terminal
                    print("LLM Results:")
                    if "error" in llm_grade:
                        print(f"  ❌ ERROR: {llm_grade['error']}")
                        if "raw_response_preview" in llm_grade:
                            print(f"  Response preview: {llm_grade['raw_response_preview']}")
                    else:
                        successful_parses += 1
                        print(f"  ✅ Final Mark: {llm_grade['final_mark']}/5")
                        print(f"  ✅ Agrees with SymPy: {llm_grade['agrees_with_sympy']}")
                        if llm_grade['mistakes_made']:
                            print(f"  ❌ Mistakes: {', '.join(llm_grade['mistakes_made'])}")
                        else:
                            print(f"  ✅ No mistakes found")
                    
                    processed_students += 1
                    
                    # Add small delay to avoid overwhelming the system
                    time.sleep(1)
                    
                except Exception as e:
                    error_msg = f"Exception during LLM grading: {str(e)}"
                    print(f"  ❌ ERROR: {error_msg}")
                    student_result["llm_grading"] = {
                        "error": error_msg
                    }
            else:
                student_result["llm_grading"] = {
                    "message": "LLM grading disabled"
                }
                print("LLM grading disabled for this run")

            llm_results["questions"][question_key]["student_verdicts"].append(student_result)

    # Calculate statistics
    elapsed_time = time.time() - start_time
    llm_results["metadata"]["processing_time_seconds"] = round(elapsed_time, 2)
    llm_results["metadata"]["students_processed"] = processed_students
    llm_results["metadata"]["total_students"] = total_students
    llm_results["metadata"]["successful_json_parses"] = successful_parses
    llm_results["metadata"]["parse_success_rate"] = f"{(successful_parses/processed_students)*100:.1f}%" if processed_students > 0 else "N/A"

    # Save results
    output_file = Path(output_path)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(llm_results, f, indent=2, ensure_ascii=False)

    # Display final summary in terminal
    print(f"\n{'='*60}")
    print(f"🎯 LLM AUTOMARKING COMPLETED")
    print(f"{'='*60}")
    print(f"📊 SUMMARY STATISTICS:")
    print(f"   Total time: {elapsed_time:.2f} seconds")
    print(f"   Students processed: {processed_students}/{total_students}")
    print(f"   Successful JSON parses: {successful_parses}/{processed_students} ({llm_results['metadata']['parse_success_rate']})")
    
    if use_ollama and processed_students > 0:
        print(f"   Average time per student: {elapsed_time/processed_students:.2f}s")
    
    print(f"💾 Results saved to: {output_path}")
    print(f"{'='*60}")

def analyze_agreement_statistics(results_path: str = OUTPUT_FILE):
    """
    Analyze agreement statistics between SymPy and LLM grading.
    """
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    total_comparisons = 0
    agreements = 0
    mark_distribution = {}
    parse_errors = 0
    
    print(f"\n{'='*60}")
    print(f"📈 AGREEMENT ANALYSIS")
    print(f"{'='*60}")
    
    for question_key, question_data in results["questions"].items():
        print(f"\n📚 {question_key}: {question_data['metadata']['field']} - {question_data['metadata']['topic']}")
        
        for student in question_data["student_verdicts"]:
            llm_grade = student.get("llm_grading", {})
            
            if "error" in llm_grade:
                parse_errors += 1
                print(f"   Student {student['student_id']}: ❌ PARSE ERROR")
                continue
                
            if "agrees_with_sympy" in llm_grade:
                total_comparisons += 1
                
                # Get SymPy final answer status
                sympy_final_correct = any(
                    step.get("step") == "Final Answer" and step.get("status") == "Correct" 
                    for step in student["sympy_grading"]
                )
                
                # Get LLM agreement
                agrees = llm_grade["agrees_with_sympy"]
                
                if agrees:
                    agreements += 1
                
                # Track marks
                final_mark = llm_grade.get("final_mark", 0)
                mark_range = f"{int(final_mark)}"
                mark_distribution[mark_range] = mark_distribution.get(mark_range, 0) + 1
                
                status_symbol = "✅" if agrees else "❌"
                print(f"   Student {student['student_id']}: {status_symbol} Agreement (Mark: {final_mark}/5)")
    
    if total_comparisons > 0:
        agreement_rate = (agreements / total_comparisons) * 100
        print(f"\n{'='*60}")
        print(f"📊 SUMMARY STATISTICS")
        print(f"{'='*60}")
        print(f"   Total comparisons: {total_comparisons}")
        print(f"   ✅ Agreements: {agreements} ({agreement_rate:.1f}%)")
        print(f"   ❌ Disagreements: {total_comparisons - agreements} ({100-agreement_rate:.1f}%)")
        print(f"   ⚠️  Parse errors: {parse_errors}")
        
        # Display mark distribution in a nice format
        print(f"\n   📝 Mark Distribution:")
        for mark, count in sorted(mark_distribution.items()):
            percentage = (count / total_comparisons) * 100
            print(f"      {mark}/5: {count} students ({percentage:.1f}%)")
        
        print(f"{'='*60}")

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Improved LLM Automarker: Use Mistral to grade math answers")
    parser.add_argument("--input", "-i", default=INPUT_FILE, help="Input JSON file with SymPy results")
    parser.add_argument("--output", "-o", default=OUTPUT_FILE, help="Output JSON file for LLM results")
    parser.add_argument("--no-llm", dest="use_llm", action="store_false", help="Skip LLM grading (analysis only)")
    parser.add_argument("--model", default=OLLAMA_MODEL, help="Ollama model name")
    parser.add_argument("--analyze", action="store_true", help="Analyze agreement statistics from existing results")
    
    args = parser.parse_args()

    if args.analyze:
        analyze_agreement_statistics(args.output)
    else:
        OLLAMA_MODEL = args.model
        USE_OLLAMA = args.use_llm
        grade_with_llm(input_path=args.input, output_path=args.output, use_ollama=USE_OLLAMA)