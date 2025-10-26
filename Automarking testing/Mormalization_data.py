import json
import random
from datetime import datetime

def create_normalization_data():
    """
    Create diverse normalization data including math questions, 
    general knowledge, and normal conversations
    """
    normalization_data = []
    
    # Math questions and answers (20 examples)
    math_qa = [
        {
            "question": "What is 15 + 27?",
            "answer": "42"
        },
        {
            "question": "Solve for x: 2x + 5 = 17",
            "answer": "x = 6"
        },
        {
            "question": "What is the area of a circle with radius 7?",
            "answer": "153.94 or 49π"
        },
        {
            "question": "Simplify: (3x² + 2x - 1) + (x² - 3x + 4)",
            "answer": "4x² - x + 3"
        },
        {
            "question": "What is 25% of 80?",
            "answer": "20"
        },
        {
            "question": "Factor: x² - 9",
            "answer": "(x + 3)(x - 3)"
        },
        {
            "question": "Calculate: 7 × 8 ÷ 4",
            "answer": "14"
        },
        {
            "question": "What is the square root of 144?",
            "answer": "12"
        },
        {
            "question": "Solve: 3(x - 4) = 15",
            "answer": "x = 9"
        },
        {
            "question": "What is 3/4 of 100?",
            "answer": "75"
        },
        {
            "question": "Find the slope of line through points (2,3) and (5,11)",
            "answer": "8/3"
        },
        {
            "question": "What is 5! (factorial)?",
            "answer": "120"
        },
        {
            "question": "Solve: 2³ × 2²",
            "answer": "32"
        },
        {
            "question": "What is the perimeter of a square with side 6?",
            "answer": "24"
        },
        {
            "question": "Convert 3/5 to decimal",
            "answer": "0.6"
        },
        {
            "question": "What is 15 × 12?",
            "answer": "180"
        },
        {
            "question": "Solve: 4x - 7 = 3x + 5",
            "answer": "x = 12"
        },
        {
            "question": "What is 7² + 8²?",
            "answer": "113"
        },
        {
            "question": "Find the mean of 4, 8, 15, 16, 23, 42",
            "answer": "18"
        },
        {
            "question": "What is 3/8 as a percentage?",
            "answer": "37.5%"
        }
    ]
    
    # General knowledge questions (10 examples)
    general_knowledge = [
        {
            "question": "What is the capital of France?",
            "answer": "Paris"
        },
        {
            "question": "Who wrote 'Romeo and Juliet'?",
            "answer": "William Shakespeare"
        },
        {
            "question": "What is the largest planet in our solar system?",
            "answer": "Jupiter"
        },
        {
            "question": "What year did World War II end?",
            "answer": "1945"
        },
        {
            "question": "What is the chemical symbol for gold?",
            "answer": "Au"
        },
        {
            "question": "Who painted the Mona Lisa?",
            "answer": "Leonardo da Vinci"
        },
        {
            "question": "What is the hardest natural substance on Earth?",
            "answer": "Diamond"
        },
        {
            "question": "How many continents are there?",
            "answer": "7"
        },
        {
            "question": "What is the largest ocean on Earth?",
            "answer": "Pacific Ocean"
        },
        {
            "question": "Who developed the theory of relativity?",
            "answer": "Albert Einstein"
        }
    ]
    
    # Normal conversations (10 examples)
    conversations = [
        {
            "user": "Hi, how are you today?",
            "assistant": "I'm doing well, thank you! How can I help you?"
        },
        {
            "user": "What's the weather like?",
            "assistant": "I don't have access to real-time weather data, but I hope it's nice where you are!"
        },
        {
            "user": "Can you help me with my homework?",
            "assistant": "I'd be happy to help you with your homework! What subject are you working on?"
        },
        {
            "user": "Tell me a joke",
            "assistant": "Why was the equal sign so humble? Because he knew he wasn't less than or greater than anyone else!"
        },
        {
            "user": "What time is it?",
            "assistant": "I don't have access to real-time clock data, but you can check your device's clock for the current time."
        },
        {
            "user": "Thank you for your help",
            "assistant": "You're welcome! I'm glad I could assist you."
        },
        {
            "user": "What can you do?",
            "assistant": "I can help with math problems, answer general knowledge questions, have conversations, and assist with various educational topics!"
        },
        {
            "user": "How does photosynthesis work?",
            "assistant": "Photosynthesis is the process where plants convert light energy, usually from the sun, into chemical energy in the form of glucose. They use carbon dioxide and water to produce oxygen and sugars."
        },
        {
            "user": "Good morning!",
            "assistant": "Good morning! I hope you're having a wonderful day. How can I assist you today?"
        },
        {
            "user": "What's your favorite color?",
            "assistant": "As an AI, I don't have personal preferences, but many people find blue to be a pleasant and calming color!"
        }
    ]
    
    # Convert to fine-tuning format
    for math_item in math_qa:
        normalization_data.append({
            "messages": [
                {"role": "user", "content": f"MATHEMATICS QUESTION: {math_item['question']}"},
                {"role": "assistant", "content": math_item['answer']}
            ]
        })
    
    for gk_item in general_knowledge:
        normalization_data.append({
            "messages": [
                {"role": "user", "content": f"GENERAL KNOWLEDGE: {gk_item['question']}"},
                {"role": "assistant", "content": gk_item['answer']}
            ]
        })
    
    for conv_item in conversations:
        normalization_data.append({
            "messages": [
                {"role": "user", "content": conv_item['user']},
                {"role": "assistant", "content": conv_item['assistant']}
            ]
        })
    
    return normalization_data

def process_single_example(example):
    """
    Process a single example from the automarking dataset with improved formatting
    """
    conversations = []
    
    metadata = example.get("metadata", {})
    student_verdicts = example.get("student_verdicts", [])
    
    for verdict in student_verdicts:
        # Create user message with improved structure
        user_content = f"""BEGIN MATHS AUTOMARKING TASK. You are a math teacher grader. Analyze the student's derivation step by step, identify any mistakes, provide a final mark (0.0-5.0), list all mistakes made, give constructive feedback, and indicate whether you agree with SymPy's assessment.

QUESTION CONTEXT:
Field: {metadata.get('field', 'Unknown')}
Topic: {metadata.get('topic', 'Unknown')}
Difficulty: {metadata.get('difficulty', 'Unknown')}
Question: {metadata.get('question_text', 'Unknown')}
Expected Answer: {metadata.get('expected_answer', 'Unknown')}

STUDENT WORK:
{chr(10).join(verdict.get('submitted_derivation', []))}

SYMPY GRADING:"""
        
        # Add SymPy grading without comments
        sympy_grading = verdict.get('sympy_grading', [])
        for sg in sympy_grading:
            stp = sg.get("step", "")
            sw = sg.get("student_work", "")
            st = sg.get("status", "")
            user_content += f"\nStep: {stp} | Student Work: {sw} | Status: {st}"
        
        user_content += "\n\nYOUR TASK: Provide grading analysis in JSON format with fields: final_mark, mistakes_made, feedback, agrees_with_sympy, reasoning"
        
        # Create assistant message with compact JSON formatting
        llm_grading = verdict.get('llm_grading', {})
        assistant_content = json.dumps({
            "final_mark": llm_grading.get('final_mark', 0),
            "mistakes_made": llm_grading.get('mistakes_made', []),
            "feedback": llm_grading.get('feedback', ''),
            "agrees_with_sympy": llm_grading.get('agrees_with_sympy', False),
            "reasoning": llm_grading.get('reasoning', '')
        }, ensure_ascii=False, separators=(',', ':'))
        
        conversation = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        }
        
        conversations.append(conversation)
    
    return conversations

def convert_to_finetuning_format(original_data, include_normalization=True):
    """
    Convert the original automarking dataset to fine-tuning format
    with optional normalization data
    """
    finetuning_data = []
    
    # Process automarking data
    if isinstance(original_data, list):
        for example in original_data:
            finetuning_data.extend(process_single_example(example))
    else:
        finetuning_data.extend(process_single_example(original_data))
    
    # Add normalization data if requested
    if include_normalization:
        normalization_data = create_normalization_data()
        finetuning_data.extend(normalization_data)
    
    # Shuffle the data to mix automarking and normalization examples
    random.shuffle(finetuning_data)
    
    return finetuning_data

def load_and_process_file(input_file_path, output_file_path, include_normalization=True):
    """
    Load the original JSON file and convert it to fine-tuning format
    """
    try:
        # Read the original file
        with open(input_file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Handle the case where multiple JSON objects are concatenated
        json_objects = []
        start_idx = 0
        
        while start_idx < len(content):
            if content[start_idx] == '{':
                # Try to parse from this position
                try:
                    # Find matching closing brace
                    brace_count = 0
                    end_idx = start_idx
                    for i in range(start_idx, len(content)):
                        if content[i] == '{':
                            brace_count += 1
                        elif content[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i
                                break
                    
                    if brace_count == 0:
                        json_str = content[start_idx:end_idx+1]
                        json_obj = json.loads(json_str)
                        json_objects.append(json_obj)
                        start_idx = end_idx + 1
                    else:
                        start_idx += 1
                except json.JSONDecodeError:
                    start_idx += 1
            else:
                start_idx += 1
        
        print(f"Found {len(json_objects)} JSON objects in the file")
        
        # Process all objects
        all_conversations = []
        for json_obj in json_objects:
            conversations = process_single_example(json_obj)
            all_conversations.extend(conversations)
        
        # Add normalization data
        if include_normalization:
            normalization_data = create_normalization_data()
            all_conversations.extend(normalization_data)
            print(f"Added {len(normalization_data)} normalization examples")
        
        # Shuffle the data
        random.shuffle(all_conversations)
        
        # Save to output file with compact JSON formatting
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for conversation in all_conversations:
                f.write(json.dumps(conversation, ensure_ascii=False, separators=(',', ':')) + '\n')
        
        print(f"Successfully converted {len(all_conversations)} examples to fine-tuning format")
        print(f"Output saved to: {output_file_path}")
        
        return all_conversations
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return []

def analyze_dataset(output_file_path):
    """
    Analyze the final dataset composition
    """
    try:
        with open(output_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        automarking_count = 0
        math_qa_count = 0
        general_knowledge_count = 0
        conversation_count = 0
        
        for line in lines:
            data = json.loads(line)
            user_content = data['messages'][0]['content']
            
            if "BEGIN MATHS AUTOMARKING TASK" in user_content:
                automarking_count += 1
            elif "MATHEMATICS QUESTION:" in user_content:
                math_qa_count += 1
            elif "GENERAL KNOWLEDGE:" in user_content:
                general_knowledge_count += 1
            else:
                conversation_count += 1
        
        print("\n=== DATASET COMPOSITION ===")
        print(f"Automarking examples: {automarking_count}")
        print(f"Math Q&A examples: {math_qa_count}")
        print(f"General knowledge examples: {general_knowledge_count}")
        print(f"Conversation examples: {conversation_count}")
        print(f"Total examples: {len(lines)}")
        print("============================\n")
        
    except Exception as e:
        print(f"Error analyzing dataset: {e}")

# Usage
if __name__ == "__main__":
    input_file = r"C:\Users\Naseeka\Desktop\Thesis work\MathStral\automarking_results_diversified.json"
    output_file = r"C:\Users\Naseeka\Desktop\Thesis work\MathStral\fine_tuning_data3.jsonl"
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Convert with normalization data
    converted_data = load_and_process_file(input_file, output_file, include_normalization=True)
    
    # Analyze the final dataset
    analyze_dataset(output_file)
    
    # Create a sample preview
    print("\n=== SAMPLE EXAMPLES ===")
    for i, example in enumerate(converted_data[:3]):
        print(f"\nExample {i+1}:")
        print(json.dumps(example, indent=2, ensure_ascii=False))