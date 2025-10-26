import json

def convert_to_chat_messages(record):
    """
    Convert one record with metadata + student_verdicts into chat-style entries.
    Prepends a short instruction after BEGIN MATHS AUTOMARKING TASK.
    """
    meta = record.get("metadata", {})
    field = meta.get("field", "")
    topic = meta.get("topic", "")
    difficulty = meta.get("difficulty", "")
    question = meta.get("question_text", "")
    expected = meta.get("expected_answer", "")

    entries = []
    for sv in record.get("student_verdicts", []):
        sid = sv.get("student_id", "")
        deriv = sv.get("submitted_derivation", [])
        sym = sv.get("sympy_grading", [])
        llm = sv.get("llm_grading", None)
        if llm is None:
            continue

        # Build the “user” content string
        lines = []
        instruction = ("BEGIN MATHS AUTOMARKING TASK, you are a math teacher grader — "
                       "analyze student solution, find mistakes, score it, and indicate if you agree with SymPy.")
        lines.append(instruction)
        lines.append(f"Field: {field}")
        lines.append(f"Topic: {topic}")
        lines.append(f"Difficulty: {difficulty}")
        lines.append(f"Question: {question}")
        lines.append(f"Expected Answer: {expected}")
        lines.append("")
        lines.append("Student Work:")
        for st in deriv:
            lines.append(st)
        lines.append("")
        lines.append("SymPy Grading:")
        for sg in sym:
            stp = sg.get("step", "")
            sw = sg.get("student_work", "")
            st = sg.get("status", "")
            cm = sg.get("comment", "")
            lines.append(f"{stp} => {sw} | status: {st} | comment: {cm}")

        user_content = "\n".join(lines)

        # Build the assistant’s grading JSON string
        assistant_json = json.dumps({
            "final_mark": float(llm.get("final_mark", 0)),
            "mistakes_made": llm.get("mistakes_made", []),
            "feedback": llm.get("feedback", ""),
            "agrees_with_sympy": bool(llm.get("agrees_with_sympy", False)),
            "reasoning": llm.get("reasoning", "")
        }, ensure_ascii=False)

        entry = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_json}
            ]
        }
        entries.append(entry)

    return entries

def convert_json_to_chat_jsonl(input_json_path: str , output_jsonl_path: str):
    """
    Reads a JSON file with your dataset, converts it, and writes chat-style JSONL.
    """
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Determine record list
    if "questions" in data and isinstance(data["questions"], dict):
        records = []
        for qkey, qval in data["questions"].items():
            rec = {
                "metadata": qval.get("metadata", {}),
                "student_verdicts": qval.get("student_verdicts", [])
            }
            records.append(rec)
    elif isinstance(data, list):
        records = data
    else:
        records = list(data.values())

    all_entries = []
    for rec in records:
        chats = convert_to_chat_messages(rec)
        all_entries.extend(chats)

    with open(output_jsonl_path, "w", encoding="utf-8") as fout:
        for e in all_entries:
            fout.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"Converted dataset: wrote {len(all_entries)} chat entries to {output_jsonl_path}")

# Example usage:
# convert_json_to_chat_jsonl("your_dataset.json", "dataset_for_finetune.jsonl")
if __name__ == "__main__":
    # call with defaults; change args if you want different files
    convert_json_to_chat_jsonl( r"C:\Users\Naseeka\Desktop\Thesis work\MathStral\automarking_results_diversified.json", r"C:\Users\Naseeka\Desktop\Thesis work\MathStral\dataset_for_finetune.jsonl")