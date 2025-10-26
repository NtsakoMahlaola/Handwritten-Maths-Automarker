import os
import google.generativeai as genai
from pathlib import Path
import json

# --- CONFIGURE ---
genai.configure(api_key="AIzaSyDk-vLW9ZMKuniHFKIYbsFwCvb3YZoUHg4")
model = genai.GenerativeModel("gemini-2.5-flash")

# --- FOLDER PATH ---
input_folder = Path("images")       # Folder containing images
output_file = Path("results.json")  # Store all Sympy expressions here

# --- CREATE OUTPUT CONTAINER ---
results = {}

# --- PROCESS EACH IMAGE ---
for img_path in input_folder.glob("*.*"):
    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
        continue

    print(f"Processing: {img_path.name} ...")

    with open(img_path, "rb") as f:
        image_data = f.read()

    # PROMPT: ask Gemini for Sympy-compatible output
    prompt = (
        "Extract the mathematical equation or expression from this image "
        "and convert it to Python Sympy-compatible format. "
        "Use syntax like Eq(sin(x) + cos(x), 0), ** for powers, "
        "and avoid LaTeX or Markdown. Output only the expression."
    )

    try:
        response = model.generate_content([
            {"mime_type": "text/plain", "text": prompt},
            {"mime_type": "image/jpeg", "data": image_data}
        ])
        sympy_expr = response.text.strip()
        results[img_path.name] = sympy_expr
        print(f" → {sympy_expr}")

    except Exception as e:
        print(f"Error processing {img_path.name}: {e}")

# --- SAVE OUTPUT ---
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"\n✅ All results saved to {output_file}")
