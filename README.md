# 🧮 MathAutoMark: Automated Mathematics Assessment System

MathAutoMark is an intelligent system for automated grading of handwritten mathematical derivations.  
It integrates **computer vision**, **symbolic mathematics**, and **large language models (LLMs)** to provide accurate and interpretable grading feedback.

---

## 🚀 Features
- 📸 **Image Processing** – Preprocess and enhance handwritten derivations.
- 🔍 **OCR Conversion** – Convert handwriting to LaTeX via **Google Gemini API**.
- ✅ **SymPy Verification** – Verify mathematical correctness symbolically.
- 🤖 **LLM Analysis** – Generate feedback using **Ollama** and **mathstral**.
- ⚡ **Batch Processing** – Evaluate multiple students at once.
- 🎯 **Step-by-Step Analysis** – Identify and explain errors in derivations.

---

## 🧩 Installation

### Prerequisites
- Python 3.8+
- Ollama (for local LLMs)
- Google Gemini API key

### Install Dependencies
```bash
pip install customtkinter Pillow opencv-python sympy numpy google-generativeai
