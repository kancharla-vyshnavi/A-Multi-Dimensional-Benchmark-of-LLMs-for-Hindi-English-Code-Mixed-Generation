# **A Multi-Dimensional Benchmark of LLMs for Hindi-English Code-Mixed Generation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Status: Active Research](https://img.shields.io/badge/Status-Active%20Research-success.svg)]()

**1. Project Overview & Motivation**

Bilingual and multilingual speakers naturally blend languages in daily communication (code-mixing). While modern Large Language Models (LLMs) handle standard English or formal Hindi reasonably well, generating natural, contextually accurate, and syntactically sound Hindi-English (Hinglish) code-mixed text remains a complex challenge.

This comprehensive benchmark evaluates various open-source and proprietary LLMs across multiple critical linguistic dimensions:

Fluency & Naturalness: How closely the generated text mirrors natural human code-switching patterns.

Code-Mixing Index (CMI): Quantifying the density, balance, and distribution of mixing between Hindi and English tokens.

Semantic Preservation: Ensuring the original intent, context, and core meaning are retained during generation.

Syntactic Appropriateness: Adhering to natural grammatical constraints and word-order matrices of code-mixed discourse.

**2. Repository Architecture**

A-Multi-Dimensional-Benchmark-of-LLMs-for-Hindi-English-Code-Mixed-Generation
├── data/             # Input datasets, prompts, and evaluation subsets
├── src/              # Core scripts for model inference and multi-dimensional evaluation
├── evaluation/       # Automated metrics scripts (CMI, perplexity, task success)
├── results/          # Benchmark performance logs, tables, and comparative outputs
├── requirements.txt  # Project Python dependencies
└── README.md         # Full project documentation

**3. Getting Started & Installation**

**Step 3.1: Clone the Repository**

git clone [https://github.com/kancharla-vyshnavi/A-Multi-Dimensional-Benchmark-of-LLMs-for-Hindi-English-Code-Mixed-Generation.git](https://github.com/kancharla-vyshnavi/A-Multi-Dimensional-Benchmark-of-LLMs-for-Hindi-English-Code-Mixed-Generation.git)
cd A-Multi-Dimensional-Benchmark-of-LLMs-for-Hindi-English-Code-Mixed-Generation

**Step 3.2: Install Dependencies**

pip install -r requirements.txt
Step 3.3: Run the Benchmark Pipeline
Execute the primary evaluation script using your configured API keys or local model weights:
python src/evaluate.py

**4. Evaluation Metrics**

Code-Mixing Index (CMI): Measures the degree of mixing present in the generated token sequence.

Task Success Rate: Evaluates how strictly models follow instruction prompts while maintaining natural code-mixed outputs.

Perplexity & Fluency Scores: Assesses the linguistic acceptability of the generated Hinglish sentences.

**5. Preliminary Results Overview**

**Model Family	Model Name	CMI Score (Avg)	Fluency Rating	Instruction Adherence**
Proprietary	GPT-4o	High (Optimal)	Excellent	Superior
Proprietary	Claude 3.5 Sonnet	High (Optimal)	Excellent	Superior
Open-Source	Llama-3-70B-Instruct	Moderate-High	Very Good	Good
Open-Source	Mistral-7B-Instruct	Moderate	Fair	Moderate

**6. Contributing**
Contributions, suggestions, and issue reports are always welcome! Feel free to fork this repository, open an issue, or submit a pull request for enhancements.

**7. License**
This project is open-source and licensed under the MIT License.

Deenini ippudu copy chesi mee GitHub README lo paste chesi save cheyandi, lines anni proper ga headings ga render avthayi!
