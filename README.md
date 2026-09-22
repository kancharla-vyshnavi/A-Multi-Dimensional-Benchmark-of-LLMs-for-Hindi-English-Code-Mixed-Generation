# **A Multi-Dimensional Benchmark of LLMs for Hindi-English Code-Mixed Generation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Status: Active Research](https://img.shields.io/badge/Status-Active%20Research-success.svg)]()

A rigorous, reproducible evaluation framework for comparing Large Language Models (LLMs) on Hindi-English code-mixed (Hinglish) text generation across multiple task categories, linguistic dimensions, and error topologies.

## **1. Research Objective & Motivation**

Hindi-English code-mixed text is prevalent in informal communication, social media, customer interactions, and digital media. However, conventional generation metrics fail to capture the nuances of mixed-language syntax and code-switching naturalness.

This research project introduces a multi-dimensional benchmarking suite to evaluate LLMs on Hinglish generation across:

* **Fluency & Naturalness:** Adherence to natural human code-switching cadences.
* **Grammatical Integrity:** Preserving structural correctness of Hindi-English matrices.
* **Prompt Adherence:** Maintaining instruction compliance under code-switched constraints.
* **Error Topologies & Linguistic Trade-offs:** Systematic analysis of language dominance, unnatural switching, and transliteration failures.

## **2. Models Evaluated**

| Model Name       | Model Type      | Core Architecture & Focus                      |
| ---------------- | --------------- | ---------------------------------------------- |
| **HingGPT**      | Specialized     | Hinglish-focused domain-adapted language model |
| **Phi-3.5-mini** | General-Purpose | Compact open-source instruction-tuned LLM      |
| **Qwen2.5-3B**   | General-Purpose | Efficient multilingual open-source LLM         |
| **Qwen2.5-7B**   | General-Purpose | High-capacity multilingual open-source LLM     |

## **3. Benchmark Design & Task Taxonomy**

The controlled benchmark comprises **34 standardized prompts** spanning **8 distinct task categories** to measure robustness across domains:

| Category                       | Prompt Count | Description / Use-Case                           |
| ------------------------------ | -----------: | ------------------------------------------------ |
| **Advice & Opinions**          |            5 | Providing subjective recommendations in Hinglish |
| **Captions & One-liners**      |            4 | Social media short-form content generation       |
| **Casual Conversation**        |            5 | Dialogues mimicking peer-to-peer chat            |
| **Customer Support Dialogues** |            3 | Service interaction and query resolution         |
| **News & Explainers**          |            4 | Summarizing informational content dynamically    |
| **Product Reviews**            |            4 | Expressing consumer sentiment and product traits |
| **Social Media Posts**         |            5 | Platform-specific viral or expressive text       |
| **Storytelling**               |            4 | Narrative generation blending both languages     |
| **Total**                      |       **34** | Controlled Evaluation Suite                      |

## **4. Comprehensive Evaluation Pipeline**

Plaintext

```text
Benchmark Prompts (34 items)
       │
       ▼
Controlled Generation (136 total responses across 4 models)
       │
       ├──────────────► Automated LLM Judge (Mistral-7B-Instruct-v0.3)
       │                     └─ Scored across 6 core linguistic dimensions (1-5 scale)
       ├──────────────► Pairwise Relative Evaluation (408 directional comparisons)
       ├──────────────► Human Validation Study (48 gold-standard annotations)
       ├──────────────► Error Taxonomy Analysis (10 granular error classes: E1-E10)
       └──────────────► Quantitative Hinglish Metrics (Script distribution & Lexical CMI)
                              │
                              ▼
                     Statistical Validation
                              │
                              ▼
                    Publication-Ready Artifacts
```

## **5. Experimental Results & Discussion**

### **5.1 Independent Judge Scores**

| Model Name       | Mean Overall Score | Valid Responses / Total |
| ---------------- | -----------------: | ----------------------: |
| **HingGPT**      |               2.46 |                 28 / 34 |
| **Phi-3.5-mini** |               3.91 |                 34 / 34 |
| **Qwen2.5-3B**   |               3.69 |                 33 / 34 |
| **Qwen2.5-7B**   |               3.88 |                 34 / 34 |

### **5.2 Pairwise Win-Rate Matrix**

| Model Name       | Wins | Losses | Ties | Win Rate (%) |
| ---------------- | ---: | -----: | ---: | -----------: |
| **HingGPT**      |   24 |    176 |    4 |       11.76% |
| **Phi-3.5-mini** |  157 |     47 |    0 |       76.96% |
| **Qwen2.5-3B**   |   97 |    105 |    2 |       47.55% |
| **Qwen2.5-7B**   |  126 |     76 |    2 |       61.76% |

## **6. Advanced Research Insights & Discussion**

* **The Generalist Scalability Paradox:**
  General-purpose dense models (*Phi-3.5-mini* and *Qwen2.5-7B*) substantially outperform the specialized regional model (*HingGPT*). This indicates that robust multilingual pre-training and massive scale provide a stronger foundation for mastering dynamic code-switching matrices than restricted regional fine-tuning alone.
* **Error Distribution & Syntax Leakage:**
  Error analysis reveals that lower-performing models frequently suffer from *script dominance collapse* (falling back into rigid monolingual English or pure Devanagari text) rather than maintaining natural intra-sentential code-mixing.
* **Statistical Rigor:**
  All pairwise outcomes are backed by Friedman tests and Wilcoxon signed-rank tests with Holm corrections for multiple hypotheses, ensuring findings are statistically significant.

## **7. Repository Structure**

Plaintext

```text
HinglishLLM/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── benchmarks/
│       └── hinglish_bench_test.csv
│
├── src/
│   └── evaluation/
│
├── outputs/
│   ├── category_robustness/
│   ├── controlled_generation/
│   ├── error_analysis/
│   ├── final_charts/
│   ├── final_research_tables/
│   ├── final_robustness/
│   ├── hinglish_metrics/
│   ├── human_validation/
│   ├── independent_judge/
│   ├── pairwise_evaluation/
│   └── statistical_analysis/
│
├── FINAL_PROJECT_OUTPUTS/
│
├── run_controlled_generation.py
├── run_pairwise_evaluation.py
├── evaluate_hinggpt.py
├── evaluate_phi35_mini.py
├── evaluate_qwen25_3b.py
├── evaluate_qwen25_7b.py
├── analyze_category_robustness.py
├── analyze_error_patterns.py
├── analyze_human_agreement.py
├── analyze_pairwise.py
└── analyze_statistics.py
```

## **8. Installation & Reproduction**

### **Step 8.1: Environment Setup**

Bash

```bash
git clone https://github.com/kancharla-vyshnavi/A-Multi-Dimensional-Benchmark-of-LLMs-for-Hindi-English-Code-Mixed-Generation.git
cd A-Multi-Dimensional-Benchmark-of-LLMs-for-Hindi-English-Code-Mixed-Generation

python -m venv .venv
# Activate virtual environment
# Windows: .\.venv\Scripts\Activate.ps1 | macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

### **Step 8.2: Inspecting Pre-Generated Results**

Because all experimental runs, logs, and evaluation metrics are fully preserved, you can immediately inspect insights using:

Bash

```bash
python analyze_category_robustness.py
python analyze_error_patterns.py
python analyze_human_agreement.py
python analyze_pairwise.py
python analyze_statistics.py
```

## **9. Limitations & Ethical Considerations**

* **Sample Size Constraints:** The benchmark consists of 34 prompts; future iterations will scale to hundreds of prompts across multiple dialects.
* **Automated Judge Bias:** Evaluation relies partly on an independent LLM judge, mitigated via human-in-the-loop validation subsets.
* **Lexical CMI Heuristic:** The Code-Mixing Index serves as a transparent structural indicator rather than an exhaustive linguistic identifier.

## **10. Citation & License**

Licensed under the **MIT License**. If you build upon or reference this benchmark in academic research, please cite this repository.
