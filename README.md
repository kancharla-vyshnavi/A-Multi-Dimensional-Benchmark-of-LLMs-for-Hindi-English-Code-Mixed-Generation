# A Multi-Dimensional Benchmark of LLMs for Hindi-English Code-Mixed Generation

A reproducible benchmark for evaluating Large Language Models (LLMs) on **Hindi-English code-mixed (Hinglish) text generation** using controlled prompting, independent LLM-as-a-Judge evaluation, pairwise comparison, category-level robustness analysis, error analysis, and non-parametric statistical testing.

---

## Overview

Hindi-English code-mixed text is widely used in informal digital communication. However, evaluating LLMs on code-mixed generation requires more than a single automatic metric.

This project presents a **multi-dimensional benchmark for Hindi-English code-mixed generation**, comparing four language models across:

* Independent LLM-as-a-Judge evaluation
* Pairwise model comparison
* Category-level robustness
* Prompt adherence and linguistic quality
* Systematic error analysis
* Statistical significance testing
* Reproducible research outputs

The evaluation uses the **same 34 benchmark prompts** across all four models and applies a controlled evaluation protocol.

---

# Models Evaluated

| Model        | Type                                             |
| ------------ | ------------------------------------------------ |
| HingGPT      | Hindi-English code-mixed language model          |
| Phi-3.5-mini | General-purpose instruction-tuned language model |
| Qwen2.5-3B   | General-purpose language model                   |
| Qwen2.5-7B   | General-purpose language model                   |

### Methodological Note

The evaluated models differ in architecture, parameter scale, and pretraining/instruction-tuning characteristics. Therefore, the results are interpreted as a **comparative benchmark evaluation**, not as a controlled causal experiment isolating model size or architecture.

---

# Benchmark

The benchmark contains **34 prompts across 8 generation categories**.

| Category                   | Prompts |
| -------------------------- | ------: |
| Advice & Opinions          |       5 |
| Captions & One-liners      |       4 |
| Casual Conversation        |       5 |
| Customer Support Dialogues |       3 |
| News & Explainers          |       4 |
| Product Reviews            |       4 |
| Social Media Posts         |       5 |
| Storytelling               |       4 |
| **Total**                  |  **34** |

The same benchmark prompts are evaluated across all four models.

---

# Controlled Generation

Generation was performed using a controlled evaluation protocol to maintain consistency across models.

The generation configuration includes:

* Maximum new tokens: 50
* Greedy decoding for the controlled benchmark generation
* Consistent benchmark prompts
* Identical evaluation samples across models

The complete generation outputs are stored in:

```text
outputs/
└── controlled_generation/
    ├── benchmark_generations.csv
    ├── benchmark_generations.jsonl
    └── generation_config.json
```

---

# Evaluation Methodology

## 1. Independent LLM-as-a-Judge Evaluation

Generated responses were evaluated using:

**Mistral-7B-Instruct-v0.3**

The judge is independent of the four evaluated models.

Each response was evaluated on six 1–5 quality dimensions:

1. Fluency
2. Code-Mixing Naturalness
3. Hindi Grammar
4. Prompt Adherence
5. Spelling Consistency
6. Overall Quality

Two additional error indicators were recorded:

* **E6:** Prompt misunderstanding
* **E10:** Irrelevant response

### Evaluation Coverage

There are:

```text
4 models × 34 prompts = 136 model responses
```

The independent judge produced valid scores for most responses. A small number of judge outputs remained missing after controlled recovery attempts and are explicitly accounted for in the analysis.

Judge outputs are stored in:

```text
outputs/
└── independent_judge/
    ├── independent_mistral_judge_results.csv
    └── independent_mistral_judge_results.jsonl
```

---

# 2. Pairwise Evaluation

Pairwise evaluation compares two model responses for the **same benchmark prompt**.

With four models:

```text
4 models
   ↓
6 unique model pairs
   ↓
34 benchmark prompts
   ↓
2 presentation directions
   ↓
408 pairwise comparisons
```

The evaluation uses an independent Mistral-7B-Instruct-v0.3 judge.

Both presentation orders are evaluated to reduce potential position bias:

```text
Model A vs Model B
Model B vs Model A
```

The pairwise evaluation records:

* Model wins
* Model losses
* Ties
* Confidence
* Swap consistency

Outputs are stored in:

```text
outputs/
└── pairwise_evaluation/
    ├── pairwise_results.csv
    ├── pairwise_results.jsonl
    ├── pairwise_config.json
    ├── pairwise_model_comparison.csv
    ├── pairwise_model_aggregate.csv
    └── pairwise_swap_consistency.csv
```

---

# 3. Category-Level Robustness Analysis

Performance is also examined across the eight benchmark categories.

The category analysis connects each generated response to its benchmark category and summarizes independent-judge scores within each category.

The analysis reports:

* Category-level overall scores
* Model-level category means
* Variation across categories
* Category-wise pairwise comparisons
* Valid evaluation counts

Outputs are stored in:

```text
outputs/
└── category_robustness/
    ├── category_annotated_judge_rows.csv
    ├── category_error_flags.csv
    ├── category_independent_judge_summary.csv
    ├── category_overall_results.csv
    ├── category_pairwise_comparison.csv
    └── category_prompt_counts.csv
```

Because individual categories contain only 3–5 prompts, category-level results should be interpreted as **descriptive robustness evidence** rather than as large-sample estimates.

---

# 4. Statistical Analysis

The statistical analysis treats benchmark prompts as repeated evaluation units across models.

The primary non-parametric analysis includes:

* Friedman test
* Pairwise Wilcoxon signed-rank tests
* Holm correction for multiple comparisons
* Rank-biserial effect size
* Kendall's W

### Overall Friedman Test

For the independent-judge overall-quality scores:

```text
Complete prompts: 27
Friedman χ²(3) = 55.6141
p < 0.000001
Kendall's W = 0.6866
```

The Friedman test indicates statistically detectable differences among the four models over the complete paired subset.

### Holm-Corrected Pairwise Results

| Comparison                 |  n | Holm-adjusted p | Significant |
| -------------------------- | -: | --------------: | ----------- |
| HingGPT vs Phi-3.5-mini    | 28 |         <0.0001 | Yes         |
| HingGPT vs Qwen2.5-3B      | 27 |          0.0001 | Yes         |
| HingGPT vs Qwen2.5-7B      | 28 |         <0.0001 | Yes         |
| Phi-3.5-mini vs Qwen2.5-3B | 33 |          0.2997 | No          |
| Phi-3.5-mini vs Qwen2.5-7B | 34 |          0.7389 | No          |
| Qwen2.5-3B vs Qwen2.5-7B   | 33 |          0.3211 | No          |

Statistical results are stored in:

```text
outputs/
└── statistical_analysis/
    ├── friedman_results.csv
    └── wilcoxon_holm_results.csv
```

---

# 5. Error Analysis

Generation errors are analyzed using ten error categories.

| Code | Error Category                 |
| ---- | ------------------------------ |
| E1   | English-dominant output        |
| E2   | Hindi-dominant output          |
| E3   | Unnatural code-switching       |
| E4   | Grammatical error              |
| E5   | Repetition                     |
| E6   | Prompt misunderstanding        |
| E7   | Incomplete response            |
| E8   | Spelling/transliteration issue |
| E9   | Hallucination/factual error    |
| E10  | Irrelevant response            |

Error categories are **not mutually exclusive**.

### Model-Level Error Patterns

Among valid independent-judge records:

| Model        | Valid | Missing |
| ------------ | ----: | ------: |
| HingGPT      |    28 |       6 |
| Phi-3.5-mini |    34 |       0 |
| Qwen2.5-3B   |    33 |       1 |
| Qwen2.5-7B   |    34 |       0 |

Selected observed error indicators:

| Model        | E6 Prompt Misunderstanding | E10 Irrelevant Response |
| ------------ | -------------------------: | ----------------------: |
| HingGPT      |              9/27 (33.33%) |          19/27 (70.37%) |
| Phi-3.5-mini |                  0/34 (0%) |               0/34 (0%) |
| Qwen2.5-3B   |               3/33 (9.09%) |           6/33 (18.18%) |
| Qwen2.5-7B   |               1/33 (3.03%) |           4/33 (12.12%) |

Outputs are stored in:

```text
outputs/
└── error_analysis/
    ├── dimension_error_summary.csv
    ├── E10_irrelevant_response.csv
    ├── E6_prompt_misunderstanding.csv
    ├── low_score_responses.csv
    ├── missing_judge_summary.csv
    └── overall_score_distribution.csv
```

---

# Overall Independent-Judge Results

The independent Mistral judge produced the following overall-quality means:

| Model        | Valid N | Overall Mean |
| ------------ | ------: | -----------: |
| HingGPT      |      28 |        2.464 |
| Phi-3.5-mini |      34 |        3.912 |
| Qwen2.5-3B   |      33 |        3.697 |
| Qwen2.5-7B   |      34 |        3.882 |

These values are descriptive results from the independent-judge evaluation and should be interpreted together with the statistical, pairwise, category, and error analyses.

---

# Pairwise Aggregate Results

Across the 204 comparisons involving each model:

| Model        | Wins | Losses | Ties | Total | Win Rate |
| ------------ | ---: | -----: | ---: | ----: | -------: |
| HingGPT      |   24 |    176 |    4 |   204 |   0.1176 |
| Phi-3.5-mini |  157 |     47 |    0 |   204 |   0.7696 |
| Qwen2.5-3B   |   97 |    105 |    2 |   204 |   0.4755 |
| Qwen2.5-7B   |  126 |     76 |    2 |   204 |   0.6176 |

Pairwise results are based on the 408 direction-controlled comparisons and should be considered alongside swap-consistency measurements.

---

# Reproducibility

## Clone the Repository

```bash
git clone https://github.com/kancharla-vyshnavi/A-Multi-Dimensional-Benchmark-of-LLMs-for-Hindi-English-Code-Mixed-Generation.git

cd A-Multi-Dimensional-Benchmark-of-LLMs-for-Hindi-English-Code-Mixed-Generation
```

## Create a Virtual Environment

### Windows

```powershell
python -m venv .venv311
.venv311\Scripts\activate
```

### Linux/macOS

```bash
python -m venv .venv311
source .venv311/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Analysis Scripts

The repository contains scripts for the complete evaluation workflow.

### Controlled Generation

```bash
python run_controlled_generation.py
```

### Independent Judge

The independent judge evaluation uses the generated benchmark responses and Mistral-7B-Instruct-v0.3.

### Pairwise Evaluation

```bash
python run_pairwise_evaluation.py
```

### Pairwise Analysis

```bash
python analyze_pairwise.py
```

### Statistical Analysis

```bash
python analyze_statistics.py
```

### Error Analysis

```bash
python analyze_error_patterns.py
```

### Category Robustness

```bash
python analyze_category_robustness.py
```

---

# Repository Structure

```text
HinglishLLM/
│
├── configs/
│
├── data/
│   └── benchmarks/
│       └── hinglish_bench_test.csv
│
├── models/
│
├── src/
│   └── evaluation/
│
├── outputs/
│   ├── category_robustness/
│   ├── controlled_generation/
│   ├── error_analysis/
│   ├── final_research_tables/
│   ├── independent_judge/
│   ├── pairwise_evaluation/
│   └── statistical_analysis/
│
├── run_controlled_generation.py
├── run_pairwise_evaluation.py
├── analyze_pairwise.py
├── analyze_statistics.py
├── analyze_error_patterns.py
├── analyze_category_robustness.py
│
├── evaluate_hinggpt.py
├── evaluate_phi35_mini.py
├── evaluate_qwen25_3b.py
├── evaluate_qwen25_7b.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Final Research Outputs

The consolidated research tables are stored in:

```text
outputs/
└── final_research_tables/
    ├── table_overall.csv
    ├── table_category.csv
    ├── table_statistics.csv
    └── table_error_analysis.csv
```

These tables provide compact research-ready summaries of:

* Overall model evaluation
* Category-level performance
* Statistical testing
* Error patterns

---

# Limitations

1. The benchmark contains 34 prompts, so category-level estimates are based on relatively small numbers of examples.

2. A small number of independent-judge outputs remained missing after controlled recovery attempts. Statistical analyses therefore use the available paired observations where appropriate.

3. LLM-as-a-Judge scores represent automated evaluation evidence and should not be interpreted as equivalent to human annotation.

4. Pairwise judgments are subject to possible judge and presentation-order effects. Both model orders were evaluated to assess swap consistency.

5. Error categories are not mutually exclusive.

6. Automatic generation and linguistic metrics capture specific properties of generated text and should not be interpreted as complete measures of semantic or conversational quality.

7. Results are specific to the benchmark prompts, generation configuration, judge model, and evaluation protocol used in this study.

---

# Research Contribution

This project provides a **multi-dimensional and reproducible evaluation framework for Hindi-English code-mixed LLM generation**.

The framework combines:

* Controlled benchmark generation
* Independent LLM-as-a-Judge evaluation
* Six-dimensional quality assessment
* Pairwise model comparison
* Position-swap evaluation
* Category-level robustness analysis
* Systematic error analysis
* Friedman statistical testing
* Wilcoxon signed-rank testing
* Holm multiple-comparison correction
* Effect-size analysis
* Reproducible research outputs

The resulting evaluation framework is designed to provide a broader view of Hindi-English code-mixed generation quality than relying on a single metric or evaluation method.
