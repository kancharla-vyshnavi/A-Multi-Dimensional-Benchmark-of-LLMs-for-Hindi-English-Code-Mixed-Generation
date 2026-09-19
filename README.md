# A-Multi-Dimensional-Benchmark-of-LLMs-for-Hindi-English-Code-Mixed-Generation
 
A reproducible benchmark for evaluating Large Language Models (LLMs) on Hindi-English code-mixed text generation using automatic generation metrics, linguistic measures, LLM-as-a-Judge evaluation, pairwise comparison, error analysis, and statistical significance testing.

## Overview

Hindi-English code-mixed text is widely used in informal digital communication, but evaluating LLMs on such language requires more than conventional text-generation metrics.

This project presents a multi-dimensional benchmark for Hindi-English code-mixed generation, evaluating four language models across automatic generation quality, linguistic characteristics, human-like quality dimensions, pairwise preferences, and generation errors.

The benchmark combines automatic metrics, linguistic metrics, LLM-based evaluation, pairwise judgments, manual error annotation, and non-parametric statistical analysis.

---

## Research Objectives

1. Benchmark LLMs for Hindi-English code-mixed text generation.
2. Evaluate generated text using multiple complementary dimensions.
3. Measure lexical diversity and repetition using automatic generation metrics.
4. Quantify code-mixing behavior using linguistic metrics.
5. Assess generation quality using an LLM-as-a-Judge framework.
6. Compare model outputs using pairwise judgments.
7. Identify common generation errors through systematic error analysis.
8. Statistically test differences between models.
9. Provide reproducible evaluation outputs for Hindi-English code-mixed NLP research.

---

## Models Evaluated

| Model        | Type                           |
| ------------ | ------------------------------ |
| HingGPT      | Hindi-English code-mixed model |
| Phi-3.5-mini | General-purpose language model |
| Qwen2.5-3B   | General-purpose language model |
| Qwen2.5-7B   | General-purpose language model |

---

## Benchmark Configuration

| Parameter                    |  Value |
| ---------------------------- | -----: |
| Validation samples           | 14,601 |
| Generation samples           |  1,000 |
| Maximum sequence length      |    128 |
| Maximum new tokens           |     50 |
| Batch size                   |      4 |
| Temperature                  |    0.7 |
| Top-p                        |    0.9 |
| Repetition penalty           |    1.1 |
| LLM-as-a-Judge samples/model |     34 |
| Models                       |      4 |
| Model-response evaluations   |    136 |
| Judge criteria               |      6 |
| Criterion-level scores       |    816 |
| Pairwise judgments           |    204 |

---

# Evaluation Methodology

## 1. Automatic Generation Metrics

The benchmark evaluates generated text using four automatic metrics.

### Perplexity

Perplexity measures the uncertainty of a language model over the evaluated text. Lower perplexity indicates lower predictive uncertainty under the evaluated model and setup.

### Distinct-1

Distinct-1 measures unigram diversity:

`Distinct-1 = unique unigrams / total unigrams`

Higher values indicate greater lexical diversity.

### Distinct-2

Distinct-2 measures bigram diversity:

`Distinct-2 = unique bigrams / total bigrams`

Higher values indicate greater phrase-level diversity.

### Repetition Rate

Repetition rate measures the proportion of repeated generated content. Lower values indicate less repetition.

---

## 2. LLM-as-a-Judge Evaluation

Generated responses were evaluated using Phi-3.5-mini-Instruct as the automated judge.

Each response was scored on a 1–5 scale across six dimensions:

1. Fluency
2. Code-Mixing Naturalness
3. Hindi Grammar
4. Prompt Adherence
5. Spelling Consistency
6. Overall Quality

The evaluation contains 34 benchmark samples per model, resulting in 136 model-response evaluations and 816 criterion-level scores.

The judge-based evaluation is treated as automated evidence rather than independent human ground truth because Phi-3.5-mini is also one of the evaluated models.

---

## 3. Linguistic Evaluation

The benchmark measures code-mixing behavior using:

* Code-Mixing Index (CMI)
* M-index
* SyMCoM

The analysis also records Hindi, English, and other token counts.

---

## 4. Pairwise Evaluation

Pairwise judgments compare model outputs for the same benchmark samples.

There are 6 unique model pairs with 34 samples per pair, resulting in 204 pairwise judgments.

The evaluation records model wins, losses, ties, total evaluated comparisons, and Holm-adjusted significance values.

---

## 5. Error Analysis

Generated responses were annotated using ten error categories:

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

Error categories are not mutually exclusive.

---

## 6. Statistical Analysis

The benchmark uses non-parametric statistical tests for repeated model evaluations.

* Friedman test
* Wilcoxon signed-rank test
* Holm correction for multiple comparisons
* Kendall's W as an effect-size measure

---

# Evaluation Results

## Automatic Generation Results

| Model        | Validation Loss | Perplexity | Distinct-1 | Distinct-2 | Repetition Rate |
| ------------ | --------------: | ---------: | ---------: | ---------: | --------------: |
| HingGPT      |          5.1430 |   171.2320 |     0.0777 |     0.4055 |          0.4358 |
| Phi-3.5-mini |          3.9277 |    50.7891 |     0.2021 |     0.7509 |          0.0040 |
| Qwen2.5-3B   |          4.3044 |    74.0219 |     0.2569 |     0.7778 |          0.0545 |
| Qwen2.5-7B   |          4.0097 |    55.1290 |     0.3401 |     0.8532 |          0.0332 |

Generation configuration:

* Temperature: 0.7
* Top-p: 0.9
* Repetition penalty: 1.1
* Maximum length: 128
* Maximum new tokens: 50
* Generation samples: 1,000

---

## LLM-as-a-Judge Results

Values are reported as mean ± standard deviation over 34 benchmark samples per model.

| Model        |       Fluency | Code-Mixing Naturalness | Hindi Grammar | Prompt Adherence | Spelling Consistency |       Overall |
| ------------ | ------------: | ----------------------: | ------------: | ---------------: | -------------------: | ------------: |
| HingGPT      | 2.088 ± 0.866 |           1.882 ± 0.844 | 1.941 ± 0.814 |    1.853 ± 0.925 |        1.971 ± 0.937 | 1.735 ± 0.864 |
| Phi-3.5-mini | 2.265 ± 0.567 |           2.118 ± 0.478 | 2.118 ± 0.478 |    2.912 ± 0.753 |        2.206 ± 0.538 | 2.206 ± 0.538 |
| Qwen2.5-3B   | 3.176 ± 1.336 |           3.000 ± 1.326 | 3.059 ± 1.347 |    3.412 ± 1.438 |        3.206 ± 1.431 | 2.647 ± 1.515 |
| Qwen2.5-7B   | 3.265 ± 1.310 |           3.294 ± 1.194 | 3.147 ± 1.209 |    3.559 ± 1.353 |        3.441 ± 1.440 | 2.765 ± 1.558 |

---

## Overall Descriptive Statistics

| Model        |  N | Mean Overall | Median |    SD |
| ------------ | -: | -----------: | -----: | ----: |
| HingGPT      | 34 |        1.735 |    2.0 | 0.864 |
| Phi-3.5-mini | 34 |        2.206 |    2.0 | 0.538 |
| Qwen2.5-3B   | 34 |        2.647 |    2.0 | 1.515 |
| Qwen2.5-7B   | 34 |        2.765 |    2.0 | 1.558 |

---

## Linguistic Results

| Model        |     CMI | M-index | SyMCoM Imbalance | Hindi Tokens | English Tokens | Other Tokens |
| ------------ | ------: | ------: | ---------------: | -----------: | -------------: | -----------: |
| HingGPT      |  6.4182 |  0.1468 |           0.8499 |          187 |          2,251 |        1,915 |
| Phi-3.5-mini | 25.5962 |  0.5738 |           0.7925 |          634 |          1,744 |           86 |
| Qwen2.5-3B   |  6.0182 |  0.1355 |           0.8019 |           97 |          1,312 |           77 |
| Qwen2.5-7B   |  9.7465 |  0.2248 |           0.7439 |          214 |            794 |          109 |

---

## Friedman Test Results

| Criterion               |  N |      χ² |  p-value | Kendall's W | Significant (α=.05) |
| ----------------------- | -: | ------: | -------: | ----------: | ------------------- |
| Fluency                 | 34 | 29.1960 | 2.04e-06 |      0.2862 | Yes                 |
| Code-Mixing Naturalness | 34 | 38.7808 | 1.93e-08 |      0.3802 | Yes                 |
| Hindi Grammar           | 34 | 34.0941 | 1.89e-07 |      0.3343 | Yes                 |
| Prompt Adherence        | 34 | 33.9760 | 2.00e-07 |      0.3331 | Yes                 |
| Spelling Consistency    | 34 | 32.7198 | 3.69e-07 |      0.3208 | Yes                 |
| Overall Quality         | 34 | 11.4089 |   0.0097 |      0.1119 | Yes                 |

All six Friedman tests are statistically significant at α = 0.05.

---

## Post-Hoc Wilcoxon + Holm Results

Only comparisons remaining significant after Holm correction are listed.

### Fluency

| Comparison                 | Holm-adjusted p |
| -------------------------- | --------------: |
| HingGPT vs Qwen2.5-3B      |          0.0022 |
| HingGPT vs Qwen2.5-7B      |          <0.001 |
| Phi-3.5-mini vs Qwen2.5-3B |          0.0044 |
| Phi-3.5-mini vs Qwen2.5-7B |          0.0022 |

### Code-Mixing Naturalness

| Comparison                 | Holm-adjusted p |
| -------------------------- | --------------: |
| HingGPT vs Qwen2.5-3B      |          <0.001 |
| HingGPT vs Qwen2.5-7B      |          <0.001 |
| Phi-3.5-mini vs Qwen2.5-3B |          0.0054 |
| Phi-3.5-mini vs Qwen2.5-7B |          <0.001 |

### Hindi Grammar

| Comparison                 | Holm-adjusted p |
| -------------------------- | --------------: |
| HingGPT vs Qwen2.5-3B      |          0.0026 |
| HingGPT vs Qwen2.5-7B      |          <0.001 |
| Phi-3.5-mini vs Qwen2.5-3B |          0.0032 |
| Phi-3.5-mini vs Qwen2.5-7B |          0.0013 |

### Prompt Adherence

| Comparison                 | Holm-adjusted p |
| -------------------------- | --------------: |
| HingGPT vs Phi-3.5-mini    |          <0.001 |
| HingGPT vs Qwen2.5-3B      |          <0.001 |
| HingGPT vs Qwen2.5-7B      |          <0.001 |
| Phi-3.5-mini vs Qwen2.5-7B |          0.0382 |

### Spelling Consistency

| Comparison                 | Holm-adjusted p |
| -------------------------- | --------------: |
| HingGPT vs Qwen2.5-3B      |          0.0017 |
| HingGPT vs Qwen2.5-7B      |          <0.001 |
| Phi-3.5-mini vs Qwen2.5-3B |          0.0048 |
| Phi-3.5-mini vs Qwen2.5-7B |          <0.001 |

### Overall Quality

| Comparison              | Holm-adjusted p |
| ----------------------- | --------------: |
| HingGPT vs Phi-3.5-mini |          0.0477 |
| HingGPT vs Qwen2.5-3B   |          0.0400 |
| HingGPT vs Qwen2.5-7B   |          0.0051 |

---

## Pairwise Overall Results

| Comparison                 | Model A Wins | Model B Wins | Ties | Evaluated | Holm-adjusted p |
| -------------------------- | -----------: | -----------: | ---: | --------: | --------------: |
| HingGPT vs Phi-3.5-mini    |           19 |            8 |    7 |        27 |          0.3134 |
| HingGPT vs Qwen2.5-3B      |           17 |            8 |    9 |        25 |          0.5388 |
| HingGPT vs Qwen2.5-7B      |           16 |           13 |    5 |        29 |          1.0000 |
| Phi-3.5-mini vs Qwen2.5-3B |           13 |           17 |    4 |        30 |          1.0000 |
| Phi-3.5-mini vs Qwen2.5-7B |           16 |           14 |    4 |        30 |          1.0000 |
| Qwen2.5-3B vs Qwen2.5-7B   |           12 |           12 |   10 |        24 |          1.0000 |

---

## Error Analysis Results

### Overall Error Distribution

Total evaluated model responses: 136.

| Error                            | Count | Percentage |
| -------------------------------- | ----: | ---------: |
| E1 — English-dominant            |    93 |     68.38% |
| E2 — Hindi-dominant              |    18 |     13.24% |
| E3 — Unnatural code-switching    |    73 |     53.68% |
| E4 — Grammatical error           |    69 |     50.74% |
| E5 — Repetition                  |    66 |     48.53% |
| E6 — Prompt misunderstanding     |   129 |     94.85% |
| E7 — Incomplete response         |    33 |     24.26% |
| E8 — Spelling/transliteration    |     8 |      5.88% |
| E9 — Hallucination/factual error |     2 |      1.47% |
| E10 — Irrelevant response        |   118 |     86.76% |

Error categories are not mutually exclusive.

### Model-Wise Error Counts

| Model        | E1 | E2 | E3 | E4 | E5 | E6 | E7 | E8 | E9 | E10 |
| ------------ | -: | -: | -: | -: | -: | -: | -: | -: | -: | --: |
| HingGPT      | 27 |  0 | 28 | 27 | 25 | 34 |  3 |  7 |  0 |  34 |
| Phi-3.5-mini | 27 |  7 | 25 | 24 |  0 | 34 |  7 |  1 |  1 |  33 |
| Qwen2.5-3B   | 20 |  5 |  7 |  7 | 21 | 29 | 10 |  0 |  1 |  25 |
| Qwen2.5-7B   | 19 |  6 | 13 | 11 | 20 | 32 | 13 |  0 |  0 |  26 |

---

# Reproducibility

## Clone the Repository

```bash
git clone https://github.com/kancharla-vyshnavi/A-Multi-Dimensional-Benchmark-of-LLMs-for-Hindi-English-Code-Mixed-Generation.git
cd A-Multi-Dimensional-Benchmark-of-LLMs-for-Hindi-English-Code-Mixed-Generation
```

## Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Evaluation Scripts

The repository contains scripts for LLM-as-a-Judge evaluation, pairwise evaluation, statistical testing, error analysis, results generation, judge-output recovery, and final report generation.

Example:

```bash
python scripts/generate_results_report.py
```

---

# Repository Structure

```text
HinglishLLM/
│
├── benchmark/
│   └── data/
│
├── src/
│   └── evaluation/
│
├── scripts/
│   ├── final_llm_judge_stats.py
│   ├── fix_last_3.py
│   ├── generate_results_report.py
│   ├── pairwise_criterion_significance.py
│   ├── pairwise_significance.py
│   ├── pairwise_statistics.py
│   ├── recover_last_2.py
│   ├── recover_llm_judge_34.py
│   ├── recover_llm_judge_v3.py
│   └── regenerate_error_analysis_summary.py
│
├── outputs/
│   ├── final_figures/
│   ├── llm_judge/
│   ├── pairwise_judge/
│   └── MASTER_RESULTS.csv
│
├── FINAL_PROJECT_OUTPUTS/
│   ├── error_analysis_model_comparison_final.csv
│   ├── error_analysis_summary_final.csv
│   ├── figure_1_overall_quality.png
│   ├── figure_2_criterion_comparison.png
│   ├── figure_3_linguistic_metrics.png
│   ├── figure_4_kendall_w.png
│   ├── figure_5_pairwise_win_rates.png
│   ├── figure_6_error_analysis.png
│   ├── friedman_results.csv
│   ├── hinglish_bench_llm_judge_results_final.csv
│   ├── manual_error_annotation_final.csv
│   ├── pairwise_criterion_results.csv
│   ├── pairwise_overall_results.csv
│   ├── pairwise_preference_matrix.csv
│   ├── pairwise_significance.csv
│   ├── results_report.txt
│   ├── table_*.csv
│   └── wilcoxon_holm_results.csv
│
├── README.md
├── requirements.txt
└── .gitignore
```


# Limitations

1. LLM-as-a-Judge evaluation uses Phi-3.5-mini-Instruct, which is also one of the evaluated models. Therefore, judge-based scores should not be interpreted as fully independent human ground truth.

2. The LLM-as-a-Judge and pairwise evaluations use 34 benchmark samples per model or pair.

3. Perplexity and lexical-diversity metrics capture specific properties of generated text and do not independently measure semantic quality or code-mixing appropriateness.

4. Error categories are not mutually exclusive, so their percentages should not be summed as a single-label distribution.

5. Two HingGPT judge records required constrained score decoding after emoji-only judge outputs. These recovered records are retained in the final evaluation.

6. Automatic generation metrics are based on the stated generation configuration and should therefore be interpreted within that experimental setup.

---

# Research Contribution

This project provides a multi-dimensional evaluation framework for Hindi-English code-mixed LLM generation by combining automatic generation metrics, lexical diversity, repetition analysis, code-mixing metrics, LLM-as-a-Judge evaluation, pairwise comparison, manual error analysis, statistical testing, and reproducible research artifacts.

