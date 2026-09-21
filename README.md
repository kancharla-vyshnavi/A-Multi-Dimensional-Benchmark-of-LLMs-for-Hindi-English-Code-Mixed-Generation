# A Multi-Dimensional Benchmark of LLMs for Hindi-English Code-Mixed Generation

A systematic benchmark for evaluating Large Language Models (LLMs) on **Hindi-English code-mixed (Hinglish) text generation** across multiple dimensions, including fluency, code-mixing quality, Hindi grammaticality, prompt adherence, spelling consistency, and overall response quality.

## Overview

Hindi-English code-mixed language is widely used in informal communication, social media, customer interactions, and online content. However, evaluating LLMs on code-mixed generation requires more than conventional language-generation metrics.

This project presents a **multi-dimensional evaluation framework** for comparing LLMs on controlled Hindi-English code-mixed generation tasks.

The benchmark evaluates:

* Linguistic fluency
* Code-mixing naturalness
* Hindi grammaticality
* Prompt adherence
* Spelling consistency
* Overall response quality
* Script-level characteristics
* Lexicon-based code-mixing characteristics
* Error patterns
* Pairwise model preferences
* Human–LLM judge agreement

## Models Evaluated

The benchmark evaluates four models:

| Model        | Type                            |
| ------------ | ------------------------------- |
| HingGPT      | Hinglish-focused language model |
| Phi-3.5-mini | General-purpose LLM             |
| Qwen2.5-3B   | General-purpose LLM             |
| Qwen2.5-7B   | General-purpose LLM             |

## Benchmark Design

The benchmark contains **34 prompts** distributed across **8 categories**:

1. Advice & Opinions
2. Captions & One-liners
3. Casual Conversation
4. Customer Support Dialogues
5. News & Explainers
6. Product Reviews
7. Social Media Posts
8. Storytelling

Each model was evaluated on the same benchmark prompts to enable controlled comparison.

## Evaluation Framework

### 1. Independent LLM Judge

Generated responses were evaluated using an independent **Mistral-7B-Instruct-v0.3** judge.

Each response was assessed on a 1–5 scale for:

* Fluency
* Code-mixing quality
* Hindi grammar
* Prompt adherence
* Spelling consistency
* Overall quality

The independent judge produced valid scores for 129 of the 136 generated responses. Missing responses were retained as missing rather than silently replaced.

### 2. Pairwise Evaluation

Pairwise comparison was performed for all six unique model pairs.

With 34 prompts and both comparison directions, this produced:

**408 pairwise comparisons**

The pairwise evaluation also included a swapped-order consistency analysis to examine whether changing the position of the two models affected the comparison.

### 3. Human Validation

A human validation subset containing **48 responses** was annotated:

* 12 responses per model
* 6 evaluation dimensions

Human scores were compared with the independent Mistral judge to measure agreement.

The human validation is intended as a validation analysis rather than a replacement for the full automated evaluation.

### 4. Missing-Data Sensitivity

A worst-case sensitivity analysis was performed for responses missing independent-judge scores.

For this analysis, missing overall scores were conservatively assigned a score of **1/5** to examine how much the model-level mean could change under an adverse missing-data assumption.

### 5. Hinglish-Specific Analysis

Additional linguistic analyses were performed using:

* Script-level statistics
* Devanagari/Latin script proportions
* Mixed-script rates
* Roman-dominant rates
* A transparent lexicon-based approximate Code-Mixing Index (CMI)

The CMI is treated as a **heuristic indicator**, not as gold-standard language identification.

## Main Results

### Independent Judge

| Model        | Mean Overall Score | Valid Responses |
| ------------ | -----------------: | --------------: |
| HingGPT      |              2.464 |              28 |
| Phi-3.5-mini |              3.912 |              34 |
| Qwen2.5-3B   |              3.697 |              33 |
| Qwen2.5-7B   |              3.882 |              34 |

The results show differences across models and evaluation dimensions rather than a single uniform performance pattern.

### Pairwise Evaluation

Across 204 comparisons per model:

| Model        | Wins | Losses | Ties | Win Rate |
| ------------ | ---: | -----: | ---: | -------: |
| HingGPT      |   24 |    176 |    4 |   0.1176 |
| Phi-3.5-mini |  157 |     47 |    0 |   0.7696 |
| Qwen2.5-3B   |   97 |    105 |    2 |   0.4755 |
| Qwen2.5-7B   |  126 |     76 |    2 |   0.6176 |

Pairwise results are reported together with statistical tests and swap-order consistency analysis.

### Category-Level Analysis

The benchmark also reports model performance separately across the eight task categories.

This allows differences between models to be examined at the task-category level rather than relying only on an aggregate score.

## Human Validation

Human annotation produced a substantially stricter scoring pattern than the independent LLM judge.

For the paired human–Mistral validation subset, the analysis reports:

* Mean-score differences
* Spearman correlation
* Weighted quadratic Cohen's kappa
* Paired sample counts

The results indicate **weak agreement between the single human annotator and the LLM judge** across the evaluated dimensions.

Because the validation contains a single human annotator, human–human inter-annotator agreement cannot be estimated from this dataset.

## Reproducibility

The repository contains the generated benchmark outputs and evaluation artifacts required to inspect the experimental results.

### Directory Structure

```text
HinglishLLM/
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
├── analyze_human_agreement.py
├── project_structure.txt
└── README.md
```

## Final Research Tables

The `outputs/final_research_tables/` directory contains consolidated tables for:

* Overall model results
* Statistical comparisons
* Category-level performance
* Error analysis

## Visualizations

The `outputs/final_charts/` directory contains visual summaries of:

* Independent judge scores
* Pairwise win rates
* Category-level performance
* Code-mixing index

## Statistical Analysis

Pairwise model comparisons were analyzed using statistical tests with Holm correction for multiple comparisons.

The statistical outputs are available under:

```text
outputs/statistical_analysis/
```

The results distinguish statistically significant comparisons from comparisons for which statistical significance was not established.

## Limitations

Several limitations should be considered when interpreting the results:

1. **Small benchmark size**
   The benchmark contains 34 prompts, so results should not be interpreted as universally representative of all Hindi-English code-mixed usage.

2. **Single independent LLM judge**
   Automated evaluation depends partly on the behavior and scoring tendencies of the Mistral judge.

3. **Human validation size**
   Human validation contains 48 responses and one human annotator.

4. **Human–LLM score differences**
   The human annotator was substantially stricter than the independent LLM judge, and agreement was weak across several dimensions.

5. **Script-based analysis limitations**
   Latin-script output does not necessarily mean English-language output because Romanized Hindi is also written using Latin characters.

6. **Lexicon-based CMI limitations**
   The Code-Mixing Index is a transparent heuristic and should not be interpreted as a gold-standard language-identification system.

7. **Pairwise order sensitivity**
   Swap-order consistency varied across model pairs, indicating that pairwise comparisons should be interpreted together with the order-sensitivity analysis.

## Conclusion

This project provides a structured framework for evaluating Hindi-English code-mixed generation using multiple complementary evaluation approaches.

Rather than relying on a single metric, the benchmark combines:

* Independent LLM judging
* Pairwise comparison
* Category-level analysis
* Error analysis
* Human validation
* Missing-data sensitivity analysis
* Script-level analysis
* Code-mixing analysis
* Statistical significance testing

The resulting evaluation artifacts provide a reproducible basis for studying how different LLMs handle Hindi-English code-mixed generation across diverse task types and evaluation dimensions.

