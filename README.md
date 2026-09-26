# A Multi-Dimensional Benchmark of LLMs for Hindi-English Code-Mixed Generation

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A rigorous, reproducible multi-dimensional benchmark for evaluating Large Language Models (LLMs) on Hindi-English code-mixed (Hinglish) generation across linguistic quality, prompt adherence, spelling, code-mixing naturalness, robustness, statistical significance, error patterns, and Hinglish-specific linguistic characteristics.

---

## 1. Research Objective & Motivation

Hindi-English code-mixed language is widely used in informal communication, social media, digital services, and everyday conversations. However, conventional language-generation evaluation metrics do not fully capture the linguistic characteristics of Hinglish, particularly code-switching naturalness, Hindi grammatical correctness, transliteration and spelling behavior, and instruction adherence.

This research introduces a multi-dimensional benchmark designed specifically to evaluate Hinglish generation quality across multiple complementary dimensions.

The study evaluates four language models:

* **HingGPT**
* **Phi-3.5-mini**
* **Qwen2.5-3B**
* **Qwen2.5-7B**

The evaluation combines controlled generation, independent LLM judging, inter-judge reliability analysis, robustness testing, statistical analysis, structured error analysis, and Hinglish-specific linguistic metrics.

---

## 2. Models Evaluated

| Model            | Model Type                       | Evaluation Role                  |
| ---------------- | -------------------------------- | -------------------------------- |
| **HingGPT**      | Hinglish-focused model           | Domain-adapted comparison model  |
| **Phi-3.5-mini** | General-purpose LLM              | Compact multilingual model       |
| **Qwen2.5-3B**   | General-purpose multilingual LLM | Small/efficient comparison model |
| **Qwen2.5-7B**   | General-purpose multilingual LLM | Higher-capacity comparison model |

The benchmark uses the same controlled prompt set for all four models to ensure comparable evaluation conditions.

---

## 3. Dataset & Benchmark

The project uses a cleaned and deduplicated natural Hinglish corpus containing approximately **14,601 usable records**.

The benchmark, **Hinglish-Bench**, contains:

* **1,000 standardized prompts**
* **4 evaluated models**
* **4,000 controlled model generations**
* Identical benchmark prompts across all four models

The benchmark covers multiple Hinglish generation scenarios and is designed to evaluate model behavior under consistent prompting conditions.

### Benchmark Structure

```text
Hinglish Corpus
      │
      ▼
Data Cleaning & Deduplication
      │
      ▼
Train / Validation Preparation
      │
      ▼
QLoRA Continued Pretraining
      │
      ▼
CPT-Adapted Models
      │
      ▼
Hinglish-Bench
1,000 Controlled Prompts
      │
      ▼
4 Models × 1,000 Prompts
      │
      ▼
4,000 Controlled Generations
```

---

## 4. Comprehensive Evaluation Pipeline

```text
Pretrained LLMs
      │
      ▼
Natural Hinglish Corpus
      │
      ▼
Data Cleaning + Deduplication
      │
      ▼
Train / Validation Preparation
      │
      ▼
QLoRA Continued Pretraining
      │
      ▼
CPT-Adapted Models
      │
      ▼
Hinglish-Bench
1,000 Prompts
      │
      ▼
4,000 Controlled Generations
      │
      ├──────────────────────────────┐
      ▼                              ▼
Independent Judge 1            Independent Judge 2
Mistral-7B-Instruct-v0.3       OLMo-2-0425-1B-Instruct
      │                              │
      └──────────────┬───────────────┘
                     ▼
          Inter-Judge Reliability
             QWK + Spearman
                     │
                     ▼
          Robustness Evaluation
          Perturbed Mistral Rubric
                     │
                     ▼
          Statistical Analysis
     Friedman + Wilcoxon + Holm
       Effect Sizes + Bootstrap CI
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   Error Analysis          Hinglish Metrics
      E1–E10             CMI + Script Analysis
          │                     │
          └──────────┬──────────┘
                     ▼
          Final Comparative Analysis
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   Research Tables          Research Figures
          │                     │
          └──────────┬──────────┘
                     ▼
              Research Paper
                     │
                     ▼
              Hinglish Chatbot
```

---

## 5. Controlled Generation

Each of the four models generates a response for the same **1,000 benchmark prompts**.

This produces:

**1,000 prompts × 4 models = 4,000 controlled generations**

The controlled generation artifacts are stored under:

```text
outputs/controlled_1000/
```

The final generation dataset contains the model identity, prompt information, and generated response required for downstream evaluation.

---

## 6. Independent LLM Evaluation

Two independent LLM judges are used to reduce dependence on a single evaluator.

### Judge 1

**Mistral-7B-Instruct-v0.3**

### Judge 2

**OLMo-2-0425-1B-Instruct**

Both judges evaluate the generated responses across six core dimensions:

1. Fluency
2. Code-mixing naturalness
3. Hindi grammar
4. Prompt adherence
5. Spelling consistency
6. Overall quality

Scores are represented on a consistent **1–5 evaluation scale**.

The OLMo evaluation includes a dedicated repair step for the Phi-3.5-mini generation issue. The corrected OLMo master contains all **4,000 evaluation rows**, with the final verification showing **zero `[GENERATION_ERROR]` responses for Phi-3.5-mini**.

---

## 7. Inter-Judge Reliability

Agreement between the two independent judges is evaluated using:

* **Weighted Cohen's Kappa / QWK**
* **Spearman rank correlation**
* Pairwise complete-case analysis

Reliability is reported separately for each evaluation dimension.

Low agreement is treated as an empirical finding rather than artificially corrected or removed.

Final reliability outputs are available under:

```text
outputs/reliability_1000_FINAL/
```

---

## 8. Robustness Evaluation

A robustness analysis evaluates whether model judgments remain stable under a perturbed evaluation rubric.

The robustness protocol includes:

* Response-only evaluation
* No model identity inference
* No direct model comparison during judging
* Independent evaluation of each dimension
* Explicit consideration of natural code-switching
* Hindi grammar evaluation only when Hindi is present
* Actual spelling evaluation
* No reward or penalty based solely on response length

Robustness analysis includes:

* Spearman correlation
* Weighted Cohen's kappa
* Mean absolute score change
* Signed mean score change
* Model-level changes
* Rank/order stability
* Bootstrap 95% confidence intervals

Final robustness artifacts are stored under:

```text
outputs/final_robustness/
```

---

## 9. Statistical Analysis

Statistical analysis is used to complement descriptive model-level results.

The final analysis includes:

* **Friedman tests** for matched multi-model comparisons
* **Wilcoxon signed-rank tests** for pairwise comparisons
* **Holm correction** for multiple comparisons
* **Rank-biserial effect sizes**
* **Bootstrap 95% confidence intervals**

The analysis is based on complete-prompt cases where the required judge scores are available.

Because the two judges show limited agreement on some dimensions, combined two-judge results are treated as an exploratory composite rather than as an unquestionable ground truth.

Final statistical outputs are stored under:

```text
outputs/statistical_analysis_1000_FINAL/
```

---

## 10. Error Analysis

A structured E1–E10 error taxonomy is used to characterize common model failure patterns.

| Code    | Error Type                     |
| ------- | ------------------------------ |
| **E1**  | English-dominant output        |
| **E2**  | Hindi-dominant output          |
| **E3**  | Unnatural code-switching       |
| **E4**  | Grammatical error              |
| **E5**  | Repetition                     |
| **E6**  | Prompt misunderstanding        |
| **E7**  | Incomplete response            |
| **E8**  | Spelling/transliteration error |
| **E9**  | Hallucination/factual error    |
| **E10** | Irrelevant response            |

The error analysis is intended to explain **how** generated responses fail, complementing the numerical quality scores that describe differences across models.

The repository contains both response-level annotations and aggregated error summaries.

---

## 11. Hinglish-Specific Linguistic Analysis

The project includes additional linguistic analysis designed specifically for Hinglish generation.

The analysis covers:

* Latin-script usage
* Devanagari usage
* Mixed-script behavior
* Roman-dominant behavior
* English/Hindi lexical mixing
* Code-Mixing Index (CMI)

The CMI is treated as a **heuristic structural indicator**, not as a gold-standard language-identification system.

Final linguistic metrics are stored under:

```text
outputs/hinglish_metrics/
```

---

## 12. Final Research Tables

The final publication-oriented tables are stored under:

```text
outputs/final_research_tables/
```

### Required Tables

```text
table_1_dataset_benchmark_summary.csv
table_2_generation_coverage.csv
table_3_overall_model_performance.csv
table_4_dimension_wise_performance.csv
table_5_inter_judge_reliability.csv
table_6_robustness_analysis.csv
table_7_statistical_significance_pairwise.csv
table_8_error_analysis_e1_e10.csv
table_9_hinglish_linguistic_metrics.csv
```

These tables provide the consolidated numerical results used for the final research analysis.

---

## 13. Final Research Figures

The final publication-oriented figures are stored under:

```text
outputs/final_charts/
```

### Required Figures

```text
figure_1_methodology_pipeline.png
figure_2_model_dimension_heatmap.png
figure_3_inter_judge_agreement.png
figure_4_robustness_analysis.png
figure_5_pairwise_effect_sizes.png
figure_6_e1_e10_error_distribution.png
figure_7_hinglish_linguistic_characteristics.png
```

The figures provide visual summaries of:

* Research methodology
* Model × dimension performance
* Inter-judge agreement
* Robustness behavior
* Pairwise statistical effects
* E1–E10 error distribution
* Hinglish linguistic characteristics

---

## 14. Repository Structure

```text
HinglishLLM/
│
├── README.md
├── .gitignore
├── new_requirements.txt
├── paper.md
│
├── chatbot/
│   ├── app.py
│   └── chatbot.py
│
├── data/
│   ├── benchmarks/
│   │   └── hinglish_bench_.csv
│   └── processed/
│
├── models/
│
├── data/model/
│   └── CPT model adapter and tokenizer artifacts
│
├── outputs/
│   ├── controlled_1000/
│   ├── independent_judge_1000/
│   ├── reliability_1000_FINAL/
│   ├── statistical_analysis_1000_FINAL/
│   ├── final_robustness/
│   ├── final_charts/
│   ├── final_research_tables/
│   ├── error_analysis/
│   └── hinglish_metrics/
│
└── src/
    ├── run_independent_mistral_judge.py
    ├── run_independent_olmo2_judge.py
    ├── run_olmo2_phi_repair.py
    ├── run_final_inter_judge_reliability.py
    ├── run_final_statistical_analysis.py
    ├── run_mistral_robustness.py
    ├── robustness_analysis.py
    └── metrics/
```

Large model-weight files are excluded from version control through `.gitignore`.

---

## 15. Reproducibility

The repository preserves the generated benchmark outputs and final analysis artifacts so that the reported results can be inspected without regenerating all model responses.

The principal research artifacts are available under:

```text
outputs/final_research_tables/
outputs/final_charts/
outputs/final_robustness/
outputs/reliability_1000_FINAL/
outputs/statistical_analysis_1000_FINAL/
```

The environment dependencies are documented in:

```text
new_requirements.txt
```

Model checkpoints and adapter artifacts are subject to the repository's Git ignore rules and local storage configuration.

---

## 16. Limitations

The benchmark has several important limitations:

* **Benchmark scope:** The 1,000-prompt benchmark represents a controlled evaluation sample and cannot capture every form of Hindi-English code-mixed communication.
* **Judge dependence:** Automated evaluation depends partly on the behavior and rubric interpretation of the selected LLM judges.
* **Inter-judge disagreement:** Agreement between the two independent judges is limited for some dimensions and should be considered when interpreting combined results.
* **Missing evaluations:** Some evaluation records may be unavailable or invalid for particular analyses; complete-case requirements are therefore reported explicitly.
* **CMI limitation:** Lexical CMI is a transparent heuristic rather than exhaustive language identification.
* **Causal interpretation:** Observed differences between models should not be interpreted as causal evidence for architecture, pretraining scale, or specialization because the benchmark does not isolate individual causal factors experimentally.
* **Model adaptation:** The evaluated models and CPT/adaptation configurations represent a particular experimental setup and should not be generalized to all possible model configurations.

---

## 17. Research Paper

The complete research paper is provided in:

```text
paper.md
```

The paper incorporates the final benchmark methodology, independent judging, reliability analysis, robustness evaluation, statistical testing, error analysis, Hinglish-specific metrics, tables, figures, limitations, and final comparative discussion.

---

## 18. Hinglish Chatbot

A demonstration chatbot is included under:

```text
chatbot/
```

The chatbot provides an interactive interface for generating Hindi-English code-mixed responses using the selected model/adaptation configuration.

The chatbot is treated as a demonstration application built from the research outputs rather than as a separate benchmark.

---

## 19. License

This project is licensed under the **MIT License**.

If you build upon or reference this benchmark in academic research, please cite the repository and associated research work.
