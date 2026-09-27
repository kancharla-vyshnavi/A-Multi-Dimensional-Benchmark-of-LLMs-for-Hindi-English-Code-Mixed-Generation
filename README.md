# A Multi-Dimensional Benchmark of LLMs for Hindi-English Code-Mixed Generation

## 1. Project Overview

This project presents a multi-dimensional benchmark for evaluating Large Language Models (LLMs) on **Hindi-English code-mixed (Hinglish) generation**.

The study evaluates four pretrained language models after **QLoRA-based continued pretraining (CPT)** on a natural Hinglish corpus:

1. **HingGPT**
2. **Phi-3.5-mini**
3. **Qwen2.5-3B**
4. **Qwen2.5-7B**

The evaluation combines controlled generation, independent multi-dimensional judging, inter-judge reliability analysis, robustness analysis, statistical testing, automated error analysis, and Hinglish-specific linguistic metrics.

The overall objective is to understand not only which models generate better Hinglish, but also **which dimensions of Hinglish quality are difficult for current LLMs, how evaluator disagreement affects interpretation, and what linguistic error patterns occur in generated responses**.

---

## 2. Research Objectives

The project addresses the following objectives:

* Evaluate LLM performance on Hindi-English code-mixed generation.
* Compare CPT-adapted models using a common controlled benchmark.
* Measure multiple dimensions of Hinglish generation quality.
* Compare independent evaluator judgments.
* Quantify inter-judge reliability.
* Test the robustness of evaluation results under rubric perturbation.
* Identify common generation error patterns.
* Analyze Hinglish-specific linguistic characteristics.
* Evaluate automatic generation quality using a separate 17K-prompt evaluation.
* Select a model for downstream Hinglish chatbot deployment based on combined evidence.

---

## 3. Models Evaluated

| Model        | Model Family           | Evaluation  |
| ------------ | ---------------------- | ----------- |
| HingGPT      | Hinglish-focused model | CPT-adapted |
| Phi-3.5-mini | Phi family             | CPT-adapted |
| Qwen2.5-3B   | Qwen family            | CPT-adapted |
| Qwen2.5-7B   | Qwen family            | CPT-adapted |

All benchmark evaluations use the corresponding trained/adapted checkpoints rather than treating the original pretrained base models as the final evaluated systems.

---

## 4. Dataset and Benchmark

### Natural Hinglish Corpus

The training corpus consists of naturally occurring Hindi-English code-mixed utterances.

After preprocessing and deduplication:

* Usable rows: **14,601**
* Approximately unique utterances: **14,600**

The corpus is used for continued pretraining of the evaluated models.

### Hinglish-Bench

A controlled benchmark containing:

* **1,000 prompts**
* Columns:

  * `prompt_id`
  * `row_index`
  * `prompt`
  * `source_text`

The same 1,000 prompts are used for all four models to ensure controlled comparison.

Importantly, the benchmark generation stage uses the **prompt** as the model input.

---

## 5. Research Pipeline

The complete evaluation pipeline is:

```text
Pretrained LLMs
      ↓
Natural Hinglish Corpus
      ↓
Data Cleaning + Deduplication
      ↓
Train / Validation Split
      ↓
QLoRA Continued Pretraining (CPT)
      ↓
CPT-Adapted Models
      ↓
Hinglish-Bench (1,000 prompts)
      ↓
Same 1,000 Prompts × 4 Models
      ↓
4,000 Controlled Generations
      ↓
Independent Judge 1
(Mistral-7B-Instruct-v0.3)
      ↓
Independent Judge 2
(OLMo-2-0425-1B-Instruct)
      ↓
Inter-Judge Reliability
      ↓
Robustness Evaluation
      ↓
Statistical Analysis
      ↓
Error Analysis (E1–E10)
      ↓
Hinglish-Specific Linguistic Metrics
      ↓
17K Automatic Evaluation
      ↓
Final Comparative Analysis
      ↓
Model Selection
      ↓
Hinglish Chatbot
```

---

# 6. Controlled Generation

The controlled benchmark evaluates:

* 4 models
* 1,000 common prompts per model
* Total expected generations: **4,000**

Output:

```text
outputs/controlled_1000/benchmark_generations_.csv
```

The controlled generation dataset contains the generated responses used for the independent judge evaluations.

The benchmark is designed so that model comparisons are based on the same prompt set rather than different subsets of prompts.

---

# 7. Independent Multi-Dimensional Evaluation

Two independent LLM judges were used.

### Judge 1

**Mistral-7B-Instruct-v0.3**

Final evaluation:

* 4,000 rows
* 3,990 valid evaluations
* 10 invalid evaluations

Final file:

```text
outputs/independent_judge_1000/independent_mistral_judge__FINAL.csv
```

### Judge 2

**OLMo-2-0425-1B-Instruct**

Corrected final evaluation:

* 3,999 valid evaluations
* 1 missing evaluation

Final file:

```text
outputs/independent_judge_1000/independent_olmo2_1b_judge__CORRECTED_FINAL.csv
```

The corrected OLMo evaluation is used for final analysis rather than the earlier corrupted evaluation output.

---

## 7.1 Evaluation Dimensions

Each generation is evaluated across six dimensions:

1. **Fluency**
2. **Code-Mixing**
3. **Hindi Grammar**
4. **Prompt Adherence**
5. **Spelling**
6. **Overall Quality**

Each dimension is evaluated using a five-point scoring scale.

---

## 7.2 Inter-Judge Reliability

Final inter-judge reliability results:

| Dimension        | Quadratic Weighted Kappa | Spearman |
| ---------------- | -----------------------: | -------: |
| Fluency          |                    0.090 |    0.204 |
| Code-Mixing      |                    0.003 |    0.020 |
| Hindi Grammar    |                   -0.077 |   -0.292 |
| Prompt Adherence |                   -0.017 |   -0.060 |
| Spelling         |                   -0.183 |   -0.322 |
| Overall          |                   -0.024 |   -0.067 |

The agreement between the two independent judges is low and varies substantially across dimensions.

This is treated as an important methodological finding rather than being hidden or artificially corrected.

### Interpretation for Model-Level Results

Because inter-judge agreement is limited across several dimensions, the model-level scores reported in the benchmark tables should be interpreted as **judge-specific evaluation signals rather than definitive ground-truth quality scores**.

The results remain useful for comparing how the two evaluators assessed the same controlled generations. However, the disagreement indicates evaluator sensitivity and motivates the separate robustness analysis.

The two-judge results should therefore be interpreted together with the reliability statistics rather than as a single unquestionable quality score.

---

# 8. Model Performance

## 8.1 Mistral Judge

Mean scores from the final Mistral evaluation:

| Model        | Fluency | Code-Mixing | Hindi Grammar | Adherence | Spelling | Overall |
| ------------ | ------: | ----------: | ------------: | --------: | -------: | ------: |
| HingGPT      |    2.12 |        2.40 |          1.64 |      1.78 |     2.87 |    1.99 |
| Phi-3.5-mini |    1.03 |        1.04 |          1.01 |      1.03 |     1.05 |    1.03 |
| Qwen2.5-3B   |    3.53 |        3.98 |          2.94 |      3.74 |     4.65 |    3.56 |
| Qwen2.5-7B   |    3.25 |        3.68 |          2.61 |      3.07 |     4.66 |    3.17 |

## 8.2 OLMo Judge

Mean scores from the corrected OLMo evaluation:

| Model        | Fluency | Code-Mixing | Hindi Grammar | Adherence | Spelling | Overall |
| ------------ | ------: | ----------: | ------------: | --------: | -------: | ------: |
| HingGPT      |    3.44 |        4.21 |          3.88 |      3.85 |     4.34 |    4.18 |
| Phi-3.5-mini |    3.95 |        4.83 |          3.89 |      4.79 |     4.84 |    4.89 |
| Qwen2.5-3B   |    3.44 |        4.05 |          3.88 |      3.70 |     4.17 |    4.01 |
| Qwen2.5-7B   |    3.26 |        3.91 |          4.00 |      3.74 |     4.21 |    3.96 |

Because the two judges show substantial disagreement, these tables should be interpreted as **separate evaluator views** rather than collapsed into a single definitive ranking.

---

# 9. Robustness Analysis

A perturbed version of the Mistral evaluation rubric was used to test evaluator sensitivity.

The analysis compares original and perturbed rubric scores for paired generations.

Figure 4 contains **3,979 paired observations**.

Approximate original → perturbed means:

| Dimension        | Original | Perturbed |
| ---------------- | -------: | --------: |
| Fluency          |     2.48 |      2.91 |
| Code-Mixing      |     2.76 |      3.18 |
| Hindi Grammar    |     2.05 |      2.29 |
| Prompt Adherence |     2.41 |      2.99 |
| Spelling         |     3.30 |      3.43 |
| Overall          |     2.44 |      2.86 |

Approximate signed shifts:

| Dimension        |  Shift |
| ---------------- | -----: |
| Fluency          | +0.432 |
| Code-Mixing      | +0.408 |
| Hindi Grammar    | +0.245 |
| Prompt Adherence | +0.577 |
| Spelling         | +0.134 |
| Overall          | +0.419 |

The results indicate that model-quality scores can be sensitive to changes in evaluator rubric wording.

This reinforces the importance of reporting evaluator reliability and robustness rather than relying only on raw judge scores.

---

# 10. Statistical Analysis

A Friedman test was used to compare the four models across repeated prompt-level evaluations.

Final Friedman statistics:

| Dimension        | Chi-Square |
| ---------------- | ---------: |
| Fluency          |    1739.64 |
| Code-Mixing      |    1423.18 |
| Hindi Grammar    |     616.50 |
| Prompt Adherence |     900.86 |
| Spelling         |    1562.22 |
| Overall          |    1236.96 |

The corresponding p-values are extremely small.

Post-hoc pairwise comparisons were conducted using Wilcoxon signed-rank tests with Holm correction.

The pairwise effect-size analysis is reported separately in Figure 5.

### Overall Pairwise Effect Sizes

| Comparison                 | Effect Size |
| -------------------------- | ----------: |
| Qwen2.5-3B vs Qwen2.5-7B   |      +0.300 |
| Phi-3.5-mini vs Qwen2.5-7B |      -0.779 |
| Phi-3.5-mini vs Qwen2.5-3B |      -0.909 |
| HingGPT vs Qwen2.5-7B      |      -0.577 |
| HingGPT vs Qwen2.5-3B      |      -0.792 |
| HingGPT vs Phi-3.5-mini    |      +0.272 |

These effect sizes describe the magnitude and direction of pairwise differences in the analyzed evaluation scores.

---

# 11. Error Analysis

Automated error analysis uses ten error categories.

| Code | Error Type                       |
| ---- | -------------------------------- |
| E1   | English-dominant                 |
| E2   | Hindi-dominant                   |
| E3   | Unnatural code-switching         |
| E4   | Grammatical error                |
| E5   | Repetition                       |
| E6   | Prompt misunderstanding          |
| E7   | Incomplete response              |
| E8   | Spelling / transliteration error |
| E9   | Hallucination / factual error    |
| E10  | Irrelevant response              |

## 11.1 Error Distribution

| Error | Count | Percentage |
| ----- | ----: | ---------: |
| E1    |    61 |       1.5% |
| E2    |   648 |      16.2% |
| E3    |    26 |       0.7% |
| E4    |   284 |       7.1% |
| E5    |   242 |       6.0% |
| E6    |  1045 |      26.1% |
| E7    |   679 |      17.0% |
| E8    |   865 |      21.6% |
| E9    |     0 |       0.0% |
| E10   |   485 |      12.1% |

The most frequent detected categories were:

* **E6 Prompt misunderstanding — 26.1%**
* **E8 Spelling / transliteration error — 21.6%**
* **E7 Incomplete response — 17.0%**

### Error Pattern Interpretation

These patterns indicate that a substantial share of detected failures was concentrated around:

* instruction following,
* orthographic/transliteration consistency,
* response completeness.

However, the automated E1–E10 taxonomy identifies **observed error patterns rather than proven causal mechanisms**.

Therefore, these frequencies should not be interpreted as definitive explanations for why a model produced a particular error.

For example, a response classified as E6 indicates that the output was detected as not adequately following the prompt, but the category alone does not establish the underlying cause.

Similarly, E8 identifies spelling or transliteration problems but does not establish whether the problem originated from tokenization, training data, language mixing, or generation behavior.

### E9 Hallucination Caveat

E9 was detected **0 times** by the automated detector.

This should not be interpreted as proof that the models never hallucinated or produced factual errors. It means only that the automated E1–E10 detector did not flag any E9 instances in the evaluated outputs.

---

# 12. Hinglish-Specific Linguistic Characteristics

The project additionally measures linguistic characteristics of generated Hinglish.

## 12.1 Lexical Code-Mixing

| Model        | Avg. Lexical CMI | Both Languages Present |
| ------------ | ---------------: | ---------------------: |
| HingGPT      |            19.1% |                  77.3% |
| Phi-3.5-mini |            15.7% |                  51.3% |
| Qwen2.5-3B   |            15.6% |                  67.0% |
| Qwen2.5-7B   |            13.7% |                  68.3% |

## 12.2 Classified Lexicon Proportions

| Model        | English | Hindi |
| ------------ | ------: | ----: |
| HingGPT      |    0.59 |  0.26 |
| Phi-3.5-mini |    0.34 |  0.45 |
| Qwen2.5-3B   |    0.55 |  0.35 |
| Qwen2.5-7B   |    0.69 |  0.29 |

These proportions are calculated over the **classified lexicon**, rather than all generated tokens.

---

# 13. Qualitative Examples

Qualitative inspection complements the aggregate benchmark statistics by making common generation strengths and failure patterns easier to interpret.

Representative examples can be used to illustrate:

* prompt adherence,
* code-mixing balance,
* grammaticality,
* spelling/transliteration,
* response completeness.

| Type                             | What to Inspect                                                                          |
| -------------------------------- | ---------------------------------------------------------------------------------------- |
| Strong generation                | Natural Hinglish, clear prompt adherence, appropriate code-mixing, and complete response |
| Prompt misunderstanding          | Output addresses a different task or misses an important instruction                     |
| Spelling / transliteration issue | Hinglish wording contains inconsistent or incorrect Romanized Hindi forms                |
| Incomplete response              | Generation stops before fully addressing the requested task                              |

These examples are **illustrative qualitative evidence** and are not treated as additional quantitative benchmark scores.

Detailed examples and discussion should be interpreted together with the automated E1–E10 analysis.

---

# 14. Final Research Figures

The project includes seven final research figures.

### Figure 1 — Methodology Pipeline

```text
outputs/final_charts/figure_1_methodology_pipeline.png
```

### Figure 2 — Model × Dimension Performance

```text
outputs/final_charts/figure_2_model_dimension_heatmap.png
```

### Figure 3 — Inter-Judge Agreement

```text
outputs/final_charts/figure_3_inter_judge_agreement.png
```

### Figure 4 — Robustness Analysis

```text
outputs/final_charts/figure_4_robustness_analysis.png
```

### Figure 5 — Pairwise Effect Sizes

```text
outputs/final_charts/figure_5_pairwise_effect_sizes.png
```

### Figure 6 — E1–E10 Error Distribution

```text
outputs/final_charts/figure_6_e1_e10_error_distribution.png
```

### Figure 7 — Hinglish Linguistic Characteristics

```text
outputs/final_charts/figure_7_hinglish_linguistic_characteristics.png
```

---

# 15. Final Research Tables

The final research tables are stored in:

```text
outputs/final_research_tables/
```

| Table    | File                                                |
| -------- | --------------------------------------------------- |
| Table 1  | `table_1_dataset_benchmark_summary.csv`             |
| Table 2  | `table_2_generation_coverage.csv`                   |
| Table 3  | `table_3_overall_model_performance.csv`             |
| Table 4  | `table_4_dimension_wise_performance.csv`            |
| Table 5  | `table_5_inter_judge_reliability.csv`               |
| Table 6  | `table_6_robustness_analysis.csv`                   |
| Table 7  | `table_7_statistical_significance_pairwise.csv`     |
| Table 8  | `table_8_error_analysis_e1_e10.csv`                 |
| Table 9  | `table_9_hinglish_linguistic_metrics.csv`           |
| Table 10 | `table_17k_automatic_evaluation_trained_models.csv` |

---

# 16. 17K Automatic Evaluation

A separate automatic evaluation was performed on a larger 17K evaluation dataset.

This evaluation is **separate from the 1,000-prompt benchmark and independent judge analysis**.

## Evaluation Rules

* Only the **`prompt`** column is used as model input.
* `source_text` is not used as model input.
* The trained/adapted checkpoints are evaluated.
* The original base models are not used as the evaluated systems.
* One response is generated per prompt for each model.
* Only four automatic metrics are reported.

## Metrics

1. Perplexity
2. Distinct-1
3. Distinct-2
4. Repetition Rate

## Results

| Model        | Perplexity | Distinct-1 | Distinct-2 | Repetition Rate |
| ------------ | ---------: | ---------: | ---------: | --------------: |
| HingGPT      |    171.232 |     0.0777 |     0.4055 |          0.4358 |
| Phi-3.5-mini |    50.7891 |     0.2021 |     0.7509 |          0.0040 |
| Qwen2.5-3B   |    74.0219 |     0.2569 |     0.7778 |          0.0545 |
| Qwen2.5-7B   |    55.1290 |     0.3401 |     0.8532 |          0.0332 |

These automatic metrics provide a complementary view of generation behavior.

In particular, lexical diversity and repetition should not be interpreted as direct substitutes for the multi-dimensional judge evaluation.

The 17K evaluation is therefore treated as a **separate automatic evaluation track** rather than merged into the human-style judge scores.

### 17K Evaluation Files

```text
outputs/17k_automatic_evaluation_trained_models.csv
outputs/17k_automatic_evaluation_trained_models.png
outputs/final_charts/17k_automatic_evaluation_trained_models.png
outputs/final_research_tables/table_17k_automatic_evaluation_trained_models.csv
src/generate_17k_automatic_evaluation.py
```

---

# 17. Final Model Selection

Considering the combined evidence from:

* benchmark quality,
* dimension-wise evaluation,
* statistical analysis,
* robustness analysis,
* Hinglish-specific linguistic characteristics,
* automated 17K evaluation,
* and downstream deployment,

the **CPT-adapted Qwen2.5-3B model was selected as the final model for the Hinglish chatbot**.

The selection is based on the overall multi-dimensional evidence rather than a claim that Qwen2.5-3B is highest on every individual metric.

In the 17K automatic evaluation, **Qwen2.5-7B shows higher Distinct-1 and Distinct-2**, indicating higher lexical diversity on those metrics.

Therefore, model selection should be understood as a combined research decision across multiple evaluation dimensions rather than a single-metric ranking.

---

# 18. Model Checkpoints

Expected CPT adapter/checkpoint locations include:

### HingGPT

```text
models/hinggpt_cpt/hinggpt_277k_cpt/checkpoint-17340
```

### Phi-3.5-mini

```text
models/phi35_mini_cpt/phi35_mini_final_17340/content/drive/MyDrive/HindiHinglish_LLM/outputs/phi35_mini_final_17340
```

### Qwen2.5-3B

```text
models/qwen25_3b_cpt/qwen25_3b_cpt/checkpoint-17340
```

### Qwen2.5-7B

```text
models/qwen25_7b_cpt/content/drive/MyDrive/HindiHinglish_LLM/outputs/qwen25_7b_cpt/checkpoint-17340
```

Large pretrained model weight files are excluded from normal Git tracking through the repository's Git ignore rules.

---

# 19. Chatbot Deployment

The final selected model is used in the downstream Hinglish chatbot.

### Base Model

```text
Qwen/Qwen2.5-3B-Instruct
```

### Adaptation

The chatbot uses the corresponding CPT adapter/checkpoint.

### Environment

* Python: 3.11.9
* PyTorch: 2.14.0+cu132
* PEFT: 0.21.0
* CUDA: Enabled
* GPU: NVIDIA RTX 4050 6GB
* Device mapping: automatic

### Generation Configuration

```text
max_new_tokens = 128
do_sample = True
temperature = 0.7
top_p = 0.9
repetition_penalty = 1.05
device_map = "auto"
```

The chatbot loads and generates successfully using the adapted model.

Generated responses may still contain occasional awkward or repetitive Hinglish, which is consistent with the limitations observed during evaluation.

---

# 20. Research Limitations

## 20.1 Inter-Judge Disagreement

The two independent judges show low agreement across several dimensions.

Therefore, judge-derived model scores should be interpreted as evaluator-specific signals rather than definitive ground-truth measurements.

## 20.2 Rubric Sensitivity

The robustness experiment shows that changing the evaluation rubric can shift model scores.

This indicates that evaluator formulation is an important source of measurement variability.

## 20.3 Automated Error Detection

The E1–E10 analysis is based on an automated error taxonomy.

It identifies detected error patterns but does not establish causal mechanisms.

The absence of detected E9 errors should not be interpreted as proof of zero factual errors.

## 20.4 Automatic Metrics

Perplexity, Distinct-1, Distinct-2, and repetition rate measure specific aspects of generated text.

They do not fully capture:

* semantic correctness,
* contextual appropriateness,
* natural code-switching,
* prompt interpretation,
* or overall conversational quality.

## 20.5 Qualitative Analysis

Qualitative examples are used to illustrate observed strengths and failure patterns.

They are complementary evidence rather than additional quantitative benchmark scores.

## 20.6 Resource Constraints

The models were evaluated under practical GPU and memory constraints.

The project therefore emphasizes reproducible comparative evaluation under a fixed experimental setup rather than claiming unrestricted maximum-scale model performance.

---

# 21. Project Outputs

The major outputs include:

```text
outputs/
│
├── controlled_1000/
│   ├── benchmark_generations_.csv
│   ├── benchmark_generations_.jsonl
│   └── generation_config.json
│
├── independent_judge_1000/
│   ├── independent_mistral_judge__FINAL.csv
│   └── independent_olmo2_1b_judge__CORRECTED_FINAL.csv
│
├── final_charts/
│   ├── figure_1_methodology_pipeline.png
│   ├── figure_2_model_dimension_heatmap.png
│   ├── figure_3_inter_judge_agreement.png
│   ├── figure_4_robustness_analysis.png
│   ├── figure_5_pairwise_effect_sizes.png
│   ├── figure_6_e1_e10_error_distribution.png
│   ├── figure_7_hinglish_linguistic_characteristics.png
│   └── 17k_automatic_evaluation_trained_models.png
│
├── final_research_tables/
│   ├── table_1_dataset_benchmark_summary.csv
│   ├── table_2_generation_coverage.csv
│   ├── table_3_overall_model_performance.csv
│   ├── table_4_dimension_wise_performance.csv
│   ├── table_5_inter_judge_reliability.csv
│   ├── table_6_robustness_analysis.csv
│   ├── table_7_statistical_significance_pairwise.csv
│   ├── table_8_error_analysis_e1_e10.csv
│   ├── table_9_hinglish_linguistic_metrics.csv
│   └── table_17k_automatic_evaluation_trained_models.csv
│
├── 17k_automatic_evaluation_trained_models.csv
└── 17k_automatic_evaluation_trained_models.png
```

---

# 22. Research Paper

The project research paper documents:

* research motivation,
* dataset construction,
* CPT methodology,
* benchmark design,
* controlled generation,
* independent judge evaluation,
* inter-judge reliability,
* robustness analysis,
* statistical testing,
* E1–E10 error analysis,
* Hinglish-specific linguistic analysis,
* 17K automatic evaluation,
* qualitative interpretation,
* model selection,
* limitations,
* and conclusions.

The README provides the reproducibility-oriented overview, while the research paper provides the detailed methodology and interpretation.

---

# 23. Reproducibility

The repository contains the scripts and configuration required to reproduce the major stages of the study.

The main experimental stages are:

```text
Dataset Preparation
        ↓
CPT Training
        ↓
Benchmark Construction
        ↓
Controlled Generation
        ↓
Independent Evaluation
        ↓
Reliability Analysis
        ↓
Robustness Analysis
        ↓
Statistical Analysis
        ↓
Error Analysis
        ↓
Linguistic Analysis
        ↓
17K Automatic Evaluation
        ↓
Figures + Tables
        ↓
Chatbot Deployment
```

Large model weights and local checkpoint artifacts are not intended to be committed directly to the repository.

---

# 24. Repository Structure

```text
HinglishLLM/
│
├── data/
│   ├── benchmarks/
│   ├── datasets/
│   └── model/
│
├── models/
│
├── outputs/
│   ├── controlled_1000/
│   ├── independent_judge_1000/
│   ├── final_charts/
│   ├── final_research_tables/
│   └── ...
│
├── src/
│   ├── training/
│   ├── evaluation/
│   ├── analysis/
│   ├── chatbot/
│   └── generate_17k_automatic_evaluation.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 25. Conclusion

This project provides a multi-dimensional evaluation framework for Hindi-English code-mixed generation.

Rather than relying on a single automatic metric or a single evaluator, the study combines:

* controlled benchmark evaluation,
* two independent LLM judges,
* inter-judge reliability,
* robustness testing,
* statistical comparison,
* automated error analysis,
* Hinglish-specific linguistic metrics,
* and a separate 17K automatic evaluation.

The results show that evaluation of Hinglish generation is multi-dimensional and evaluator-sensitive.

The low inter-judge agreement demonstrates that model-quality measurements can vary substantially depending on the evaluator.

The robustness analysis further shows that changes in rubric formulation can shift evaluation scores.

The E1–E10 analysis identifies recurring failure patterns, particularly prompt misunderstanding, spelling/transliteration issues, and incomplete responses.

The 17K automatic evaluation provides an additional perspective on perplexity, lexical diversity, and repetition behavior.

Considering the combined evidence, the **CPT-adapted Qwen2.5-3B model was selected for downstream Hinglish chatbot deployment**, while recognizing that Qwen2.5-7B demonstrates higher lexical diversity on the 17K Distinct-1 and Distinct-2 metrics.

Overall, the project demonstrates the importance of evaluating code-mixed language generation through multiple complementary dimensions rather than relying on a single score.

---

# 26. License

This project is intended for research and educational purposes.

Please refer to the licenses and usage terms of the underlying pretrained models and datasets before redistribution or commercial use.

---

# 27. Repository

GitHub repository:

```text
https://github.com/kancharla-vyshnavi/A-Multi-Dimensional-Benchmark-of-LLMs-for-Hindi-English-Code-Mixed-Generation
```
