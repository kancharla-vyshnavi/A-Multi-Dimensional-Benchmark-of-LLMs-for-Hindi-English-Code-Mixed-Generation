# A Multi-Dimensional Benchmark of LLMs for Hindi-English Code-Mixed Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Status: Research Complete](https://img.shields.io/badge/Status-Research%20Complete-success.svg)]()

A reproducible multi-dimensional benchmark for evaluating Large Language Models (LLMs) on **Hindi-English code-mixed (Hinglish) generation** across linguistic quality, prompt adherence, robustness, statistical significance, error patterns, and Hinglish-specific linguistic characteristics.

---

# 1. Research Objective & Motivation

Hindi-English code-mixed language is widely used in informal communication, social media, digital services, education, and everyday online interactions. However, conventional text-generation evaluation metrics do not fully capture the characteristics of code-mixed language.

This project develops a controlled evaluation framework for studying how different LLMs generate Hinglish and how their outputs differ across multiple quality dimensions.

The benchmark evaluates:

* **Fluency**
* **Code-mixing quality**
* **Hindi grammatical quality**
* **Prompt adherence**
* **Spelling / transliteration quality**
* **Overall response quality**
* **Evaluator robustness**
* **Error patterns**
* **Hinglish-specific linguistic characteristics**

The objective is not to reduce Hinglish generation quality to a single metric, but to provide a **multi-dimensional and statistically supported comparison framework**.

---

# 2. Models Evaluated

Four models were evaluated under the same controlled benchmark:

| Model            | Model Type           | Role                               |
| ---------------- | -------------------- | ---------------------------------- |
| **HingGPT**      | Hinglish-specialized | Domain-focused Hinglish generation |
| **Phi-3.5-mini** | General-purpose      | Compact instruction-tuned LLM      |
| **Qwen2.5-3B**   | General-purpose      | Multilingual LLM                   |
| **Qwen2.5-7B**   | General-purpose      | Higher-capacity multilingual LLM   |

The evaluation uses the same benchmark prompts and controlled generation procedure across all four models.

---

# 3. Dataset & Benchmark

The project uses a cleaned and deduplicated **Natural Hinglish Corpus** containing approximately **14,601 usable rows**.

The benchmark used for final controlled evaluation contains exactly:

* **1,000 benchmark prompts**
* **4 models**
* **4,000 controlled generations**
* The same 1,000 prompts are evaluated across all models

Benchmark file:

```text
data/benchmarks/hinglish_bench_.csv
```

## Benchmark Structure

| Field         | Description                           |
| ------------- | ------------------------------------- |
| `prompt_id`   | Unique benchmark prompt identifier    |
| `row_index`   | Source row reference                  |
| `prompt`      | Controlled Hinglish generation prompt |
| `source_text` | Corresponding source text             |

The benchmark is designed so that model outputs can be compared under the same prompt conditions.

---

# 4. Research Pipeline

The complete evaluation pipeline is:

```text
Natural Hinglish Corpus
        │
        ▼
Data Cleaning & Deduplication
        │
        ▼
Train / Validation Preparation
        │
        ▼
QLoRA Continued Pretraining (CPT)
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
        │
        ├──────────────────────────────┐
        │                              │
        ▼                              ▼
Mistral-7B-Instruct-v0.3        OLMo-2-1B-Instruct
Independent Judge 1             Independent Judge 2
        │                              │
        └──────────────┬───────────────┘
                       ▼
              Inter-Judge Reliability
               QWK + Spearman ρ
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
                       ▼
                Error Analysis
                  E1 – E10
                       │
                       ▼
         Hinglish Linguistic Metrics
                       │
                       ▼
         Final Tables + Figures
                       │
                       ▼
              Research Paper
                       │
                       ▼
              Hinglish Chatbot
```

---

# 5. Controlled Generation

The final benchmark produces **4,000 controlled generations**:

```text
1,000 prompts × 4 models = 4,000 generations
```

The main controlled-generation outputs are stored under:

```text
outputs/controlled_1000/
```

Primary files include:

```text
benchmark_generations_.csv
benchmark_generations_.jsonl
generation_config_.json
```

The benchmark uses the same prompts across all models to maintain controlled comparison conditions.

---

# 6. Independent LLM Evaluation

Two independent instruction-tuned LLM judges were used.

## Judge 1

**Mistral-7B-Instruct-v0.3**

## Judge 2

**OLMo-2-1B-Instruct**

The judges evaluate the generated responses across six core dimensions:

1. Fluency
2. Code-Mixing
3. Hindi Grammar
4. Prompt Adherence
5. Spelling
6. Overall Quality

Scores use a **1–5 scale**.

Final judge outputs are stored under:

```text
outputs/independent_judge_1000/
```

The corrected OLMo evaluation is used for the final analysis rather than the earlier corrupted evaluation output.

---

# 7. Inter-Judge Reliability

Agreement between the two independent judges is evaluated using:

* **Quadratic Weighted Kappa (QWK)**
* **Spearman Rank Correlation (ρ)**

The purpose is to measure whether the two independent evaluators provide consistent assessments of the same generated responses.

The final analysis indicates that judge agreement varies substantially across dimensions. This is treated as an important methodological finding rather than being hidden or artificially corrected.

## Impact on Headline Model Results

Because inter-judge agreement is limited across several dimensions, the model-level scores reported in the headline performance tables, particularly **Table 3 (Overall Model Performance)** and **Table 4 (Dimension-Wise Performance)**, should be interpreted as **evaluator-specific quality signals rather than definitive ground-truth quality scores**.

The tables remain useful because the same benchmark prompts and controlled generations are evaluated across models. However, disagreement between the independent judges means that the absolute scores should not be interpreted as universally accepted quality measurements.

In particular:

* Table 3 should be interpreted as a summary of how the evaluators assessed overall model quality.
* Table 4 should be interpreted as a dimension-specific view of evaluator-assigned quality.
* The low inter-judge agreement means that apparent differences in individual dimensions should be considered together with the reliability analysis.
* The separate robustness experiment further evaluates how sensitive the evaluation is to rubric formulation.

Therefore, the headline model-performance results are **comparative evaluation signals under the defined judging setup**, rather than claims of absolute model quality.

Final reliability outputs are available under:

```text
outputs/reliability_1000_FINAL/
```

---

# 8. Evaluator Robustness

A robustness experiment evaluates whether changing the evaluation rubric affects the Mistral judge's scores.

The same 4,000 generated responses are evaluated under a **perturbed Mistral rubric**.

The analysis includes:

* Original vs perturbed mean scores
* Spearman correlation
* Weighted agreement
* Mean absolute score change
* Signed score change
* Model-level score changes
* Rank/order stability
* Bootstrap 95% confidence intervals

The robustness outputs are stored under:

```text
outputs/final_robustness/
```

The robustness analysis is important because the inter-judge disagreement indicates that evaluator behavior is itself a source of measurement variability.

---

# 9. Statistical Analysis

The statistical analysis evaluates whether observed differences between models are supported by matched statistical tests.

The analysis includes:

## Friedman Test

Used for multi-model repeated comparisons across benchmark observations.

## Wilcoxon Signed-Rank Test

Used for pairwise matched comparisons between models.

## Holm Correction

Applied to multiple pairwise comparisons to control the family-wise error rate.

## Rank-Biserial Effect Size

Effect sizes are reported alongside statistical significance.

## Bootstrap Confidence Intervals

Bootstrap procedures are used to estimate 95% confidence intervals for relevant comparisons.

Statistical outputs are available under:

```text
outputs/statistical_analysis_1000_FINAL/
```

Important statistical files include:

```text
final_friedman_results.csv
final_wilcoxon_pairwise_holm.csv
final_bootstrap_95ci_pairwise.csv
final_model_means_two_judge.csv
```

The exploratory two-judge composite is interpreted together with the inter-judge reliability results rather than being treated as an unquestionable ground truth.

---

# 10. Error Analysis: E1–E10

A structured error taxonomy is used to identify common failure patterns across the 4,000 controlled generations.

| Code    | Error Type                       |
| ------- | -------------------------------- |
| **E1**  | English-dominant output          |
| **E2**  | Hindi-dominant output            |
| **E3**  | Unnatural code-switching         |
| **E4**  | Grammatical error                |
| **E5**  | Repetition                       |
| **E6**  | Prompt misunderstanding          |
| **E7**  | Incomplete response              |
| **E8**  | Spelling / transliteration error |
| **E9**  | Hallucination / factual error    |
| **E10** | Irrelevant response              |

The automated analysis identified:

| Error | Count | Percentage |
| ----- | ----: | ---------: |
| E1    |    61 |       1.5% |
| E2    |   648 |      16.2% |
| E3    |    26 |       0.7% |
| E4    |   284 |       7.1% |
| E5    |   242 |       6.0% |
| E6    | 1,045 |      26.1% |
| E7    |   679 |      17.0% |
| E8    |   865 |      21.6% |
| E9    |     0 |       0.0% |
| E10   |   485 |      12.1% |

**Important:** E9 = 0 means that the automated detector flagged zero cases. It does **not** establish that the models produced no true hallucinations.

## 10.1 Why Are E6, E8, and E7 the Most Frequent?

The three most frequent detected categories were:

* **E6 Prompt misunderstanding — 26.1%**
* **E8 Spelling / transliteration error — 21.6%**
* **E7 Incomplete response — 17.0%**

These patterns suggest that a substantial portion of detected failures is concentrated around three aspects of Hinglish generation:

### E6 — Prompt Misunderstanding

E6 represents cases where the generated response does not adequately address the requested task or misses an important instruction.

In a code-mixed generation setting, this may appear as a response that produces text but does not correctly preserve the intended task, requested format, or conversational objective.

The observed frequency therefore highlights **instruction-following as an important failure pattern in the benchmark**.

However, E6 is an observed error category and does not by itself establish the underlying causal mechanism.

### E8 — Spelling / Transliteration Error

E8 captures spelling and Romanized-Hindi/transliteration inconsistencies.

Hinglish commonly uses Roman script for Hindi words, which introduces additional variation in spelling and transliteration.

For example, semantically similar Hindi expressions can have multiple Romanized forms. Consequently, inconsistent transliteration can affect automatic detection and perceived text quality even when the intended meaning remains understandable.

The high E8 frequency therefore highlights **orthographic and transliteration consistency as an important challenge for Hinglish generation**.

This should not be interpreted as proof that the underlying cause is a particular tokenizer, training-data property, or model architecture.

### E7 — Incomplete Response

E7 identifies responses that stop before adequately completing the requested task.

This pattern can affect prompt adherence and overall usefulness even when the generated text is grammatically acceptable.

The E7 frequency therefore indicates that **response completeness is another important failure pattern** in the evaluated generations.

Again, the taxonomy identifies an observed output pattern rather than proving why a generation stopped early.

## 10.2 Interpretation of Error Frequencies

The E1–E10 taxonomy identifies **observed error patterns rather than proven causal mechanisms**.

Therefore:

* E6 indicates observed prompt-following failures, not a definitive explanation of why they occurred.
* E8 indicates spelling/transliteration problems, not a proven data or tokenizer cause.
* E7 indicates incomplete outputs, not a proven decoding or generation-length cause.

The error analysis is therefore used to identify **where failures are concentrated**, while the research paper discusses possible interpretations without treating them as experimentally established causes.

Error-analysis outputs are stored under:

```text
outputs/error_analysis/
```

---

# 11. Hinglish-Specific Linguistic Analysis

The project includes additional linguistic analysis designed specifically for Hinglish generation.

The analysis covers:

* English lexical share
* Hindi lexical share
* Code-Mixing Index (CMI)
* Presence of both languages
* Model-level lexical characteristics
* Response-level Hinglish characteristics

The CMI is treated as a **transparent heuristic indicator** of lexical mixing rather than a perfect language-identification system.

Final metric outputs:

```text
outputs/hinglish_metrics/
```

Important files:

```text
code_mixing_index.csv
code_mixing_summary.csv
model_level_hinglish_metrics.csv
response_level_hinglish_metrics.csv
```

---

# 12. Qualitative Examples

Aggregate scores and error percentages provide quantitative evidence, but representative generations make the benchmark easier to interpret.

The qualitative section is intended to provide **side-by-side examples of strong and problematic generations** before the reader moves into the detailed research paper.

The examples should focus on the same prompt being evaluated across models or on representative outputs illustrating the major error categories.

## Example Categories

| Example Type                     | What to Inspect                                                                      |
| -------------------------------- | ------------------------------------------------------------------------------------ |
| Strong generation                | Natural Hinglish, appropriate code-mixing, clear prompt adherence, complete response |
| Prompt misunderstanding          | Response does not adequately address the requested task                              |
| Spelling / transliteration issue | Incorrect or inconsistent Romanized-Hindi spelling                                   |
| Incomplete response              | Response stops before fully addressing the prompt                                    |
| Unnatural code-switching         | Hindi-English switching is awkward or unnatural                                      |
| Grammatical error                | Hindi or mixed-language grammatical construction is problematic                      |

### Side-by-Side Example Format

| Prompt                            | Model          | Generation                  | Interpretation                                      |
| --------------------------------- | -------------- | --------------------------- | --------------------------------------------------- |
| *Representative benchmark prompt* | Selected model | *Representative generation* | Strong prompt adherence and natural Hinglish        |
| *Same / similar prompt*           | Another model  | *Representative generation* | Prompt misunderstanding or incomplete response      |
| *Representative benchmark prompt* | Selected model | *Representative generation* | Spelling/transliteration issue                      |
| *Representative benchmark prompt* | Another model  | *Representative generation* | More complete and contextually appropriate response |

> **Note:** Actual generation text should be taken directly from the controlled-generation output files. No synthetic or invented model outputs are used as research evidence.

These qualitative examples are **illustrative evidence** and are not treated as additional quantitative benchmark scores.

---

# 13. Final Research Figures

All publication-oriented figures are stored under:

```text
outputs/final_charts/
```

## Figure 1 — Overall Methodology Pipeline

![Overall Methodology Pipeline](outputs/final_charts/figure_1_methodology_pipeline.png)

## Figure 2 — Multi-Dimensional Performance Heatmap

![Multi-Dimensional Performance Heatmap](outputs/final_charts/figure_2_model_dimension_heatmap.png)

This figure presents model performance across the six evaluation dimensions for both independent judges and the exploratory two-judge composite.

## Figure 3 — Inter-Judge Agreement

![Inter-Judge Agreement](outputs/final_charts/figure_3_inter_judge_agreement.png)

This figure reports QWK and Spearman agreement between the Mistral-7B and OLMo-2-1B judges across the evaluated dimensions.

## Figure 4 — Evaluator Robustness

![Evaluator Robustness](outputs/final_charts/figure_4_robustness_analysis.png)

This figure shows original versus perturbed Mistral rubric scores and bootstrap confidence intervals for score shifts.

## Figure 5 — Pairwise Model Effect Sizes

![Pairwise Model Effect Sizes](outputs/final_charts/figure_5_pairwise_effect_sizes.png)

This figure presents rank-biserial effect sizes for pairwise overall-quality comparisons after Holm correction.

## Figure 6 — E1–E10 Error Distribution

![E1–E10 Error Distribution](outputs/final_charts/figure_6_e1_e10_error_distribution.png)

This figure summarizes the automated E1–E10 error distribution across the 4,000 controlled generations.

## Figure 7 — Hinglish Linguistic Characteristics

![Hinglish Linguistic Characteristics](outputs/final_charts/figure_7_hinglish_linguistic_characteristics.png)

This figure presents model-level code-mixing characteristics and English/Hindi lexical distributions across the 4,000 controlled generations.

---

# 14. 17K Automatic Evaluation

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

The 17K evaluation is therefore treated as a **separate automatic evaluation track** rather than merged into the judge scores.

## 17K Evaluation Figure

![17K Automatic Evaluation](outputs/final_charts/17k_automatic_evaluation_trained_models.png)

## 17K Evaluation Files

```text
outputs/17k_automatic_evaluation_trained_models.csv
outputs/17k_automatic_evaluation_trained_models.png
outputs/final_charts/17k_automatic_evaluation_trained_models.png
outputs/final_research_tables/table_17k_automatic_evaluation_trained_models.csv
src/generate_17k_automatic_evaluation.py
```

---

# 15. Final Research Tables

The final research tables are stored under:

```text
outputs/final_research_tables/
```

### Table 1 — Dataset & Benchmark Summary

```text
table_1_dataset_benchmark_summary.csv
```

### Table 2 — Generation Coverage

```text
table_2_generation_coverage.csv
```

### Table 3 — Overall Model Performance

```text
table_3_overall_model_performance.csv
```

**Interpretation note:** Because inter-judge reliability is limited, Table 3 should be interpreted as evaluator-specific comparative evidence rather than definitive ground-truth model quality.

### Table 4 — Dimension-Wise Performance

```text
table_4_dimension_wise_performance.csv
```

**Interpretation note:** Dimension-wise differences in Table 4 should be considered together with the corresponding QWK/Spearman reliability results and robustness analysis.

### Table 5 — Inter-Judge Reliability

```text
table_5_inter_judge_reliability.csv
```

### Table 6 — Robustness Analysis

```text
table_6_robustness_analysis.csv
```

### Table 7 — Statistical Significance & Pairwise Comparisons

```text
table_7_statistical_significance_pairwise.csv
```

### Table 8 — E1–E10 Error Analysis

```text
table_8_error_analysis_e1_e10.csv
```

### Table 9 — Hinglish Linguistic Metrics

```text
table_9_hinglish_linguistic_metrics.csv
```

### Table 10 — 17K Automatic Evaluation

```text
table_17k_automatic_evaluation_trained_models.csv
```

These tables provide structured machine-readable versions of the main research results.

---

# 16. Repository Structure

```text
HinglishLLM/
│
├── README.md
├── paper.md
├── .gitignore
│
├── chatbot/
│   ├── app.py
│   └── chatbot.py
│
├── configs/
│
├── data/
│   ├── benchmarks/
│   │   └── hinglish_bench_.csv
│   └── model/
│
├── models/
│
├── outputs/
│   ├── controlled_1000/
│   ├── independent_judge_1000/
│   ├── reliability_1000_FINAL/
│   ├── statistical_analysis_1000_FINAL/
│   ├── error_analysis/
│   ├── hinglish_metrics/
│   ├── final_charts/
│   ├── final_research_tables/
│   └── final_robustness/
│
└── src/
    ├── evaluation/
    ├── metrics/
    ├── run_final_inter_judge_reliability.py
    ├── run_final_statistical_analysis.py
    ├── run_olmo2_phi_repair.py
    └── generate_17k_automatic_evaluation.py
```

Large pretrained model weights and checkpoint files are excluded from normal Git tracking through `.gitignore` rules.

---

# 17. Reproducibility

The repository preserves the generated evaluation artifacts so that the final analysis can be inspected without regenerating all model outputs.

The main reproducibility components are:

```text
Benchmark
    ↓
Controlled generations
    ↓
Independent judges
    ↓
Reliability analysis
    ↓
Robustness analysis
    ↓
Statistical analysis
    ↓
Error analysis
    ↓
Hinglish metrics
    ↓
17K automatic evaluation
    ↓
Final tables & figures
```

The benchmark generation configuration is stored in:

```text
outputs/controlled_1000/generation_config_.json
```

---

# 18. Environment

The project was developed and evaluated using a Python/PyTorch environment with GPU acceleration.

The recorded environment includes:

```text
Python 3.11.x
PyTorch 2.14.0+cu132
Transformers 5.17.0
PEFT 0.21.0
CUDA-enabled GPU
```

A dependency snapshot is provided in:

```text
new_requirements.txt
```

Because model inference depends on GPU memory, CUDA compatibility, model checkpoints, and local hardware configuration, exact runtime behavior may vary across machines.

---

# 19. Hinglish Chatbot

The final project also contains a chatbot interface using a CPT-adapted model.

Chatbot files:

```text
chatbot/
├── app.py
└── chatbot.py
```

The chatbot uses a Qwen-based model with a PEFT adapter for Hinglish generation.

The chatbot is treated as an application layer built on top of the research pipeline rather than as an additional benchmark result.

---

# 20. Research Limitations

Several limitations should be considered when interpreting the results.

## Benchmark Coverage

Although the final benchmark contains 1,000 prompts, it cannot represent every form of Hindi-English code-mixed communication.

## Automated Judge Dependence

LLM-based evaluation depends on the behavior and rubric interpretation of the selected judges.

## Inter-Judge Disagreement

The independent judges show limited agreement on several dimensions.

Therefore, the two-judge composite and the headline model scores are treated as **evaluation signals under the defined judging setup**, rather than unquestionable ground-truth scores.

This is particularly relevant when interpreting Tables 3 and 4.

## Evaluator Robustness

The robustness experiment demonstrates that changes in rubric formulation can alter evaluator scores.

This indicates that evaluator design is an important component of the measurement process.

## Error Detector Limitations

Automated E1–E10 detection can miss subtle or context-dependent errors.

## E9 Interpretation

The absence of automatically detected E9 cases should not be interpreted as proof that no factual errors or hallucinations exist.

## CMI Limitation

The Code-Mixing Index is a heuristic linguistic measure and does not constitute perfect language identification.

## Causal Interpretation

Observed differences between models should not be interpreted as causal evidence for model architecture, parameter count, pretraining data, or specialization because the benchmark does not isolate individual causal factors experimentally.

---

# 21. Research Outputs

The project produces the following major outputs:

```text
✓ Cleaned Hinglish corpus
✓ 1,000-prompt Hinglish benchmark
✓ 4,000 controlled generations
✓ Independent Mistral evaluation
✓ Independent OLMo evaluation
✓ Inter-judge reliability analysis
✓ Evaluator robustness analysis
✓ Friedman statistical analysis
✓ Pairwise Wilcoxon + Holm correction
✓ Rank-biserial effect sizes
✓ Bootstrap confidence intervals
✓ E1–E10 automated error analysis
✓ Hinglish-specific linguistic metrics
✓ 9 final research tables
✓ 7 publication-oriented figures
✓ Separate 17K automatic evaluation
✓ Research paper
✓ Hinglish chatbot
```

---

# 22. Research Paper

The research manuscript associated with the project is available in:

```text
paper.md
```

The paper documents the methodology, benchmark construction, controlled generation process, independent evaluation, robustness analysis, statistical analysis, error analysis, Hinglish-specific findings, and model-level interpretation.

The paper provides the detailed methodological and analytical discussion, while this README provides a concise reproducibility-oriented overview.

---

# 23. License

This project is released under the **MIT License**.

See the repository license file for the complete license text.

---

# 24. Repository

GitHub repository:

**A Multi-Dimensional Benchmark of LLMs for Hindi-English Code-Mixed Generation**

https://github.com/kancharla-vyshnavi/A-Multi-Dimensional-Benchmark-of-LLMs-for-Hindi-English-Code-Mixed-Generation
