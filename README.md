**A Multi-Dimensional Benchmark of LLMs for Hindi-English Code-Mixed Generation**

A reproducible evaluation framework for comparing Large Language Models (LLMs) on Hindi-English code-mixed (Hinglish) text generation across multiple task categories and evaluation dimensions.

The repository contains the benchmark data, model-generation scripts, evaluation scripts, statistical analyses, human-validation results, error analysis, linguistic metrics, research tables, and final visualizations.

**Research Objective**

Hindi-English code-mixed text is common in informal communication, social media, customer interactions, and online content. Conventional language-generation metrics do not fully capture the quality of such mixed-language output.

This project therefore evaluates Hinglish generation using complementary dimensions:

Fluency

Code-mixing naturalness

Hindi grammaticality

Prompt adherence

Spelling consistency

Overall response quality

Script-level characteristics

Lexicon-based code-mixing characteristics

Error patterns

Pairwise model comparisons

Human–LLM judge agreement

**Models Evaluated**

**Model**

**Type**

HingGPT

Hinglish-focused language model

Phi-3.5-mini

General-purpose LLM

Qwen2.5-3B

General-purpose LLM

Qwen2.5-7B

General-purpose LLM

**Benchmark**

The controlled benchmark contains 34 prompts across 8 task categories:

**Category**

**Prompts**

Advice & Opinions

5

Captions & One-liners

4

Casual Conversation

5

Customer Support Dialogues

3

News & Explainers

4

Product Reviews

4

Social Media Posts

5

Storytelling

4

**Total**

34

Every evaluated model receives the same benchmark prompts to support controlled comparison.

**Evaluation Pipeline**

Benchmark prompts
       │
       ▼
Controlled generation
       │
       ▼
136 model responses
(34 prompts × 4 models)
       │
       ├──────────────► Independent LLM judge
       │                     │
       │                     ├─ Fluency
       │                     ├─ Code-mixing naturalness
       │                     ├─ Hindi grammar
       │                     ├─ Prompt adherence
       │                     ├─ Spelling consistency
       │                     └─ Overall quality
       │
       ├──────────────► Pairwise evaluation
       │
       ├──────────────► Human validation
       │
       ├──────────────► Error analysis
       │
       └──────────────► Hinglish-specific metrics
                              │
                              ▼
                     Statistical analysis
                              │
                              ▼
                    Research tables & figures

**Evaluation Components**

**1. Independent LLM Judge**

Generated responses were evaluated using an independent Mistral-7B-Instruct-v0.3 judge.

Each response was scored from 1–5 for:

Fluency

Code-mixing naturalness

Hindi grammar

Prompt adherence

Spelling consistency

Overall quality

The completed judge output contains valid scores for 129 of 136 responses. Missing scores are retained as missing rather than silently replaced.

**2. Pairwise Evaluation**

All six unique model pairs were evaluated in both comparison directions.

6 model pairs

34 benchmark prompts

2 comparison directions

408 pairwise comparisons

A swap-order consistency analysis is also included.

**3. Human Validation**

A validation subset of 48 responses was manually annotated:

12 responses per model

6 evaluation dimensions

Human scores were compared with the independent LLM judge using mean differences, Spearman correlation, and weighted quadratic Cohen's kappa.

Because the validation contains one human annotator, human–human inter-annotator agreement cannot be estimated.

**4. Missing-Data Sensitivity**

A conservative worst-case analysis assigns a score of 1/5 to missing overall scores and measures the resulting change in model-level means.

**5. Hinglish-Specific Analysis**

The repository includes:

Devanagari/Latin script proportions

Mixed-script rates

Roman-dominant rates

A transparent lexicon-based approximate Code-Mixing Index (CMI)

The CMI is treated as a heuristic indicator rather than a gold-standard language-identification system.

**6. Error Analysis**

The error-analysis pipeline covers:

E1 — English-dominant output

E2 — Hindi-dominant output

E3 — Unnatural code-switching

E4 — Grammatical error

E5 — Repetition

E6 — Prompt misunderstanding

E7 — Incomplete response

E8 — Spelling/transliteration error

E9 — Hallucination/factual error

E10 — Irrelevant response

Both aggregate and model-level error summaries are provided.

**Main Results**

**Independent Judge**

**Model**

**Mean Overall Score**

**Valid Responses**

HingGPT

2.464

28

Phi-3.5-mini

3.912

34

Qwen2.5-3B

3.697

33

Qwen2.5-7B

3.882

34

These results should be interpreted together with the dimension-level, pairwise, category-level, human-validation, and missing-data analyses rather than as a single standalone score.

**Pairwise Evaluation**

**Model**

**Wins**

**Losses**

**Ties**

**Win Rate**

HingGPT

24

176

4

0.1176

Phi-3.5-mini

157

47

0

0.7696

Qwen2.5-3B

97

105

2

0.4755

Qwen2.5-7B

126

76

2

0.6176

Statistical significance results and Holm-corrected comparisons are provided in outputs/statistical_analysis/.

**Repository Structure**

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
│
├── evaluate_hinggpt.py
├── evaluate_phi35_mini.py
├── evaluate_qwen25_3b.py
├── evaluate_qwen25_7b.py
│
├── analyze_category_robustness.py
├── analyze_error_patterns.py
├── analyze_human_agreement.py
├── analyze_pairwise.py
└── analyze_statistics.py

**What each major directory contains**

**Directory**

**Purpose**

data/benchmarks/

Benchmark prompts

src/evaluation/

Reusable evaluation components

outputs/controlled_generation/

Generated benchmark responses

outputs/independent_judge/

Independent LLM-judge results

outputs/pairwise_evaluation/

Pairwise comparison results

outputs/category_robustness/

Category-level analysis

outputs/error_analysis/

Error annotations and summaries

outputs/human_validation/

Human scores and human–LLM agreement

outputs/hinglish_metrics/

Script and code-mixing metrics

outputs/statistical_analysis/

Statistical tests and corrected comparisons

outputs/final_research_tables/

Paper-ready consolidated tables

outputs/final_charts/

Final figures

outputs/final_robustness/

Missing-data sensitivity results

FINAL_PROJECT_OUTPUTS/

Consolidated final research artifacts

**Installation**

**1. Clone the repository**

git clone https://github.com/kancharla-vyshnavi/A-Multi-Dimensional-Benchmark-of-LLMs-for-Hindi-English-Code-Mixed-Generation.git
cd A-Multi-Dimensional-Benchmark-of-LLMs-for-Hindi-English-Code-Mixed-Generation

**2. Create a virtual environment**

Windows PowerShell:

python -m venv .venv
.\.venv\Scripts\Activate.ps1

Linux/macOS:

python -m venv .venv
source .venv/bin/activate

**3. Install dependencies**

pip install -r requirements.txt

**Reproducing the Existing Analyses**

The repository already contains the generated outputs and research artifacts. Therefore, you do not need to regenerate model responses simply to inspect the reported results.

Useful analysis scripts include:

python analyze_category_robustness.py
python analyze_error_patterns.py
python analyze_human_agreement.py
python analyze_pairwise.py
python analyze_statistics.py

The corresponding results are stored under outputs/.

**Running Controlled Generation**

The main controlled-generation entry point is:

python run_controlled_generation.py

Model-specific evaluation scripts are also provided:

python evaluate_hinggpt.py
python evaluate_phi35_mini.py
python evaluate_qwen25_3b.py
python evaluate_qwen25_7b.py

Generation requires the relevant model checkpoints and a compatible Python/PyTorch environment. Model weights are not assumed to be downloaded automatically by this README.

**Running Pairwise Evaluation**

python run_pairwise_evaluation.py

The resulting artifacts are stored in:

outputs/pairwise_evaluation/

**Inspecting Final Results**

For the paper/research report, start with:

outputs/final_research_tables/
outputs/final_charts/
FINAL_PROJECT_OUTPUTS/

The consolidated research tables include:

Overall model results

Statistical comparisons

Category-level performance

Error analysis

The final charts include:

Independent-judge scores

Pairwise win rates

Category-level performance

Code-mixing index

**Reproducibility Notes**

To reproduce the analysis consistently:

Use the benchmark prompts provided in data/benchmarks/.

Use the same model checkpoints/configurations documented by the corresponding generation scripts.

Keep the evaluation dimensions and scoring scales unchanged.

Keep missing judge scores as missing unless explicitly performing the documented worst-case sensitivity analysis.

Use the provided output tables as the reference for the reported experiment.

Record hardware, Python, PyTorch, and Transformers versions when rerunning generation or judge inference.

**Statistical Analysis**

The project uses statistical analysis to complement descriptive results.

**The repository contains:**

outputs/statistical_analysis/friedman_results.csv
outputs/statistical_analysis/wilcoxon_holm_results.csv

Pairwise comparisons use Holm correction for multiple comparisons. Statistical significance should be interpreted together with effect sizes, sample size, and the practical magnitude of score differences.

**Limitations**

**Benchmark size:** 34 prompts cannot represent all Hindi-English code-mixed usage.

**Single independent LLM judge:** Automated scores depend partly on the behavior of the selected judge.

**Human validation size:** 48 responses and one human annotator limit the strength of human-validation conclusions.

**Human–LLM disagreement**: Human scores were substantially stricter than the independent LLM judge in the validation subset.

**Script-based analysis:** Latin-script output can represent Romanized Hindi as well as English.

**CMI limitation:** The lexicon-based Code-Mixing Index is a transparent heuristic, not gold-standard language identification.

**Pairwise order sensitivity:** Swap-order consistency varies across model pairs and should be considered when interpreting pairwise results.

**Research Outputs**

The repository is organized so that a reader can move from:

Benchmark
   ↓
Generation
   ↓
Automated evaluation
   ↓
Pairwise evaluation
   ↓
Human validation
   ↓
Error analysis
   ↓
Linguistic analysis
   ↓
Statistical analysis
   ↓
Final tables & figures

This separation makes the experimental workflow easier to inspect, reproduce, and extend.

**Citation**

If you use this benchmark, code, or reported evaluation artifacts in academic work, please cite the associated research work.

**License**

See the repository license and the licenses of the individual model checkpoints/datasets before redistribution or commercial use.
