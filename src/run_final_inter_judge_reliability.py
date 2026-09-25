import csv
import os
import math
from collections import defaultdict


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"C:\Users\vyshu\OneDrive\Desktop\HinglishLLM"

JUDGE_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "independent_judge_1000",
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "reliability_1000_FINAL",
)

MISTRAL_FILE = os.path.join(
    JUDGE_DIR,
    "independent_mistral_judge_1000_FINAL.csv",
)

OLMO_FILE = os.path.join(
    JUDGE_DIR,
    "independent_olmo2_1b_judge_1000_FINAL.csv",
)


# ============================================================
# DIMENSIONS
# ============================================================

DIMENSIONS = [
    "fluency",
    "code_mixing_naturalness",
    "hindi_grammar",
    "prompt_adherence",
    "spelling_consistency",
    "overall",
]


MODELS = [
    "HingGPT",
    "Phi-3.5-mini",
    "Qwen2.5-3B",
    "Qwen2.5-7B",
]


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nFile not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        return list(
            csv.DictReader(f)
        )


# ============================================================
# SAFE SCORE
# ============================================================

def get_score(row, dimension):

    value = row.get(
        dimension,
        "",
    )

    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return None

    try:

        score = float(value)

    except Exception:

        return None

    if score < 1 or score > 5:
        return None

    return score


# ============================================================
# RANK DATA
# ============================================================

def rankdata(values):

    """
    Average ranks for ties.
    """

    indexed = list(
        enumerate(values)
    )

    indexed.sort(
        key=lambda x: x[1]
    )

    ranks = [0.0] * len(values)

    i = 0

    while i < len(indexed):

        j = i

        while (
            j + 1 < len(indexed)
            and indexed[j + 1][1]
            == indexed[i][1]
        ):
            j += 1

        avg_rank = (
            (i + 1) + (j + 1)
        ) / 2.0

        for k in range(i, j + 1):

            original_index = indexed[k][0]

            ranks[original_index] = avg_rank

        i = j + 1

    return ranks


# ============================================================
# SPEARMAN
# ============================================================

def spearman(x, y):

    n = len(x)

    if n < 2:
        return float("nan")

    rx = rankdata(x)
    ry = rankdata(y)

    mean_x = sum(rx) / n
    mean_y = sum(ry) / n

    numerator = sum(
        (a - mean_x) * (b - mean_y)
        for a, b in zip(rx, ry)
    )

    denominator_x = math.sqrt(
        sum(
            (a - mean_x) ** 2
            for a in rx
        )
    )

    denominator_y = math.sqrt(
        sum(
            (b - mean_y) ** 2
            for b in ry
        )
    )

    denominator = (
        denominator_x
        * denominator_y
    )

    if denominator == 0:
        return float("nan")

    return numerator / denominator


# ============================================================
# QUADRATIC WEIGHTED COHEN'S KAPPA
# ============================================================

def quadratic_weighted_kappa(
    x,
    y,
):

    """
    Quadratic weighted Cohen's kappa
    for ratings 1-5.
    """

    n = len(x)

    if n == 0:
        return float("nan")

    categories = [
        1,
        2,
        3,
        4,
        5,
    ]

    # --------------------------------------------------------
    # Observed matrix
    # --------------------------------------------------------

    observed = [
        [0.0 for _ in categories]
        for _ in categories
    ]

    for a, b in zip(x, y):

        ai = int(a) - 1
        bi = int(b) - 1

        observed[ai][bi] += 1.0


    # --------------------------------------------------------
    # Expected matrix
    # --------------------------------------------------------

    row_totals = [
        sum(row)
        for row in observed
    ]

    col_totals = [
        sum(
            observed[i][j]
            for i in range(5)
        )
        for j in range(5)
    ]


    expected = [
        [
            (
                row_totals[i]
                * col_totals[j]
                / n
            )
            for j in range(5)
        ]
        for i in range(5)
    ]


    # --------------------------------------------------------
    # Quadratic weights
    # --------------------------------------------------------

    weights = [
        [
            (
                (i - j) ** 2
                / (4 ** 2)
            )
            for j in range(5)
        ]
        for i in range(5)
    ]


    observed_disagreement = 0.0
    expected_disagreement = 0.0

    for i in range(5):

        for j in range(5):

            observed_disagreement += (
                weights[i][j]
                * observed[i][j]
            )

            expected_disagreement += (
                weights[i][j]
                * expected[i][j]
            )


    if expected_disagreement == 0:

        return float("nan")


    return (
        1.0
        - (
            observed_disagreement
            / expected_disagreement
        )
    )


# ============================================================
# PREPARE JUDGE LOOKUPS
# ============================================================

print()
print("=" * 70)
print("FINAL INTER-JUDGE RELIABILITY")
print("=" * 70)
print()


print("Loading Mistral Judge-1...")

mistral_rows = load_csv(
    MISTRAL_FILE
)

print(
    f"Mistral rows: {len(mistral_rows)}"
)


print()
print("Loading OLMo Judge-2...")

olmo_rows = load_csv(
    OLMO_FILE
)

print(
    f"OLMo rows: {len(olmo_rows)}"
)


# ============================================================
# BUILD LOOKUPS
# ============================================================

mistral_lookup = {}
olmo_lookup = {}


for row in mistral_rows:

    key = (
        row.get(
            "row_index",
            "",
        ).strip(),
        row.get(
            "model",
            "",
        ).strip(),
    )

    mistral_lookup[key] = row


for row in olmo_rows:

    key = (
        row.get(
            "row_index",
            "",
        ).strip(),
        row.get(
            "model",
            "",
        ).strip(),
    )

    olmo_lookup[key] = row


# ============================================================
# CHECK KEY OVERLAP
# ============================================================

mistral_keys = set(
    mistral_lookup.keys()
)

olmo_keys = set(
    olmo_lookup.keys()
)

common_keys = (
    mistral_keys
    & olmo_keys
)


print()
print(
    f"Mistral unique keys: {len(mistral_keys)}"
)

print(
    f"OLMo unique keys:    {len(olmo_keys)}"
)

print(
    f"Common keys:         {len(common_keys)}"
)


# ============================================================
# COLLECT PAIRS
# ============================================================

all_results = []

model_results = []


# ============================================================
# OVERALL + MODEL LEVEL
# ============================================================

for model in (
    ["ALL"]
    + MODELS
):

    if model == "ALL":

        keys = sorted(
            common_keys
        )

    else:

        keys = sorted(
            key
            for key in common_keys
            if key[1] == model
        )


    for dimension in DIMENSIONS:

        x = []
        y = []

        valid_keys = []

        for key in keys:

            mistral_row = (
                mistral_lookup[key]
            )

            olmo_row = (
                olmo_lookup[key]
            )

            mistral_score = get_score(
                mistral_row,
                dimension,
            )

            olmo_score = get_score(
                olmo_row,
                dimension,
            )


            # ------------------------------------------------
            # Pairwise complete-case handling
            # ------------------------------------------------

            if (
                mistral_score is None
                or olmo_score is None
            ):

                continue


            x.append(
                mistral_score
            )

            y.append(
                olmo_score
            )

            valid_keys.append(
                key
            )


        n = len(x)


        if n >= 2:

            kappa = quadratic_weighted_kappa(
                x,
                y,
            )

            rho = spearman(
                x,
                y,
            )

        else:

            kappa = float("nan")
            rho = float("nan")


        result = {
            "scope": model,
            "dimension": dimension,
            "n": n,
            "quadratic_weighted_kappa": kappa,
            "spearman_rho": rho,
        }


        all_results.append(
            result
        )


        if model != "ALL":

            model_results.append(
                result
            )


# ============================================================
# SAVE OVERALL SUMMARY
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "final_inter_judge_reliability_summary.csv",
)


with open(
    summary_file,
    "w",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "scope",
            "dimension",
            "n",
            "quadratic_weighted_kappa",
            "spearman_rho",
        ],
    )

    writer.writeheader()

    writer.writerows(
        all_results
    )


# ============================================================
# SAVE MODEL-LEVEL RESULTS
# ============================================================

model_file = os.path.join(
    OUTPUT_DIR,
    "final_inter_judge_reliability_by_model.csv",
)


with open(
    model_file,
    "w",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "scope",
            "dimension",
            "n",
            "quadratic_weighted_kappa",
            "spearman_rho",
        ],
    )

    writer.writeheader()

    writer.writerows(
        model_results
    )


# ============================================================
# MISSING DATA AUDIT
# ============================================================

missing_results = []


for model in (
    ["ALL"]
    + MODELS
):

    if model == "ALL":

        keys = sorted(
            common_keys
        )

    else:

        keys = sorted(
            key
            for key in common_keys
            if key[1] == model
        )


    for dimension in DIMENSIONS:

        mistral_missing = 0
        olmo_missing = 0
        pair_missing = 0

        for key in keys:

            mistral_score = get_score(
                mistral_lookup[key],
                dimension,
            )

            olmo_score = get_score(
                olmo_lookup[key],
                dimension,
            )


            if mistral_score is None:

                mistral_missing += 1


            if olmo_score is None:

                olmo_missing += 1


            if (
                mistral_score is None
                or olmo_score is None
            ):

                pair_missing += 1


        missing_results.append(
            {
                "scope": model,
                "dimension": dimension,
                "total_common_rows": len(keys),
                "mistral_missing": mistral_missing,
                "olmo_missing": olmo_missing,
                "pairwise_excluded": pair_missing,
            }
        )


missing_file = os.path.join(
    OUTPUT_DIR,
    "final_inter_judge_missing_data_audit.csv",
)


with open(
    missing_file,
    "w",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "scope",
            "dimension",
            "total_common_rows",
            "mistral_missing",
            "olmo_missing",
            "pairwise_excluded",
        ],
    )

    writer.writeheader()

    writer.writerows(
        missing_results
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("FINAL RELIABILITY RESULTS — OVERALL")
print("=" * 70)

for result in all_results:

    if result["scope"] != "ALL":
        continue

    print()

    print(
        f"{result['dimension']}"
    )

    print(
        f"  N      = {result['n']}"
    )

    print(
        "  QWK    = "
        f"{result['quadratic_weighted_kappa']:.6f}"
    )

    print(
        "  Spearman = "
        f"{result['spearman_rho']:.6f}"
    )


# ============================================================
# MODEL RESULTS
# ============================================================

print()
print("=" * 70)
print("MODEL-LEVEL RELIABILITY")
print("=" * 70)


for model in MODELS:

    print()
    print(
        f"--- {model} ---"
    )

    for result in model_results:

        if result["scope"] != model:
            continue

        print(
            f"{result['dimension']}: "
            f"N={result['n']}, "
            f"QWK={result['quadratic_weighted_kappa']:.6f}, "
            f"rho={result['spearman_rho']:.6f}"
        )


# ============================================================
# OUTPUT PATHS
# ============================================================

print()
print("=" * 70)
print("FILES CREATED")
print("=" * 70)

print()
print(
    "Overall reliability:"
)

print(
    summary_file
)

print()
print(
    "Model-level reliability:"
)

print(
    model_file
)

print()
print(
    "Missing-data audit:"
)

print(
    missing_file
)

print()
print("=" * 70)
print(
    "SUCCESS — FINAL INTER-JUDGE RELIABILITY COMPLETE"
)
print("=" * 70)