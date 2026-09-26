import csv
import os
import math
import random
from itertools import combinations


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
    "statistical_analysis_1000_FINAL",
)

MISTRAL_FILE = os.path.join(
    JUDGE_DIR,
    "independent_mistral_judge__FINAL.csv",
)

OLMO_FILE = os.path.join(
    JUDGE_DIR,
    "independent_olmo2_1b_judge__CORRECTED_FINAL.csv",
)


# ============================================================
# SETTINGS
# ============================================================

MODELS = [
    "HingGPT",
    "Phi-3.5-mini",
    "Qwen2.5-3B",
    "Qwen2.5-7B",
]

DIMENSIONS = [
    "fluency",
    "code_mixing_naturalness",
    "hindi_grammar",
    "prompt_adherence",
    "spelling_consistency",
    "overall",
]

JUDGES = [
    "Mistral-7B-Instruct-v0.3",
    "OLMo-2-0425-1B-Instruct",
]

N_BOOTSTRAP = 2000

RANDOM_SEED = 42


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# CSV
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


def save_csv(
    path,
    rows,
    fieldnames,
):

    with open(
        path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# SCORE
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
# MEAN
# ============================================================

def mean(values):

    if not values:

        return float("nan")

    return sum(values) / len(values)


# ============================================================
# MEDIAN
# ============================================================

def median(values):

    if not values:

        return float("nan")

    values = sorted(values)

    n = len(values)

    middle = n // 2

    if n % 2 == 1:

        return values[middle]

    return (
        values[middle - 1]
        + values[middle]
    ) / 2.0


# ============================================================
# RANKS WITH TIES
# ============================================================

def rankdata(values):

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
            (i + 1)
            + (j + 1)
        ) / 2.0

        for k in range(
            i,
            j + 1,
        ):

            original_index = (
                indexed[k][0]
            )

            ranks[
                original_index
            ] = avg_rank

        i = j + 1

    return ranks


# ============================================================
# NORMAL CDF
# ============================================================

def normal_cdf(x):

    return (
        0.5
        * (
            1.0
            + math.erf(
                x / math.sqrt(2.0)
            )
        )
    )


# ============================================================
# WILCOXON SIGNED-RANK
# ============================================================

def wilcoxon_signed_rank(
    differences
):

    """
    Paired Wilcoxon signed-rank test.

    Zero differences are removed.
    Exact permutation is used when n <= 20.
    Normal approximation with continuity correction
    is used for larger n.
    """

    nonzero = [
        d
        for d in differences
        if d != 0
    ]

    n = len(nonzero)

    if n == 0:

        return {
            "n": 0,
            "statistic": 0.0,
            "p_value": 1.0,
        }


    absolute = [
        abs(d)
        for d in nonzero
    ]

    ranks = rankdata(
        absolute
    )


    positive_rank_sum = sum(
        rank
        for rank, diff
        in zip(
            ranks,
            nonzero,
        )
        if diff > 0
    )


    negative_rank_sum = sum(
        rank
        for rank, diff
        in zip(
            ranks,
            nonzero,
        )
        if diff < 0
    )


    statistic = min(
        positive_rank_sum,
        negative_rank_sum,
    )


    # --------------------------------------------------------
    # Exact test for small n
    # --------------------------------------------------------

    if n <= 20:

        total_rank = sum(ranks)

        count = 0

        total = 2 ** n

        observed = (
            positive_rank_sum
        )

        for mask in range(total):

            rank_sum = 0.0

            for i in range(n):

                if mask & (
                    1 << i
                ):

                    rank_sum += ranks[i]

            if (
                rank_sum
                <= observed + 1e-12
                or
                rank_sum
                >= total_rank
                - observed
                - 1e-12
            ):

                count += 1

        p_value = count / total

        return {
            "n": n,
            "statistic": statistic,
            "p_value": min(
                1.0,
                p_value,
            ),
        }


    # --------------------------------------------------------
    # Normal approximation
    # --------------------------------------------------------

    mean_w = (
        n * (n + 1)
    ) / 4.0

    variance_w = (
        n
        * (n + 1)
        * (2 * n + 1)
    ) / 24.0


    if variance_w <= 0:

        return {
            "n": n,
            "statistic": statistic,
            "p_value": 1.0,
        }


    # Continuity correction.
    if (
        positive_rank_sum
        < mean_w
    ):

        z = (
            positive_rank_sum
            + 0.5
            - mean_w
        ) / math.sqrt(
            variance_w
        )

    else:

        z = (
            positive_rank_sum
            - 0.5
            - mean_w
        ) / math.sqrt(
            variance_w
        )


    p_value = (
        2.0
        * (
            1.0
            - normal_cdf(
                abs(z)
            )
        )
    )


    return {
        "n": n,
        "statistic": statistic,
        "p_value": min(
            1.0,
            p_value,
        ),
    }


# ============================================================
# HOLM CORRECTION
# ============================================================

def holm_adjust(
    p_values
):

    """
    Holm-Bonferroni adjusted p-values.
    """

    m = len(p_values)

    indexed = sorted(
        enumerate(p_values),
        key=lambda x: x[1],
    )

    adjusted = [
        0.0
        for _ in p_values
    ]

    previous = 0.0

    for rank, (
        original_index,
        p_value,
    ) in enumerate(
        indexed
    ):

        adjusted_value = (
            (m - rank)
            * p_value
        )

        adjusted_value = max(
            adjusted_value,
            previous,
        )

        adjusted_value = min(
            adjusted_value,
            1.0,
        )

        adjusted[
            original_index
        ] = adjusted_value

        previous = adjusted_value

    return adjusted


# ============================================================
# RANK-BISERIAL EFFECT SIZE
# ============================================================

def rank_biserial(
    differences
):

    nonzero = [
        d
        for d in differences
        if d != 0
    ]

    n = len(nonzero)

    if n == 0:

        return float("nan")


    absolute = [
        abs(d)
        for d in nonzero
    ]

    ranks = rankdata(
        absolute
    )


    positive = sum(
        rank
        for rank, diff
        in zip(
            ranks,
            nonzero,
        )
        if diff > 0
    )


    negative = sum(
        rank
        for rank, diff
        in zip(
            ranks,
            nonzero,
        )
        if diff < 0
    )


    total = positive + negative

    if total == 0:

        return 0.0

    return (
        positive - negative
    ) / total


# ============================================================
# FRIEDMAN TEST
# ============================================================

def friedman_test(
    samples
):

    """
    Friedman test for k related samples.

    samples = [model1_values, model2_values, ...]
    """

    k = len(samples)

    if k < 2:

        return {
            "n": 0,
            "chi_square": float("nan"),
            "p_value": float("nan"),
        }


    n = min(
        len(sample)
        for sample in samples
    )


    if n < 2:

        return {
            "n": n,
            "chi_square": float("nan"),
            "p_value": float("nan"),
        }


    # Truncate to common paired observations.
    samples = [
        sample[:n]
        for sample in samples
    ]


    rank_sums = [
        0.0
        for _ in range(k)
    ]


    for i in range(n):

        values = [
            samples[j][i]
            for j in range(k)
        ]

        ranks = rankdata(
            values
        )

        for j in range(k):

            rank_sums[j] += (
                ranks[j]
            )


    chi_square = (
        12.0
        / (
            n * k * (k + 1)
        )
        * sum(
            rank_sum ** 2
            for rank_sum
            in rank_sums
        )
        - 3.0
        * n
        * (k + 1)
    )


    # Chi-square survival function
    # with df = k-1.
    #
    # For k=4, df=3.
    #
    # We calculate the survival probability
    # using the regularized gamma function
    # implementation below.

    df = k - 1

    p_value = chi_square_sf(
        chi_square,
        df,
    )


    return {
        "n": n,
        "chi_square": chi_square,
        "p_value": p_value,
    }


# ============================================================
# CHI-SQUARE SURVIVAL FUNCTION
# ============================================================

def gammaincc(
    a,
    x,
):

    """
    Regularized upper incomplete gamma function.

    Numerical implementation using:
    - series expansion for x < a+1
    - continued fraction otherwise
    """

    if x < 0 or a <= 0:

        return float("nan")

    if x == 0:

        return 1.0


    ITMAX = 200
    EPS = 3e-14
    FPMIN = 1e-300


    # --------------------------------------------------------
    # Series for lower gamma
    # --------------------------------------------------------

    if x < a + 1.0:

        ap = a

        summation = 1.0 / a

        delta = summation

        for _ in range(
            ITMAX
        ):

            ap += 1.0

            delta *= (
                x / ap
            )

            summation += delta

            if abs(delta) < (
                abs(summation)
                * EPS
            ):

                break

        log_term = (
            -x
            + a * math.log(x)
            - math.lgamma(a)
        )

        lower = (
            summation
            * math.exp(log_term)
        )

        return max(
            0.0,
            min(
                1.0,
                1.0 - lower,
            ),
        )


    # --------------------------------------------------------
    # Continued fraction for upper gamma
    # --------------------------------------------------------

    b = (
        x
        + 1.0
        - a
    )

    c = 1.0 / FPMIN

    d = 1.0 / b

    h = d


    for i in range(
        1,
        ITMAX + 1,
    ):

        an = (
            -i
            * (i - a)
        )

        b += 2.0

        d = (
            an * d
            + b
        )

        if abs(d) < FPMIN:
            d = FPMIN

        c = (
            b
            + an / c
        )

        if abs(c) < FPMIN:
            c = FPMIN

        d = 1.0 / d

        delta = d * c

        h *= delta

        if abs(
            delta - 1.0
        ) < EPS:

            break


    log_term = (
        -x
        + a * math.log(x)
        - math.lgamma(a)
    )

    result = (
        math.exp(log_term)
        * h
    )

    return max(
        0.0,
        min(
            1.0,
            result,
        ),
    )


def chi_square_sf(
    x,
    df,
):

    if x < 0:
        return 1.0

    return gammaincc(
        df / 2.0,
        x / 2.0,
    )


# ============================================================
# BOOTSTRAP CI
# ============================================================

def bootstrap_mean_difference_ci(
    x,
    y,
    n_bootstrap=2000,
    seed=42,
):

    """
    Bootstrap 95% CI for paired mean difference:

        mean(x - y)
    """

    pairs = [
        (
            a,
            b,
        )
        for a, b in zip(x, y)
        if a is not None
        and b is not None
    ]


    n = len(pairs)

    if n == 0:

        return (
            float("nan"),
            float("nan"),
            float("nan"),
        )


    differences = [
        a - b
        for a, b in pairs
    ]


    observed = mean(
        differences
    )


    rng = random.Random(
        seed
    )


    bootstrap_values = []


    for _ in range(
        n_bootstrap
    ):

        sample = [
            differences[
                rng.randrange(n)
            ]
            for _ in range(n)
        ]

        bootstrap_values.append(
            mean(sample)
        )


    bootstrap_values.sort()


    lower_index = int(
        0.025
        * n_bootstrap
    )

    upper_index = int(
        0.975
        * n_bootstrap
    )


    lower_index = max(
        0,
        min(
            lower_index,
            n_bootstrap - 1,
        ),
    )

    upper_index = max(
        0,
        min(
            upper_index,
            n_bootstrap - 1,
        ),
    )


    return (
        observed,
        bootstrap_values[
            lower_index
        ],
        bootstrap_values[
            upper_index
        ],
    )


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("FINAL STATISTICAL ANALYSIS")
print("=" * 70)
print()


print(
    "Loading corrected Mistral Judge-1..."
)

mistral_rows = load_csv(
    MISTRAL_FILE
)

print(
    f"Mistral rows: {len(mistral_rows)}"
)


print()
print(
    "Loading final OLMo Judge-2..."
)

olmo_rows = load_csv(
    OLMO_FILE
)

print(
    f"OLMo rows: {len(olmo_rows)}"
)


# ============================================================
# LOOKUPS
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


common_keys = (
    set(mistral_lookup)
    & set(olmo_lookup)
)


print()
print(
    f"Common keys: {len(common_keys)}"
)


# ============================================================
# TWO-JUDGE MODEL MEANS
# ============================================================

mean_results = []


for model in MODELS:

    model_keys = [
        key
        for key in common_keys
        if key[1] == model
    ]


    for dimension in DIMENSIONS:

        mistral_values = []

        olmo_values = []

        combined_values = []


        for key in model_keys:

            m = get_score(
                mistral_lookup[key],
                dimension,
            )

            o = get_score(
                olmo_lookup[key],
                dimension,
            )


            if m is not None:

                mistral_values.append(m)


            if o is not None:

                olmo_values.append(o)


            if (
                m is not None
                and o is not None
            ):

                combined_values.append(
                    (m + o) / 2.0
                )


        mean_results.append(
            {
                "model": model,
                "dimension": dimension,
                "mistral_n": len(
                    mistral_values
                ),
                "mistral_mean": mean(
                    mistral_values
                ),
                "olmo_n": len(
                    olmo_values
                ),
                "olmo_mean": mean(
                    olmo_values
                ),
                "combined_n": len(
                    combined_values
                ),
                "combined_mean": mean(
                    combined_values
                ),
            }
        )


# ============================================================
# SAVE MODEL MEANS
# ============================================================

means_file = os.path.join(
    OUTPUT_DIR,
    "final_model_means_two_judge.csv",
)


save_csv(
    means_file,
    mean_results,
    [
        "model",
        "dimension",
        "mistral_n",
        "mistral_mean",
        "olmo_n",
        "olmo_mean",
        "combined_n",
        "combined_mean",
    ],
)


# ============================================================
# FRIEDMAN TEST
# ============================================================

friedman_results = []


for dimension in DIMENSIONS:

    # --------------------------------------------------------
    # Build complete paired observations.
    # A prompt is retained only if BOTH judges
    # have scores for ALL four models.
    # --------------------------------------------------------

    row_indices = sorted(
        set(
            key[0]
            for key in common_keys
        )
    )


    complete_prompts = []


    for row_index in row_indices:

        complete = True

        for model in MODELS:

            key = (
                row_index,
                model,
            )


            if key not in common_keys:

                complete = False
                break


            m = get_score(
                mistral_lookup[key],
                dimension,
            )

            o = get_score(
                olmo_lookup[key],
                dimension,
            )


            if (
                m is None
                or o is None
            ):

                complete = False
                break


        if complete:

            complete_prompts.append(
                row_index
            )


    # --------------------------------------------------------
    # Combined two-judge score
    # --------------------------------------------------------

    model_samples = {
        model: []
        for model in MODELS
    }


    for row_index in complete_prompts:

        for model in MODELS:

            key = (
                row_index,
                model,
            )


            m = get_score(
                mistral_lookup[key],
                dimension,
            )

            o = get_score(
                olmo_lookup[key],
                dimension,
            )


            combined = (
                m + o
            ) / 2.0


            model_samples[
                model
            ].append(
                combined
            )


    samples = [
        model_samples[model]
        for model in MODELS
    ]


    result = friedman_test(
        samples
    )


    friedman_results.append(
        {
            "dimension": dimension,
            "n_complete_prompts": result[
                "n"
            ],
            "chi_square": result[
                "chi_square"
            ],
            "p_value": result[
                "p_value"
            ],
        }
    )


# ============================================================
# SAVE FRIEDMAN
# ============================================================

friedman_file = os.path.join(
    OUTPUT_DIR,
    "final_friedman_results.csv",
)


save_csv(
    friedman_file,
    friedman_results,
    [
        "dimension",
        "n_complete_prompts",
        "chi_square",
        "p_value",
    ],
)


# ============================================================
# PAIRWISE WILCOXON
# ============================================================

pairwise_results = []


for dimension in DIMENSIONS:

    for model_a, model_b in combinations(
        MODELS,
        2,
    ):

        # ----------------------------------------------------
        # Paired complete observations
        # ----------------------------------------------------

        row_indices = sorted(
            set(
                key[0]
                for key in common_keys
            )
        )


        values_a = []
        values_b = []


        for row_index in row_indices:

            key_a = (
                row_index,
                model_a,
            )

            key_b = (
                row_index,
                model_b,
            )


            if (
                key_a not in common_keys
                or key_b not in common_keys
            ):

                continue


            ma = get_score(
                mistral_lookup[key_a],
                dimension,
            )

            oa = get_score(
                olmo_lookup[key_a],
                dimension,
            )

            mb = get_score(
                mistral_lookup[key_b],
                dimension,
            )

            ob = get_score(
                olmo_lookup[key_b],
                dimension,
            )


            if (
                ma is None
                or oa is None
                or mb is None
                or ob is None
            ):

                continue


            combined_a = (
                ma + oa
            ) / 2.0


            combined_b = (
                mb + ob
            ) / 2.0


            values_a.append(
                combined_a
            )

            values_b.append(
                combined_b
            )


        differences = [
            a - b
            for a, b
            in zip(
                values_a,
                values_b,
            )
        ]


        test = wilcoxon_signed_rank(
            differences
        )


        effect = rank_biserial(
            differences
        )


        delta = mean(
            differences
        )


        pairwise_results.append(
            {
                "dimension": dimension,
                "model_a": model_a,
                "model_b": model_b,
                "n": len(
                    differences
                ),
                "wilcoxon_statistic": test[
                    "statistic"
                ],
                "p_value": test[
                    "p_value"
                ],
                "mean_difference_a_minus_b": delta,
                "rank_biserial": effect,
            }
        )


# ============================================================
# HOLM CORRECTION
# ============================================================

for dimension in DIMENSIONS:

    indices = [
        i
        for i, result
        in enumerate(
            pairwise_results
        )
        if result[
            "dimension"
        ] == dimension
    ]


    p_values = [
        pairwise_results[i][
            "p_value"
        ]
        for i in indices
    ]


    adjusted = holm_adjust(
        p_values
    )


    for i, adjusted_p in zip(
        indices,
        adjusted,
    ):

        pairwise_results[i][
            "holm_adjusted_p"
        ] = adjusted_p


        pairwise_results[i][
            "significant_holm_0_05"
        ] = (
            "yes"
            if adjusted_p < 0.05
            else "no"
        )


# ============================================================
# SAVE PAIRWISE RESULTS
# ============================================================

pairwise_file = os.path.join(
    OUTPUT_DIR,
    "final_wilcoxon_pairwise_holm.csv",
)


save_csv(
    pairwise_file,
    pairwise_results,
    [
        "dimension",
        "model_a",
        "model_b",
        "n",
        "wilcoxon_statistic",
        "p_value",
        "holm_adjusted_p",
        "significant_holm_0_05",
        "mean_difference_a_minus_b",
        "rank_biserial",
    ],
)


# ============================================================
# BOOTSTRAP CIs
# ============================================================

bootstrap_results = []


for dimension in DIMENSIONS:

    for model_a, model_b in combinations(
        MODELS,
        2,
    ):

        row_indices = sorted(
            set(
                key[0]
                for key in common_keys
            )
        )


        values_a = []
        values_b = []


        for row_index in row_indices:

            key_a = (
                row_index,
                model_a,
            )

            key_b = (
                row_index,
                model_b,
            )


            if (
                key_a not in common_keys
                or key_b not in common_keys
            ):

                continue


            ma = get_score(
                mistral_lookup[key_a],
                dimension,
            )

            oa = get_score(
                olmo_lookup[key_a],
                dimension,
            )

            mb = get_score(
                mistral_lookup[key_b],
                dimension,
            )

            ob = get_score(
                olmo_lookup[key_b],
                dimension,
            )


            if (
                ma is None
                or oa is None
                or mb is None
                or ob is None
            ):

                continue


            values_a.append(
                (
                    ma + oa
                ) / 2.0
            )

            values_b.append(
                (
                    mb + ob
                ) / 2.0
            )


        observed, lower, upper = (
            bootstrap_mean_difference_ci(
                values_a,
                values_b,
                n_bootstrap=N_BOOTSTRAP,
                seed=RANDOM_SEED,
            )
        )


        bootstrap_results.append(
            {
                "dimension": dimension,
                "model_a": model_a,
                "model_b": model_b,
                "n": len(values_a),
                "mean_difference_a_minus_b": observed,
                "bootstrap_ci_95_lower": lower,
                "bootstrap_ci_95_upper": upper,
                "bootstrap_iterations": N_BOOTSTRAP,
                "seed": RANDOM_SEED,
            }
        )


# ============================================================
# SAVE BOOTSTRAP
# ============================================================

bootstrap_file = os.path.join(
    OUTPUT_DIR,
    "final_bootstrap_95ci_pairwise.csv",
)


save_csv(
    bootstrap_file,
    bootstrap_results,
    [
        "dimension",
        "model_a",
        "model_b",
        "n",
        "mean_difference_a_minus_b",
        "bootstrap_ci_95_lower",
        "bootstrap_ci_95_upper",
        "bootstrap_iterations",
        "seed",
    ],
)


# ============================================================
# PRINT MODEL MEANS
# ============================================================

print()
print("=" * 70)
print("FINAL TWO-JUDGE MODEL MEANS")
print("=" * 70)


for dimension in DIMENSIONS:

    print()
    print(
        f"--- {dimension} ---"
    )


    for model in MODELS:

        row = next(
            r
            for r in mean_results
            if (
                r["model"] == model
                and
                r["dimension"]
                == dimension
            )
        )


        print(
            f"{model}: "
            f"Mistral={row['mistral_mean']:.4f}, "
            f"OLMo={row['olmo_mean']:.4f}, "
            f"Combined={row['combined_mean']:.4f}"
        )


# ============================================================
# PRINT FRIEDMAN
# ============================================================

print()
print("=" * 70)
print("FINAL FRIEDMAN TESTS")
print("=" * 70)


for row in friedman_results:

    print()
    print(
        f"{row['dimension']}: "
        f"N={row['n_complete_prompts']}, "
        f"chi2={row['chi_square']:.6f}, "
        f"p={row['p_value']:.8g}"
    )


# ============================================================
# PRINT PAIRWISE
# ============================================================

print()
print("=" * 70)
print("FINAL PAIRWISE WILCOXON + HOLM")
print("=" * 70)


for row in pairwise_results:

    print(
        f"{row['dimension']} | "
        f"{row['model_a']} vs "
        f"{row['model_b']} | "
        f"N={row['n']} | "
        f"p={row['p_value']:.8g} | "
        f"Holm={row['holm_adjusted_p']:.8g} | "
        f"RB={row['rank_biserial']:.6f} | "
        f"Delta={row['mean_difference_a_minus_b']:.6f}"
    )


# ============================================================
# OUTPUT FILES
# ============================================================

print()
print("=" * 70)
print("FILES CREATED")
print("=" * 70)

print()
print(
    "Model means:"
)

print(
    means_file
)

print()
print(
    "Friedman:"
)

print(
    friedman_file
)

print()
print(
    "Wilcoxon + Holm:"
)

print(
    pairwise_file
)

print()
print(
    "Bootstrap 95% CI:"
)

print(
    bootstrap_file
)


# ============================================================
# SUCCESS
# ============================================================

print()
print("=" * 70)
print(
    "SUCCESS — FINAL STATISTICAL ANALYSIS COMPLETE"
)
print("=" * 70)