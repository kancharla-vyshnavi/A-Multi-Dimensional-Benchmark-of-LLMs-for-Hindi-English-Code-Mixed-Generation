import os
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = r".\outputs\error_analysis\manual_error_annotation.csv"
OUTPUT_FILE = r".\outputs\error_analysis\manual_error_annotation.csv"


# ============================================================
# ERROR DEFINITIONS
# ============================================================

ERRORS = {
    "E1": "English-dominant output",
    "E2": "Hindi-dominant output",
    "E3": "Unnatural code-switching",
    "E4": "Grammatical error",
    "E5": "Repetition",
    "E6": "Prompt misunderstanding",
    "E7": "Incomplete response",
    "E8": "Spelling/transliteration error",
    "E9": "Hallucination/factual error",
    "E10": "Irrelevant response",
}


# ============================================================
# COLUMN MAPPING
# ============================================================

ERROR_COLUMNS = {
    "E1": "E1_English_dominant",
    "E2": "E2_Hindi_dominant",
    "E3": "E3_Unnatural_code_switching",
    "E4": "E4_Grammatical_error",
    "E5": "E5_Repetition",
    "E6": "E6_Prompt_misunderstanding",
    "E7": "E7_Incomplete_response",
    "E8": "E8_Spelling_transliteration_error",
    "E9": "E9_Hallucination_factual_error",
    "E10": "E10_Irrelevant_response",
}


# ============================================================
# CHECK INPUT FILE
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nFile not found:\n{INPUT_FILE}\n"
        "\nPlease run manual_error_analysis.py first."
    )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)


# ============================================================
# FIX DATA TYPES
# ============================================================
# Important:
# Blank columns may be read by pandas as float64.
# We explicitly convert text columns to string and
# error-label columns to integers.

TEXT_COLUMNS = [
    "Final_Error_Category",
    "Manual_Notes",
    "Review_Status",
]

for column in TEXT_COLUMNS:

    if column not in df.columns:
        df[column] = ""

    df[column] = df[column].fillna("").astype(str)


for code, column in ERROR_COLUMNS.items():

    if column not in df.columns:
        df[column] = 0

    df[column] = (
        pd.to_numeric(df[column], errors="coerce")
        .fillna(0)
        .astype(int)
    )


# ============================================================
# MAIN HEADER
# ============================================================

print("=" * 75)
print("HINGLISH ERROR ANALYSIS - MANUAL REVIEW")
print("=" * 75)

print(f"\nTotal responses: {len(df)}")


# ============================================================
# REVIEW LOOP
# ============================================================

for i in range(len(df)):

    # --------------------------------------------------------
    # Skip already reviewed responses
    # --------------------------------------------------------

    status = str(
        df.loc[i, "Review_Status"]
    ).strip().upper()

    if status == "REVIEWED":
        continue


    # --------------------------------------------------------
    # Get response information
    # --------------------------------------------------------

    model = df.loc[i, "model"]
    sample = df.loc[i, "sample"]
    category = df.loc[i, "category"]
    prompt = df.loc[i, "prompt"]
    response = df.loc[i, "response"]
    automatic = df.loc[
        i,
        "Automatic_Screening_Reference"
    ]


    # --------------------------------------------------------
    # Display response
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print(f"RESPONSE {i + 1} / {len(df)}")
    print("=" * 75)

    print(f"\nMODEL    : {model}")
    print(f"SAMPLE   : {sample}")
    print(f"CATEGORY : {category}")


    print("\nPROMPT")
    print("-" * 75)
    print(prompt)


    print("\nGENERATED RESPONSE")
    print("-" * 75)
    print(response)


    print("\nAUTOMATIC SCREENING")
    print("-" * 75)

    if (
        pd.isna(automatic)
        or str(automatic).strip() == ""
        or str(automatic).lower() == "nan"
    ):
        print("No automatic error detected")
    else:
        print(automatic)


    # --------------------------------------------------------
    # Show error definitions
    # --------------------------------------------------------

    print("\nERROR CATEGORIES")
    print("-" * 75)

    for code, description in ERRORS.items():
        print(f"{code:<4} - {description}")


    print("-" * 75)


    # ========================================================
    # GET FINAL ERROR LABEL
    # ========================================================

    while True:

        final_error = input(
            "\nEnter final error code(s) "
            "(example: E4 or E3,E8 or NONE): "
        ).strip().upper()


        # ----------------------------------------------------
        # Empty input
        # ----------------------------------------------------

        if final_error == "":
            print(
                "Please enter NONE or one/more codes "
                "from E1-E10."
            )
            continue


        # ----------------------------------------------------
        # No error
        # ----------------------------------------------------

        if final_error == "NONE":
            codes = []
            break


        # ----------------------------------------------------
        # Multiple error codes
        # ----------------------------------------------------

        codes = [
            code.strip()
            for code in final_error.split(",")
            if code.strip()
        ]


        # ----------------------------------------------------
        # Validate codes
        # ----------------------------------------------------

        invalid_codes = [
            code
            for code in codes
            if code not in ERRORS
        ]


        if invalid_codes:

            print(
                f"\nInvalid code(s): {invalid_codes}"
            )

            print(
                "Valid codes are E1, E2, E3, ..., E10."
            )

            continue


        # ----------------------------------------------------
        # Remove duplicates while preserving order
        # ----------------------------------------------------

        codes = list(dict.fromkeys(codes))

        final_error = ",".join(codes)

        break


    # ========================================================
    # NOTES
    # ========================================================

    notes = input(
        "Notes (optional, press Enter to skip): "
    ).strip()


    # ========================================================
    # SAVE E1-E10 LABELS
    # ========================================================

    for code, column in ERROR_COLUMNS.items():

        if code in codes:
            df.loc[i, column] = 1
        else:
            df.loc[i, column] = 0


    # ========================================================
    # SAVE FINAL CATEGORY
    # ========================================================

    if len(codes) == 0:

        df.loc[
            i,
            "Final_Error_Category"
        ] = "NONE"

    else:

        df.loc[
            i,
            "Final_Error_Category"
        ] = ",".join(codes)


    # ========================================================
    # SAVE NOTES
    # ========================================================

    df.loc[
        i,
        "Manual_Notes"
    ] = notes


    # ========================================================
    # MARK AS REVIEWED
    # ========================================================

    df.loc[
        i,
        "Review_Status"
    ] = "REVIEWED"


    # ========================================================
    # SAVE AFTER EVERY RESPONSE
    # ========================================================

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )


    print("\n✓ Response saved successfully.")


# ============================================================
# FINAL REVIEW STATUS
# ============================================================

reviewed = (
    df["Review_Status"]
    .astype(str)
    .str.upper()
    .eq("REVIEWED")
    .sum()
)

remaining = len(df) - reviewed


print("\n")
print("=" * 75)
print("REVIEW STATUS")
print("=" * 75)

print(f"Total responses : {len(df)}")
print(f"Reviewed        : {reviewed}")
print(f"Remaining       : {remaining}")


if remaining == 0:

    print("\n✓ ALL 136 RESPONSES REVIEWED.")
    print(
        "\nNext step: generate final error-analysis "
        "counts, percentages, and representative examples."
    )

else:

    print(
        "\nYou can stop anytime and run this script again."
    )

    print(
        "It will automatically continue from the "
        "first unreviewed response."
    )


print("\nOutput file:")
print(OUTPUT_FILE)