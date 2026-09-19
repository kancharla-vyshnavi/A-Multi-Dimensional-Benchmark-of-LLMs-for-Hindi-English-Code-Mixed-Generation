import pandas as pd
import os
import shutil

INPUT = r".\outputs\error_analysis\manual_error_annotation_final.csv"
BACKUP = r".\outputs\error_analysis\manual_error_annotation_final_backup.csv"

# ============================================================
# 1. BACKUP
# ============================================================

if not os.path.exists(BACKUP):
    shutil.copy2(INPUT, BACKUP)
    print("Backup created:", BACKUP)

# ============================================================
# 2. READ CSV
# ============================================================

df = pd.read_csv(INPUT)

print("Input shape:", df.shape)

# ============================================================
# 3. FINAL ERROR CODES FOR ALL 136 RESPONSES
# ============================================================

codes_text = {

"HingGPT": """
1:E3,E4,E5,E6,E8,E10
2:E1,E3,E4,E5,E6,E10
3:E5,E6,E10
4:E5,E6,E10
5:E5,E6,E10
6:E1,E6,E10
7:E1,E3,E4,E6,E8,E10
8:E1,E3,E5,E6,E10
9:E3,E4,E6,E8,E10
10:E1,E3,E4,E5,E6,E10
11:E1,E3,E4,E5,E6,E10
12:E1,E3,E4,E5,E6,E7,E10
13:E1,E3,E4,E5,E6,E10
14:E1,E3,E4,E5,E6,E10
15:E1,E3,E4,E5,E6,E10
16:E5,E6,E7,E10
17:E1,E3,E4,E5,E6,E10
18:E1,E3,E4,E6,E10
19:E1,E3,E4,E6,E8,E10
20:E1,E3,E4,E5,E6,E10
21:E1,E3,E4,E5,E6,E10
22:E1,E3,E4,E5,E6,E10
23:E5,E6,E7,E10
24:E1,E3,E4,E5,E6,E8,E10
25:E1,E3,E4,E6,E8,E10
26:E1,E3,E4,E5,E6,E8,E10
27:E1,E3,E4,E5,E6,E10
28:E1,E3,E4,E5,E6,E10
29:E1,E3,E4,E5,E6,E10
30:E1,E3,E4,E5,E6,E10
31:E1,E3,E4,E6,E10
32:E1,E3,E4,E6,E10
33:E1,E3,E4,E5,E6,E10
34:E1,E3,E4,E6,E10
""",

"Phi-3.5-mini": """
1:E2,E3,E4,E6,E8,E10
2:E2,E3,E4,E6,E7,E10
3:E2,E3,E4,E6,E7,E10
4:E1,E3,E4,E6,E10
5:E2,E3,E4,E6,E7,E10
6:E1,E6,E10
7:E1,E3,E4,E6,E10
8:E1,E3,E4,E6,E9,E10
9:E1,E6,E10
10:E1,E3,E4,E6,E10
11:E1,E3,E4,E6,E10
12:E2,E3,E4,E6,E7,E10
13:E1,E3,E4,E6,E10
14:E1,E3,E4,E6,E10
15:E1,E6,E7,E10
16:E2,E3,E4,E6,E10
17:E1,E6,E10
18:E1,E3,E4,E6,E10
19:E1,E3,E4,E6,E10
20:E2,E3,E4,E6,E10
21:E1,E3,E4,E6,E10
22:E1,E3,E4,E6,E10
23:E1,E6,E10
24:E1,E6,E10
25:E1,E3,E4,E6,E10
26:E1,E3,E4,E6,E10
27:E1,E6,E10
28:E1,E3,E4,E6,E10
29:E1,E3,E4,E6,E10
30:E1,E6,E7,E10
31:E1,E3,E4,E6,E10
32:E1,E3,E4,E6,E10
33:E1,E6,E7,E10
34:E1,E3,E6
""",

"Qwen2.5-3B": """
1:E1,E5,E6,E10
2:E5
3:E5,E7
4:E7
5:E2,E3,E4,E5,E6,E10
6:E1,E5,E6,E10
7:E1,E6,E10
8:E1,E5
9:E1,E5,E6,E10
10:E5,E7,E6,E10
11:E1,E6
12:E1,E5,E6
13:E5,E6,E10
14:E1,E6,E10
15:E2,E3,E4,E6,E10
16:E1,E6,E7,E10
17:E1,E3,E4,E6,E10
18:E1,E6,E7,E10
19:E1,E6,E10
20:E5,E7,E6,E10
21:E2,E3,E4,E5,E6,E10
22:E1,E6
23:E5,E7,E6,E10
24:E1,E5,E6,E10
25:E1,E6
26:E1,E6,E10
27:E5,E7,E6,E10
28:E1,E5,E6,E10
29:E2,E3,E4,E5,E6,E7,E10
30:E2,E3,E4,E5,E6,E10
31:E1,E5,E6,E9,E10
32:E1,E6,E10
33:E1,E3,E4,E5
34:E5,E7,E6,E10
""",

"Qwen2.5-7B": """
1:E1,E3,E4,E5,E6,E10
2:E5,E7,E6,E10
3:E2,E3,E4,E5,E6,E10
4:E5,E7,E6,E10
5:E2,E3,E4,E5,E6
6:E1,E6,E10
7:E1,E6,E10
8:E5,E7
9:E1,E5,E6,E10
10:E3,E4,E6
11:E5,E7,E6,E10
12:E1,E5,E6,E10
13:E5,E7,E6,E10
14:E1,E6,E10
15:E1,E6,E10
16:E1,E3,E5,E6,E7,E10
17:E1,E6,E10
18:E2,E3,E4,E5,E6,E10
19:E5,E7,E6,E10
20:E1,E5,E7,E6,E10
21:E1,E6,E10
22:E1,E3,E6,E7
23:E1,E6
24:E1,E6,E10
25:E7
26:E2,E3,E4,E5,E6,E10
27:E2,E3,E4,E5,E6,E10
28:E1,E5,E6,E10
29:E1,E3,E4,E5,E6,E10
30:E1,E3,E4,E6,E7
31:E2,E3,E4,E6,E10
32:E1,E3,E4,E6,E10
33:E5,E7,E6,E10
34:E1,E5,E6,E7
"""
}

# ============================================================
# 4. CONVERT TEXT TO DICTIONARY
# ============================================================

codes = {}

for model, text in codes_text.items():
    codes[model] = {}

    for line in text.strip().splitlines():
        sample, error_codes = line.strip().split(":")
        codes[model][int(sample)] = error_codes

# ============================================================
# 5. ERROR COLUMN NAMES
# ============================================================

error_columns = {
    "E1": "E1_English_dominant",
    "E2": "E2_Hindi_dominant",
    "E3": "E3_Unnatural_code_switching",
    "E4": "E4_Grammatical_error",
    "E5": "E5_Repetition",
    "E6": "E6_Prompt_misunderstanding",
    "E7": "E7_Incomplete_response",
    "E8": "E8_Spelling_transliteration_error",
    "E9": "E9_Hallucination_factual_error",
    "E10": "E10_Irrelevant_response"
}

# ============================================================
# 6. FILL ALL 136 ROWS
# ============================================================

for index, row in df.iterrows():

    model = str(row["model"]).strip()
    sample = int(row["sample"])

    if model not in codes:
        raise ValueError(f"Unknown model: {model}")

    if sample not in codes[model]:
        raise ValueError(f"Missing code for {model} sample {sample}")

    final_code = codes[model][sample]

    # Fill Final_Error_Category
    df.at[index, "Final_Error_Category"] = final_code

    # Convert codes to set
    selected_errors = set(final_code.split(","))

    # Fill E1-E10 indicator columns
    for error_code, column_name in error_columns.items():
        df.at[index, column_name] = (
            1 if error_code in selected_errors else 0
        )

# ============================================================
# 7. SAVE
# ============================================================

df.to_csv(INPUT, index=False)

# ============================================================
# 8. VERIFICATION
# ============================================================

print()
print("=" * 60)
print("DONE")
print("=" * 60)

print("Total responses:", len(df))
print("Expected responses: 136")

print()
print("Model counts:")
print(df["model"].value_counts())

print()
print("Final error categories filled:")
print(df["Final_Error_Category"].notna().sum(), "/ 136")

print()
print("Error category counts:")

all_errors = []

for value in df["Final_Error_Category"].dropna():
    all_errors.extend(value.split(","))

print(pd.Series(all_errors).value_counts().sort_index())

print()
print("Review status:")
print(df["Review_Status"].value_counts())

print()
print("Output:")
print(INPUT)

print()
print("Backup:")
print(BACKUP)

print()
print("=" * 60)
print("ALL 136 ERROR CODES FILLED")
print("=" * 60)