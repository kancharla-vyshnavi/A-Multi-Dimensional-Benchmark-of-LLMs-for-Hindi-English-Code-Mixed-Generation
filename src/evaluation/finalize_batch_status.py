import pandas as pd

INPUT = r".\outputs\error_analysis\manual_error_annotation_final.csv"

df = pd.read_csv(INPUT)

# Mark previously unreviewed rows as batch-assisted
mask = df["Review_Status"].astype(str).str.strip() != "REVIEWED"

df.loc[mask, "Review_Status"] = "BATCH_ASSISTED"

# Add a note only where notes are empty
notes_empty = (
    df["Manual_Notes"].isna()
    | (df["Manual_Notes"].astype(str).str.strip() == "")
)

df.loc[mask & notes_empty, "Manual_Notes"] = (
    "Batch-assisted error annotation; requires human verification "
    "before being reported as fully human-reviewed."
)

df.to_csv(INPUT, index=False)

print("=" * 60)
print("BATCH FINALIZATION COMPLETE")
print("=" * 60)

print("Total responses:", len(df))
print()
print("Review status:")
print(df["Review_Status"].value_counts())

print()
print("Final error categories filled:",
      df["Final_Error_Category"].notna().sum(), "/ 136")

print()
print("Output:")
print(INPUT)

print("=" * 60)