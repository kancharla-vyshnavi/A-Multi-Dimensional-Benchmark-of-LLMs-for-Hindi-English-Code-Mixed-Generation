import tkinter as tk
from tkinter import messagebox
import pandas as pd
import os

SRC = "outputs/controlled_generation/benchmark_generations_v3.csv"
OUTDIR = "outputs/human_validation_v3"
OUT = OUTDIR + "/human_validation_results_v3.csv"

os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(SRC, encoding="utf-8")
df = df.groupby("model", group_keys=False).sample(n=12, random_state=42).reset_index(drop=True)

criteria = [
    "human_fluency",
    "human_code_mixing_naturalness",
    "human_hindi_grammar",
    "human_prompt_adherence",
    "human_spelling_consistency",
    "human_overall"
]

if os.path.exists(OUT):
    saved = pd.read_csv(OUT)
    for c in criteria:
        if c in saved.columns:
            df[c] = saved[c]
else:
    for c in criteria:
        df[c] = pd.NA

root = tk.Tk()
root.title("Hinglish Human Validation V3")
root.geometry("1200x850")

idx = 0
vars = [tk.IntVar(value=0) for _ in criteria]

tk.Label(root, text="", font=("Arial", 16, "bold")).pack(pady=8)
model_label = tk.Label(root, text="", font=("Arial", 13, "bold"))
model_label.pack()

prompt_box = tk.Text(root, height=7, wrap="word", font=("Arial", 11))
prompt_box.pack(fill="x", padx=20, pady=8)

response_box = tk.Text(root, height=16, wrap="word", font=("Arial", 11))
response_box.pack(fill="both", expand=True, padx=20, pady=8)

frame = tk.Frame(root)
frame.pack(pady=8)

names = [
    "Fluency",
    "Code-mixing naturalness",
    "Hindi grammar",
    "Prompt adherence",
    "Spelling consistency",
    "Overall"
]

for i, name in enumerate(names):
    f = tk.Frame(frame)
    f.pack(side="left", padx=10)
    tk.Label(f, text=name, font=("Arial", 9, "bold")).pack()
    for n in range(1, 6):
        tk.Radiobutton(f, text=str(n), variable=vars[i], value=n).pack(side="left")

status = tk.Label(root, text="", font=("Arial", 11))
status.pack(pady=5)

def show():
    global idx
    r = df.iloc[idx]

    model_label.config(
        text=f"Response {idx+1} / {len(df)}    |    {r['model']}    |    row_index={r['row_index']}"
    )

    prompt_box.config(state="normal")
    prompt_box.delete("1.0", "end")
    prompt_box.insert("1.0", str(r["prompt"]))
    prompt_box.config(state="disabled")

    response_box.config(state="normal")
    response_box.delete("1.0", "end")
    response_box.insert("1.0", str(r["response"]))
    response_box.config(state="disabled")

    for i, c in enumerate(criteria):
        vars[i].set(0 if pd.isna(r[c]) else int(r[c]))

    status.config(text="Rate all 6 criteria, then click SAVE & NEXT")

def save_next():
    global idx

    scores = [v.get() for v in vars]

    if any(x not in [1,2,3,4,5] for x in scores):
        messagebox.showwarning("Missing rating", "Please rate all 6 criteria from 1 to 5.")
        return

    for i, c in enumerate(criteria):
        df.loc[idx, c] = scores[i]

    df.to_csv(OUT, index=False, encoding="utf-8-sig")

    if idx < len(df) - 1:
        idx += 1
        show()
    else:
        messagebox.showinfo(
            "Completed!",
            f"All 48 responses are annotated!\n\nSaved to:\n{OUT}"
        )
        status.config(text="DONE — 48/48 completed.")

tk.Button(
    root,
    text="SAVE & NEXT",
    command=save_next,
    font=("Arial", 14, "bold"),
    padx=30,
    pady=10
).pack(pady=10)

show()
root.mainloop()
