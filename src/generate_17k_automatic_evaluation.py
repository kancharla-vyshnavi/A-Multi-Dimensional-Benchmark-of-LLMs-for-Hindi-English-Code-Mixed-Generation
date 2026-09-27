"""
Script: generate_17k_automatic_evaluation.py
Purpose: Generates the official CSV table and high-resolution comparison chart for:
         "17K Automatic Evaluation of Trained/Adapted Models"
Metrics:
  1. Perplexity
  2. Distinct-1
  3. Distinct-2
  4. Repetition Rate
Models:
  1. HingGPT (checkpoint-17340)
  2. Phi-3.5-mini (checkpoint-17340)
  3. Qwen2.5-3B (checkpoint-17340)
  4. Qwen2.5-7B (checkpoint-17340)
"""

import os
import csv
import matplotlib.pyplot as plt
import numpy as np

# Output directories
OUTPUT_DIR = "outputs"
CHARTS_DIR = os.path.join(OUTPUT_DIR, "final_charts")
TABLES_DIR = os.path.join(OUTPUT_DIR, "final_research_tables")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# Official Evaluation Data for Trained/Adapted Checkpoints (CPT checkpoint-17340)
EVAL_DATA = [
    {
        "Model": "HingGPT",
        "Perplexity": 171.2320,
        "Distinct-1": 0.0777,
        "Distinct-2": 0.4055,
        "Repetition Rate": 0.4358,
    },
    {
        "Model": "Phi-3.5-mini",
        "Perplexity": 50.7891,
        "Distinct-1": 0.2021,
        "Distinct-2": 0.7509,
        "Repetition Rate": 0.0040,
    },
    {
        "Model": "Qwen2.5-3B",
        "Perplexity": 74.0219,
        "Distinct-1": 0.2569,
        "Distinct-2": 0.7778,
        "Repetition Rate": 0.0545,
    },
    {
        "Model": "Qwen2.5-7B",
        "Perplexity": 55.1290,
        "Distinct-1": 0.3401,
        "Distinct-2": 0.8532,
        "Repetition Rate": 0.0332,
    },
]

def save_csv():
    csv_paths = [
        os.path.join(OUTPUT_DIR, "17k_automatic_evaluation_trained_models.csv"),
        os.path.join(TABLES_DIR, "table_17k_automatic_evaluation_trained_models.csv")
    ]
    fieldnames = ["Model", "Perplexity", "Distinct-1", "Distinct-2", "Repetition Rate"]
    
    for path in csv_paths:
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in EVAL_DATA:
                writer.writerow(row)
        print(f"Saved CSV: {path}")

def generate_chart():
    models = [d["Model"] for d in EVAL_DATA]
    ppl = [d["Perplexity"] for d in EVAL_DATA]
    dist1 = [d["Distinct-1"] for d in EVAL_DATA]
    dist2 = [d["Distinct-2"] for d in EVAL_DATA]
    rep_rate = [d["Repetition Rate"] for d in EVAL_DATA]

    # Professional academic color scheme
    # HingGPT: Coral/Red, Phi-3.5-mini: Emerald/Green, Qwen2.5-3B: Sky Blue, Qwen2.5-7B: Deep Indigo
    colors = ["#e74c3c", "#27ae60", "#2980b9", "#8e44ad"]
    
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    
    # Subplot 1: Perplexity (Lower is better)
    ax1 = axes[0, 0]
    bars1 = ax1.bar(models, ppl, color=colors, width=0.55, edgecolor="#2c3e50", linewidth=0.7, zorder=3)
    ax1.set_title("Perplexity (Lower is better $\\downarrow$)", fontsize=13, fontweight="bold", pad=12, color="#2c3e50")
    ax1.set_ylabel("Perplexity (PPL)", fontsize=11, fontweight="bold", color="#34495e")
    ax1.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax1.set_ylim(0, max(ppl) * 1.15)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 3.0, f"{yval:.2f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#2c3e50")
    ax1.tick_params(axis="x", labelsize=10.5)
    ax1.tick_params(axis="y", labelsize=10)

    # Subplot 2: Distinct-1 (Higher is better)
    ax2 = axes[0, 1]
    bars2 = ax2.bar(models, dist1, color=colors, width=0.55, edgecolor="#2c3e50", linewidth=0.7, zorder=3)
    ax2.set_title("Distinct-1 (Higher is better $\\uparrow$)", fontsize=13, fontweight="bold", pad=12, color="#2c3e50")
    ax2.set_ylabel("Distinct-1 Ratio", fontsize=11, fontweight="bold", color="#34495e")
    ax2.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax2.set_ylim(0, max(dist1) * 1.18)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.008, f"{yval:.4f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#2c3e50")
    ax2.tick_params(axis="x", labelsize=10.5)
    ax2.tick_params(axis="y", labelsize=10)

    # Subplot 3: Distinct-2 (Higher is better)
    ax3 = axes[1, 0]
    bars3 = ax3.bar(models, dist2, color=colors, width=0.55, edgecolor="#2c3e50", linewidth=0.7, zorder=3)
    ax3.set_title("Distinct-2 (Higher is better $\\uparrow$)", fontsize=13, fontweight="bold", pad=12, color="#2c3e50")
    ax3.set_ylabel("Distinct-2 Ratio", fontsize=11, fontweight="bold", color="#34495e")
    ax3.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax3.set_ylim(0, max(dist2) * 1.18)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.018, f"{yval:.4f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#2c3e50")
    ax3.tick_params(axis="x", labelsize=10.5)
    ax3.tick_params(axis="y", labelsize=10)

    # Subplot 4: Repetition Rate (Lower is better)
    ax4 = axes[1, 1]
    bars4 = ax4.bar(models, rep_rate, color=colors, width=0.55, edgecolor="#2c3e50", linewidth=0.7, zorder=3)
    ax4.set_title("Repetition Rate (Lower is better $\\downarrow$)", fontsize=13, fontweight="bold", pad=12, color="#2c3e50")
    ax4.set_ylabel("Repetition Rate Ratio", fontsize=11, fontweight="bold", color="#34495e")
    ax4.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax4.set_ylim(0, max(rep_rate) * 1.18)
    for bar in bars4:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.4f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#2c3e50")
    ax4.tick_params(axis="x", labelsize=10.5)
    ax4.tick_params(axis="y", labelsize=10)

    # Super Title
    fig.suptitle("17K Automatic Evaluation of Trained/Adapted Models", fontsize=16, fontweight="bold", color="#1a252f", y=0.98)
    
    # Add subtitle / footnote note
    plt.figtext(0.5, 0.01, "Evaluated on CPT checkpoints (checkpoint-17340) under identical decoding conditions (Seed: 42, Temp: 0.7, Top-p: 0.9, Rep-penalty: 1.1, Max-new-tokens: 50).", 
               ha="center", fontsize=10, style="italic", color="#555555")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    chart_paths = [
        os.path.join(CHARTS_DIR, "17k_automatic_evaluation_trained_models.png"),
        os.path.join(OUTPUT_DIR, "17k_automatic_evaluation_trained_models.png")
    ]
    for path in chart_paths:
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved Chart: {path}")
    plt.close()

if __name__ == "__main__":
    save_csv()
    generate_chart()
    print("17K Automatic Evaluation completed successfully.")
