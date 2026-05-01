from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pipeline import MODELS_DIR, ROOT_DIR, train_and_save_models


def _save_visualizations() -> None:
    processed_path = ROOT_DIR / "data" / "processed" / "processed_aviation_data.csv"
    images_dir = ROOT_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if not processed_path.exists():
        return

    df = pd.read_csv(processed_path)
    sns.set_theme(style="whitegrid")

    if "high_risk" in df.columns:
        plt.figure(figsize=(7, 4))
        sns.countplot(x="high_risk", data=df, palette="viridis")
        plt.title("High Risk Distribution")
        plt.xlabel("High Risk (1=True)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(images_dir / "risk_distribution.png", dpi=160)
        plt.close()

    if {"year", "fatalities"}.issubset(df.columns):
        yearly = df.groupby("year", as_index=False)["fatalities"].sum()
        plt.figure(figsize=(10, 4))
        sns.lineplot(data=yearly, x="year", y="fatalities", linewidth=2)
        plt.title("Total Fatalities by Year")
        plt.xlabel("Year")
        plt.ylabel("Fatalities")
        plt.tight_layout()
        plt.savefig(images_dir / "fatalities_over_time.png", dpi=160)
        plt.close()

    if {"aircraft_category", "high_risk"}.issubset(df.columns):
        category_risk = (
            df.groupby("aircraft_category", as_index=False)["high_risk"].mean()
            .sort_values("high_risk", ascending=False)
        )
        plt.figure(figsize=(9, 4))
        sns.barplot(data=category_risk, x="aircraft_category", y="high_risk", palette="magma")
        plt.title("Average High-Risk Rate by Aircraft Category")
        plt.xlabel("Aircraft Category")
        plt.ylabel("Average High-Risk Rate")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(images_dir / "aircraft_risk_rate.png", dpi=160)
        plt.close()


def main() -> None:
    metrics = train_and_save_models()
    _save_visualizations()

    print("Training complete.")
    print(f"Best Model : {metrics['best_model']}")
    print(f"Accuracy   : {metrics['best_accuracy']:.4f}")
    print(f"AUC Score  : {metrics['best_auc']:.4f}")
    print(f"Rows       : {metrics['total_rows']} total")
    print(f"Saved to   : {MODELS_DIR}")

    metrics_path = Path(MODELS_DIR) / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
