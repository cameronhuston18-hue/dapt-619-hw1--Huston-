# ---- Scoring Script for DAPT-619 Assignment ----
# Author: Cameron Huston
# Date: November 8 2025
# Description: Loads the trained model and Geyser data, scores the dataset,
# and saves predictions + plot to output folders.

import pandas as pd
import joblib
import os
from pathlib import Path
import matplotlib.pyplot as plt

# ------------------------------------------------
# Setup paths
# ------------------------------------------------
models_dir = Path("models")
data_dir = Path("data")

# Model path
model_path = models_dir / "linear_regression_pipeline.joblib"

# Handle either CSV or TSV data formats
raw_dir = data_dir / "raw"
if (raw_dir / "geyser.tsv").exists():
    input_data_path = raw_dir / "geyser.tsv"
    data = pd.read_csv(input_data_path, sep="\t")
elif (raw_dir / "geyser.csv").exists():
    input_data_path = raw_dir / "geyser.csv"
    data = pd.read_csv(input_data_path)
else:
    raise FileNotFoundError(f"No geyser data found in {raw_dir}")

# ------------------------------------------------
# Load model
# ------------------------------------------------
if not model_path.exists():
    raise FileNotFoundError(f"Model not found at {model_path}")
model = joblib.load(model_path)

# ------------------------------------------------
# Make predictions
# ------------------------------------------------
data["predicted_waiting"] = model.predict(data[["eruptions"]])

# ------------------------------------------------
# Save scored data
# ------------------------------------------------
output_dir = data_dir / "scored"
os.makedirs(output_dir, exist_ok=True)
output_path = output_dir / "geyser_scored.csv"
data.to_csv(output_path, index=False)

print(f"✅ Scored data saved to {output_path}")

# ------------------------------------------------
# Plot actual vs predicted waiting time
# ------------------------------------------------
plots_dir = Path("plots")
plots_dir.mkdir(parents=True, exist_ok=True)
png_path = plots_dir / "geyser_predictions.png"

if "waiting" in data.columns:
    plt.figure()
    plt.scatter(data["waiting"], data["predicted_waiting"])
    plt.xlabel("Actual waiting")
    plt.ylabel("Predicted waiting")
    plt.title("Geyser: Actual vs Predicted Waiting")
    # Optional 45° line
    lo = min(data["waiting"].min(), data["predicted_waiting"].min())
    hi = max(data["waiting"].max(), data["predicted_waiting"].max())
    plt.plot([lo, hi], [lo, hi], color="red", linestyle="--")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"🖼️ Plot saved to {png_path}")
else:
    print("ℹ️ No 'waiting' column found — plot skipped.")
