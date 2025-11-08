import pandas as pd
import joblib
import numpy as np
from pathlib import Path

def _load_model():
    # Try both common names; use the one that exists
    for p in [Path("models/linear_regression_pipeline.joblib"),
              Path("models/simple_linear_model.pkl")]:
        if p.exists():
            return joblib.load(p)
    raise FileNotFoundError("No model artifact found under models/")

def test_scoring():
    pipeline = _load_model()

    df = pd.DataFrame({"eruptions": [1.5, 2.0, 3.0]})
    preds = pipeline.predict(df[["eruptions"]])

    # a) same number of predictions as inputs
    assert len(preds) == len(df["eruptions"])
    # b) predictions are finite
    assert np.isfinite(preds).all()
    # c) predictions are positive
    assert (preds > 0).all()
