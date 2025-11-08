import pandas as pd
import joblib
import numpy as np

def test_scoring():
    model_path = "./models/linear_regression_pipeline.joblib"
    model = joblib.load(model_path)

    df = pd.DataFrame({"eruptions": [1.5, 2.0, 3.0]})
    preds = model.predict(df[["eruptions"]])

    # a) same number of predictions as inputs
    assert len(preds) == len(df["eruptions"])
    # b) predictions are finite
    assert np.all(np.isfinite(preds))
    # c) predictions are positive
    assert np.all(preds > 0)
