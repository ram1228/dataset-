from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def predict_one(bundle: dict, features: dict[str, float]) -> float:
    """Predict progression from a mapping of feature names to values."""
    expected = bundle["feature_names"]
    missing = set(expected) - set(features)
    extra = set(features) - set(expected)

    if missing or extra:
        raise ValueError(
            f"Feature mismatch. Missing: {sorted(missing)}; "
            f"unexpected: {sorted(extra)}"
        )

    row = np.array([[features[name] for name in expected]], dtype=float)
    return float(bundle["pipeline"].predict(row)[0])


def main() -> None:
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
    )

    # Scaling is inside the pipeline to prevent data leakage during cross-validation.
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )

    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=5, scoring="r2"
    )
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    print(f"5-fold CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"Test MAE:     {mean_absolute_error(y_test, predictions):.3f}")
    print(f"Test RMSE:    {np.sqrt(mean_squared_error(y_test, predictions)):.3f}")
    print(f"Test R²:      {r2_score(y_test, predictions):.3f}")

    bundle = {
        "pipeline": pipeline,
        "feature_names": list(data.feature_names),
    }
    model_path = Path(__file__).resolve().parent / "models" / "diabetes_ridge.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    print(f"Saved model to: {model_path}")

    example = dict(zip(data.feature_names, X_test[0]))
    print(f"Example prediction: {predict_one(bundle, example):.2f}")


if __name__ == "__main__":
    main()
