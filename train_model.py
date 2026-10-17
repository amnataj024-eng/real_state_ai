"""Train, compare and save the price model. Run: python -m src.train_model"""
import json, joblib, numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from src.data import load_clean, ROOT

FEATURES = ["location", "area_sqft", "bedrooms", "bathrooms", "age_years", "parking", "furnished", "near_metro", "year_listed"]
MODEL_PATH = ROOT / "models" / "price_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

def make_pipeline(model):
    pre = ColumnTransformer([("loc", OneHotEncoder(handle_unknown="ignore"), ["location"])], remainder="passthrough")
    return Pipeline([("pre", pre), ("model", model)])

def main():
    df = load_clean()
    X, y = df[FEATURES], df["price"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    candidates = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=14, random_state=42, n_jobs=-1),
    }
    results, best_name, best_pipe, best_r2 = {}, None, None, -1
    for name, model in candidates.items():
        pipe = make_pipeline(model).fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        r2 = r2_score(y_te, pred)
        results[name] = {"MAE": round(mean_absolute_error(y_te, pred)),
                         "RMSE": round(float(np.sqrt(mean_squared_error(y_te, pred)))), "R2": round(r2, 4)}
        print(f"{name:18s} {results[name]}")
        if r2 > best_r2:
            best_name, best_pipe, best_r2 = name, pipe, r2
    joblib.dump(best_pipe, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({"best_model": best_name, "results": results}, indent=2))
    print(f"Saved best model: {best_name} -> {MODEL_PATH}")

if __name__ == "__main__":
    main()
