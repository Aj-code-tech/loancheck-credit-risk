import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

df = pd.read_csv("credit_risk_dataset.csv").drop_duplicates()

target_col = "loan_status"
X = df.drop(columns=[target_col])
y = df[target_col]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            cat_cols,
        ),
    ]
)

xgb = XGBClassifier(
    n_estimators=220,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.85,
    eval_metric="logloss",
    random_state=42,
)

base_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", xgb),
])

calibrated_model = CalibratedClassifierCV(base_pipeline, method="sigmoid", cv=3)
calibrated_model.fit(X, y)

joblib.dump(calibrated_model, "credit_risk_model.pkl")
joblib.dump(0.38, "best_threshold.pkl")
print("Linux-native model artifacts generated successfully.")