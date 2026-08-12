import os
import time

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

DATA_DIR = "./data"

ID = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]

test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                        encoding="utf-8-sig", nrows=0).columns
FEATURES = [c for c in test_cols if c != ID]
NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                    encoding="utf-8-sig", usecols=FEATURES + [TARGET])

print("train:", train.shape, "| 피처:", len(FEATURES),
      f"(범주형 {len(CAT_COLS)}, 수치형 {len(NUM_COLS)})")
print("시즌:", train["season"].min(), "~", train["season"].max())
print(f"제구 성공률: {train[TARGET].mean():.4f}")

preprocessor = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                           unknown_value=-1), CAT_COLS),
    ("num", SimpleImputer(strategy="median"), NUM_COLS),
])

model = Pipeline([
    ("pre", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=200,
        n_jobs=-1,
        random_state=42,
    )),
])

# 2024 시즌을 검증용으로 떼어 두고 2019~2023 으로 학습합니다.
is_val = train["season"] == 2024
X_train, y_train = train.loc[~is_val, FEATURES], train.loc[~is_val, TARGET]
X_val, y_val = train.loc[is_val, FEATURES], train.loc[is_val, TARGET]
print("train:", len(X_train), "| val:", len(X_val))

t = time.time()
model.fit(X_train, y_train)
print(f"학습 완료 :: {time.time() - t:.1f}s")

val_pred = model.predict_proba(X_val)[:, 1]

r = y_val.mean()
brier = ((val_pred - y_val) ** 2).mean()
baseline_brier = r * (1 - r)
score = max(0, 100000 * (1 - brier / baseline_brier))

print(f"Brier: {brier:.6f} | 기준선 r(1-r): {baseline_brier:.6f}")
print(f"Validation Score: {score:.2f}")

t = time.time()
model.fit(train[FEATURES], train[TARGET])
print(f"재학습 완료 :: {time.time() - t:.1f}s")

os.makedirs("./model", exist_ok=True)
joblib.dump(model, "./model/rf.pkl", compress=3)
print("저장 완료: ./model/rf.pkl")