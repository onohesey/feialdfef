"""CatBoost baseline: original columns only, without engineered features."""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "submission_catboost" / "model"
ID_COL, TARGET_COL = "row_id", "control_success"

# 숫자로 저장되어 있어도 순서가 없는 식별자/상태 코드는 범주형으로 전달합니다.
CAT_COLS = [
    "top_bottom", "game_type",
    "pitcher_id", "batter_id",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
    "pitcher_experience_stage", "batter_experience_stage",
]


def make_features(df):
    """Original columns plus the selected baseball-context features."""
    x = df.copy().drop(columns=[
        ID_COL, "run_top_before", "run_bot_before", "base_state",
    ], errors="ignore")

    # 투수 팀 점수 차를 이용해 양 팀의 실제 점수를 복원합니다.
    x["pitcher_team_runs_before"] = (
        x["run_total_before"] + x["score_diff_pitcher_team"]
    ) / 2
    x["batter_team_runs_before"] = (
        x["run_total_before"] - x["score_diff_pitcher_team"]
    ) / 2

    # 2루 또는 3루에 주자가 있으면 즉시 득점권 상황입니다.
    x["runner_on_scoring_position"] = (
        x["runner_on_2b"].eq(1) | x["runner_on_3b"].eq(1)
    ).astype("int8")

    x["pitcher_team_win_expectancy"] = np.where(
        x["top_bottom"].eq("T"), x["home_win_expectancy"], x["away_win_expectancy"]
    )
    x["home_away_win_expectancy_diff"] = (
        x["home_win_expectancy"] - x["away_win_expectancy"]
    )
    x = x.drop(columns=["home_win_expectancy", "away_win_expectancy"])

    x["pitcher_experience_stage"] = np.select(
        [x["asof_pitcher_n"] <= 500, x["asof_pitcher_n"] >= 4_000],
        ["rookie", "veteran"], default="regular",
    )
    x["batter_experience_stage"] = np.select(
        [x["asof_batter_n"] <= 600, x["asof_batter_n"] >= 5_000],
        ["rookie", "veteran"], default="regular",
    )
    x["pitcher_batter_experience_gap"] = (
        x["asof_pitcher_n"] - x["asof_batter_n"]
    )

    for n in (1, 3, 5):
        x[f"pitcher_form_gap_{n}"] = (
            x[f"asof_pitcher_prev{n}_game_success_rate"]
            - x["asof_pitcher_success_rate"]
        )

    for col in CAT_COLS:
        x[col] = x[col].fillna("__MISSING__").astype(str)
    return x


def make_catboost(iterations):
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="BrierScore",
        iterations=iterations,
        learning_rate=0.04,
        depth=8,
        l2_leaf_reg=8,
        random_seed=42,
        thread_count=-1,
        verbose=100,
        allow_writing_files=False,
    )


def brier(y, pred):
    return float(np.mean((pred - y.to_numpy()) ** 2))


def main():
    train = pd.read_csv(DATA_DIR / "train.csv", encoding="utf-8-sig")
    X = make_features(train.drop(columns=[TARGET_COL]))
    y = train[TARGET_COL]

    # 2019~2023으로 학습하고 2024로 검증합니다.
    is_val = train["season"].eq(2024)
    X_train, y_train = X.loc[~is_val], y.loc[~is_val]
    X_val, y_val = X.loc[is_val], y.loc[is_val]
    print(f"Features: {X.shape[1]} | categorical: {len(CAT_COLS)}")
    print(f"Train: {len(X_train):,} | Validation (2024): {len(X_val):,}")

    started = time.time()
    model = make_catboost(iterations=1500)
    model.fit(
        X_train, y_train,
        cat_features=CAT_COLS,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        use_best_model=True,
    )

    pred = model.predict_proba(X_val)[:, 1]
    best_iterations = max(1, model.get_best_iteration() + 1)
    validation_brier = brier(y_val, pred)
    baseline = y_val.mean() * (1 - y_val.mean())
    score = max(0, 100000 * (1 - validation_brier / baseline))
    print(f"Validation Brier: {validation_brier:.6f}")
    print(f"Validation Score: {score:.2f}")
    print(f"Best trees: {best_iterations} | Elapsed: {time.time() - started:.1f}s")

    # 제출용 모델은 검증에 사용한 2024까지 포함해 다시 학습합니다.
    final_model = make_catboost(iterations=best_iterations)
    final_model.fit(X, y, cat_features=CAT_COLS)

    os.makedirs(OUT_DIR, exist_ok=True)
    final_model.save_model(OUT_DIR / "catboost.cbm")
    with open(OUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"feature_columns": X.columns.tolist()}, f, ensure_ascii=False, indent=2)
    print("Saved CatBoost model to submission_catboost/model/")


if __name__ == "__main__":
    main()
