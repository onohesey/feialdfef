"""Contest inference entry point for the original-column CatBoost baseline."""
import json
import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ID_COL, TARGET_COL = "row_id", "control_success"
CAT_COLS = [
    "top_bottom", "game_type",
    "pitcher_id", "batter_id",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
    "pitcher_experience_stage", "batter_experience_stage",
]


def make_features(df):
    """Use the same original columns and selected features as training."""
    x = df.copy().drop(columns=[
        ID_COL, "run_top_before", "run_bot_before", "base_state",
    ], errors="ignore")

    x["pitcher_team_runs_before"] = (
        x["run_total_before"] + x["score_diff_pitcher_team"]
    ) / 2
    x["batter_team_runs_before"] = (
        x["run_total_before"] - x["score_diff_pitcher_team"]
    ) / 2
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


def main():
    test = pd.read_csv("./data/test.csv", encoding="utf-8-sig")
    sub = pd.read_csv("./data/sample_submission.csv", encoding="utf-8-sig")

    with open("./model/metadata.json", encoding="utf-8") as f:
        feature_columns = json.load(f)["feature_columns"]
    X = make_features(test).reindex(columns=feature_columns)

    model = CatBoostClassifier()
    model.load_model("./model/catboost.cbm")
    pred_by_id = pd.Series(model.predict_proba(X)[:, 1], index=test[ID_COL])
    sub[TARGET_COL] = sub[ID_COL].map(pred_by_id)

    if sub[TARGET_COL].isna().any() or not sub[TARGET_COL].between(0, 1).all():
        raise ValueError("Invalid predictions")
    os.makedirs("./output", exist_ok=True)
    sub.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(f"Saved ./output/submission.csv ({len(sub)} rows)")


if __name__ == "__main__":
    main()
