"""
script.py — 평가 서버가 자동 실행하는 추론 코드 (FE_S 앙상블 v2)
================================================================
입력:  ./data/test.csv, ./data/sample_submission.csv,
       ./model/model.pkl, ./model/feature_engineering.py
출력:  ./output/submission.csv

행 단위 독립 예측
  파생 피처는 각 행의 입력만으로 계산된다.
  보정 상수(오프셋 / seg_mean_logit / anchor / lam)와 이력 테이블은 전부
  학습 시점에 확정되어 model.pkl 에 들어 있다.
  test.csv 의 다른 행이나 전체 분포를 참조하는 연산은 하지 않는다.

버전 안전성
  아티팩트에 pandas 객체를 넣지 않았다. 범주형은 정수 코드로 다루므로
  학습/평가 서버의 pandas 버전이 달라도 깨지지 않는다.
"""
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

# 평가 서버 로케일이 POSIX/C 면 한글 로그에서 UnicodeEncodeError 로 죽는다.
for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "model"))
import feature_engineering as fe  # noqa: E402

ID_COL = "row_id"
TARGET_COL = "control_success"
EPS = 1e-6
BATCH = 100_000


def _resolve(*rel):
    r = os.path.join(*rel)
    for base in (os.getcwd(), _HERE):
        c = os.path.join(base, r)
        if os.path.exists(c):
            return c
    return os.path.join(_HERE, r)


TEST_PATH = _resolve("data", "test.csv")
SAMPLE_SUB_PATH = _resolve("data", "sample_submission.csv")
MODEL_PATH = _resolve("model", "model.pkl")
_OUT_BASE = os.getcwd() if os.path.isdir(os.path.join(os.getcwd(), "output")) else _HERE
OUT_PATH = os.path.join(_OUT_BASE, "output", "submission.csv")


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------- 그룹 키
def key_n(df):
    return pd.cut(df["asof_pitcher_n"], [-1, 50, 200, 500, 1000, 2000,
                                         5000, 10000, 1e9]).astype(str).to_numpy()


def key_count(df):
    return (df["balls_before"].astype(int).astype(str) + "-"
            + df["strikes_before"].astype(int).astype(str)).to_numpy()


def key_mu_count(df):
    return (df["pitcher_hand"].astype(int).astype(str) + "v"
            + df["batter_hand"].astype(int).astype(str) + "_"
            + df["balls_before"].astype(int).astype(str)
            + df["strikes_before"].astype(int).astype(str)).to_numpy()


KEYFN = {"n": key_n, "count": key_count, "mu_count": key_mu_count}


# ---------------------------------------------------------------- 로드
def restore_meta(pm):
    """저장된 순수 배열을 fe.transform 이 기대하는 형태로 복원."""
    meta = {"priors": dict(pm["priors"]), "trackman": None,
            "max_hist_season": int(pm["max_hist_season"]), "cat_levels": {}}
    for k in ("pitcher_hist", "batter_hist"):
        h = pm.get(k)
        meta[k] = None if h is None else pd.DataFrame({c: np.asarray(v)
                                                       for c, v in h.items()})
    return meta


def load_test(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음: {list(df.columns)[:5]}")
    return df


def load_sample_submission(path, test_ids=None, fallback=0.5):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
            if list(df.columns[:2]) == [ID_COL, TARGET_COL]:
                return df
            print(" [warn] sample_submission 컬럼 불일치 -> test 기준 재생성")
        except Exception as e:
            print(f" [warn] sample_submission 로드 실패({type(e).__name__}) -> 재생성")
    else:
        print(" [warn] sample_submission.csv 없음 -> test 기준 생성")
    return pd.DataFrame({ID_COL: list(test_ids or []), TARGET_COL: fallback})


# ---------------------------------------------------------------- 추론
def build_matrix(chunk, art, meta):
    """행 단위 파생 피처 -> 학습 때와 동일한 컬럼 순서/범주 코드의 float32 행렬."""
    cols, cat_maps = art["cols"], art["cat_maps"]
    part = fe.transform(chunk, meta)
    X = np.empty((len(chunk), len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        if c == "season":
            X[:, j] = pd.to_numeric(chunk["season"], errors="coerce").to_numpy(np.float32)
        elif c in cat_maps:
            lut = {v: i for i, v in enumerate(cat_maps[c])}
            src = part[c] if c in part.columns else chunk.get(c)
            X[:, j] = pd.Series(np.asarray(src, dtype=object)).astype(str) \
                        .map(lut).fillna(-1).to_numpy(np.float32)
        else:
            src = part[c] if c in part.columns else np.nan
            X[:, j] = pd.to_numeric(pd.Series(src, index=part.index),
                                    errors="coerce").to_numpy(np.float32)
    return X


def predict_pool(art, X):
    w = art["weights"]
    z = np.zeros(len(X), dtype=np.float64)
    for name, (_, m) in art["models"].items():
        z += w[name] * logit(m.predict_proba(X)[:, 1])
    return z / sum(w.values())


def apply_offsets(z, chunk, art):
    """구간 상대 오프셋. 각 행은 자기 그룹의 상수만 받는다."""
    g = art["offset_gamma"]
    for k in art["offset_keys"]:
        off = art["offsets"][k]
        kv = KEYFN[k](chunk)
        z = z + np.array([off.get(str(x), 0.0) for x in kv]) * g
    return z


def apply_post(z, seg, art):
    seg = np.asarray(seg)
    anchors, segm = art["anchors"], art["seg_mean_logit"]
    lam, lam_d = art["lam"], art["lam_default"]
    fb_a = anchors.get("R", float(np.mean(list(anchors.values()))))
    fb_m = segm.get("R", float(np.mean(list(segm.values()))))
    out = np.empty_like(z)
    for g in np.unique(seg):
        k = seg == g
        out[k] = (logit(anchors.get(g, fb_a))
                  + lam.get(g, lam_d) * (z[k] - segm.get(g, fb_m)))
    return np.clip(sigmoid(out), 0.0, 1.0)


def fallback_rate(art):
    anc = art.get("anchors") or {}
    if "R" in anc:
        return float(anc["R"])
    return float(np.mean(list(anc.values()))) if anc else 0.5


def infer(art, test, meta):
    fb = fallback_rate(art)
    preds = np.full(len(test), fb, dtype=np.float64)
    seg_col = art["seg_col"]
    n_failed = 0
    for s in range(0, len(test), BATCH):
        e = min(s + BATCH, len(test))
        chunk = test.iloc[s:e]
        try:
            X = build_matrix(chunk, art, meta)
            z = predict_pool(art, X)
            z = apply_offsets(z, chunk, art)
            preds[s:e] = apply_post(z, chunk[seg_col].astype(str).to_numpy(), art)
        except Exception as ex:
            n_failed += e - s
            print(f"   [warn] chunk {s:,}-{e:,} 실패 -> {fb:.4f} 대체: "
                  f"{type(ex).__name__}: {ex}", flush=True)
        print(f"   {e:,}/{len(test):,}", flush=True)
    if n_failed:
        print(f" 경고: {n_failed:,}행 대체값 사용")
    return preds


# ---------------------------------------------------------------- 제출
def merge_predictions(sub, ids, preds):
    pm = dict(zip(ids, preds))
    vals, miss = [], 0
    for rid, cur in zip(sub[ID_COL], sub[TARGET_COL]):
        p = pm.get(rid)
        if p is None:
            miss += 1
            vals.append(cur)
        else:
            vals.append(p)
    if miss:
        print(f" 경고: 예측 없는 row_id {miss}건")
    sub[TARGET_COL] = vals
    return sub


def sanitize(sub, fb):
    v = pd.to_numeric(sub[TARGET_COL], errors="coerce")
    bad = int(v.isna().sum())
    if bad:
        print(f" 경고: NaN {bad}건 -> {fb:.4f}")
    sub[TARGET_COL] = v.fillna(fb).clip(0.0, 1.0)
    return sub


def main():
    t0 = time.time()
    print("Load model...")
    art = joblib.load(MODEL_PATH)
    meta = restore_meta(art["fe_meta"])
    print(f" models={list(art['models'])} weights={art['weights']}")
    print(f" anchors={ {k: round(v,4) for k,v in art['anchors'].items()} } lam={art['lam']}")

    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH, test[ID_COL].tolist(), fallback_rate(art))
    print(f" test={len(test):,}  submission={len(sub):,}")

    print("Inference...")
    ids = test[ID_COL].tolist()
    preds = infer(art, test, meta) if len(test) else np.array([])
    print(f" preds={len(preds):,}")

    print("Build submission...")
    sub = sanitize(merge_predictions(sub, ids, preds), fallback_rate(art))
    d = os.path.dirname(OUT_PATH)
    if d:
        os.makedirs(d, exist_ok=True)
    sub.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Saved: {OUT_PATH} (rows={len(sub):,}) in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
