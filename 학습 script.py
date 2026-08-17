"""
학습 script.py — 제출 아티팩트 생성 (FE_S 앙상블 v2)
====================================================
실행:  python "학습 script.py"
입력:  ./data/train.csv, ./data/test.csv
출력:  ./model/model.pkl, ./model/fe_runtime.py 사용

구성
  1) 파생 피처 (feature_engineering.transform) + season 복원 = 101 피처
     - season 을 되살리는 것이 중요하다. 제거 시 fold B 656.7 -> 27.3 붕괴.
       트리가 2025 를 외삽하진 못해도 마지막 구간에 클램프되어 최근 수준을 붙잡는다.
  2) 4모델 앙상블 (가중 로짓 평균)
       c_hreg 0.45  HistGradientBoosting 강정규화   <- 개별 최고 (foldB 724.4)
       c_cat  0.40  CatBoost                        <- 체제붕괴 방어 (foldA 429.2)
       c_wide 0.15  LightGBM 넓은트리+강한 샘플링

     56개 3모델 조합 전수 탐색에서 선택. 계열이 셋 다 다르다.
     로버스트 상위 10개 중 9개가 CatBoost 를 포함했다.
  3) 구간 상대 오프셋 (n / count / matchup x count), gamma=0.2
  4) 세그먼트 앵커 (game_type) + 분산 수축

검증 (2019-2023 -> 2024 = 제출과 동형)
  베이스라인 RandomForest 415.6 / foldA 0.0
  본 구성              744.6 / foldA 449.4

버전 안전성
  아티팩트에 pandas 객체를 넣지 않는다. 범주형은 정수 코드로 다루고
  카테고리 목록은 순수 list[str] 로 저장한다. 학습/평가 서버의 pandas
  버전이 달라도 추론이 깨지지 않는다.
"""
import gc
import json
import os
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
sys.path.insert(0, MODEL_DIR)
import feature_engineering as fe  # noqa: E402

for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DATA_DIR = os.environ.get("DATA_DIR", "./data")
OUT = os.path.join(MODEL_DIR, "model.pkl")

ID, TARGET = "row_id", "control_success"
SEG_COL = "game_type"
TARGET_SEASON = 2025
CHUNK = 250_000
# 최종 학습에 쓸 행 비율. 1.0 이 정상이며 반드시 1.0 으로 제출용 모델을 만들 것.
# 메모리가 부족한 환경에서 파이프라인을 검증할 때만 낮춘다.
# (sklearn HistGradientBoosting 은 입력을 float64 로 변환한다:
#  1.47M x 101 x 8B = 1.19GB. 8GB 이상 환경이면 1.0 으로 문제없다.)
TRAIN_FRAC = float(os.environ.get("TRAIN_FRAC", "1.0"))

# ---- 확정 하이퍼파라미터 (fold A/B 교차 검증 결과) ----
WEIGHTS = {"c_hreg": 0.45, "c_cat": 0.40, "c_wide": 0.15}
OFFSET_KEYS = ["n", "count", "mu_count"]
OFFSET_K = 500.0        # 표본수 수축
OFFSET_GAMMA = 0.2      # 전역 감쇠 — holdout 측정치의 20%만 사용
ALPHA = 0.50            # 앵커 추세 감쇠
LAM = {"R": 1.00, "F": 0.20}
LAM_DEFAULT = 0.50
ANCHOR_CLIP = (0.30, 0.70)
EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bss(y, p):
    y = np.asarray(y, float)
    r = y.mean()
    return 100000 * (1 - np.mean((np.clip(p, 0, 1) - y) ** 2) / (r * (1 - r)))


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


# ---------------------------------------------------------------- 모델
def build(name, cat_idx):
    import lightgbm as lgb
    from sklearn.ensemble import HistGradientBoostingClassifier
    if name == "c_hreg":
        return "hgb", HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=1000,
            l2_regularization=20.0, categorical_features=cat_idx,
            early_stopping=False, random_state=11)
    if name == "c_cat":
        from catboost import CatBoostClassifier
        return "cat", CatBoostClassifier(
            iterations=700, learning_rate=0.05, depth=6, l2_leaf_reg=20.0,
            rsm=0.6, subsample=0.8, bootstrap_type="Bernoulli",
            border_count=64, boosting_type="Plain", max_ctr_complexity=0,
            random_seed=5, verbose=0, thread_count=1, allow_writing_files=False)
    if name == "c_wide":
        return "lgb", lgb.LGBMClassifier(
            objective="binary", n_estimators=700, learning_rate=0.03, num_leaves=63,
            min_child_samples=2000, reg_lambda=20.0, colsample_bytree=0.4,
            subsample=0.7, subsample_freq=1, random_state=2025, n_jobs=-1, verbose=-1)
    raise ValueError(name)


def spill_rows(X, mask, path, chunk=200_000):
    """마스크로 고른 행을 디스크 memmap 으로 복사한다.

    X[mask] 는 한 번에 사본을 만들어 (1.2M x 101 float32 = 0.49GB) 학습과 겹치면
    메모리가 부족해진다. 청크로 나눠 쓰고 memmap 으로 다시 연다.
    """
    idx = np.flatnonzero(mask)
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                    shape=(len(idx), X.shape[1]))
    for s in range(0, len(idx), chunk):
        e = min(s + chunk, len(idx))
        out[s:e] = X[idx[s:e]]
    out.flush()
    del out
    gc.collect()
    return np.load(path, mmap_mode="r")


def fit_pool(X, y, cat_idx, tag, spill_dir=None):
    """모델을 하나씩 학습하고 즉시 디스크로 내보낸 뒤 메모리에서 해제한다.

    4개를 동시에 들고 있으면 학습 행렬과 겹쳐 메모리가 부족해질 수 있다.
    (본 컨테이너 3GB 기준. 대회 서버 28GB 에서는 여유가 있다.)
    """
    paths = {}
    for name in WEIGHTS:
        kind, m = build(name, cat_idx)
        t = time.time()
        if kind == "lgb":
            m.fit(X, y, categorical_feature=cat_idx)
        else:
            m.fit(X, y)
        print(f"    [{tag}] {name} ({kind}) {time.time()-t:.0f}s", flush=True)
        if spill_dir:
            fp = os.path.join(spill_dir, f"_spill_{name}.pkl")
            joblib.dump((kind, m), fp, compress=0)
            paths[name] = fp
        else:
            paths[name] = (kind, m)
        del m
        gc.collect()
    if not spill_dir:
        return paths
    out = {}
    for name, fp in paths.items():
        out[name] = joblib.load(fp)
        os.remove(fp)
    return out


def _worker(name, xtr, ytr, xpr, cat_idx, out_pkl, out_npy):
    """모델 1개를 별도 프로세스에서 학습. 종료 시 OS 가 메모리를 완전히 회수한다."""
    import subprocess
    code = (
        "import sys,json,joblib,numpy as np;"
        "sys.path.insert(0,%r);"
        "import importlib.util as iu;"
        "spec=iu.spec_from_file_location('trainmod',%r);"
        "M=iu.module_from_spec(spec);spec.loader.exec_module(M);"
        "X=np.load(%r,mmap_mode='r');y=np.load(%r);"
        "ci=json.loads(%r);"
        "kind,m=M.build(%r,ci);"
        "m.fit(X,y,categorical_feature=ci) if kind=='lgb' else m.fit(X,y);"
        "Xp=np.load(%r,mmap_mode='r');"
        "np.save(%r, M.logit(m.predict_proba(Xp)[:,1]).astype('float64'));"
        "joblib.dump((kind,m),%r,compress=0) if %r else None"
    ) % (MODEL_DIR, os.path.abspath(__file__), xtr, ytr,
         json.dumps(cat_idx), name, xpr, out_npy, out_pkl, bool(out_pkl))
    r = subprocess.run([sys.executable, "-c", code],
                       env={**os.environ, "TRAIN_WORKER": "1",
                            "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    if r.returncode != 0:
        raise RuntimeError(f"worker {name} 실패 (code {r.returncode})")


def fit_pool_proc(xtr, ytr, xpr, cat_idx, tag, keep_models):
    """모델별 프로세스 격리 학습. 가중 로짓 평균과 (선택적) 모델 객체를 반환."""
    tot = sum(WEIGHTS.values())
    z = None
    models = {}
    for name in WEIGHTS:
        pk = os.path.join(MODEL_DIR, f"_m_{name}.pkl") if keep_models else ""
        zp = os.path.join(MODEL_DIR, f"_z_{name}.npy")
        t = time.time()
        _worker(name, xtr, ytr, xpr, cat_idx, pk, zp)
        zi = np.load(zp)
        os.remove(zp)
        z = WEIGHTS[name] * zi if z is None else z + WEIGHTS[name] * zi
        print(f"    [{tag}] {name} {time.time()-t:.0f}s", flush=True)
        if keep_models:
            models[name] = joblib.load(pk)
            os.remove(pk)
        gc.collect()
    return z / tot, models


def fit_predict_stream(Xtr, ytr, Xpr, cat_idx, tag):
    """모델을 하나씩 학습 -> 즉시 예측 -> 해제. stage1 은 모델을 보관할 필요가 없다.
    4개를 동시에 들고 있으면 1.2M x 101 행렬 사본과 겹쳐 OOM 이 난다."""
    tot = sum(WEIGHTS.values())
    z = np.zeros(len(Xpr), dtype=np.float64)
    for name in WEIGHTS:
        kind, m = build(name, cat_idx)
        t = time.time()
        if kind == "lgb":
            m.fit(Xtr, ytr, categorical_feature=cat_idx)
        else:
            m.fit(Xtr, ytr)
        z += WEIGHTS[name] * logit(m.predict_proba(Xpr)[:, 1])
        print(f"    [{tag}] {name} ({kind}) {time.time()-t:.0f}s", flush=True)
        del m
        gc.collect()
    return z / tot


def predict_pool(pool, X):
    z = np.zeros(len(X), dtype=np.float64)
    for name, (_, m) in pool.items():
        z += WEIGHTS[name] * logit(m.predict_proba(X)[:, 1])
    return z / sum(WEIGHTS.values())


# ---------------------------------------------------------------- 피처 행렬
def build_matrix(df, meta, cat_maps, cols, path=None):
    """청크 변환. 원본 transform 을 1.47M 행에 한 번에 돌리면 OOM 이 난다.

    행렬 자체도 디스크 memmap 에 쓴다. 메모리에 들고 있으면 마스크 슬라이싱
    사본과 겹쳐 학습 단계에서 OOM 이 난다.
    """
    n = len(df)
    X = (np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                   shape=(n, len(cols)))
         if path else np.empty((n, len(cols)), dtype=np.float32))
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        part = fe.transform(df.iloc[s:e], meta)
        for j, c in enumerate(cols):
            if c == "season":
                X[s:e, j] = df["season"].to_numpy(np.float32)[s:e]
            elif c in cat_maps:
                X[s:e, j] = part[c].astype(str).map(
                    {v: i for i, v in enumerate(cat_maps[c])}).fillna(-1).to_numpy(np.float32)
            else:
                X[s:e, j] = pd.to_numeric(part[c], errors="coerce").to_numpy(np.float32)
        del part
        gc.collect()
        print(f"    transform {e:,}/{n:,}", flush=True)
    if path:
        X.flush()
        del X
        gc.collect()
        return np.load(path, mmap_mode="r")
    return X


def plain_meta(meta):
    """pandas 객체를 제거해 버전 안전한 형태로 변환."""
    out = {"priors": {k: float(v) for k, v in meta["priors"].items()},
           "max_hist_season": int(meta["max_hist_season"]), "trackman": None}
    for k in ("pitcher_hist", "batter_hist"):
        h = meta[k]
        out[k] = None if h is None or len(h) == 0 else {
            c: h[c].to_numpy() for c in h.columns}
    return out


# ---------------------------------------------------------------- 오프셋
def rel_offsets(z_h, y_h, keys_h):
    p = sigmoid(z_h)
    g_all = logit(np.clip(y_h.mean(), EPS, 1 - EPS)) - logit(p.mean())
    d = pd.DataFrame({"k": keys_h, "p": p, "y": y_h})
    g = d.groupby("k", observed=True).agg(n=("y", "size"), ym=("y", "mean"), pm=("p", "mean"))
    g = g[g["n"] >= 300]
    raw = logit(g["ym"].clip(0.02, 0.98)) - logit(g["pm"]) - g_all
    return {str(k): float(v) for k, v in
            (raw * g["n"] / (g["n"] + OFFSET_K)).items()}


def compute_anchors(tr, target_season):
    seg = tr[SEG_COL].astype(str)
    rates = tr.groupby([seg, "season"], observed=True)[TARGET].mean()
    lv = rates.index.get_level_values(0).unique()
    rR = rates.loc["R"] if "R" in lv else rates.groupby(level=1).mean()
    sl, ic = np.polyfit(rR.index.astype(float), rR.to_numpy(), 1)
    last, trend = float(rR.iloc[-1]), float(sl * target_season + ic)
    aR = float(np.clip(last + ALPHA * (trend - last), *ANCHOR_CLIP))
    out = {"R": aR}
    for g in lv:
        if g != "R":
            out[str(g)] = aR      # delta=0: F 앵커를 R 과 동일하게 고정
    return out, {"r_last_R": last, "trend_R": trend, "alpha": ALPHA}


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                            encoding="utf-8-sig", nrows=0).columns.tolist()
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig",
                        usecols=test_cols + [TARGET])
    seasons = sorted(train["season"].unique())
    hold = int(seasons[-1])
    print(f"train {train.shape} | 시즌 {int(seasons[0])}~{hold}")
    for s, r in train.groupby("season", observed=True)[TARGET].mean().items():
        print(f"  {int(s)}: {r:.4f}")

    print("\n[1/5] 피처 메타 생성")
    meta = fe.fit_meta(train, None)
    probe = fe.transform(train.iloc[:2000], meta)
    cat_cols = [c for c in probe.columns if str(probe[c].dtype) == "category"]
    cols = list(probe.columns) + ["season"]
    cat_maps = {}
    for c in cat_cols:
        src = train[c] if c in train.columns else probe[c]
        cat_maps[c] = sorted(set(map(str, pd.unique(src.astype(str)))))
    cat_idx = [cols.index(c) for c in cat_cols]
    print(f"  피처 {len(cols)}개 (범주형 {len(cat_cols)})")

    print("\n[2/5] 피처 행렬 변환")
    MATPATH = os.path.join(MODEL_DIR, "_X.npy")
    if os.path.exists(MATPATH):
        X = np.load(MATPATH, mmap_mode="r")
        print(f"  캐시 재사용 {X.shape}")
    else:
        X = build_matrix(train, meta, cat_maps, cols, MATPATH)
    y = train[TARGET].to_numpy()
    season = train["season"].astype(int).to_numpy()
    print(f"  X {X.shape} {X.nbytes/1e9:.2f}GB")

    # 자식 프로세스가 학습하는 동안 부모가 train DataFrame(약 0.6GB)을 들고 있으면
    # 합산 메모리가 한계를 넘는다. 이후 필요한 최소 컬럼만 남기고 해제한다.
    m_pre, m_hold = season < hold, season == hold
    KEYCOLS = ["game_type", "asof_pitcher_n", "balls_before", "strikes_before",
               "pitcher_hand", "batter_hand"]
    hold_df = train.loc[m_hold, KEYCOLS].reset_index(drop=True)
    anchor_df = train[["season", "game_type", TARGET]].copy()
    del train
    gc.collect()

    CKPT = os.path.join(MODEL_DIR, "_const.pkl")
    print(f"\n[3/5] 보정 상수 산출 (fit <={hold-1} -> predict {hold})")
    if os.path.exists(CKPT):
        seg_mean_logit, offsets = joblib.load(CKPT)
        print("  체크포인트 재사용")
        PRE = HLD = None
    else:
        PRE = os.path.join(MODEL_DIR, "_Xpre.npy")
        HLD = os.path.join(MODEL_DIR, "_Xhold.npy")
        Xpre = spill_rows(X, m_pre, PRE)
        Xhold = spill_rows(X, m_hold, HLD)
        del Xpre, Xhold
        gc.collect()
        YPRE = os.path.join(MODEL_DIR, "_ypre.npy")
        np.save(YPRE, y[m_pre])
        z_hold, _ = fit_pool_proc(PRE, YPRE, HLD, cat_idx, "stage1", keep_models=False)
        for f in (YPRE, PRE, HLD):
            if os.path.exists(f):
                os.remove(f)

        seg_hold = hold_df[SEG_COL].astype(str).to_numpy()
        seg_mean_logit = {g: float(z_hold[seg_hold == g].mean())
                          for g in np.unique(seg_hold)}
        offsets = {k: rel_offsets(z_hold, y[m_hold], KEYFN[k](hold_df))
                   for k in OFFSET_KEYS}
        print(f"  [{hold} holdout 보정 전] BSS={bss(y[m_hold], sigmoid(z_hold)):.1f}")
        del z_hold
        gc.collect()
        joblib.dump((seg_mean_logit, offsets), CKPT, compress=0)
    print(f"  seg_mean_logit={ {k: round(v,4) for k,v in seg_mean_logit.items()} }")
    print(f"  offsets: " + ", ".join(f"{k}({len(v)}그룹)" for k, v in offsets.items()))

    print(f"\n[4/5] 앵커 (목표 시즌 {TARGET_SEASON})")
    anchors, ameta = compute_anchors(anchor_df, TARGET_SEASON)
    print(f"  r_last_R={ameta['r_last_R']:.4f} trend={ameta['trend_R']:.4f} "
          f"alpha={ALPHA} -> { {k: round(v,4) for k,v in anchors.items()} }")

    print(f"\n[5/5] 최종 모델 학습 (전체 {len(X):,}행)")
    del hold_df, anchor_df
    gc.collect()
    YALL = os.path.join(MODEL_DIR, "_yall.npy")
    HEAD = os.path.join(MODEL_DIR, "_Xhead.npy")
    np.save(HEAD, np.ascontiguousarray(X[:1000]))
    if TRAIN_FRAC >= 1.0:
        FITMAT = MATPATH
        np.save(YALL, y)
    else:
        rs = np.random.RandomState(0)
        sel = np.sort(rs.choice(len(y), int(len(y) * TRAIN_FRAC), replace=False))
        FITMAT = os.path.join(MODEL_DIR, "_Xsub.npy")
        sub = np.lib.format.open_memmap(FITMAT, mode="w+", dtype=np.float32,
                                        shape=(len(sel), X.shape[1]))
        for a in range(0, len(sel), 200_000):
            b = min(a + 200_000, len(sel))
            sub[a:b] = X[sel[a:b]]
        sub.flush()
        del sub
        np.save(YALL, y[sel])
        print(f"  ⚠️ TRAIN_FRAC={TRAIN_FRAC} — 검증용 축소 학습 "
              f"({len(sel):,}행). 제출 모델은 반드시 TRAIN_FRAC=1.0 으로 생성할 것.")
    del X
    gc.collect()
    _, final = fit_pool_proc(FITMAT, YALL, HEAD, cat_idx, "final", keep_models=True)
    if FITMAT != MATPATH and os.path.exists(FITMAT):
        os.remove(FITMAT)
    for f in (YALL, HEAD):
        if os.path.exists(f):
            os.remove(f)
    if os.path.exists(MATPATH):
        os.remove(MATPATH)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({
        "models": {k: v for k, v in final.items()},
        "weights": dict(WEIGHTS),
        "fe_meta": plain_meta(meta),
        "cols": cols,
        "cat_maps": cat_maps,
        "cat_idx": cat_idx,
        "seg_col": SEG_COL,
        "seg_mean_logit": seg_mean_logit,
        "offsets": offsets,
        "offset_keys": OFFSET_KEYS,
        "offset_gamma": OFFSET_GAMMA,
        "anchors": anchors,
        "lam": LAM,
        "lam_default": LAM_DEFAULT,
        "meta": {**ameta, "target_season": TARGET_SEASON, "train_frac": TRAIN_FRAC,
                 "train_seasons": [int(s) for s in seasons], "n_train": int(len(y))},
    }, OUT, compress=3)
    for f in ("_X.npy", "_const.pkl", "_yall.npy", "_Xhead.npy"):
        fp = os.path.join(MODEL_DIR, f)
        if os.path.exists(fp):
            os.remove(fp)
    print(f"\n저장: {OUT} ({os.path.getsize(OUT)/1e6:.1f}MB) | 총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
