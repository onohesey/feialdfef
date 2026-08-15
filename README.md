# aimers — KBO 투구 제구 성공 확률 예측

> 투구 단위의 **제구 성공 확률(`control_success`)** 을 예측하는 해커톤 프로젝트입니다.
> 경기 후 집계 지표(평균자책점, 볼넷수 등) 대신 **투구 직전까지 확인 가능한 정보**(볼카운트, 주자 상황, 타자·투수 특성, 과거 이력 등)만으로 해당 투구의 제구 성공 확률을 예측합니다.

---

## 1. 문제 정의

- **Target**: `control_success` (학습 데이터 기준 성공=1, 실패=0 / 예측은 0~1 확률값)
- **제구 실패**로 판단하는 경우 (아래 중 하나)
  1. 스트라이크존 가운데 부근으로 들어간 공
  2. 스트라이크존에서 크게 벗어난 공
  3. 포수의 요구 방향과 반대로 들어간 공
- 그 외 유효 투구는 제구 성공으로 간주

## 2. 평가 방식

코드 제출형 대회이며, **Brier Skill Score**로 평가합니다.

```
Score = max(0, 100000 × (1 - Brier Score / 평균 제구율 Brier Score))

Brier Score        = mean((p_i - y_i)^2)
r                   = mean(y_i)              # 비공개 수치
평균 제구율 Brier Score = r × (1 - r)
```

| 항목 | 제약 |
|---|---|
| 추론 실행 시간 | ≤ 10분 (실 평가 245,789 샘플 기준) |
| 패키지 설치 시간 | ≤ 10분 |
| 제출 파일 용량 | ≤ 10GB (압축 해제 후 최대 32GB) |
| 실행 환경 | 오프라인 (패키지 설치 외 인터넷 불가) |
| 서버 스펙 | 6 vCPU, 28GB RAM, L4 GPU 22.4GiB VRAM, Python 3.11.15, CUDA 12.8 |

## 3. 저장소 구성

```
aimers/
├── 학습 script.py     # 베이스라인 학습 코드 (RandomForest 파이프라인 학습 → ./model/rf.pkl 저장)
├── 추론 script.py     # 평가 서버가 자동 실행하는 추론 코드 (./data → ./output/submission.csv)
├── requirements.txt   # 추론에 필요한 패키지/버전
├── data_description.md # 데이터 컬럼 상세 설명
└── README.md
```

> 대회 제출용 `submit.zip`은 `model/`, `script.py`, `requirements.txt`로 구성되어야 하며,
> 이 저장소의 `추론 script.py`가 제출용 `script.py`에 해당합니다.

### 제출 시 평가 서버가 자동으로 추가하는 항목

```
submit.zip
├── model/                    # 참가자 구성 (학습 script.py 실행 결과)
├── script.py                 # 참가자 구성 (추론 script.py)
├── requirements.txt          # 참가자 구성
├── data/                     # 평가용 테스트 데이터 (읽기 전용, 자동 생성)
└── output/submission.csv     # 추론 결과 저장 경로 (자동 생성)
```

## 4. 데이터

`data_description.md`에 컬럼별 상세 설명이 정리되어 있습니다. 요약:

| 파일 | 설명 |
|---|---|
| `train.csv` | 학습 데이터 (1,475,092행 × 49컬럼, 2019~2023 시즌) |
| `test.csv` | 평가 입력 (실 평가는 245,789행, 배포본은 형식 확인용 5건 샘플) |
| `sample_submission.csv` | 제출 양식 (`row_id`, `control_success`) |
| `trackman_history.csv` | 2019~2024 Trackman 로그 (1,793,078행 × 30컬럼, `train`/`test`와 1:1 결합 테이블 아님) |

핵심 피처군은 운영 측이 **투구 직전 시점까지의 기록으로 사전 계산**해 제공하는 `asof_*` 컬럼(투수/타자 성공률, middle rate, 최근 경기 추세 등)입니다. 사용할 피처는 반드시 **`test.csv` 컬럼 기준**으로 결정해야 합니다(`train.csv`에만 있는 컬럼은 평가 시 존재하지 않음).

## 5. 베이스라인 모델

- **입력**: `test.csv`의 47개 컬럼 (`row_id` 제외) — 범주형 3개(`top_bottom`, `game_type`, `base_state`) + 수치형 44개
- **전처리**: `ColumnTransformer` — 범주형 `OrdinalEncoder`(미지 범주 -1), 수치형 `SimpleImputer`(median)
- **모델**: `RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=200, random_state=42)`
- **검증 분할**: `train.csv`에서 `season==2024`를 검증셋으로 분리, 나머지로 학습
- **저장**: 파이프라인 전체를 `./model/rf.pkl`로 joblib 저장(`compress=3`)
- `trackman_history.csv`는 베이스라인에서 미사용 → 개선 여지

### 실행 방법

```bash
pip install -r requirements.txt

# 1) 학습 — ./data/train.csv, test.csv 필요 → ./model/rf.pkl 생성
python "학습 script.py"

# 2) 추론 — ./data/test.csv, sample_submission.csv, ./model/rf.pkl 필요
#            → ./output/submission.csv 생성
python "추론 script.py"
```

## 6. 규칙 (⚠️ 반드시 준수)

1. **사전학습 모델/가중치**: 공식 공개 + 최소 비상업적 이용 허용 라이선스(MIT, Apache 2.0 등)만 사용 가능
2. **외부 API 금지**: OpenAI API, Gemini API 등 원격 서버 기반 API 사용 불가 — 모든 작업은 로컬에서 실행/재현 가능해야 함
3. **외부 데이터 금지**: 공식 제공 데이터 외 외부 데이터 사용 불가
4. **행 단위 독립 예측 원칙**: 평가 데이터의 각 행은 독립적 예측 대상 — 다른 행이나 전체 분포를 이용한 보정/생성 금지 (data leakage)

특정 행 A의 예측값은 아래 정보만으로 생성되어야 합니다.

- 행 A에 포함된 입력 변수
- 행 A의 입력 변수만을 이용해 생성한 파생변수
- 주최 측이 제공한 공식 학습 데이터
- 공식 학습 데이터만을 이용해 생성한 통계·모델·파생변수

### 명시적으로 금지되는 leakage 패턴

- `test.csv` 내부 행들을 이용한 선수별/팀별/월별 누적 통계, 빈도값/분포 통계, target encoding
- `test.csv` 행 순서 기반 rolling/expanding feature
- 평가 데이터 전체를 보고 만든 사후 보정값
- 같은 선수·팀·월·경기 단위로 평가 데이터 내 다른 행을 집계하는 방식
- 평가 데이터상 시점이 더 과거로 보이는 행을 이용해 현재 행의 피처를 생성하는 방식 (`test.csv` 내 시점 정보가 있어도 다른 행 사용은 전부 불가)

> ✅ 단, 운영 측이 제공한 `asof_*` 컬럼은 투구 직전 시점까지의 과거 기록으로 사전 계산된 공식 입력 피처이므로 그대로 사용 가능합니다.

행 독립성은 다음과 같이 판별합니다: 어떤 행의 예측값은 `test.csv`에 그 행 하나만 있을 때와 전체 평가 데이터가 함께 있을 때 **동일해야** 합니다. 다르면 다른 행이 추론에 영향을 준 것으로 간주됩니다. 새 피처를 추가할 때마다 아래처럼 직접 검증하는 것을 권장합니다.

```python
# 전체 test로 추론
full_preds = model.predict_proba(build_features(test))[:, 1]

# 동일 행 1개만 떼어 추론
single = test.iloc[[0]]
single_pred = model.predict_proba(build_features(single))[:, 1]

assert abs(full_preds[0] - single_pred[0]) < 1e-9, "행 독립성 위반 가능성 — test 내 다른 행이 영향을 줌"
```

### 입력으로 사용 금지된 정보

현재 투구 이후에 확정되는 모든 정보, 실제 위치/코스, 실제 판정·결과·제구 성공 여부, 실제 구종, 해당 투구 자체의 Trackman 측정값, 2025년 Trackman 데이터(미제공)는 입력으로 사용할 수 없습니다.

### 생성 AI 사용 및 실격 관련 유의사항

생성 AI를 활용하여 코드를 작성하는 것 자체는 금지되지 않습니다. 다만 생성 AI가 작성한 코드라 하더라도 최종 제출 코드가 위 규칙을 준수하는지 확인할 책임은 참가자 본인에게 있으며, 검토 없이 그대로 제출하여 발생하는 불이익은 참가자에게 귀속됩니다.

운영진은 제출된 코드의 추론 과정을 상시 모니터링하며, 규칙 위반이 확인되는 경우 제출 시점부터 적용되는 규정에 따라 실격될 수 있습니다. 규칙을 충분히 숙지하지 않았거나 제출 전 점검을 하지 않았다는 사유, 그리고 제출 이후의 삭제·제외 요청은 면책 사유로 인정되지 않습니다.

## 7. 제출 전 체크리스트

- [ ] `submit.zip` 최상위 구조가 `model/`, `script.py`, `requirements.txt`와 정확히 일치하는가
- [ ] `requirements.txt`가 서버 기본 설치 패키지와 버전 충돌 없는가
- [ ] `script.py`가 `./data/test.csv` → `./output/submission.csv` 경로를 정확히 사용하는가
- [ ] 인터넷 연결 없이 로컬에서 전체 파이프라인이 재현 가능한가
- [ ] 사용 모델/가중치가 라이선스 조건(MIT, Apache 2.0 등)을 만족하는가
- [ ] 외부 API, 외부 데이터를 사용하지 않았는가
- [ ] 특정 행 예측 시 다른 행/전체 분포 정보를 사용하지 않았는가 (data leakage 없는가): 같은 선수·팀·월·경기 단위 집계, 시점상 과거로 보이는 다른 행 이용까지 포함해서 확인
- [ ] `test.csv` 행 1개만으로 추론했을 때와 전체로 추론했을 때 동일 행의 예측값이 일치하는지 직접 테스트했는가 (6장 스니펫 참고)
- [ ] GroupBy, rolling, target encoding, `test` 전체 통계(mean/std/rank 등) 코드가 추론 스크립트 어디에도 없는지 전체 코드 검색으로 확인했는가
- [ ] 추론 시간이 10분 이내(245,789 샘플 기준)로 예상되는가
