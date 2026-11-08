# A2_1 — AI가 어떻게 학습하는지 수학으로 직접 풀어보기

선형대수·미적분·역전파·최적화·확률의 수식을 직접 유도하고 NumPy로 구현·검증한 프로젝트입니다. [미션 원문](A2_1_mission.md) 2절의 필수 결과물과 4절의 기능 요구를 대상으로 하며, 코드·실행된 노트북·PNG·측정 JSON을 함께 제공합니다. Adam·Newton·정보 이론 보너스는 필수 범위에 포함하지 않습니다.

**제출 저장소:** [https://github.com/ntt65/A2_1](https://github.com/ntt65/A2_1). PR #11~#18을 기본 브랜치 `main`에 병합했습니다. 평가·재현은 `main`의 코드와 아래 실행 방법을 기준으로 합니다.

**기본 미션 통합 검증:** 2026-11-08 새 Python 3.11.4 가상환경에서 잠금 의존성 설치, `pip check`, 전체 **129개 테스트(20.417초)**, 레지스트리 검사를 통과했습니다. 세 실험과 두 노트북을 재실행해 생성한 **PNG 10개·JSON 5개는 제출 산출물과 바이트 단위로 일치**했고 노트북 원본은 변경되지 않았습니다. [최종 구현 PR #18](https://github.com/ntt65/A2_1/pull/18)까지 병합했으며, 병합 전 최종 커밋의 Windows·Linux 검사도 통과했습니다. 병합 후 기본 브랜치 검사와 제출 마무리 기록은 [이슈 #19](https://github.com/ntt65/A2_1/issues/19)에서 확인할 수 있습니다. 명령·환경·감사 결과는 [최종 재현 기록](docs/worklogs/step-7.md)에 있습니다.

## Naito_AI 평가 보완: 고유값 비교 증거

사용자가 제공한 **2026-11-08 11:52:57 평가 결과는 93%(14/15 통과)**입니다. [평가 원문·15개 항목 분석·보완 계획](Naito_AI_eval_result.md)을 보존했습니다. 유일한 FAIL #3은 기존 테스트의 `np.linalg.eig` 비교가 제출용 실행 출력에 명확히 드러나지 않은 문제였습니다. [이슈 #23](https://github.com/ntt65/A2_1/issues/23)에서 전용 실행 보고서와 기본 실험 출력을 보완했습니다. **외부 재평가는 아직 실행하지 않았습니다.**

아래 설치를 마친 프로젝트 환경에서 `.\.venv\Scripts\python.exe -m scripts.verify_power_iteration`을 실행합니다(Linux: `.venv/bin/python -m scripts.verify_power_iteration`). 선형대수 기본 실행 명령도 같은 비교 증거를 생성합니다.

| 비교 항목 | 실제 실행 결과 |
|---|---:|
| 직접 구현한 Power Iteration λ | 4.618033988749896 |
| `np.linalg.eig` 기준 λ | 4.618033988749895 |
| 상대오차 `abs(λ_manual−λ_numpy)/abs(λ_numpy)` | 1.923282552410481e-16 ≤ 0.05(5%): PASS |
| 업데이트 횟수 / 상대 잔차 | 32 / 6.399563856969149e-11 |

입력 행렬은 `[[4,1],[1,3]]`, 초기 벡터는 `[1,1]`, dtype은 float64입니다. 두 고유벡터·부호 정렬·독립 닫힌 해 교차 검증은 [실행 보고서](outputs/power_iteration_eig_comparison.md)와 [전체 JSON](outputs/power_iteration_eig_comparison.json)에 있습니다. 검증 코드는 [scripts/verify_power_iteration.py](scripts/verify_power_iteration.py)에 있으며 알고리즘 본체는 그대로입니다. 1차 보완 후 **132개 테스트(26.959초)**를 통과했습니다. [작업 기록](docs/worklogs/issue-23.md)에 명령·보완 과정을 기록합니다.

기존 유도 증거도 함께 안내합니다. [역전파 노트북](notebooks/backprop_derivation.ipynb)의 `backprop-forward-derivation`, `backprop-chain-rule` 셀과 [확률 노트북](notebooks/probability_loss.ipynb)의 `gaussian-mle-derivation`, `bernoulli-mle-derivation`, `categorical-mle-derivation` 셀에 단계별 수식 전개가 있습니다. 같은 유도를 [일반 Markdown 수식 문서](docs/MATH_DERIVATIONS.md)에서도 읽을 수 있습니다.

### 나머지 평가 권고 재현

[이슈 #25](https://github.com/ntt65/A2_1/issues/25)는 출력 이미지의 실제 크기·DPI, 수치 미분 h와 Softmax 축별 오차, 의도적 불일치 진단 및 여러 초기점의 최적화 결과를 추가합니다.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_evaluation_evidence
.\.venv\Scripts\python.exe -m scripts.run_optimizer_sensitivity --no-cache
```

Linux에서는 실행 파일을 `.venv/bin/python`으로 바꿉니다. 기본 실험 산출물이 있는 저장소 루트에서 실행합니다. [수치·이미지 증거 보고서](outputs/evaluation_evidence.md)와 [최적화 민감도 보고서](outputs/optimizer_sensitivity.md)에 재현 조건·실패 진단·집계 모집단을 기록합니다. PNG 캔버스 크기는 SVD 입력의 64×64 조건과 구분하고, 여러 초기점의 결과는 동일 입력 반복실행의 평균으로 해석하지 않습니다. 전체 **144개 테스트(24.250초)**와 환경·색인 검사를 통과했으며 [검증 기록](docs/worklogs/issue-25.md)에 결과를 남겼습니다. 보너스 구현은 이 단계 다음에 이어집니다.

## 1. 설치와 재현

Python **3.11**과 Git이 필요합니다. 아래 명령은 새 작업 폴더에서 기본 브랜치의 완성본을 내려받고, 저장소 루트에서 실행합니다. 전체 의존성은 [requirements-lock.txt](requirements-lock.txt)로 고정했습니다. 구현·시각화 직접 의존성은 [requirements.txt](requirements.txt), 노트북 실행 도구는 [requirements-dev.txt](requirements-dev.txt)에 구분합니다.

### Windows PowerShell

```powershell
git clone --branch main --single-branch https://github.com/ntt65/A2_1.git
cd A2_1
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/update_registry.py --check
.\.venv\Scripts\python.exe -m scripts.run_linear_algebra --no-cache
.\.venv\Scripts\python.exe -m scripts.run_calculus
.\.venv\Scripts\python.exe -m scripts.run_optimizer --no-cache
.\.venv\Scripts\python.exe -m scripts.run_notebooks
.\.venv\Scripts\python.exe scripts/update_registry.py --check
```

### Linux / macOS

```sh
git clone --branch main --single-branch https://github.com/ntt65/A2_1.git
cd A2_1
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/update_registry.py --check
.venv/bin/python -m scripts.run_linear_algebra --no-cache
.venv/bin/python -m scripts.run_calculus
.venv/bin/python -m scripts.run_optimizer --no-cache
.venv/bin/python -m scripts.run_notebooks
.venv/bin/python scripts/update_registry.py --check
```

가상환경 활성화는 필요하지 않습니다. `python -O`는 assert를 제거하므로 사용하지 않습니다. 실험 세 개와 노트북 두 개가 `outputs/`의 PNG·JSON을 다시 생성합니다. 이슈 #23에서 선형대수 실험에 eig 비교 JSON·Markdown 각 1개가 추가되어 현재 산출물은 PNG 10개·JSON 6개·비교 Markdown 1개입니다. 선형대수·최적화의 `--no-cache`는 캐시 읽기와 쓰기를 모두 생략합니다. 미적분 CLI에는 이 옵션이 없습니다. 외부 데이터 다운로드 없이 seed 42의 합성 grayscale 이미지를 사용합니다.

`run_notebooks`에 경로를 생략하면 두 노트북을 각각 **현재 가상환경의 새 커널**에서 처음부터 실행합니다. 기본 실행은 `.ipynb` 원본을 변경하지 않으며, 기존 셀 출력의 성공 여부에 의존하지 않습니다. 재현 명령에는 `--write`를 사용하지 않습니다. 개발자가 실행 결과를 노트북에 다시 저장하려면 `--write` 후 `scripts/update_registry.py --update`로 파일 해시를 갱신해야 합니다.

세 실험과 노트북 실행 도구는 모두 `--output-dir`을 지원합니다. 예를 들어 Windows에서 다음 명령은 기존 제출 PNG·JSON을 덮어쓰지 않고 별도 폴더에 결과를 만듭니다.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_linear_algebra --no-cache --output-dir .cache/reproduce
.\.venv\Scripts\python.exe -m scripts.run_calculus --output-dir .cache/reproduce
.\.venv\Scripts\python.exe -m scripts.run_optimizer --no-cache --output-dir .cache/reproduce
.\.venv\Scripts\python.exe -m scripts.run_notebooks --output-dir .cache/reproduce
```

### 콘솔 결과 예시

기본 미션의 새 환경 실행 결과에 이슈 #23에서 실제 실행한 eig 비교 출력을 반영했습니다. 아래 저장 경로는 기본 명령의 `outputs`로 표기했으며, 최종 재현 검증은 별도 폴더 `.cache/integration-artifacts-20261108`에 생성해 기존 산출물과 비교했습니다. 저장 경로는 실행 위치와 `--output-dir`에 따라 달라집니다.

```text
Power Iteration: lambda=4.618033988750, relative error=1.923e-16, residual=6.400e-11
np.linalg.eig: lambda=4.618033988749895; relative error=1.923283e-16 <= 0.05 (5%): PASS
Figures and metrics saved in outputs
Derivative error: 3.931e-11; gradient error: 3.296e-11; orthogonality: 0.000e+00
Figures and metrics saved in outputs
circle: GD radius=1.440402e-09, loss=2.074758e-18; Momentum radius=2.016252e-02, loss=4.065273e-04
ellipse: GD radius=1.328070e-04, loss=1.763770e-08; Momentum radius=1.987508e-02, loss=8.051093e-04
Figures and metrics saved in outputs
Executed backprop_derivation.ipynb: 8 code cells passed; source unchanged
Executed probability_loss.ipynb: 9 code cells passed; source unchanged
Registry check passed.
```

## 2. 요구 사항별 결과와 증거

측정 환경은 Python 3.11.4 / NumPy 2.2.6 / Matplotlib 3.10.3, 수치 계산은 `float64`입니다. 아래는 반올림한 값이며, 전체 자릿수·입력·경로는 연결된 JSON에 있습니다. 허용오차는 [수용 기준](docs/ACCEPTANCE.md)을 따릅니다. 원문의 잘못된 참고 수치·상충 조건은 [결정 기록](docs/DECISIONS.md)에 근거를 남겼습니다.

| 원문 요구 | 실험 조건 | 실측 결과 | 허용오차·판정 기준 | 증거 |
|---|---|---|---|---|
| 4.1 회전·스케일·전단과 면적비 | 단위 원, 회전 π/4, 스케일 (2,0.5), 전단 k=1 | 면적비와 `abs(det(A))`의 최대 상대오차 5.55e-16 | ≤1% | [모듈](src/linear_algebra.py), [변환 PNG](outputs/linear_transforms.png), [JSON](outputs/linear_algebra_metrics.json) |
| 4.1 직접 구현한 Power Iteration | A=[[4,1],[1,3]], 초기 벡터 [1,1] | λ_manual=4.618033988749896, λ_numpy=4.618033988749895, v≈[0.850651,0.525731], 32회, 상대오차 1.92e-16, 상대잔차 6.40e-11 | 고유값 상대오차 ≤5%; 잔차 ≤1e-6 | [eig 비교 실행 보고서](outputs/power_iteration_eig_comparison.md), [두 계산 비교 JSON](outputs/power_iteration_eig_comparison.json), [비교 테스트](tests/test_power_iteration_verification.py) |
| 4.1 SVD 이미지 복원 | 64×64: k=10,50,64; 보충 128×128: k=10,50,100 | 완전 rank 복원 상대오차 2.36e-15 / 1.10e-15; rank 증가 시 오차 감소 | 완전 rank ≤1e-10, 유효 rank·비증가 오차 확인 | [64 PNG](outputs/svd_64.png), [128 PNG](outputs/svd_128.png), [JSON](outputs/linear_algebra_metrics.json) |
| 4.2 중심차분 | f(x)=x², x=3, h=1e-5 | 6.000000000039306, 절대오차 3.93e-11 | ≤1e-4 | [모듈](src/calculus.py), [오차 PNG](outputs/derivative_error.png), [JSON](outputs/calculus_metrics.json) |
| 4.2 등고선과 Gradient | f=x²+y², 반경 2의 8개 점과 원점, gradient shape (9,2) | 수치 gradient 최대오차 3.30e-11; 비원점 gradient·접선 정규화 내적 0 | gradient ≤1e-8; 수직성 ≤1e-12; 원점은 방향 없는 영벡터 | [Gradient PNG](outputs/gradient_geometry.png), [JSON](outputs/calculus_metrics.json) |
| 4.3 순전파·역전파 손계산 | 원문 고정 예제, 단일 표본 Sigmoid+BCE | 13개 체크포인트 네 자리 표시 일치; 최대 절대오차 2.22e-16 | ≤5e-5 및 네 자리 표시 일치 | [역전파 노트북](notebooks/backprop_derivation.ipynb), [JSON](outputs/backprop_metrics.json) |
| 4.3 역전파 추가 독립 검증 | 총 9개 매개변수, 중심차분 h=1e-5 | 해석·수치 gradient 최대오차 7.84e-12 | ≤1e-6 | [역전파 노트북](notebooks/backprop_derivation.ipynb), [JSON](outputs/backprop_metrics.json) |
| 4.4 Vanilla GD 수렴 | f=x²+y², 초기 (5,5), lr=0.1, 100회 | 최종 [1.0185e-9,1.0185e-9], 반경 1.44e-9 | 반경 ≤0.1 | [모듈](src/optimizer.py), [원형 PNG](outputs/optimizer_circle.png), [JSON](outputs/optimizer_metrics.json) |
| 4.4 학습률별 경로 | 원형, 초기 (5,5), lr=0.5 / 1.0 / 1.1, 20회 | 최종 반경 0 / 7.0711 / 271.0878 | 즉시 수렴 / 진동 / 발산, 닫힌 반복식과 비교 | [학습률 PNG](outputs/optimizer_learning_rates.png), [D10](docs/DECISIONS.md#d10--원형-함수의-학습률-발산-조건을-정정한다) |
| 4.4 GD와 Momentum 비교 | β=0.9, 초기 (5,5), 100회; 원형 lr=0.1, 타원형 x²+10y² lr=0.05 | 최종 loss: 원형 GD 2.07e-18 / Momentum 4.07e-4; 타원형 GD 1.76e-8 / Momentum 8.05e-4 | 동일 조건·예산으로 비교; Momentum 우월을 가정하지 않음 | [원형 PNG](outputs/optimizer_circle.png), [타원형 PNG](outputs/optimizer_ellipse.png), [JSON](outputs/optimizer_metrics.json) |
| 4.5 정규분포 PDF 2종 | N(0,1), N(2,0.5); 두 번째 인자는 분산; 각 μ±8σ에서 10,001점 적분 | 적분 오차 1.11e-15 / 9.99e-16, PDF 비음수 | 각각 ≤1e-4 | [정규분포 PNG](outputs/probability_normal.png), [JSON](outputs/probability_metrics.json) |
| 4.5 Bernoulli PMF 2종 | p=0.3 / 0.7, support [0,1] | PMF [0.7,0.3] / [0.3,0.7], 합 오차 0 | 합 오차 ≤1e-12, 확률 [0,1] | [Bernoulli PNG](outputs/probability_bernoulli.png), [JSON](outputs/probability_metrics.json) |
| 4.5 Softmax | shape (2,3), axis=-1; 큰 logit ±1000, axis=0도 검증 | 확률 합 최대오차 1.11e-16, 유한 출력 | ≤1e-6 | [공통 구현](src/utils/math.py), [확률 노트북](notebooks/probability_loss.ipynb), [JSON](outputs/probability_metrics.json) |
| 4.5 MLE와 손실 연결 | iid Gaussian 오차·고정 양의 공통 분산, Bernoulli, one-hot categorical을 각각 유도 | n=3, 분산 0.5, SSE=1.5에서 NLL=3.217094828774, MSE=0.5; BCE합=0.713349887877, CE=0.693147180560 | 독립 상수 오차 Gaussian ≤1e-14, BCE·CE ≤1e-15; BCE gradient 오차 8.63e-12 ≤1e-8 | [확률 노트북](notebooks/probability_loss.ipynb), [JSON](outputs/probability_metrics.json) |

## 3. 역전파 고정 예제

단일 표본을 사용하며 `z1=W1@x+b1`, `a1=sigmoid(z1)`, `z2=W2@a1+b2`, `p=sigmoid(z2)`입니다. 손실은 자연로그의 BCE `L=−y log(p)−(1−y)log(1−p)`입니다. 중간값은 반올림하지 않고 계산하고, 아래 표만 소수점 네 자리로 표시합니다.

| 입력·매개변수 | 고정값 | shape |
|---|---|---|
| x | [1,0] | (2,) |
| W1 | [[0.1,0.2],[0.3,0.4]] | (2,2) |
| b1 | [0,0] | (2,) |
| W2 | [0.5,0.6] | (2,) |
| b2 / y_true | 0 / 1 | 각각 scalar () |

| 순전파·필수 미분 | 식 | NumPy 결과(4자리) | shape |
|---|---|---|---|
| z1 | W1@x+b1 | [0.1000,0.3000] | (2,) |
| a1 | sigmoid(z1) | [0.5250,0.5744] | (2,) |
| z2 | W2@a1+b2 | 0.6072 | () |
| p=y_pred | sigmoid(z2) | 0.6473 | () |
| L | BCE(y,p) | 0.4350 | () |
| dL/dp | (p−y)/(p(1−p)) | −1.5449 | () |
| δ2=dL/dz2 | p−y | −0.3527 | () |
| dL/dW2 | δ2·a1 | [−0.1852,−0.2026] | (2,) |
| dL/da1 | δ2·W2 | [−0.1764,−0.2116] | (2,) |
| δ1=dL/dz1 | (δ2·W2)·a1·(1−a1) | [−0.0440,−0.0517] | (2,) |
| dL/dW1 | outer(δ1,x) | [[−0.0440,0.0000],[−0.0517,0.0000]] | (2,2) |
| dL/db1 / dL/db2 | δ1 / δ2 | [−0.0440,−0.0517] / −0.3527 | (2,) / () |

곱 `·`는 해당 벡터식의 원소별 곱입니다. 두 번째 입력이 0이므로 `dL/dW1`의 두 번째 열은 0입니다. 수식별 손계산 과정과 독립 기준값, 9개 매개변수의 중심차분 비교는 [실행된 역전파 노트북](notebooks/backprop_derivation.ipynb)과 [상세 설명](docs/BACKPROP.md)에 있습니다.

## 4. 원문과의 차이 및 결과 해석

**SVD 크기와 rank:** 64×64 이하 이미지에서는 k=100을 유효한 rank로 지정할 수 없습니다. 본 실험은 64×64의 k=10,50,64, 보충 실험은 128×128의 k=10,50,100으로 분리했습니다. 두 조건을 하나의 실험으로 동시에 만족한다고 해석하지 않습니다. 저장 원소 수는 `k(m+n+1)`이므로 원본 `mn`보다 커질 수 있습니다. 128×128의 k=100은 복원 오차가 약 1.22%지만 저장 원소 비율은 약 1.5686배여서 저장 공간 절감이 아닙니다. PNG 파일 크기와 SVD 인자 원소 수를 혼동하지 않습니다. [D09](docs/DECISIONS.md#d09--svd-이미지-크기와-rank-충돌을-두-실험으로-해결한다)

**학습률 0.5:** 원형 함수의 GD는 `θ_next=(1−2lr)θ`이므로 수렴 범위는 `0<lr<1`입니다. lr=0.5는 한 번에 원점 도착, lr=1은 일정 크기의 진동, lr=1.1은 발산입니다. 원문의 “0.5 이상 발산”을 구현 결과로 주장하지 않습니다. lr=0.1의 100회 후 각 좌표는 `5×0.8^100≈1.0185e-9`이며 참고 예시의 0.08과 다릅니다. [D10](docs/DECISIONS.md#d10--원형-함수의-학습률-발산-조건을-정정한다)

**Momentum의 이점과 한계:** `v_next=βv+gradient(θ)`, `θ_next=θ−lr·v_next`, `v0=0`을 사용합니다. 원형에서 반경 0.1에 처음 들어가는 시점은 Momentum 10회, GD 20회로 Momentum이 빠르지만, 이후 다시 밖으로 진동합니다. 기록된 100회까지 계속 반경 안에 머무르기 시작하는 시점은 각각 77회와 20회입니다. 두 지정 함수 모두 100회 후 loss는 GD가 더 작습니다. 타원형 lr=0.05에서는 GD의 y가 한 번에 0이 되므로 지속적인 지그재그를 기대하지 않습니다. “항상 더 빠르고 부드럽다”는 결론 대신 동일 조건의 측정 결과를 제시합니다. [D11](docs/DECISIONS.md#d11--momentum-비교는-실측-결과로-판단한다)

**역전파 참고 수치:** 고정 입력과 `W1@x` 방향을 유지하여 다시 계산하면 z2는 0.6072, 두 번째 은닉 기울기는 −0.0517입니다. 원문의 0.6071, −0.0513을 정답으로 복사하지 않았습니다. `dL/dW1` 두 번째 행 첫 원소도 −0.0517로 같습니다. [D12](docs/DECISIONS.md#d12--역전파는-고정-예제를-유지하고-참고-수치를-재계산한다)

**확률과 MLE:** N(μ,σ²)의 두 번째 인자는 분산입니다. iid Gaussian 오차와 고정 양의 공통 분산이면 `NLL=n/2·log(2πσ²)+SSE/(2σ²)`이므로 MSE와 최적해가 같습니다. 두 손실의 수치가 같은 것은 아니며 연속분포의 NLL은 음수일 수도 있습니다. Bernoulli BCE와 categorical CE는 각각 유도하고 합 손실과 평균 손실을 구분합니다. 확률 경계는 `0 log 0=0`, 불가능 사건은 `+∞`로 다루며 임의 clipping을 하지 않습니다. 큰 logit은 안정적 식을 쓰고, BCE의 큰 양수 logit에서 생길 수 있는 뺄셈 상쇄도 검증합니다. [D13](docs/DECISIONS.md#d13--확률-표기와-mle의-가정을-고정한다)

## 5. 파일과 실행 규칙

| 필수 산출물 | 구현·본문 | 결과·상세 문서 |
|---|---|---|
| 선형대수 | [src/linear_algebra.py](src/linear_algebra.py) | [선형대수 설명](docs/LINEAR_ALGEBRA.md), [측정 JSON](outputs/linear_algebra_metrics.json) |
| 미적분 | [src/calculus.py](src/calculus.py) | [미적분 설명](docs/CALCULUS.md), [측정 JSON](outputs/calculus_metrics.json) |
| 역전파 유도 | [notebooks/backprop_derivation.ipynb](notebooks/backprop_derivation.ipynb) | [역전파 설명](docs/BACKPROP.md), [측정 JSON](outputs/backprop_metrics.json) |
| GD·Momentum | [src/optimizer.py](src/optimizer.py) | [최적화 설명](docs/OPTIMIZATION.md), [측정 JSON](outputs/optimizer_metrics.json) |
| 확률·MLE 유도 | [notebooks/probability_loss.ipynb](notebooks/probability_loss.ipynb) | [확률·손실 설명](docs/PROBABILITY_LOSS.md), [측정 JSON](outputs/probability_metrics.json) |

알고리즘과 수학 계산은 **NumPy만** 사용합니다. Matplotlib·Seaborn은 시각화용이며 PyTorch·TensorFlow·JAX·scikit-learn·SciPy 알고리즘은 사용하지 않습니다. `np.linalg.svd`는 SVD 복원에 사용하고, `np.linalg.eig`는 Power Iteration 비교 검증 코드(테스트와 전용 검증 스크립트)에서만 사용합니다. 파일·JSON·해시·테스트에는 Python 표준 라이브러리를 사용하며 노트북 실행 도구는 개발 의존성입니다.

각 독립 실험·노트북 시작점에서 공통 `set_seed(42)`로 `np.random.seed(42)`를 한 번 설정합니다. import나 수학 함수 호출마다 seed를 초기화하지 않습니다. 공통 함수와 캐시는 [src/utils/](src/utils/)에 있고, 캐시는 Redis 서버 없이 `.cache/`에 저장합니다. [레지스트리](.agent/registry.json)는 파일·심볼·개념·의존성·해시와 노트북 셀 ID를 연결합니다. 상세 사용법은 [공통 기반 문서](docs/UTILITIES.md)에 있습니다.

## 6. 단계별 이슈·PR·검증 기록

작업은 **이슈 → 전용 브랜치 → 기초·상세·실행계획 → 구현 → 점검·보완 → 검증 → push·PR → 이슈 기록** 순서로 진행했습니다. 아래 PR #11~#18은 사용자의 병합 지시에 따라 선행 단계부터 순서대로 `main`에 병합했습니다. 각 이슈와 작업 기록에 계획·검증·보완·병합 이력을 남겼습니다. 제출 안내의 최종 정리는 [이슈 #19](https://github.com/ntt65/A2_1/issues/19)에서 추적합니다.

| 단계 | 이슈 | PR | 실행·보완 기록 |
|---|---|---|---|
| 0 환경·운영 준비 | [#3](https://github.com/ntt65/A2_1/issues/3) | [#11](https://github.com/ntt65/A2_1/pull/11) | [Step 0](docs/worklogs/step-0.md) |
| 1 공통 기반 | [#4](https://github.com/ntt65/A2_1/issues/4) | [#12](https://github.com/ntt65/A2_1/pull/12) | [Step 1](docs/worklogs/step-1.md) |
| 2 선형대수 | [#5](https://github.com/ntt65/A2_1/issues/5) | [#13](https://github.com/ntt65/A2_1/pull/13) | [Step 2](docs/worklogs/step-2.md) |
| 3 미적분 | [#6](https://github.com/ntt65/A2_1/issues/6) | [#14](https://github.com/ntt65/A2_1/pull/14) | [Step 3](docs/worklogs/step-3.md) |
| 4 최적화 | [#7](https://github.com/ntt65/A2_1/issues/7) | [#15](https://github.com/ntt65/A2_1/pull/15) | [Step 4](docs/worklogs/step-4.md) |
| 5 역전파 | [#8](https://github.com/ntt65/A2_1/issues/8) | [#16](https://github.com/ntt65/A2_1/pull/16) | [Step 5](docs/worklogs/step-5.md) |
| 6 확률·손실 | [#9](https://github.com/ntt65/A2_1/issues/9) | [#17](https://github.com/ntt65/A2_1/pull/17) | [Step 6](docs/worklogs/step-6.md) |
| 7 통합·최종 제출 | [#10](https://github.com/ntt65/A2_1/issues/10) | [#18](https://github.com/ntt65/A2_1/pull/18) | [Step 7](docs/worklogs/step-7.md) |

[전체 추적 이슈 #2](https://github.com/ntt65/A2_1/issues/2) · [전체 계획·체크리스트](docs/PROJECT_PLAN.md) · [수용 기준](docs/ACCEPTANCE.md) · [결정 기록](docs/DECISIONS.md) · [운영 규칙](AGENTS.md) · [원문 PDF](docs/A2_1_mission.pdf)
