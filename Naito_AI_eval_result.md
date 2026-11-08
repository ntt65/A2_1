# Naito_AI 사전평가 결과와 보완 계획

평가 원문은 사용자가 제공한 내용이다. 아래 원문은 오류로 의심되는 표현도 그대로 보존한다. 이 문서의 분석·보완 기록은 원문과 구분한다.
작업 추적: [평가 보완 1 — 이슈 #23](https://github.com/ntt65/A2_1/issues/23).

## 사용자 제공 평가 원문

```text
AI 사전평가 결과
평가 시각: 2026-11-08 11:52:57

93%
(14 / 15 항목 통과)
종합 평가
대부분의 실험과 시각화, 수치검증 산출물이 코드와 결과 파일에 명확히 포함되어 있습니다. 누락된 항목은 Power Iteration의 NumPy 고유 고유값(eig)과의 명시적 비교로, 해당 비교 결과 또는 호출이 자료에 없습니다.

항목별 평가 (15개)
PASS
평가 항목 #1
근거
src/linear_algebra.py > 'Return a Figure and three Axes comparing rotation, scaling and shear.'
잘한 점
세 가지 변환(rotation/scale/shear)을 하나의 비교 도식으로 구현해 시각화함.
부족한 점
개선점: 각 패널 이미지 크기 및 저장 시 최대 64픽셀 권고 준수 안내 미기재.
보완
figure 생성 시 호출자에게 이미지 크기 제약(예: max 64px) 또는 저장 옵션을 문서화하라.
PASS
평가 항목 #2
근거
src/linear_algebra.py > 'Compare independent area ratio with abs(det(A)) and return error metrics.'
잘한 점
독립적 폴리곤 면적비와 2x2 행렬식(abs(det))을 비교하는 검증 구현.
부족한 점
검증 보고서에 tolerance 적용 근거는 있으나 추가적 실패 케이스 예시는 없음.
보완
tolerance 실패 시 대표적 입력 사례와 진단 메시지를 함께 기록하라.
FAIL
평가 항목 #3
부족한 점
np.linalg.eig을 이용한 기준 고유값과의 비교 실행/출력 미제출
보완
Power Iteration 결과를 numpy.linalg.eig로 계산한 참값과 함께 기록하여 상대오차를 보고하라.
PASS
평가 항목 #4
근거
outputs/linear_algebra_metrics.json > '128x128 supplemental experiment permits required ranks 10,50,100.'
잘한 점
요구된 k값(10,50,100)을 포함한 SVD 실험과 비교 결과를 산출·저장함.
부족한 점
출력에 저장비율/오차가 있지만 각 패널 이미지 해상도 설명은 제한적임.
보완
svd 비교 이미지 생성 시 각 패널의 픽셀 크기와 저장포맷을 메타데이터로 기록하라.
PASS
평가 항목 #5
근거
src/calculus.py > 'Return central-difference estimates and errors at x=3 for 61 step sizes.'
잘한 점
중심차분으로 단계별 오차 스터디를 수행해 해석값과 비교함.
부족한 점
오차 허용치 및 최적 h 선택 근거는 문서화되어 있으나 자동 권장 h 제안은 없음.
보완
오차 최소화를 위한 권장 h 범위를 계산해 요약 표로 제공하라.
PASS
평가 항목 #6
근거
src/calculus.py > 'Return a Figure and Axes showing gradients normal to circular contours.'
잘한 점
등고선 위에 정규화된 gradient 화살표와 접선 화살표를 명확히 표시함.
부족한 점
이미지 출력 크기/품질 관련 메타데이터(권장 픽셀 등)가 없음.
보완
시각화 생성 시 권장 figure.set_size_inches 또는 출력 DPI 권고를 추가하라.
PASS
평가 항목 #7
근거
outputs/backprop_metrics.json > '"hand_reference": {'
잘한 점
고정 예제에 대한 손계산 레퍼런스가 JSON으로 상세히 기록되어 있음.
부족한 점
손계산 유도 과정의 단계별 수식이 노트북(또는 텍스트)로 포함되어 있지 않음.
보완
손계산 유도의 단계별 워크를 노트북이나 텍스트 파일로 함께 첨부하라.
PASS
평가 항목 #8
근거
outputs/backprop_metrics.json > '"four_decimal_match": true'
잘한 점
손계산과 NumPy 결과의 소수 네 자리 일치 여부를 명시적으로 검증함.
부족한 점
일치 불일치 발생 시 재현용 RNG/환경 정보 외 추가 진단 데이터가 제한적임.
보완
불일치 케이스에 대해 차분 벡터와 재현 스크립트를 함께 제공하라.
PASS
평가 항목 #9
근거
outputs/optimizer_metrics.json > '"lr": 0.1'
잘한 점
Vanilla GD 실험에서 lr=0.1을 사용한 장기(100회) 수렴 기록을 제공함.
부족한 점
증거 인용은 lr만 보여주므로 실행 횟수(100회)와 반경1 내부 유지 지표의 인용을 병기하면 명확함.
보완
요약표에 lr, steps, 반경(예: remaining_inside_radius_0_1) 값을 함께 노출하라.
PASS
평가 항목 #10
근거
src/optimizer.py > 'Return Figure/Axes(3,) for circle GD learning rates 0.5, 1.0, and 1.1.'
잘한 점
lr>=0.5 조건(0.5,1.0,1.1)에 대한 학습률별 발산/진동 사례를 시각화함.
부족한 점
시각적으로 발산을 보이나 수치적 발산 판단(예: 발산 임계치) 정의는 문서화되지 않음.
보완
발산 판단 기준(예: 반경/에너지 증가율 임계값)을 메타데이터로 명시하라.
PASS
평가 항목 #11
근거
src/optimizer.py > '"Vanilla_GD": ["VanillaGD"], "Momentum_Update": ["Momentum"]'
잘한 점
Vanilla GD와 Momentum 구현이 모두 존재하고 비교용 시각화 함수도 제공됨.
부족한 점
두 방법 비교 시 통계적 요약(예: 평균 수렴속도)은 별도 집계로 제공되지 않음.
보완
다수 재시도 결과의 평균/분산 등 요약 통계를 추가로 출력하라.
PASS
평가 항목 #12
근거
src/optimizer.py > 'Return x**2+10*y**2 for a finite real point (2,)'
잘한 점
타원형 목적함수(x²+10y²)와 그라디언트를 구현해 수렴경로 실험에 사용함.
부족한 점
초기조건 다양성에 따른 수렴 경로의 민감도 분석 표시는 제한적임.
보완
다른 초기점에서의 경로 표본(여러 시작점)을 함께 시각화하라.
PASS
평가 항목 #13
근거
outputs/probability_metrics.json > '"figures": ["probability_normal.png", "probability_bernoulli.png"]'
잘한 점
정규분포 2종 및 베르누이 PMF 2종의 시각화 파일을 생성해 포함함.
부족한 점
각 플롯의 해상도/픽셀 크기 메타데이터가 명시되어 있지 않음(이미지 크기 제한 권고 미반영).
보완
시각화 파일에 이미지 크기(픽셀)와 DPI 메타데이터를 함께 저장하라.
PASS
평가 항목 #14
근거
outputs/probability_metrics.json > '"row_sums": [ 0.9999999999999999, 1.0 ]'
잘한 점
softmax 출력의 행 합(합=1)과 최대 오차를 수치로 검증해 기록함.
부족한 점
softmax 허용오차(예: 1e-6) 대비 결과 표시가 있으나 다양한 축(axis) 테스트 요약은 제한적임.
보완
다양한 축(axis) 케이스에 대한 합 오차 요약(최대/평균)을 추가하라.
PASS
평가 항목 #15
근거
outputs/probability_metrics.json > 'Gaussian NLL and MSE share minimizers only under the stated fixed-variance assumptions.'
잘한 점
Gaussian NLL↔MSE 및 CE↔MLE 관계를 결과와 노트로 해석·검증함.
부족한 점
유도(수식 전개) 원문 노트북이나 수학적 유도가 제출물에 명시적으로 포함되어 있지 않음.
보완
유도 과정(수식 전개)을 담은 노트북이나 텍스트 파일을 명시적으로 첨부하라.
```

## 분석과 처리 원칙

기준은 평가 시점의 제출 구현과 [미션 원문](A2_1_mission.md)이다. PASS 항목의 추가 권고와 실제 필수 누락을 구분하고, 사용자 지시에 따라 보너스보다 평가 보완을 먼저 순차 진행한다.

1. **#3은 제출 증거를 보강해야 한다.** 기존 [비교 테스트](tests/test_power_iteration.py)의 `test_mission_matrix_matches_analytic_value_and_eig`에 `np.linalg.eig` 호출이 있었지만, 기본 실험 JSON·콘솔에는 닫힌 해 기준만 기록했다. 따라서 “호출이 전혀 없다”는 판단과 달리 구현은 존재했고, 평가자가 바로 볼 수 있는 실행 결과가 부족했다.
2. **#7·#15의 수식 유도는 이미 존재한다.** [역전파 노트북](notebooks/backprop_derivation.ipynb)의 `backprop-forward-derivation`, `backprop-chain-rule` 셀과 [확률 노트북](notebooks/probability_loss.ipynb)의 `gaussian-mle-derivation`, `bernoulli-mle-derivation`, `categorical-mle-derivation` 셀에 전개가 있다. 후속 보완은 유도 접근 경로와 일반 텍스트 증거를 강화한다.
3. **64×64는 SVD 입력 grayscale 이미지 가이드다.** 원문은 모든 Figure의 출력이나 각 그래프 패널을 64픽셀 이하로 제한하지 않는다. 축·수식을 읽을 수 없게 전체 그래프를 축소하지 않는다. 입력 배열 크기, Figure 픽셀 크기, DPI를 구분해 기록한다. rank 100과 입력 크기의 충돌은 [D09](docs/DECISIONS.md#d09--svd-이미지-크기와-rank-충돌을-두-실험으로-해결한다)의 두 실험 방식을 유지한다.
4. **반경 기준은 0.1이다.** #9의 “반경1” 표현은 원문의 0.1을 대체하지 않는다. #10도 lr=0.5는 즉시 수렴, lr=1은 진동, lr>1은 발산이라는 직접 유도를 유지한다.
5. 동일 초기점의 결정론적 GD를 반복 실행해 평균·분산을 만드는 것은 새로운 정보를 주지 않는다. #11·#12를 보완할 때는 명시한 여러 초기점의 민감도 실험으로 설계하고, 미수렴 표본을 성공으로 간주하거나 임의 반복수로 대체하지 않는다.
6. 수치 미분의 권장 h는 함수·점·dtype·오차 정의에 종속된다. x²의 중심차분 절단오차는 이론상 0이므로 특정 격자의 우연한 최소오차를 보편적인 최적 h라고 주장하지 않는다.

## 항목별 대책과 우선순위

| 항목 | 기존 상태 대조 | 대책 | 순서/상태 |
|---|---|---|---|
| #1 | 변환 Figure는 존재, 64px 전체 출력 제한은 원문에 없음 | 저장 크기·DPI와 입력 이미지 제한의 적용 범위 설명 | 완료: 이슈 #25 |
| #2 | 면적비·행렬식 비교 및 경계 검증 존재 | 실패를 재현하는 입력·오차·진단 예시를 결과에 연결 | 완료: 이슈 #25 |
| #3 | eig 비교 테스트 존재, 기본 실행 출력은 닫힌 해만 제시 | 검증 전용 eig 비교를 실행하고 두 고유값·벡터·오차·판정을 JSON/텍스트/README에 공개 | 완료: 이슈 #23 |
| #4 | 64/128 입력 배열과 rank 비교 존재 | 입력 배열 shape와 실제 저장 패널/캔버스 픽셀·PNG 형식을 구분 | 완료: 이슈 #25 |
| #5 | 61개 h 오차 배열과 설명 존재 | 해당 함수·점의 경험적 허용 h 구간/최소오차를 한계와 함께 표로 제시 | 완료: 이슈 #25 |
| #6 | Gradient·접선 Figure 존재 | 권장 inches/DPI 및 실제 PNG 크기 노출 | 완료: 이슈 #25 |
| #7 | 노트북의 단계별 손계산 유도 존재 | 명시적 셀 위치와 일반 Markdown 유도본 연결 | 완료: 이슈 #25 |
| #8 | 기준값·계산값·절대오차·수치 gradient 존재 | 불일치 시 부호 있는 차분과 재현 입력을 보여주는 진단 추가 | 완료: 이슈 #25 |
| #9 | steps=100·반경0.1 도달/유지 지표 존재 | lr·steps·최종반경·유지시작을 한 표에 집계 | 완료: 이슈 #25 |
| #10 | 원형 GD 닫힌 반복식과 증가하는 loss 검증 존재 | `(1-2lr)` 배율과 이론적 분류/유한 관측 근거 명시 | 완료: 이슈 #25 |
| #11 | 고정 예제의 결정론적 비교 존재 | 여러 초기점에서 도달/유지·최종오차 통계, 집계 모집단과 미수렴 처리 명시 | 완료: 이슈 #25 |
| #12 | 타원형 고정 초기점 비교 존재 | #11과 같은 명시적 초기점 집합의 경로 비교 | 완료: 이슈 #25 |
| #13 | PDF·PMF PNG 존재 | PNG의 실제 크기·DPI 기록 | 완료: 이슈 #25 |
| #14 | 노트북 axis=-1/0 검증과 column_sum_error 존재 | 축별 shape·최대/평균 합 오차와 허용치 요약 | 완료: 이슈 #25 |
| #15 | Gaussian/Bernoulli/categorical 유도 셀 존재 | 일반 Markdown 전개본과 README의 직접 링크 강화 | 완료: 이슈 #25 |

## 진행·재평가 상태

- [x] 사용자 제공 평가 원문 저장
- [x] 15개 항목의 기존 증거·권고·수학적 해석 분류
- [x] 1차: #3 eig 비교 실행·출력·독립 검증(전체 132개 테스트 통과)
- [x] 후속: 나머지 권고의 수치·유도·시각화 증거 보완(이슈 #25)
- [ ] 변경이 반영된 제출본으로 Naito_AI 재평가

**현재 외부 평가 결과는 93%(14/15) 그대로다.** 자체 테스트 통과를 Naito_AI의 재평가 통과 또는 100% 점수로 표현하지 않는다. 보너스 Adam·Newton·정보 이론 구현도 사용자 추가 지시에 따라 [이슈 #27](https://github.com/ntt65/A2_1/issues/27)에서 완료했다. [보너스 수식·검증·결과](docs/BONUS.md)와 [작업 기록](docs/worklogs/issue-27.md)에 근거를 남겼다.

1차 보완의 재개·검증 증거는 [작업 기록](docs/worklogs/issue-23.md)에 정리했다. 이 문서 커밋 이후의 게시 커밋·PR·자동 검사·검토 및 병합 상태는 [이슈 #23](https://github.com/ntt65/A2_1/issues/23)에서 확인한다. 구현·검증 완료와 main 반영·외부 재평가를 구분한다.

2차 증거: [수치·PNG 메타데이터·의도적 실패 진단](outputs/evaluation_evidence.md), [고정 초기점 최적화 민감도](outputs/optimizer_sensitivity.md), [역전파·MLE 전체 수식 유도](docs/MATH_DERIVATIONS.md). 실제 구현과 추가 권고의 해석을 구분한 근거는 [D15](docs/DECISIONS.md#d15--평가-권고는-실측-증거와-적용-범위를-구분한다)에 있다.

최종 전달은 사용자 승인에 따른 관련 PR의 순차 main 병합과 README 정리를 포함한다. 현재 자체 전체 검증은 186개 통과이며, 최종 main CI·병합 커밋·동기화 상태는 이슈 #27을 따른다. 외부 평가 점수는 실제 재평가 결과를 받을 때만 갱신한다.
