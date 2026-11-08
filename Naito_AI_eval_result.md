# Naito_AI 사전평가 결과와 하락 원인 분석

**최신 외부 평가: 2026-11-08 12:58:12(KST), 60%(9/15).** 이전 11:52:57의 93%(14/15)와 구분한다. 두 원문을 이 문서 하단에 그대로 보존한다. [작업 이슈 #29](https://github.com/ntt65/A2_1/issues/29), [15항목 실행 증거 요약](MISSION_EVIDENCE.md), [검증 기록](docs/worklogs/issue-29.md).

## 왜 PASS가 FAIL로 바뀌었는가

**직접적인 점수 하락 이유는 기존 PASS 6개를 이번에 ‘제출 증거 없음’으로 판정했기 때문이다.** 수치 오답·실행 예외·알고리즘 퇴행을 근거로 실패시킨 내용은 이 6개 지적에 없다. 기존 FAIL #3은 새 eig 비교 증거가 인정되어 PASS가 됐다. 따라서 14−6+1=9개 통과이며, 통과 수가 5개 줄었다. 표시 점수는 93→60으로 33%p 하락했고, 정확한 비율 차이는 5/15=33.333…%p다.

**지적된 핵심 노트북·그림·측정값은 이전 main에도 있었고 현재 main에도 있다.** 두 노트북과 관련 기존 PNG·역전파/최적화/확률 JSON의 Git blob이 이전 main 3f03488과 감사 기준 main d43b076에서 동일하다. 선형대수 JSON은 eig 결과가 추가됐지만 SVD/변환 측정값은 동일하다. 현재 원격 main의 Git tree에서도 해당 파일을 확인했다. 따라서 해당 파일의 삭제나 변경으로 이번 ‘없음’ 판정이 발생했다는 설명은 확인 사실과 맞지 않는다.

다만 **GitHub에 파일이 있다는 사실과 평가기가 그 파일을 입력으로 받았다는 사실은 다르다.** 평가 대상 SHA·수집 파일 목록·추출 오류는 제공받지 못했다. 기존 PASS 역시 평가기가 당시 모든 증거를 정확히 읽었다는 보장은 없다. 원인 후보는 아래처럼 구분하며 ‘평가기 버그 확정’, ‘구현이 모두 외부 평가를 통과함’으로 결론 내리지 않는다.

## 항목별 판정 변화와 실물 대조

| 항목 | 11:52 → 12:58 | 이번 판정의 이유 또는 유지 근거 | 현재 main에서 확인한 증거 |
|---|---|---|---|
| #1 변환 | PASS → PASS | 변환 함수 인정, 저장 그림은 못 봄 | outputs/linear_transforms.png 존재 |
| #2 행렬식·면적 | PASS → PASS | 비교 구현 인정 | JSON의 회전·스케일·전단·반사·특이 사례 존재 |
| #3 Power Iteration | FAIL → PASS | 새 eig 비교 JSON 직접 인용 | λ=4.618033988749896, 상대오차1.923282552410481e-16 |
| #4 SVD | **PASS → FAIL** | k=10/50/100 실행 결과·이미지 없음 | outputs/svd_128.png와 k별 상대오차 .12751847/.05938687/.01218301 존재 |
| #5 수치 미분 | PASS → PASS | 중심차분 구현 인정 | 고정 h=1e-5의 오차3.930633596382904e-11 존재 |
| #6 Gradient | PASS → PASS | quiver 인정, 저장 그림은 못 봄 | outputs/gradient_geometry.png 존재 |
| #7 역전파 유도* | **PASS → FAIL** | 노트북/손계산 예제 없음 | notebooks/backprop_derivation.ipynb, 실행 코드8셀과 단계별 유도 존재 |
| #8 손계산 비교 | **PASS → FAIL** | NumPy와 네 자리 비교 증거 없음 | 저장 출력13항목 비교, JSON four_decimal_match=true, 최대오차2.220446049250313e-16 |
| #9 GD 수렴 | PASS → PASS | 최종 norm 인용 | 최종반경1.4404019523870759e-9, 원문 .1 기준 충족 |
| #10 학습률 | **PASS → FAIL** | 실행 수렴/발산 그래프 없음 | outputs/optimizer_learning_rates.png와 lr=.5/1/1.1 최종반경0/7.0711/271.0878 존재 |
| #11 Momentum | PASS → PASS | JSON 인정, 비교 그림은 못 봄 | outputs/optimizer_circle.png 존재 |
| #12 타원형 | PASS → PASS | JSON 인정, 경로 그림은 못 봄 | outputs/optimizer_ellipse.png 존재 |
| #13 분포 | **PASS → FAIL** | 정규·Bernoulli 노트북/그림 없음 | notebooks/probability_loss.ipynb, 실행 코드9셀과 PNG2개 존재 |
| #14 Softmax | PASS → PASS | 구현 인정, 수치 검증은 못 봄 | 확률 JSON의 row/column 합오차1.1102230246251565e-16 존재 |
| #15 MLE | **PASS → FAIL** | MSE/CE↔MLE 유도 없음 | 확률 노트북의 Gaussian/Bernoulli/categorical 유도 셀과 docs/MATH_DERIVATIONS.md 존재 |

* 새 원문은 해당 항목을 실제로 ‘평가 항목 #’라고만 적었다. 위치·내용상 #7로 대응했으며 원문 번호를 고치지 않았다. 위 대조는 저장소 감사 결과이며 외부 FAIL을 임의로 PASS로 수정한 표가 아니다.

## 확인 사실과 원인 가설

### 두 평가가 직접 충돌하는 지점

| 항목 | 11:52 평가가 인정한 증거 | 12:58 평가의 주장 | 해석 |
|---|---|---|---|
| #4 | linear_algebra_metrics.json의 ‘128x128 supplemental experiment permits required ranks 10,50,100.’를 인용하고 실행·저장을 인정 | 같은 rank의 실행 결과·이미지가 없음 | 같은 SVD 데이터가 유지됐으므로 파일 변경으로 설명되지 않음 |
| #8 | backprop_metrics.json의 ‘four_decimal_match: true’를 직접 인정 | 네 자리 비교 증거가 없음 | 이전에 인정했던 동일 JSON의 증거를 이번에 놓쳤거나 다르게 처리함 |
| #13 | probability_metrics.json의 두 그림 파일명과 생성·포함을 인정 | 분포 그림·노트북이 없음 | 실제 PNG·JSON이 동일하므로 증거 인식의 차이를 조사해야 함 |
| #7·#15 | 유도 과정이 부족하다고 쓰면서 PASS | 유도 과정이 없다는 이유로 FAIL | 처음부터 PASS/FAIL 적용 기준이 일관되지 않았을 가능성. 첫 PASS를 완전한 검증으로 해석할 수 없음 |

따라서 **‘첫 평가보다 구현이 나빠졌다’는 결론은 두 보고서와 Git 비교에서 나오지 않는다.** 보고서가 보여주는 것은 증거 인식 또는 판정 기준의 변화다. 그 변화가 수집 단계에서 생겼는지 모델 판단에서 생겼는지는 별도 확인이 필요하다.

감사 기준은 [main d43b076](https://github.com/ntt65/A2_1/tree/d43b076319ac7fcf1bde5f1fecf48188bd00727a)이다. 이 커밋의 병합 시각은 12:53:54 KST로 새 평가 시각보다 앞선다. 그러나 평가 시작/수집 시점과 대상 SHA는 원문에 없으므로 ‘이 커밋 전체가 평가됐다’고 단정하지 않는다.

| 구분 | 판단 | 근거 및 확인 방법 |
|---|---|---|
| 확인됨 | 기존 핵심 제출물이 삭제되거나 퇴행한 것은 아님 | 이전/현재 Git blob 동일, 원격 Git tree 존재 확인 |
| 확인됨 | 업데이트 일부는 평가 근거에 나타남 | #3이 새 outputs/power_iteration_eig_comparison.json과 실제 고유값을 인용함 |
| 확인됨 | 지정된 그림 폴더를 준수함 | 미션 원문은 outputs/에 PNG 저장을 명시하며 현재 PNG15개가 존재함 |
| 확인됨 | 기존 노트북의 그림은 상대경로 연결 방식 | 코드 실행/stream 출력은 저장됐지만 image/png 출력은0개; 확률 그림은 Markdown에서 ../outputs/를 참조함 |
| 우선 조사 | 파일 수집 범위 또는 노트북·그림 추출 누락 | 코드·일부 JSON은 인용하면서 노트북·PNG를 반복해서 없다고 함. 실제 수집 목록·지원 확장자·추출 오류를 확인해야 함 |
| 가능성 | 평가 판단의 변동 또는 증거 선택의 비일관성 | 같은 산출물에 대한 PASS/FAIL이 달라짐. 동일한 입력·평가기 설정을 고정한 비교 없이는 변동 원인을 확정할 수 없음 |
| 미확인 | 일부 파일만 읽음, 길이 제한, 캐시, 브랜치/수집 시점 차이 | 직접 확인할 입력 목록·로그가 없음. 보너스 추가가 분량을 늘렸다는 사실만으로 길이 제한 탓이라고 결론 내리지 않음 |
| 설명력 낮음 | 완전히 예전 입력만 재사용함 | 새 eig JSON이 인용돼 전부 구버전이라는 설명만으로는 충분하지 않음. 부분 캐시는 미확인 |

기존 README에 파일 링크와 유도가 있었지만 핵심 결과표는 뒤쪽에 분산돼 있었고 그림은 링크로 제공했다. 이는 개선 가능한 전달 구조다. 이전 완료 검증은 코드·산출물의 계산/존재 검증이었으며, 평가기가 읽은 입력의 완전성까지 확인한 것은 아니었다. 저장소 감사만으로 외부 평가 점수나 입력 전달 완료를 보장하지 않는다.

## 대책과 완료 기준

| 우선순위 | 조치 | 상태·완료 기준 |
|---|---|---|
| P0 최신 기록 정정 | 이 문서와 README를60%로 갱신,93% 원문·이력 유지 | 이번 문서 작업에 반영. 점수 하락을 숨기거나 자체 검사로 대체하지 않음 |
| P0 평가 입력 확인 | 제출 URL·main/커밋·평가 실행ID·실제 포함 파일/추출 실패 목록 확보 | 미확인. 사용자 제공 원문에는 해당 정보가 없음. 확보 시 GitHub 실물과1:1 대조 |
| P1 증거 접근 단순화 | 루트 MISSION_EVIDENCE.md에15항목별 실제값·원본경로·고정커밋·핵심유도·그림 연결, README 앞부분에서 안내 | 이번 문서 작업에 반영. 예전 평가 원문과 제출 증거를 별도 문서로 분리 |
| P1 누락 파일의 직접 재제출 | 평가 서비스가 파일 선택을 지원한다면 두 노트북·기본 PNG10개·JSON5개·수식 문서·eig 보고서를 실제 포함 목록에서 확인 | 다음 평가 준비 절차. 저장소URL만 제출한다면 수집 범위/지원형식 확인이 우선 |
| P2 형식 호환 보완 | 상대이미지 참조를 못 읽는 것이 확인되면 실행 결과 image/png를 내장한 제출 노트북 또는 HTML 보조본 생성 | 아직 구현하지 않음. 기존 outputs/PNG는 유지. 내장은 원문 필수가 아닌 조건부 보조책 |
| P2 재평가 검증 | 입력 SHA·파일 목록을 기록한 동일 제출본으로 재평가하고6개FAIL 각각의 인용 근거 확인 | 미실행. 목표15/15이며 결과를 받기 전에 점수 회복/100%를 주장하지 않음 |

추가 구현이 필요하면 입력 처리 문제를 확인한 뒤 별도 작업으로 진행한다. 파일을 다른 폴더로 옮기거나 기존 알고리즘을 다시 만드는 것은 현재 증거가 지지하는 해결책이 아니다.

## 평가 문구에 맞춰 바꾸면 안 되는 조건

- **그림 폴더:** [미션 원문](A2_1_mission.md)은 outputs/PNG를 지정한다. 현재 저장 방식은 요구에 맞다. 노트북 내장 그림이 없다는 이유로 폴더 규격을 위반했다고 해석하지 않는다.
- **SVD:** 64×64 입력에 rank100을 강제할 수 없다. 64×64 본 실험과128×128의 k10/50/100 보충 실험을 유지한다([D09](docs/DECISIONS.md#d09--svd-이미지-크기와-rank-충돌을-두-실험으로-해결한다)).
- **학습률:** 원형 GD는 theta_next=(1−2lr)theta다. lr=.5는 한 번에 수렴, lr=1은 진동, lr>1은 발산한다. ‘lr>=.5 전부 발산’이라는 권고를 그대로 따르지 않는다([D10](docs/DECISIONS.md#d10--원형-함수의-학습률-발산-조건을-정정한다)).
- **반경·해상도:** 원문의 수렴 반경은.1이다. 평가의 반경1로 느슨하게 바꾸지 않는다. 64px는 SVD 입력 가이드이며 전체 결과 그래프를 읽을 수 없게 축소하지 않는다.
- **미분 오차:** 고정 h=1e-5 검증과61개 h 탐색은 구분한다. 탐색 전체 최대오차는 약.670929로 모두1e-4 이내라고 주장할 수 없다. 통과한 h 표본은46/61이다.

## 검증 상태

현재 수학 구현은 직전 main CI에서 Ubuntu/Windows 각각186개 검사를 통과했다. 그 기록과 최신 외부60% 판정은 별개다. 이번 변경은 평가·증거·계획 문서이며 알고리즘, 노트북, PNG, JSON, 의존성을 변경하지 않는다. 레지스트리 대상 소스도 변경하지 않는다. 이번 문서 검사·게시·CI·main 반영 결과는 [이슈 #29](https://github.com/ntt65/A2_1/issues/29)와 [작업 기록](docs/worklogs/issue-29.md)에 기록한다.

## 최신 사용자 제공 원문 — 2026-11-08 12:58:12

아래는 사용자 첨부 텍스트다. 번호 누락·영문 표현·수학적으로 부정확한 권고까지 수정하지 않고 보존한다.

```text
AI 사전평가 결과
평가 시각: 2026-11-08 12:58:12

60%
(9 / 15 항목 통과)
종합 평가
Codebase implements many numerical algorithms and visualizations (linear transforms, power iteration, SVD utilities, numerical differentiation, optimizers) and provides executed verification for power iteration and optimizer experiments; missing submitted notebooks and explicit artifacts for backprop hand-derivation, SVD k=10/50/100 runs, distribution plots, and MLE derivations.

항목별 평가 (15개)
PASS
평가 항목 #1
근거
src/linear_algebra.py > "('Rotation: 45 degrees', rotation_matrix(np.pi / 4))"
잘한 점
Three-panel transformation_figure includes rotation, scaling, and shear examples.
부족한 점
Figure is returned but notebook/image export for submission not shown.
보완
Include saved/embedded figures or thumbnails in the notebook for submission.
PASS
평가 항목 #2
근거
src/linear_algebra.py > "\"determinant\": determinant"
잘한 점
determinant_area_check computes independent determinant and area ratio.
부족한 점
No explicit additional examples of extreme dynamic-range cases in deliverables.
보완
Add a concise notebook cell that prints determinant vs area_ratio for representative transforms.
PASS
평가 항목 #3
근거
outputs/power_iteration_eig_comparison.json > "eigenvalue": 4.618033988749896
잘한 점
Power iteration implemented and numerically compared to numpy.linalg.eig with recorded residuals.
부족한 점
Tolerance/seed usage is present but broader matrix-case tests not shown in outputs.
보완
Record a short table of tests with different matrices and relative errors in the notebook.
FAIL
평가 항목 #4
잘한 점
The svd_comparison_figure supports arbitrary ranks.
부족한 점
SVD comparison runs for k=10,50,100 are not present: executed outputs/images missing.
보완
Run svd_comparison_figure for ranks (10,50,100) and include the resulting images or numeric errors in the notebook.
PASS
평가 항목 #5
근거
src/calculus.py > "\"quadratic_error\": np.abs(quadratic_estimates - 6.0)"
잘한 점
derivative_error_study computes central differences and absolute errors against analytic derivatives.
부족한 점
No explicit single-line report of max error vs tolerance in submitted outputs.
보완
Summarize the max absolute errors and confirm they meet the 1e-4 tolerance in a notebook cell.
PASS
평가 항목 #6
근거
src/calculus.py > "axes.quiver("
잘한 점
gradient_figure overlays contour and normalized gradient/tangent arrows using quiver.
부족한 점
Figure is returned but a saved image or notebook display for review is not present in files list.
보완
Embed the returned figure in backprop_derivation.ipynb or save a thumbnail within outputs.
FAIL
평가 항목 #
부족한 점
backprop_derivation.ipynb or an explicit hand-derived backprop worked example is not included in the submission.
보완
Add backprop_derivation.ipynb containing the fixed example and step-by-step handwritten derivation (scans or markdown cells).
FAIL
평가 항목 #8
부족한 점
No artifacts showing numeric hand-calculation compared to NumPy to 4 decimal places are present.
보완
Include a reproducible notebook cell that shows hand-derived scalar/vector values and compares them to NumPy to 4 decimal places.
PASS
평가 항목 #9
근거
outputs/optimizer_metrics.json > "final_norm": 1.4404019523870759e-09
잘한 점
Optimizer run for vanilla GD lr=0.1 with 100 steps is recorded and reaches a very small final norm.
부족한 점
Radius-1 membership is implied by norms but a one-line boolean check against radius=1 is not explicit.
보완
Add an explicit assertion or printed check in the notebook verifying radius<1 by a named variable.
FAIL
평가 항목 #10
잘한 점
learning_rate_figure implements panels for rates 0.5,1.0,1.1.
부족한 점
No executed divergence/convergence graphs for lr>=0.5 are present in submitted outputs.
보완
Run learning_rate_figure with recorded runs and include the generated images (or numeric final norms) showing divergence for lr>=0.5.
PASS
평가 항목 #11
근거
outputs/optimizer_metrics.json > "Momentum": {
잘한 점
The outputs include both GD and Momentum experiment records allowing side-by-side comparison.
부족한 점
Comparison visuals (figure image files) are not attached; numeric summaries exist but a concise comparative plot may be missing.
보완
Embed the optimizer_comparison_figure output or add a summarized plot in the notebook comparing paths/losses.
PASS
평가 항목 #12
근거
outputs/optimizer_metrics.json > "objective": "ellipse"
잘한 점
Elliptic objective (x²+10y²) experiments and trajectories for GD/Momentum are recorded in outputs.
부족한 점
Convergence-path figures are provided by functions but exported figure assets for the experiment are not included in the repository outputs list.
보완
Include the produced path figure or a compact image (<=64px thumbnail) in the notebook outputs.
FAIL
평가 항목 #13
부족한 점
No distribution-visualization outputs or notebooks showing two normal PDFs and two Bernoulli PMFs are present.
보완
Add probability_loss.ipynb or cells that plot the two normal PDFs and two Bernoulli PMFs and save the images.
PASS
평가 항목 #14
근거
src/utils/math.py > "return weights / np.sum(weights, axis=axis, keepdims=True)"
잘한 점
softmax implementation subtracts max and normalizes by sum, enforcing probability normalization.
부족한 점
No explicit numeric unit-test or printed verification that sums equal 1 within 1e-6 appears in outputs.
보완
Add a unit test or notebook cell that computes softmax on sample inputs and checks sum-1 within 1e-6.
FAIL
평가 항목 #15
부족한 점
Derivations linking MSE/CE to MLE (probability_loss.ipynb) are not included in the submitted files.
보완
Provide probability_loss.ipynb with stepwise symbolic/markdown derivations for MSE↔MLE and CE↔MLE and minimal numeric checks.
```

## 이전 사용자 제공 원문 — 2026-11-08 11:52:57

이하93%는 과거 평가 이력이다. 당시 보완은 이슈 #23·#25, 보너스는 #27에서 진행했다. 과거 분석 전체는 [d43b076의 문서](https://github.com/ntt65/A2_1/blob/d43b076319ac7fcf1bde5f1fecf48188bd00727a/Naito_AI_eval_result.md)에 보존돼 있다.

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
