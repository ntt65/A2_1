# 평가 항목별 수학·실험 Q&A 사전

이 문서는 [미션 원문](../A2_1_mission.md)의 기본 요구를 15개 평가 항목에 대응시켜, 개념을 찾아보고 수식·입력·실행 결과를 함께 읽는 참고 자료다. 원문 참고 예시와 실제 계산이 다르면 [수용 기준](ACCEPTANCE.md)과 [결정 기록](DECISIONS.md)의 정정 근거를 따른다. 외부 평가자에게 판정을 지시하는 문서가 아니다.

설명 연습은 **핵심 설명 → 구현 위치 → 실제 검증 결과 → 한계·반례** 순서로 한다. 질문에 답한 뒤 각 항목의 오해·반례를 꼬리 질문으로 바꾸어 설명해 보면, 용어 암기와 실제 이해를 구분할 수 있다.

**검증 상태(2026-11-09):** 아래 실측값은 기존 outputs JSON·실행 노트북과 대조했다. 이슈 #31에서 세 실험과 두 새 커널을 재실행해 기존 JSON의 구조·입력·직접 계산값 일치(atol=rtol=1e-12)와 중심차분의 독립 정확도를 확인하고 다섯 보고서와 README 본문을 생성했다. 전체 회귀 **210개(30.826초)**를 통과했다. 최종 문구 보완 후 관련 검사·환경·레지스트리, PR/CI·main 병합 상태는 [이슈 #31](https://github.com/ntt65/A2_1/issues/31)과 [작업 기록](worklogs/issue-31.md)을 따른다. 외부 평가는 40%(6/15)이며 세 번의 기회가 소진됐다. 자체 검증은 외부 점수 회복을 뜻하지 않는다.

## 목차

| 평가 항목 | 찾아볼 질문 |
|---|---|
| [#1 행렬 변환](#qa-01) | 행렬 곱은 단위 원을 어떻게 바꾸는가? |
| [#2 행렬식과 면적](#qa-02) | 음의 행렬식·특이 행렬의 면적 오차는 어떻게 비교하는가? |
| [#3 Power Iteration](#qa-03) | 어느 고유값을 찾고, eig와 무엇을 비교하는가? |
| [#4 SVD](#qa-04) | k, 복원 오차, 저장량은 어떤 관계인가? |
| [#5 수치 미분](#qa-05) | h를 작게 하면 언제나 정확해지는가? |
| [#6 Gradient](#qa-06) | Gradient가 등고선에 수직이라는 뜻은 무엇인가? |
| [#7 역전파 유도](#qa-07) | 여섯 미분을 어떤 순서로, 어떤 shape으로 계산하는가? |
| [#8 손계산·NumPy 비교](#qa-08) | 네 자리 일치와 독립 수치 검증을 어떻게 확인하는가? |
| [#9 GD 수렴](#qa-09) | 100회 후 반경 0.1을 만족하는가? |
| [#10 학습률](#qa-10) | lr=0.5, 1, 1.1은 왜 다른가? |
| [#11 Momentum](#qa-11) | 첫 도달과 안정적으로 머무는 시점은 같은가? |
| [#12 타원형 최적화](#qa-12) | 곡률 차이와 Momentum의 이점을 어떻게 해석하는가? |
| [#13 분포](#qa-13) | PDF와 PMF, 분산과 표준편차는 어떻게 다른가? |
| [#14 Softmax](#qa-14) | 어느 축을 정규화하고 큰 logit을 어떻게 처리하는가? |
| [#15 MLE와 손실](#qa-15) | Gaussian→MSE, Bernoulli→BCE, categorical→CE를 어떻게 유도하는가? |
| [보너스 Adam](#qa-adam) | 모멘트·편향 보정·epsilon의 의미는 무엇인가? |
| [보너스 Newton](#qa-newton) | Hessian을 쓰면 왜 한 번에 풀리는 예제가 있는가? |
| [보너스 정보 이론](#qa-information) | H·CE·KL과 0확률·무한 손실은 어떻게 연결되는가? |
| [공통 재현 용어](#qa-common) | float64·seed·캐시·허용오차는 무엇을 보장하는가? |
| [검증과 외부 평가](#qa-evaluation) | 내부 검사 성공과 외부 평가 점수는 같은 사실인가? |

## 읽기와 실행의 공통 규칙

아래 python은 [README 설치 안내](../README.md)의 잠금 의존성을 설치한 Python 3.11 환경을 뜻한다. Windows에서는 .\.venv\Scripts\python.exe, Linux에서는 .venv/bin/python으로 바꿔 직접 실행할 수 있다. 명령은 저장소 루트에서 실행한다.

~~~text
python -m scripts.run_submission_reports --update-readme
~~~

통합 명령은 선형대수·미적분·최적화와 두 노트북을 새로 실행하고 기존 측정값과 대조하여 다음 보고서를 생성한다. 원본 PNG·JSON·노트북을 보존하고 README의 지정 결과 영역을 갱신한다. 아래 문서의 고정 숫자는 별도 Q&A 설명이므로 실험 설정을 변경했다면 함께 다시 대조해야 한다.

| 내용 | 읽을 보고서 | 측정 원본 |
|---|---|---|
| 선형대수 #1~4 | [linear_algebra_report.md](../outputs/linear_algebra_report.md) | [linear_algebra_metrics.json](../outputs/linear_algebra_metrics.json) |
| 미적분 #5~6 | [calculus_report.md](../outputs/calculus_report.md) | [calculus_metrics.json](../outputs/calculus_metrics.json) |
| 역전파 #7~8 | [backprop_report.md](../outputs/backprop_report.md) | [backprop_metrics.json](../outputs/backprop_metrics.json) |
| 최적화 #9~12 | [optimizer_report.md](../outputs/optimizer_report.md) | [optimizer_metrics.json](../outputs/optimizer_metrics.json) |
| 확률·손실 #13~15 | [probability_report.md](../outputs/probability_report.md) | [probability_metrics.json](../outputs/probability_metrics.json) |

식의 ln은 자연로그, ||·||₂는 벡터의 유클리드 norm, ||·||F는 행렬의 Frobenius norm, ⊙는 원소별 곱이다. scalar의 NumPy shape은 (), 길이 2 벡터는 (2,)다. 그림의 축·단위·비교 조건은 각 PNG에서도 확인한다.

<a id="qa-01"></a>
## #1 — 행렬 곱은 단위 원을 어떻게 바꾸는가?

**질문·요구사항.** 회전, 스케일링, 전단을 구현하고 원본 단위 원과 변환 결과를 하나의 Figure에서 비교할 수 있는가?

**개념과 수식.** 원 위의 점을 열벡터로 묶은 P:(2,n)에 A:(2,2)를 왼쪽에서 곱한다. 각 열 p에 대해 p′=Ap이며 결과는 (2,n)이다. 오른쪽 곱이나 전치 행렬을 사용하면 다른 변환이 되므로 곱 방향을 고정한다.

$$
R(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix},
\quad S=\begin{bmatrix}2&0\\0&0.5\end{bmatrix},
\quad Sh(k)=\begin{bmatrix}1&k\\0&1\end{bmatrix}.
$$

**실제 입력·출력.** 기본 단위 원은 끝점을 중복하지 않는 400점이다. 회전각은 π/4(45°), 전단 k=1이다. 회전은 길이와 면적을 유지한다. 스케일링은 가로 반지름 2·세로 반지름 0.5의 타원으로 바꾸며 면적은 유지한다. 전단은 x′=x+y, y′=y로 기울인다. 계산 배열은 (2,400), Figure는 회전·스케일·전단의 세 패널이다.

**허용치·검증.** 이 항목의 그림 확인과 #2의 면적 검증은 별개다. 입력·결과가 유한하고 곱 방향 및 shape이 올바른지 검사하며 면적의 1% 기준은 다음 항목에서 적용한다.

**재현·근거.** python -m scripts.run_linear_algebra. [구현](../src/linear_algebra.py), [실행 설정](../scripts/run_linear_algebra.py), [변환 PNG](../outputs/linear_transforms.png), [JSON의 transforms](../outputs/linear_algebra_metrics.json), [선형대수 보고서](../outputs/linear_algebra_report.md).

**오해·반례.** “가로를 2배 늘리면 면적도 2배”는 세로 배율을 생략한 설명이다. 여기서는 2×0.5=1이다. 전단의 면적 보존은 전단이 길이·각도까지 보존한다는 뜻이 아니다.

<a id="qa-02"></a>
## #2 — 행렬식과 실제 면적비는 어떻게 비교하는가?

**질문·요구사항.** 행렬식과 변환 전후의 면적을 독립적으로 계산하고 1% 이내에서 일치하는가?

**개념과 유도.** A=[[a,b],[c,d]]의 행렬식은 ad−bc다. 두 기저벡터가 만드는 평행사변형의 부호 있는 면적 배율이며, 일반 도형의 양의 면적 배율은 |det(A)|이다. 다각형의 실제 면적은 꼭짓점 좌표에 대한 신발끈 공식으로 구한다.

$$
\operatorname{Area}(P)=\frac12\left|\sum_i(x_i y_{i+1}-y_i x_{i+1})\right|,
\quad r=\frac{\operatorname{Area}(AP)}{\operatorname{Area}(P)}.
$$

따라서 det≠0이면 상대오차 |r−|det(A)||/|det(A)|를 비교한다. det=0이면 이 나눗셈이 정의되지 않으므로 면적비의 절대오차 |r|을 사용한다.

**실제 입력·출력.** 원본 다각형 면적은 3.141463462364136이다. 유한 다각형이므로 π와 정확히 같을 필요는 없으며 동일 다각형의 전후 비율을 비교한다.

| 행렬 | det | 면적비 r | 측정 오차 | 기준 |
|---|---:|---:|---:|---|
| R(π/4) | 1.0000000000000002 | 0.9999999999999999 | 상대 3.330669073875469e-16 | ≤0.01 |
| diag(2,0.5) | 1 | 1 | 상대 0 | ≤0.01 |
| [[1,1],[0,1]] | 1 | 0.9999999999999994 | 상대 5.551115123125783e-16 | ≤0.01 |
| diag(−2,3) | −6 | 5.999999999999997 | 상대 4.440892098500626e-16 | ≤0.01 |
| diag(2,0) | 0 | 0 | 절대 0 | ≤1e-10 |

**허용치·검증.** 위 다섯 저장 사례는 각각의 기준을 만족한다. 행렬식 계산으로 면적을 대신한 뒤 두 값을 비교하면 독립 검증이 아니므로, 좌표 기반 다각형 면적과 행렬식을 따로 구한다.

**재현·근거.** python -m scripts.run_linear_algebra. [면적 구현·입력 검증](../src/linear_algebra.py), [측정 JSON](../outputs/linear_algebra_metrics.json), [보고서](../outputs/linear_algebra_report.md), [변환 그림](../outputs/linear_transforms.png). [추가 진단](../outputs/evaluation_evidence.md)은 고의로 면적비를 바꾼 실패 입력도 별도로 보여준다.

**오해·반례.** det=−6은 면적이 −6배라는 뜻이 아니라 방향이 반전되고 양의 면적이 6배가 된다는 뜻이다. 특이 행렬에 상대오차 식을 억지로 적용하면 0으로 나누게 된다.

<a id="qa-03"></a>
## #3 — Power Iteration은 무엇을 찾고 어떻게 검증하는가?

**질문·요구사항.** 고유값을 직접 반복 계산하고 np.linalg.eig 비교 상대오차 5% 이내를 확인하는가?

**개념과 유도.** Av=λv이면 A를 반복 적용할 때 고유방향의 크기는 λ의 거듭제곱으로 변한다. 절댓값이 가장 큰 고유값이 다른 값들과 분리되고 초기 벡터에 해당 방향 성분이 있으면 그 방향이 상대적으로 우세해진다.

$$
w_{t+1}=Av_t,\qquad v_{t+1}=w_{t+1}/\|w_{t+1}\|_2,\qquad
\lambda_t=\frac{v_t^TAv_t}{v_t^Tv_t}.
$$

이것은 대수적으로 가장 큰 고유값을 항상 찾는 알고리즘이 아니다. 예를 들어 diag(−5,2)에서 지배값은 −5다. 검증용 eig 결과도 |λ|가 가장 큰 항을 선택하고 부호를 유지한다.

**실제 입력·출력.** A=[[4,1],[1,3]], 초기 v=[1,1], dtype=float64다. 특성방정식 λ²−7λ+11=0에서 독립 기준은 (7+√5)/2이다.

| 값 | 저장된 결과 |
|---|---:|
| 수동 Power Iteration λ | 4.618033988749896 |
| NumPy eig λ | 4.618033988749895 |
| 수동 고유벡터 | [0.8506508082738572, 0.5257311122456358] |
| 상대 고유값 오차 | 1.923282552410481e-16 |
| 상대 잔차 ‖Av−λv‖₂/(‖A‖F·‖v‖₂) | 6.399563856969149e-11 |
| 부호를 맞춘 고유벡터 L2 차이 | 1.4871237088596054e-10 |
| 반복 횟수·종료 | 32회, residual_tolerance, converged=true |

**허용치·검증.** 고유값 상대오차 ≤0.05, 상대 잔차 ≤1e-6, 부호 정렬한 고유벡터 차이 ≤1e-8이다. 이 고정 사례의 저장 결과는 모두 만족한다. np.linalg.eig는 검증 스크립트·테스트에서만 사용하고 알고리즘 본체에서 호출하지 않는다.

**재현·근거.** python -m scripts.verify_power_iteration 또는 python -m scripts.run_linear_algebra. [독립 비교 Markdown](../outputs/power_iteration_eig_comparison.md), [비교 JSON](../outputs/power_iteration_eig_comparison.json), [선형대수 보고서](../outputs/linear_algebra_report.md), [알고리즘](../src/linear_algebra.py), [검증 전용 코드](../scripts/verify_power_iteration.py).

**오해·반례.** 잔차가 작으면 어떤 고유쌍에 가깝다는 근거가 되지만 반드시 지배 고유값을 찾았다는 뜻은 아니다. 초기 벡터가 지배 고유공간에 직교하면 그 방향을 얻지 못할 수 있다. 참고 예시의 51회는 정답이 아니며 종료 조건·초기값에 따라 달라진다.

<a id="qa-04"></a>
## #4 — SVD의 k, 복원 오차와 저장량은 어떤 관계인가?

**질문·요구사항.** SVD 기반 복원을 구현하고 k=10,50,100 결과를 같은 Figure에 비교하는가?

**개념과 수식.** A:(m,n)을 A=UΣVᵀ로 분해하고 가장 큰 k개 특잇값만 유지한다.

$$
A_k=U_{:,:k}\operatorname{diag}(s_{:k})V^T_{:k,:},
\quad e_k=\|A-A_k\|_F/\|A\|_F,
\quad \text{저장 원소 비율}=\frac{k(m+n+1)}{mn}.
$$

U_k, s_k, V_kᵀ의 shape은 각각 (m,k), (k,), (k,n)이다. rank는 정수이고 1≤k≤min(m,n)이다. 잘못된 rank를 조용히 줄이지 않는다.

**실제 입력·출력.** seed 42로 만든 float64 합성 grayscale 배열 두 개를 사용한다. 아래 오차는 화면 표시를 위한 clipping 이전의 복원 배열로 계산했다.

| 입력 shape | k | 상대 Frobenius 오차 | 저장/원본 원소 수 |
|---|---:|---:|---:|
| 64×64 | 10 | 0.11607845060286695 | 0.31494140625 |
| 64×64 | 50 | 0.013222144059052797 | 1.57470703125 |
| 64×64 | 64 | 2.3605124574215274e-15 | 2.015625 |
| 128×128 | 10 | 0.1275184713562091 | 0.1568603515625 |
| 128×128 | 50 | 0.05938687046053636 | 0.7843017578125 |
| 128×128 | 100 | 0.01218300917179587 | 1.568603515625 |

**허용치·검증.** 완전 rank 복원 오차는 64×64에서 2.3605124574215274e-15, 128×128에서 1.1042954043230347e-15이며 둘 다 ≤1e-10이다. 비교 rank가 증가할 때 오차가 증가하지 않는지 atol=1e-12로 확인한다. 각 낮은 rank의 오차 자체에 임의의 “5% 이내” 기준을 추가하지 않는다.

**재현·근거.** python -m scripts.run_linear_algebra --no-cache. [64×64 PNG](../outputs/svd_64.png), [128×128 PNG](../outputs/svd_128.png), [JSON의 svd](../outputs/linear_algebra_metrics.json), [보고서](../outputs/linear_algebra_report.md), [D09 정정 근거](DECISIONS.md#d09--svd-이미지-크기와-rank-충돌을-두-실험으로-해결한다).

**오해·반례.** 64×64 입력에서 rank 100은 불가능하다. 원문의 작은 이미지 가이드는 64×64 본 실험으로, 지정 rank는 별도 128×128 보충 실험으로 구분한다. “k를 늘렸으므로 압축이 더 좋다”도 모호하다. 복원 오차는 줄지만 저장 비율이 1보다 크면 원소 수 기준 저장 절감은 아니다. 64×64는 입력 배열 크기이지 제목·축을 포함한 전체 PNG 해상도가 아니다.

<a id="qa-05"></a>
## #5 — 중심차분에서 h를 작게 하면 언제나 정확해지는가?

**질문·요구사항.** f(x)=x², x=3에서 수치 미분과 해석값 6의 절대오차가 1e-4 이내인가?

**개념과 유도.** x+h와 x−h의 Taylor 전개를 빼면 짝수 차수 항이 소거된다. 충분히 매끄러운 일반 함수에서는

$$
\frac{f(x+h)-f(x-h)}{2h}=f'(x)+\frac{h^2}{6}f'''(x)+O(h^4).
$$

그러나 x²에서는 f'''=0이고 정확한 산술로 ((x+h)²−(x−h)²)/(2h)=2x이므로 절단오차가 없다. 컴퓨터에서는 가까운 수의 뺄셈과 2h로 나누는 과정의 반올림 오차가 남는다.

**실제 입력·출력.** x=3, h=1e-5, float64의 수치 미분은 6.000000000039306, 절대오차는 3.930633596382904e-11이다. 고정 사례는 ≤1e-4를 만족한다.

별도 탐색은 h=1e-15부터 1까지 로그 간격 61개를 사용했다. x²의 절대오차 ≤1e-4 표본은 **46/61**, 실패 표본은 **15/61**이다. 통과한 인접 표본 묶음은 5.6234132519034904e-12~1, 관측 최소오차는 h=1에서 0, 전체 최대오차는 0.6709294817992486이다. 이는 시험한 표본의 결과이며 그 사이 모든 실수 h의 연속 구간을 검증한 것은 아니다.

**허용치·검증.** 고정 h 검증과 탐색의 실패 표본을 함께 기록한다. h는 양의 유한 수여야 한다. sin(x) 비교 곡선에는 일반적인 O(h²) 절단오차가 있어, x²의 결과를 다른 함수에 그대로 적용할 수 없다.

**재현·근거.** python -m scripts.run_calculus. [h 오차 PNG](../outputs/derivative_error.png), [전체 61개 h·미분·오차 JSON](../outputs/calculus_metrics.json), [미적분 보고서](../outputs/calculus_report.md), [표본 묶음 진단](../outputs/evaluation_evidence.md), [공통 중심차분](../src/utils/math.py).

**오해·반례.** 가장 작은 h=1e-15의 오차가 약 0.670929이므로 “h는 작을수록 좋다”는 주장은 이 실험에서도 성립하지 않는다. 반대로 h=1이 x²에서 최소오차였다는 사실도 모든 함수의 최적 h=1을 뜻하지 않는다.

<a id="qa-06"></a>
## #6 — Gradient가 등고선에 수직이라는 뜻은 무엇인가?

**질문·요구사항.** f(x,y)=x²+y²의 등고선에 gradient 화살표를 그리고 수직 관계를 확인하는가?

**개념과 유도.** 등고선을 따라 움직이는 곡선 r(t)에서 f(r(t))=c이면 연쇄법칙으로 ∇f(r(t))·r′(t)=0이다. 따라서 gradient가 영벡터가 아닌 점에서 접선과 수직이다. 이 함수의 gradient는 (2x,2y), 원의 접선 방향은 (−y,x)다.

$$
(2x,2y)\cdot(-y,x)=-2xy+2xy=0.
$$

**실제 입력·출력.** 반경 2 원의 각도 0,π/4,…,7π/4에 해당하는 8점과 원점 1개를 검사했다. 점·해석 gradient·중심차분 gradient는 모두 (9,2)다. 예를 들어 (2,0)의 해석값은 (4,0), 수치값은 (4.000000000026205,0)이다.

| 검사 | 측정 | 기준 |
|---|---:|---:|
| 수치·해석 gradient의 최대 절대오차 | 3.29634097795406e-11 | ≤1e-8 |
| 원점 제외 정규화한 gradient·접선 내적의 최대 절댓값 | 0 | ≤1e-12 |
| 원점의 해석·수치 gradient | [0,0] | 정확히 [0,0] |
| 원점의 방향 | null | 방향을 정의하지 않음 |

**허용치·검증.** gradient의 각 성분은 h=1e-5의 중심차분과 대조한다. 수직성은 원점 외의 8점에서 길이로 나눈 내적을 사용한다. 그림의 화살표는 가독성을 위해 길이가 정규화되지만 검증 배열의 실제 gradient 크기를 대체하지 않는다.

**재현·근거.** python -m scripts.run_calculus. [등고선·Gradient PNG](../outputs/gradient_geometry.png), [측정 배열 JSON](../outputs/calculus_metrics.json), [미적분 보고서](../outputs/calculus_report.md), [구현](../src/calculus.py).

**오해·반례.** 원점에서는 gradient가 0이므로 특정 화살표 방향을 붙일 수 없다. 내적이 0이라는 식만 보고 영벡터에 기하학적 방향을 부여하면 안 된다.

<a id="qa-07"></a>
## #7 — 순전파와 여섯 역전파 미분은 어떻게 유도하는가?

**질문·요구사항.** 입력 2→은닉 2→출력 1 신경망에서 고정 예제를 유지하고 필수 여섯 미분 및 각 shape을 노트북에 전개하는가?

**고정 입력.**

| 기호 | 값 | shape |
|---|---|---|
| x | [1,0] | (2,) |
| W₁ | [[0.1,0.2],[0.3,0.4]] | (2,2) |
| b₁ | [0,0] | (2,) |
| W₂ | [0.5,0.6] | (2,) |
| b₂, y | 각각 0, 1 | 각각 () |

**순전파 유도.** σ(z)=1/(1+e^(−z)), σ′(z)=σ(z)(1−σ(z))다. 첫 층의 곱 방향은 W₁@x, 출력층은 W₂@a₁이다.

$$
z_1=W_1x+b_1=[0.1,0.3],\quad a_1=\sigma(z_1),\quad
z_2=W_2a_1+b_2,\quad p=\sigma(z_2),\quad
L=-[y\ln p+(1-y)\ln(1-p)].
$$

저장된 a₁=[0.52497918747894,0.574442516811659], z₂=0.6071551038264654, p=0.6472915699946178, L=0.43495843685146063이다. z₁·a₁은 (2,), z₂·p·L은 ()다.

### 미분 1 — 예측 확률에 대한 손실

$$
\frac{\partial L}{\partial p}=-\frac yp+\frac{1-y}{1-p}.
$$

y=1을 대입하면 −1/p=−1.5448988467566709이며 shape은 ()다. 이 식은 이 고정 예제처럼 0<p<1인 내부 확률에서 사용한다.

### 미분 2 — 출력 logit

$$
\begin{aligned}
\delta_2=\frac{\partial L}{\partial z_2}
&=\frac{\partial L}{\partial p}\frac{\partial p}{\partial z_2}\\
&=\left(-\frac yp+\frac{1-y}{1-p}\right)p(1-p)\\
&=-y(1-p)+(1-y)p=p-y.
\end{aligned}
$$

δ₂=−0.3527084300053822, shape ()다. 이 간단한 결과는 BCE와 Sigmoid의 미분을 합성했기 때문에 나온다.

### 미분 3 — 출력 가중치

z₂=ΣⱼW₂,ⱼa₁,ⱼ+b₂이므로 ∂z₂/∂W₂,ⱼ=a₁,ⱼ다. 따라서

$$
\frac{\partial L}{\partial W_2}=\delta_2a_1
=[-0.18516458500119815,-0.20261071823298063],\qquad \text{shape }(2,).
$$

### 미분 4 — 은닉 활성값

∂z₂/∂a₁,ⱼ=W₂,ⱼ이므로

$$
\frac{\partial L}{\partial a_1}=\delta_2W_2
=[-0.1763542150026911,-0.2116250580032293],\qquad \text{shape }(2,).
$$

### 미분 5 — 은닉 logit

Sigmoid는 각 은닉 성분에 독립적으로 적용된다. Jacobian은 a₁⊙(1−a₁)을 대각 원소로 갖기 때문에 원소별 곱으로 쓸 수 있다.

$$
\delta_1=\frac{\partial L}{\partial z_1}
=(\delta_2W_2)\odot a_1\odot(1-a_1)
=[-0.04397851580869701,-0.051733504390925604],\quad \text{shape }(2,).
$$

### 미분 6 — 첫 층 가중치

z₁,ᵢ=ΣⱼW₁,ᵢⱼxⱼ+b₁,ᵢ이므로 ∂z₁,ᵢ/∂W₁,ᵢⱼ=xⱼ다.

$$
\frac{\partial L}{\partial W_{1,ij}}=\delta_{1,i}x_j,\qquad
\frac{\partial L}{\partial W_1}=\delta_1x^T
=\begin{bmatrix}-0.04397851580869701&0\\-0.051733504390925604&0\end{bmatrix},
\quad \text{shape }(2,2).
$$

NumPy 대응식은 outer(δ₁,x)다. 편향 미분은 ∂L/∂b₂=δ₂, ∂L/∂b₁=δ₁이며 shape은 각각 (), (2,)다.

**허용치·검증.** 유도 자체와 수치 검증은 구분한다. 위 식의 독립 손계산 상수와 NumPy를 #8의 atol=5e-5, rtol=0으로 비교하고 9개 매개변수의 중심차분도 별도로 확인한다.

**재현·근거.** python -m scripts.run_notebooks. [역전파 노트북](../notebooks/backprop_derivation.ipynb)의 backprop-fixed-input, backprop-forward-derivation, backprop-chain-rule, backprop-backward-numpy 셀; [상세 수식](MATH_DERIVATIONS.md), [JSON의 parameters·forward·backward·shapes](../outputs/backprop_metrics.json), [역전파 보고서](../outputs/backprop_report.md). 이 항목의 핵심 산출물은 유도·표·실행 출력이며 별도 역전파 PNG를 존재하는 것처럼 인용하지 않는다.

**오해·반례.** x의 두 번째 성분이 0이므로 dL/dW₁의 두 번째 열이 0이다. 이것은 두 번째 은닉 뉴런 전체가 학습하지 않는다는 뜻이 아니다. 또한 단일 표본이므로 배치 평균의 1/n을 추가하지 않는다. W₁·x의 곱 방향을 바꾸거나 중간값을 네 자리로 잘라 재사용하면 고정 예제와 다른 계산이 된다.

<a id="qa-08"></a>
## #8 — 손계산과 NumPy의 네 자리 일치를 어떻게 검증하는가?

**질문·요구사항.** #7의 순전파·역전파 중간값을 독립 기준과 비교하고 소수점 네 자리까지 일치하는지 실행 결과로 확인하는가?

**개념.** “같은 함수를 두 번 호출해 같은 값이 나왔다”는 손계산 독립 검증이 아니다. 노트북은 독립적으로 평가해 고정한 hand_reference 상수와 NumPy 계산을 비교한다. 검증에는 반올림 전 값을 사용하고 네 자리 표시는 마지막에만 만든다.

| 체크포인트 | 독립 기준의 네 자리 표시 | shape |
|---|---|---|
| z₁ | [0.1000,0.3000] | (2,) |
| a₁ | [0.5250,0.5744] | (2,) |
| z₂ | 0.6072 | () |
| p | 0.6473 | () |
| L | 0.4350 | () |
| dL/dp | −1.5449 | () |
| dL/dz₂ | −0.3527 | () |
| dL/dW₂ | [−0.1852,−0.2026] | (2,) |
| dL/da₁ | [−0.1764,−0.2116] | (2,) |
| dL/dz₁ | [−0.0440,−0.0517] | (2,) |
| dL/dW₁ | [[−0.0440,0.0000],[−0.0517,0.0000]] | (2,2) |
| dL/db₁ | [−0.0440,−0.0517] | (2,) |
| dL/db₂ | −0.3527 | () |

**실제 결과·허용치.** 13개 체크포인트의 각 성분은 절대오차 ≤5e-5, 상대 허용오차 0으로 비교하고 네 자리 표시도 대조한다. 저장된 최대 손계산 절대오차는 2.220446049250313e-16, four_decimal_match=true다. −0.0은 수학적으로 0이며 네 자리 표시에서만 부호를 정리한다.

9개 매개변수의 순서는 W₁ 네 성분, b₁ 두 성분, W₂ 두 성분, b₂다. 좌표 i마다

$$
g_i^{\mathrm{num}}=\frac{L(\theta+h e_i)-L(\theta-h e_i)}{2h},\quad h=10^{-5}
$$

와 해석 gradient를 비교했다. 최대오차는 7.842670957103337e-12로 atol=1e-6, rtol=0을 만족한다. 이 검증은 역전파 계산과 다른 경로로 손실 함수를 미분한다.

**재현·근거.** python -m scripts.run_notebooks. [원본 노트북](../notebooks/backprop_derivation.ipynb)의 backprop-hand-reference, backprop-compare, backprop-gradient-check 셀; [전체 상수·13개 오차·9개 gradient JSON](../outputs/backprop_metrics.json), [역전파 보고서](../outputs/backprop_report.md), [D12 정정](DECISIONS.md#d12--역전파는-고정-예제를-유지하고-참고-수치를-재계산한다).

**오해·반례.** 원문 참고값 z₂=0.6071과 두 번째 은닉 기울기 −0.0513은 이 고정 예제의 정확한 네 자리 반올림이 아니다. 각각 0.6072, −0.0517이다. 원문을 고치거나 입력을 바꾸지 않고 정정 근거를 결정 기록에 보관한다. 유한 표본의 수치 검증은 모든 입력에서의 보장을 뜻하지 않는다.

<a id="qa-09"></a>
## #9 — Vanilla GD는 100회 후 반경 0.1에 도달하는가?

**질문·요구사항.** f(x,y)=x²+y², 초기점 (5,5), lr=0.1에서 100회 업데이트한 결과가 원점 반경 0.1 이내인가?

**개념과 유도.** θ=(x,y), ∇f=2θ이므로

$$
\theta_{t+1}=\theta_t-\alpha\nabla f(\theta_t)
=(1-2\alpha)\theta_t=0.8\theta_t,\qquad
\theta_t=0.8^t(5,5).
$$

반경은 ||θₜ||₂=√50·0.8ᵗ, 목적함수는 f(θₜ)=50·0.64ᵗ다. 반복 구현과 독립적으로 이 닫힌 식을 대조할 수 있다.

**실제 입력·출력.**

| 값 | 저장된 결과 |
|---|---|
| 초기점·초기 손실 | [5,5], 50 |
| 100회 후 좌표 | [1.0185179881672439e-9,1.0185179881672439e-9] |
| 최종 반경 | 1.4404019523870759e-9 |
| 최종 손실 | 2.0747577844404998e-18 |
| 최종 gradient norm | 2.8808039047741517e-9 |
| 반경 0.1 첫 진입·예산 끝까지 유지 시작 | 각각 20회 |
| 경로 shape | 초기점 포함 (101,2) |

**허용치·검증.** 최종 반경 ≤0.1을 만족한다. 경로의 손실·gradient norm도 각각 길이 101이며 업데이트 100회를 초기점까지 합쳐 101회라고 부르지 않는다. 이 고정 예산 실험의 status=budget_exhausted는 “100회 예산을 사용했다”는 의미다. 반경 수용 기준 실패라는 뜻은 아니다.

**재현·근거.** python -m scripts.run_optimizer --no-cache. [구현](../src/optimizer.py), [원형 경로·손실 PNG](../outputs/optimizer_circle.png), [JSON의 comparisons.circle.GD](../outputs/optimizer_metrics.json), [최적화 보고서](../outputs/optimizer_report.md), [닫힌 식·다중 초기점 보고서](../outputs/optimizer_sensitivity.md).

**오해·반례.** 원문의 참고 좌표 (0.08,0.08)를 실제값으로 복사하지 않는다. 또한 기준 반경을 1로 느슨하게 바꾸지 않고 원문의 0.1을 사용한다. 작은 최종 손실 하나만 보고 업데이트 횟수·시작점·함수를 생략하면 다른 실험과 비교할 수 없다.

<a id="qa-10"></a>
## #10 — lr=0.5, 1, 1.1은 왜 각각 수렴·진동·발산하는가?

**질문·요구사항.** 학습률에 따른 경로 변화를 실제 그림과 수치로 설명하는가? 원문의 “0.5 이상 발산”을 그대로 따라도 되는가?

**개념과 유도.** 원형 GD의 배율은 c=1−2α다. 원점이 아닌 초기점에서 수렴하려면 |c|<1이어야 하므로 0<α<1이다. 손실의 매 단계 배율은 c²다.

| lr | 좌표 배율 1−2lr | 해석 | 20회 후 측정 반경 |
|---|---:|---|---:|
| 0.5 | 0 | 1회 후 원점 | 0 |
| 1.0 | −1 | 부호 반전, 반경 유지 | 7.0710678118654755 |
| 1.1 | −1.2 | 부호 반전과 반경 증가 | 271.0877688101304 |

초기점은 모두 (5,5), 20회이며 경로 shape은 (21,2)다. lr=1에서 짝수 회의 좌표는 시작점과 같아진다. lr=1.1의 반경은 √50·1.2²⁰와 대응한다.

**허용치·검증.** lr=0.5의 반경 0.1 첫 도달은 1회다. 나머지 두 경우의 first_radius_0_1·유지 시작은 null이며 0회나 마지막 회차로 대체하지 않는다. 발산을 보여주는 유한 20회 관측은 비유한 수치 예외를 억지로 일으키는 실험이 아니다.

**재현·근거.** python -m scripts.run_optimizer --no-cache. [학습률 비교 PNG](../outputs/optimizer_learning_rates.png), [JSON의 learning_rates](../outputs/optimizer_metrics.json), [최적화 보고서](../outputs/optimizer_report.md), [D10 수식 정정](DECISIONS.md#d10--원형-함수의-학습률-발산-조건을-정정한다).

**오해·반례.** “lr≥0.5이면 모두 발산”의 반례가 바로 lr=0.5다. 함수에 1/2을 붙이거나 gradient에서 2를 빼면 임계 학습률이 달라지지만, 그것은 원문의 f=x²+y²와 다른 설정이다. 이미 원점인 초기점은 모든 세 학습률에서 그대로 원점이므로 발산 설명에는 비영 초기점 조건이 필요하다.

<a id="qa-11"></a>
## #11 — Momentum의 첫 도달과 유지 시작은 왜 다른가?

**질문·요구사항.** β=0.9의 Momentum을 직접 구현하고 동일 조건의 GD와 경로·손실을 비교하는가?

**개념과 수식.** 이 프로젝트의 Momentum 관례는

$$
v_{t+1}=\beta v_t+\nabla f(\theta_t),\qquad
\theta_{t+1}=\theta_t-\alpha v_{t+1},\qquad v_0=0
$$

이다. 이전 gradient를 속도에 누적하므로 같은 방향의 힘을 이어갈 수 있지만 최소점을 지나치는 진동도 가능하다. gradient에 (1−β)를 곱하는 다른 관례와 섞지 않는다.

**실제 입력·출력.** 원형 함수, 시작점 (5,5), lr=0.1, 업데이트 100회다. GD와 Momentum은 모두 (101,2) 경로를 저장한다.

| 방법 | 최종 좌표 | 최종 반경 | 반경 0.1 첫 진입 | 기록 끝까지 유지 시작 |
|---|---|---:|---:|---:|
| GD | [1.0185179881672439e-9,1.0185179881672439e-9] | 1.4404019523870759e-9 | 20 | 20 |
| Momentum | [−0.01425705560591328,−0.01425705560591328] | 0.020162521397389926 | 10 | 77 |

Momentum의 최종 손실은 0.0004065272691002066, gradient norm은 0.04032504279477985다. 첫 진입은 더 빠르지만 100회 후 반경은 GD보다 크다.

**허용치·검증.** “첫 진입”은 최초로 ||θ||≤0.1인 인덱스다. “유지 시작”은 그 시점부터 기록된 마지막 시점까지 모두 반경 안에 있는 최초 인덱스다. 둘 다 원점을 초기값으로 주면 0일 수 있고 미도달은 null이다. 유한한 기록 끝까지의 유지는 무한 시간 보장이 아니다.

**재현·근거.** python -m scripts.run_optimizer --no-cache. [원형 비교 PNG](../outputs/optimizer_circle.png), [JSON의 comparisons.circle](../outputs/optimizer_metrics.json), [보고서](../outputs/optimizer_report.md), [여러 초기점의 성공수·조건부 통계](../outputs/optimizer_sensitivity.md), [Momentum 구현](../src/optimizer.py).

**오해·반례.** 첫 진입 10회만으로 “Momentum은 항상 더 빠르다”라고 결론 내리면 다시 반경 밖으로 나가는 구간을 놓친다. 최종 반경·목적함수·유지 시작·예산을 함께 비교해야 한다. 초기점별 평균을 낼 때도 미도달 null을 0으로 취급하지 않는다.

<a id="qa-12"></a>
## #12 — 타원형 함수의 곡률 차이는 최적화에 어떤 영향을 주는가?

**질문·요구사항.** f=x²+10y²에서 GD·Momentum의 경로와 실제 이점·한계를 비교하는가?

**개념과 유도.** gradient는 (2x,20y), Hessian은 diag(2,20)이다. GD의 축별 갱신은

$$
x_{t+1}=(1-2\alpha)x_t,\qquad y_{t+1}=(1-20\alpha)y_t.
$$

두 축 모두 수렴하려면 0<α<0.1이다. 원문 설정 α=0.05에서는 x 배율이 0.9이고 y 배율이 0이다. 따라서 GD의 y는 한 번에 0이 되며 이 설정에서 y축 지그재그가 계속된다고 설명할 수 없다.

Momentum은 Hessian 고유값 λ인 방향에서

$$
\theta_{t+1}=(1+\beta-\alpha\lambda)\theta_t-\beta\theta_{t-1}
$$

의 이차 점화식으로 해석한다. 특성방정식은 r²−(1+β−αλ)r+β=0이다. 복소근은 감쇠 진동을 만들 수 있으므로 경로가 매번 더 부드럽거나 오차가 매번 작다는 보장은 없다.

**실제 입력·출력.** 시작점 (5,5), lr=0.05, β=0.9, 100회다.

| 방법 | 최종 좌표 | 최종 반경 | 최종 손실 | 반경 0.1 첫 진입/유지 |
|---|---|---:|---:|---|
| GD | [0.00013280699443793718,0] | 0.00013280699443793718 | 1.7637697771638275e-8 | 38/38 |
| Momentum | [0.018693666556487035,0.0067502306888646835] | 0.019875079463438944 | 0.0008051093128540277 | 65/80 |

**허용치·검증.** 비교의 초기점·함수·예산을 일치시키고 lr·β를 공개한다. 수용 기준은 경로·손실·gradient norm의 일관된 계산과 해석이며 Momentum 우월성을 강제하는 assertion이 아니다. 이 고정 조건의 반경과 도달 지표에서는 GD가 더 좋다.

**재현·근거.** python -m scripts.run_optimizer --no-cache. [타원형 비교 PNG](../outputs/optimizer_ellipse.png), [JSON의 comparisons.ellipse](../outputs/optimizer_metrics.json), [최적화 보고서](../outputs/optimizer_report.md), [D11 비교 관례](DECISIONS.md#d11--momentum-비교는-실측-결과로-판단한다).

**오해·반례.** Momentum이 불리한 결과를 없애거나 학습률을 몰래 바꾸어 원문 예시처럼 보이게 만들지 않는다. 다른 설정에서의 이점은 별도 실험으로 검증해야 한다. 100회 기본 실험과 200회 Adam 보너스 실험의 결과도 섞지 않는다.

<a id="qa-13"></a>
## #13 — PDF·PMF와 분산·표준편차는 어떻게 다른가?

**질문·요구사항.** 정규분포 N(0,1), N(2,0.5)의 PDF와 Bernoulli(0.3), Bernoulli(0.7)의 PMF를 각각 비교 그림으로 제공하는가?

**개념과 수식.** 정규분포의 두 번째 인자는 분산 σ²다. PDF는 연속 밀도이며 점 하나의 확률이 아니라 구간 적분이 확률이다. Bernoulli는 이산 확률질량이므로 두 값의 합이 1이다.

$$
f(x;\mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left[-\frac{(x-\mu)^2}{2\sigma^2}\right],\qquad
P(X=k)=p^k(1-p)^{1-k},\quad k\in\{0,1\}.
$$

**실제 입력·출력.** 각 정규분포의 μ±8σ를 10,001점으로 나누어 사다리꼴 적분했다.

| 정규분포 | 표준편차 σ | 평균에서의 PDF | 적분값 | 1 대비 절대오차 |
|---|---:|---:|---:|---:|
| N(0,1) | 1 | 0.3989422804014327 | 0.9999999999999989 | 1.1102230246251565e-15 |
| N(2,0.5) | 0.7071067811865476 | 0.5641895835477563 | 0.999999999999999 | 9.992007221626409e-16 |

Bernoulli의 지지집합 순서는 [0,1]이다. p=0.3의 PMF는 [0.7,0.3], p=0.7은 [0.30000000000000004,0.7]이며 합 오차는 둘 다 0이다. 마지막 표현 차이는 float64의 1−0.7 결과다.

**허용치·검증.** PDF 비음수, 적분 오차 ≤1e-4를 확인한다. PMF는 각 값이 [0,1]이고 합의 절대오차 ≤1e-12여야 한다. 유한 구간의 수치 적분을 기호 적분의 증명으로 혼동하지 않는다.

**재현·근거.** python -m scripts.run_notebooks. [확률 노트북](../notebooks/probability_loss.ipynb), [정규 PDF PNG](../outputs/probability_normal.png), [Bernoulli PMF PNG](../outputs/probability_bernoulli.png), [JSON의 normals·bernoulli](../outputs/probability_metrics.json), [확률 보고서](../outputs/probability_report.md).

**오해·반례.** N(2,0.5)의 표준편차를 0.5로 넣으면 다른 분포가 된다. PDF의 높이는 1을 넘을 수 있지만 구간 전체 확률은 1을 넘지 않는다. 예를 들어 충분히 좁은 정규분포의 봉우리는 1보다 높다. PMF의 각 확률이 1 이하여야 한다는 조건을 PDF 높이에 그대로 적용하지 않는다.

<a id="qa-14"></a>
## #14 — Softmax는 어느 축을 정규화하고 큰 logit을 어떻게 처리하는가?

**질문·요구사항.** 직접 구현한 Softmax의 출력 합이 선택한 축마다 1과 1e-6 이내로 일치하는가?

**개념과 유도.** logit z의 각 지수값을 합으로 나누면 음이 아닌 값들의 합이 1이다. 공통 상수 m을 빼도 분자·분모의 eᵐ이 약분되어 같은 분포가 된다.

$$
p_j=\frac{e^{z_j}}{\sum_k e^{z_k}}
=\frac{e^{z_j-m}}{\sum_k e^{z_k-m}},\qquad m=\max_k z_k.
$$

최댓값을 빼면 양의 큰 지수의 overflow를 피한다. axis=-1은 마지막 축, 2차원 배열의 axis=0은 각 열 방향을 뜻한다. 입력 shape은 그대로 유지하고 해당 축을 따라 합을 구하면 그 축이 제거된 shape이 된다.

**실제 입력·출력.** (2,3) 입력의 axis=-1 사례다.

| logit 행 | 출력 확률 행 | 합 |
|---|---|---:|
| [1000,1001,1002] | [0.09003057317038046,0.24472847105479764,0.6652409557748218] | 0.9999999999999999 |
| [−1000,0,1000] | [0,0,1] | 1 |

저장된 행 합 최대오차는 1.1102230246251565e-16이다. 별도로 axis=0으로 계산한 열 정규화 검증의 합 오차도 1.1102230246251565e-16이다. 이것은 위 axis=-1 출력의 열 합도 1이라는 뜻이 아니다.

**허용치·검증.** 각 축의 합 오차 ≤1e-6, 유한한 음이 아닌 출력을 검사한다. 공통 API는 유한 실수의 비어 있지 않은 배열과 유효한 정수 축을 요구하고 scalar·빈 배열·잘못된 축·bool/complex 입력을 거부한다. 확장 검증에는 여러 shape과 axis −1/0/1 사례가 있다.

**재현·근거.** python -m scripts.run_notebooks. [공통 softmax 구현](../src/utils/math.py), [확률 노트북](../notebooks/probability_loss.ipynb), [원본 JSON의 softmax](../outputs/probability_metrics.json), [확률 보고서](../outputs/probability_report.md), [확장 shape/axis 보고서](../outputs/evaluation_evidence.md).

**오해·반례.** 위 둘째 행의 0은 매우 작은 지수값이 float64에서 underflow된 결과이며 clipping으로 만든 값이 아니다. 수학적 양수 확률을 모든 크기의 입력에서 양의 float64로 표현할 수 있는 것은 아니다. 잘못된 축을 선택해도 합은 1이 될 수 있으므로 shape·축의 의미를 함께 확인해야 한다.

<a id="qa-15"></a>
## #15 — MLE에서 MSE·BCE·Cross-Entropy를 어떻게 유도하는가?

**질문·요구사항.** 분포 가정과 likelihood를 명시하고, MSE와 Cross-Entropy 최소화가 언제 최대우도추정과 같은 문제인지 단계적으로 설명하는가?

### 15-A. likelihood·log-likelihood·NLL은 무엇인가?

관측 데이터는 고정하고 그 데이터를 만들 모형의 매개변수 θ를 바꾸어 보는 함수가 likelihood다. 확률값을 θ에 대한 확률분포라고 자동 해석하지 않는다. 독립 표본의 likelihood는 곱이고, 로그는 곱을 합으로 바꾼다.

$$
\mathcal L(\theta)=\prod_{i=1}^n p(y_i\mid x_i,\theta),\qquad
\operatorname{NLL}(\theta)=-\ln\mathcal L(\theta).
$$

ln은 단조증가하므로 likelihood 최대화와 log-likelihood 최대화가 같고, 앞에 음수를 붙이면 NLL 최소화가 된다. 합과 평균은 n>0인 같은 데이터셋에서 양의 상수배이므로 같은 최적해를 갖지만 손실과 gradient의 크기는 다르다.

### 15-B. 고정 분산 Gaussian에서 MSE까지

가정은 yᵢ=fθ(xᵢ)+εᵢ, 서로 독립인 εᵢ~N(0,σ²), 모든 관측에서 같은 알려진 양의 상수 σ²다.

$$
p(y_i\mid x_i,\theta)=\frac1{\sqrt{2\pi\sigma^2}}
\exp\left[-\frac{(y_i-f_\theta(x_i))^2}{2\sigma^2}\right].
$$

독립성으로 곱을 만들고 로그를 취한 다음 음수를 붙인다.

$$
\begin{aligned}
\ln\mathcal L
&=-\frac n2\ln(2\pi\sigma^2)-\frac1{2\sigma^2}\sum_i(y_i-f_\theta(x_i))^2,\\
\operatorname{NLL}
&=\frac n2\ln(2\pi\sigma^2)+\frac{\operatorname{SSE}}{2\sigma^2}\\
&=\frac n2\ln(2\pi\sigma^2)+\frac{n}{2\sigma^2}\operatorname{MSE}.
\end{aligned}
$$

첫 항은 θ와 무관하고 두 번째 항의 계수는 양수다. 따라서 같은 매개변수 집합에서 최적해가 존재하면 argmax likelihood = argmin NLL = argmin SSE = argmin MSE다. 평균 NLL은 NLL/n이다.

**실제 입력·출력.** 관측 [0,1,2], 예측 [1,0.5,1.5], 분산 0.5, n=3이다.

| 값 | 결과 |
|---|---:|
| SSE=1²+0.5²+0.5² | 1.5 |
| MSE=SSE/3 | 0.5 |
| NLL | 3.2170948287741004 |
| 평균 NLL | 1.0723649429247002 |
| 닫힌 식과 PDF 직접 로그합의 차이 | 4.440892098500626e-16 |

독립 NLL 기준과 항등식 오차의 허용치는 1e-14다. NLL=3.217…와 MSE=0.5는 다르지만 위 가정 아래에서는 같은 최적해를 가진다.

**오해·반례.** σ²가 입력·θ에 따라 변하면 로그 분산 항과 오차별 가중치를 버릴 수 없다. 분산도 함께 추정하는 문제를 이 고정 분산 유도와 동일시하지 않는다. 연속 밀도가 1보다 클 수 있으므로 Gaussian NLL은 음수일 수도 있다.

### 15-C. Bernoulli likelihood에서 BCE까지

레이블 yᵢ∈{0,1}, 성공확률 pᵢ에 대해 한 관측의 확률은 pᵢ^yᵢ(1−pᵢ)^(1−yᵢ)다. 독립 표본의 곱에 −ln을 취하면

$$
\begin{aligned}
\mathcal L&=\prod_i p_i^{y_i}(1-p_i)^{1-y_i},\\
L_{\mathrm{sum}}=-\ln\mathcal L
&=-\sum_i[y_i\ln p_i+(1-y_i)\ln(1-p_i)],\\
L_{\mathrm{mean}}&=L_{\mathrm{sum}}/n.
\end{aligned}
$$

따라서 BCE의 합은 Bernoulli NLL 그 자체다. pᵢ=σ(zᵢ)를 대입하면

$$
\ell(z_i,y_i)=\ln(1+e^{z_i})-y_i z_i,\qquad
\frac{\partial L_{\mathrm{sum}}}{\partial z_i}=p_i-y_i,\qquad
\frac{\partial L_{\mathrm{mean}}}{\partial z_i}=(p_i-y_i)/n.
$$

shape은 y,p,z:(n,), 표본별 손실:(n,), 합·평균:()다. logaddexp(0,z)−yz는 exp overflow를 피하지만 큰 양의 z, y=1에서 마지막 뺄셈이 작은 손실을 상쇄할 수 있다. **이진 레이블에 한해** 동치인 logaddexp(0,(1−2y)z)를 사용하면 이 뺄셈을 피한다.

**실제 입력·출력.** y=[0,1], p=[0.3,0.7]의 BCE 합은 0.7133498878774649, 평균은 0.35667494393873245다. 확률식·logit식의 합이 일치하며 합 손실 gradient는 [0.3,−0.30000000000000004], 중심차분 최대오차는 8.626654945942391e-12이다. 손실 비교 atol=1e-15, gradient atol=1e-8, rtol=0으로 검사한다.

y=1,z=40에서 안정적인 이진식의 양의 손실은 4.248354255291589e-18이며 logaddexp(0,40)−40의 직접 뺄셈은 0으로 상쇄된다. y=[0,1,1,0], z=[−1000,1000,−1000,1000]의 손실은 [0,0,1000,1000]이다.

**오해·반례.** logaddexp(0,(1−2y)z)를 임의의 soft target y∈(0,1)에 그대로 적용하면 원래 BCE와 다른 식이 된다. 또한 p=0,y=1 또는 p=1,y=0은 불가능한 사건이어서 손실이 +∞다. p=0,y=0 또는 p=1,y=1은 손실 0이며 0·log(0) 항을 먼저 제외한다. 무조건 clipping하면 다른 목적함수를 계산하게 된다.

### 15-D. Categorical likelihood에서 Cross-Entropy까지

K개 클래스 중 관측 클래스 cᵢ를 one-hot yᵢⱼ로 나타내고 각 표본의 확률 합이 1이라고 하자.

$$
\begin{aligned}
\mathcal L&=\prod_{i=1}^n\prod_{j=1}^K p_{ij}^{y_{ij}},\\
L_{\mathrm{sum}}
&=-\sum_i\sum_j y_{ij}\ln p_{ij}
=-\sum_i\ln p_{i,c_i},\\
L_{\mathrm{mean}}&=L_{\mathrm{sum}}/n.
\end{aligned}
$$

이것이 one-hot categorical Cross-Entropy다. 확률·logit·one-hot shape은 (n,K), 표본별 손실은 (n,), 합·평균은 ()다. 한 표본의 Softmax를 대입하면

$$
\ell(z,c)=\ln\sum_j e^{z_j}-z_c
=\ln\sum_j e^{z_j-m}-(z_c-m),\quad m=\max_j z_j.
$$

최댓값을 뺀 logit을 사용하면 exp overflow뿐 아니라 큰 공통 offset의 마지막 뺄셈으로 인한 손실도 줄인다. logit 미분은 ∂ℓ/∂zⱼ=pⱼ−yⱼ이며 평균 손실이면 n으로 나눈다.

**실제 입력·출력.** p=[0.2,0.3,0.5], 클래스 인덱스 c=2(0부터 세기)의 손실은 −ln(0.5)=0.6931471805599453이다. logit=ln([2,3,5])의 안정식과 1e-15 이내로 일치한다. logit=[−1000,0,1000], c=0의 손실은 2000이다.

**오해·반례.** [0,0,1]에서 c=2의 손실은 0이고 c=0의 손실은 +∞다. 실제 클래스가 아닌 위치의 0확률은 one-hot 계수가 0이어서 손실 항에 포함하지 않는다. Bernoulli의 두 결과를 다루는 BCE와 K개 중 하나를 다루는 categorical CE는 관련되지만 입력·분포 가정을 구분해야 한다.

**전체 재현·근거.** python -m scripts.run_notebooks. [확률·손실 노트북](../notebooks/probability_loss.ipynb)의 gaussian-mle-derivation, gaussian-mle-check, bernoulli-mle-derivation, bce-check, categorical-mle-derivation, categorical-check 셀; [단계별 상세 수식 6~8절](MATH_DERIVATIONS.md), [JSON의 gaussian·bce·categorical](../outputs/probability_metrics.json), [확률 보고서](../outputs/probability_report.md), [D13 가정](DECISIONS.md#d13--확률-표기와-mle의-가정을-고정한다).

<a id="qa-adam"></a>
## 보너스 Q&A — Adam의 모멘트·편향 보정·epsilon

**Q. Adam은 기존 Momentum과 무엇이 다른가?** Gradient의 1차 모멘트 m과 제곱의 2차 모멘트 v를 지수 이동평균으로 추정하고, 좌표별 크기로 갱신을 정규화한다. θ,g,m,v의 shape은 모두 (n,), t는 갱신 횟수다.

$$
\begin{aligned}
m_t&=\beta_1m_{t-1}+(1-\beta_1)g_t,\\
v_t&=\beta_2v_{t-1}+(1-\beta_2)g_t^2,\\
\hat m_t&=m_t/(1-\beta_1^t),\qquad \hat v_t=v_t/(1-\beta_2^t),\\
\theta_t&=\theta_{t-1}-\alpha\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
\end{aligned}
$$

기존 Momentum은 v_next=βv+g이고 Adam의 m은 (1−β₁)g를 사용한다. 두 상태 변수의 이름이나 갱신 관례를 혼합하지 않는다.

**Q. 편향 보정은 왜 하는가?** m₀=v₀=0에서 시작하면 초기 평균이 0 쪽으로 치우친다. 일정한 g에 대해 mₜ=(1−β₁ᵗ)g, vₜ=(1−β₂ᵗ)g²이므로 보정하면 각각 g, g²가 된다. epsilon은 sqrt 바깥이며 sqrt(vhat+epsilon)과 다른 식이다.

**Q. 실제 비교에서 더 빨랐는가?** 시작 (5,5), 200회, Momentum β=.9, Adam β₁=.9/β₂=.999/epsilon=1e-8이다. 원형은 모두 lr=.1, 타원형은 모두 lr=.05다. 아래 반경은 저장된 값을 짧게 반올림한 것이다.

| 함수 | 방법 | 최종 반경(근사) | 반경 0.1 첫 진입/유지 |
|---|---|---:|---|
| 원형 | GD | 2.934151e-19 | 20/20 |
| 원형 | Momentum | 9.489598e-5 | 10/77 |
| 원형 | Adam | 1.549128e-4 | 79/79 |
| 타원형 | GD | 3.527540e-9 | 38/38 |
| 타원형 | Momentum | 8.684780e-5 | 65/80 |
| 타원형 | Adam | 0.1101141 | 미도달/미도달 |

이 조건에서는 Adam의 보편적 우월성을 주장할 수 없다. 같은 숫자의 lr도 Adam의 정규화 갱신과 GD에서 같은 이동량을 뜻하지 않는다. 경로는 초기점 포함 (201,2)다.

**Q. 잘못된 상태나 작은 수는 어떻게 다루는가?** lr·epsilon은 양의 유한 수, β₁·β₂는 [0,1)이다. 상태를 만든 뒤 입력 shape을 바꾸려면 reset이 필요하다. m,v는 복사본으로 반환한다. 유효하지 않은 후보가 생기면 마지막 수락 상태로 복원하며, 모멘트 계산의 underflow로 정밀도를 잃는 경우도 잘못된 큰 갱신 대신 거부한다. 정확히 표현 가능한 subnormal을 모두 일괄 거부하는 것은 아니다.

**검증·근거.** 첫 두 단계의 독립 기대값은 atol=1e-14, rtol=0으로 비교하고 0 gradient·경계 계수·상태 복원·실패 입력을 검사한다. python -m scripts.run_bonus_optimizers --no-cache. [Adam 구현](../src/adam.py), [독립 테스트](../tests/test_adam.py), [전체 JSON](../outputs/bonus_optimizer_metrics.json), [비교 보고서](../outputs/bonus_optimizer_report.md), [원형 PNG](../outputs/bonus_adam_circle.png), [타원형 PNG](../outputs/bonus_adam_ellipse.png), [상세 보너스 문서](BONUS.md).

<a id="qa-newton"></a>
## 보너스 Q&A — Newton의 Hessian·감쇠·계산 비용

**Q. Hessian은 무엇이며 Newton 방향은 어디서 나오는가?** Hessian H는 2차 편미분의 행렬로 국소 곡률을 나타낸다. θ 근처의 이차 근사

$$
f(\theta+d)\approx f(\theta)+g^Td+\tfrac12 d^THd
$$

를 d로 미분하면 g+Hd다. 이것을 0으로 놓으면 d=−H⁻¹g를 얻는다. 구현은 역행렬을 직접 만들지 않고 H·direction=g를 np.linalg.solve로 푼 뒤 θ_next=θ−lr·direction으로 갱신한다. θ,g,direction:(n,), H:(n,n)이다.

**Q. 왜 타원형 예제는 한 번에 끝나는가?** f=x²+10y²에서는 H=diag(2,20), g=(2x,20y)이므로 direction=(x,y)다. lr=1이면 θ_next=0이다. 이 예제에서는 함수 자체가 정확한 이차식이라 이차 근사가 정확하다.

| 비교 조건 | GD | Newton |
|---|---|---|
| 초기점·최대 업데이트 | (5,5), 200 | (5,5), 200 |
| lr | 0.05 | 1 |
| 종료 기준 | ‖gradient‖₂≤1e-8 | 동일 |
| 수락한 업데이트 수 | 197 | 1 |
| 최종 gradient norm(근사) | 9.677749e-9 | 0 |

이는 고정 200회 Adam 비교와 별도의 조기 종료 실험이다. 경로 길이는 수락 업데이트 수+1이다.

**Q. 아무 Hessian이나 넣어도 되는가?** 이 교육용 최소화 API는 유한 실수·대칭·양의 정부호 Hessian을 요구한다. 대칭성 허용치는 atol=rtol=1e-12다. 특이·부정부호·잘못된 shape를 자동 정규화나 작은 상수 추가로 숨기지 않는다. lr=1은 순수 Newton, 0<lr<1은 고정 damping이다.

**Q. 반복 수가 적으면 항상 더 빠르고 안전한가?** 일반 밀집 n차 Hessian 저장에는 O(n²), 선형계 직접 풀이에는 보통 O(n³) 비용이 든다. 반복 수는 실행 시간 비교가 아니다. Cholesky가 성공하고 해가 유한하다는 사실만으로 조건수가 큰 선형계의 전방오차가 작다는 보장은 없다. 이 구현에는 backtracking이나 특이 Hessian regularization이 없고 일반 비선형·비볼록 함수의 전역 수렴도 보장하지 않는다.

**재현·근거.** python -m scripts.run_bonus_optimizers --no-cache. [Newton 구현](../src/newton.py), [독립 테스트](../tests/test_newton.py), [Hessian·정책 설명](BONUS.md), [Newton PNG](../outputs/bonus_newton_ellipse.png), [측정 JSON](../outputs/bonus_optimizer_metrics.json), [비교 보고서](../outputs/bonus_optimizer_report.md).

<a id="qa-information"></a>
## 보너스 Q&A — 엔트로피·Cross-Entropy·KL

**Q. 세 양의 정의와 단위는 무엇인가?** 자연로그를 사용하므로 단위는 nats다. 같은 클래스 집합 위의 확률분포 p,q에 대해

$$
H(p)=-\sum_i p_i\ln p_i,\qquad
CE(p,q)=-\sum_i p_i\ln q_i,\qquad
KL(p\Vert q)=\sum_i p_i(\ln p_i-\ln q_i).
$$

H는 자기 분포의 평균 정보량, CE는 p가 실제 분포일 때 q에 기반한 로그손실의 평균, KL은 그 차이다. 구현은 p/q를 먼저 만들지 않고 로그 차이를 써서 극단적인 비율의 overflow를 피한다.

**Q. CE=H+KL은 왜 성립하는가?** pᵢ>0인 위치에서 qᵢ>0이고 결과가 유한할 때

$$
H(p)+KL(p\Vert q)
=-\sum_i p_i\ln p_i+\sum_i p_i\ln p_i-\sum_i p_i\ln q_i
=CE(p,q).
$$

따라서 고정 p에서 CE 최소화와 KL 최소화는 같다. one-hot p에서는 H=0이므로 실제 클래스의 −ln q와 연결된다. pᵢ>0,qᵢ=0인 경우에는 CE와 KL이 모두 +∞이지만 ∞−∞을 수치 오차 검증식으로 계산해서는 안 된다.

**Q. 독립 예제의 실제 값은 무엇인가?**

| 예제 | H(p) | CE(p,q) | KL(p∥q) |
|---|---:|---:|---:|
| p=q=[.25,.25,.25,.25] | 1.3862943611198906 | 1.3862943611198906 | 0 |
| p=q=[0,1] | 0 | 0 | 0 |
| p=[.25,.75], q=[.5,.5] | 0.5623351446188083 | 0.6931471805599453 | 0.13081203594113697 |
| p=[.5,.5], q=[.25,.75] | 0.6931471805599453 | 0.8369882167858358 | 0.14384103622589045 |
| p=[1,0], q=[0,1] | 0 | +∞ | +∞ |
| p=[1,0], q=[5e-324,1] | 0 | 744.4400719213812 | 744.4400719213812 |

일반 유한 예제의 독립 기대값 허용치는 atol=1e-14, rtol=0이고 극소 q 사례는 atol=2e-13이다. 저장된 유한 곡선의 CE−H−KL 최대 절대오차는 6.661338147750939e-16이다. Bernoulli 엔트로피와 p=0.3을 고정하고 q를 바꾼 CE/KL 곡선을 함께 제공한다.

**Q. 0확률과 JSON의 +inf는 어떻게 해석하는가?** pᵢ=0인 항은 0·log(0)=0의 연속적 확장으로 제외하고, pᵢ>0,qᵢ=0은 +∞를 유지한다. 확률을 epsilon으로 clipping하거나 자동 정규화하지 않는다. JSON에는 비표준 Infinity/NaN 대신 문자열 '+inf'와 경계 사례 설명을 저장한다. 따라서 파일의 문자열과 함수가 반환하는 수학적 양의 무한대를 구분해서 읽어야 한다.

**Q. shape·axis 계약은 무엇인가?** entropy(p,axis=-1), cross_entropy(p,q,axis=-1), kl_divergence(p,q,axis=-1)을 제공한다. p,q는 정확히 같은 shape이며 broadcasting하지 않는다. 각 확률은 [0,1]의 유한 실수, 지정 클래스 축의 합은 1(atol=1e-12, rtol=0), 클래스 축 길이는 1 이상이어야 한다. bool·complex·잘못된 축을 거부한다. 출력은 클래스 축을 제거한 shape이며 1차원 분포는 Python float다. 빈 배치 축은 가능하지만 클래스 축이 비어 있으면 안 된다.

**Q. KL은 거리인가, 항상 구현값이 0 이상인가?** 정확히 정규화된 분포의 수학적 KL은 비음수이지만 일반적으로 비대칭이므로 거리 metric이 아니다. 위 두 유리수 예제의 방향을 바꾸면 0.130812…와 0.143841…로 다르다. 합의 절대오차 1e-12를 허용하는 입력에서는 미세한 질량 차이 때문에 작은 음의 값이 나올 수 있으며 0으로 강제하지 않는다. 정확히 정규화한 독립 테스트 표본에서의 KL≥−1e-14 검증을 모든 허용 입력의 보장으로 확대하지 않는다.

**재현·근거.** python -m scripts.run_information_theory. [정보 이론 API](../src/information_theory.py), [독립 테스트](../tests/test_information_theory.py), [전체 예제·곡선 JSON](../outputs/information_theory.json), [보고서](../outputs/information_theory.md), [정보 이론 PNG](../outputs/information_theory.png), [정책 D17](DECISIONS.md#d17--정보-이론은-지지집합과-무한-손실을-보존한다).

<a id="qa-common"></a>
## 공통 용어 Q&A — float64·seed·허용오차·캐시·재현

**Q. float64는 정확한 실수 계산인가?** 아니다. 유한 비트의 부동소수점으로 0.1 같은 대부분의 십진 소수를 정확히 표현하지 못한다. 연산 순서·크기·취소·overflow·underflow의 영향을 받는다. 이 프로젝트는 기본 수치 검증 dtype을 float64로 고정하고 허용오차를 명시한다. 정확히 0이어야 하는 구조적 값과 근사 비교할 값을 구분한다.

**Q. atol과 rtol은 무엇인가?** 일반적인 근접 비교는 |관측−기대|≤atol+rtol·|기대| 형태다. atol은 0 근처에서도 의미 있는 절대 한계, rtol은 크기에 비례하는 상대 한계다. 두 개를 동시에 썼다면 하나만 썼다고 기록하지 않는다. 수학 기능의 실제 기준은 [ACCEPTANCE.md](ACCEPTANCE.md)이며 새 통합 보고서의 기존 측정값 재현 대조(atol=rtol=1e-12)는 그 기준을 대체하지 않는 별도의 검사다.

**Q. seed 42를 왜 한 번만 설정하는가?** [공통 set_seed](../src/utils/random.py)를 각 독립 실험·스크립트·난수 테스트의 진입점에서 한 번 호출한다. 동일 실행 순서와 입력에서 합성 데이터·실험을 재현하기 위한 것이다. import나 수학 함수 호출마다 seed를 초기화하면 호출자의 난수 흐름을 몰래 바꾸고 서로 다른 호출에서 같은 값이 반복될 수 있다. seed만 같아도 입력·dtype·버전·호출 순서가 다르면 모든 결과가 동일하다고 보장할 수 없다.

**Q. 캐시는 무엇을 저장하고 무엇을 절약하는가?** [로컬 CacheManager](../src/utils/cache_manager.py)는 비용이 큰 SVD·meshgrid·실험 경로의 계산 결과를 재사용한다. Redis 서버가 아니다. 키에는 전체 입력·shape/dtype·알고리즘 버전·매개변수·관련 seed 등을 반영하고 세션마다 달라지는 Python hash 대신 SHA-256을 쓴다. JSON/NumPy 형식, allow_pickle=False, 원자적 저장, 크기 제한·정리, 손상·버전 불일치 시 재계산, 반환값 복사 정책을 갖는다.

**Q. 캐시 결과가 맞다는 근거는 무엇인가?** 캐시를 켠 값과 끈 값을 대조한다. --no-cache 옵션이 있는 실험에서는 읽기·쓰기 모두 생략하고 다시 계산할 수 있다. 캐시는 계산 시간을 절약하며 탐색 토큰 절감은 [레지스트리](../.agent/registry.json)와 필요한 파일만 읽는 절차가 담당한다. 작은 엔트로피 합 같은 연산을 무조건 캐시하지 않는다.

**Q. 새 커널 실행은 왜 필요한가?** 기존 노트북 커널에는 이전 셀의 변수·실행 순서가 남아 오류가 숨을 수 있다. [노트북 실행 도구](../scripts/run_notebooks.py)는 실행 출력·횟수를 초기화하고 같은 프로젝트 Python의 새 커널에서 처음부터 검증한다. 기본 실행은 노트북 원본을 덮어쓰지 않으며 통합 보고서는 새 실행 결과를 임시 위치에 만들고 기존 측정값과 대조한다.

**Q. 출력 PNG와 입력 이미지 크기는 같은가?** 아니다. PNG는 그래프의 제목·축·범례·패널을 포함한 캔버스다. SVD의 64×64/128×128은 입력 배열 크기다. [메타데이터 보고서](../outputs/evaluation_evidence.md)는 실제 PNG width/height/DPI와 입력 크기, 재구성 렌더의 패널 크기를 구분한다. 기존 기본 PNG 10개와 보너스·보완 PNG는 원문 지정 outputs 폴더에 유지한다.

**Q. 실패 테스트는 실제 알고리즘이 실패했다는 뜻인가?** 유효하지 않은 입력을 넣어 기대한 예외를 확인하거나 일부 값을 고의로 바꾸어 실패 판정을 검증하는 것은 의도된 테스트다. [평가 보완 진단](../outputs/evaluation_evidence.json)의 injected 사례는 실제 알고리즘 출력과 별도로 표시한다. 실패 원인을 숨기려고 테스트를 삭제하거나 허용오차를 느슨하게 바꾸지 않는다.

**Q. 전체 검증 명령은 무엇인가?** 잠금 환경에서 아래 명령을 실행한다. 이 목록은 실행 방법이며 이 문서 작성만으로 새 실행 성공을 주장하지 않는다.

~~~text
python -m unittest discover -s tests -v
python -m pip check
python scripts/update_registry.py --check
python -m scripts.run_submission_reports --update-readme
~~~

python -O는 assert를 제거할 수 있어 검증 실행에 사용하지 않는다. 전체 검사와 별도로 생성된 표·그림의 제목·축·범례·비교 조건도 읽어 본다.

<a id="qa-evaluation"></a>
## 제출·검증 Q&A — 내부 검사 성공과 외부 점수

**Q. 코드가 있고 테스트를 통과하면 외부 평가에서도 반드시 PASS인가?** 그렇지 않다. 구현의 존재, 특정 입력의 실행 성공, 원격 제출 상태, 평가기가 실제 읽은 파일, 평가기의 판정은 각각 다른 사실이다. 내부 검증은 수식·입력·허용치·실행 결과로 증명하고 외부 평가는 받은 원문과 실행 시각을 따로 기록한다.

**Q. 저장소에 있는 파일을 평가가 “없다”고 하면 원인이 확정되는가?** 저장소에 존재한다는 사실만으로 평가 입력에 포함됐다고 단정할 수 없다. 반대로 인용하지 않았다는 사실만으로 수집하지 않았다고 단정할 수도 없다. 평가 대상 URL·브랜치/커밋 SHA·실행 ID·수집 파일 목록·추출 오류가 확인돼야 파일 전달 문제와 증거 선택·판정 문제를 구분할 수 있다. README 길이·파일 형식 필터·캐시·모델 판단 변동은 해당 자료가 없으면 가설이다.

**Q. 예전 PASS와 최근 FAIL 중 어느 쪽을 문서가 따르는가?** 외부 원문은 판정을 바꾸지 않고 시점별로 보존한다. 실물 증거를 대조하여 모순을 설명하되 자체 판단으로 받은 FAIL을 PASS로 수정하지 않는다. 최신 외부 상태와 분석은 [평가 결과 기록](../Naito_AI_eval_result.md), 필수 항목의 근거 위치는 [MISSION_EVIDENCE](../MISSION_EVIDENCE.md)에서 확인한다.

**Q. README·보고서·이 Q&A는 어떻게 역할이 다른가?** README는 실행과 핵심 실측 결과의 첫 진입점이다. outputs의 보고서 5개는 각 독립 실험의 입력·계산·판정을 읽는 자료다. 이 Q&A는 개념과 유도를 자세히 설명하고 원본 JSON·PNG·노트북·구현으로 안내한다. 원문·노트북·PNG를 대체하거나 삭제하는 문서가 아니다.

**Q. 구현 완료, PR 검토 대기, main 병합 완료는 같은 상태인가?** 다르다. 로컬 구현·검증을 마친 뒤 커밋·push·PR·CI·검토·병합 상태를 구분한다. 최종 제출 대상과 실제 이슈 상태는 [README](../README.md), [전체 계획](PROJECT_PLAN.md), [이슈 #31](https://github.com/ntt65/A2_1/issues/31)을 따른다. 이 Q&A가 만들어졌다는 이유로 새 외부 평가가 성공했거나 특정 점수를 얻었다고 주장하지 않는다.

## 설명 연습과 꼬리 질문

1. 행렬을 오른쪽에서 곱하면 왜 같은 변환이라고 할 수 없는가? det<0인데 면적은 왜 양수인가?
2. 잔차가 작은 고유쌍이 지배 고유쌍이라는 보장이 없는 반례는 무엇인가?
3. SVD의 rank를 높이면 복원과 저장량 중 무엇이 좋아지고 무엇이 나빠지는가?
4. x²의 중심차분과 sin의 중심차분에서 h 해석이 왜 다른가? 원점 gradient에 방향을 붙일 수 있는가?
5. BCE와 Sigmoid를 합성하면 왜 δ₂=p−y가 되는가? dL/dW₁의 두 번째 열은 왜 0인가?
6. 손계산 독립 상수, 네 자리 표시, 중심차분은 각각 무엇을 확인하는가?
7. 원형 GD의 lr=.5는 왜 발산의 반례인가? Momentum의 첫 진입 10회만으로 우월성을 말할 수 있는가?
8. 타원형 GD에서 lr=.05일 때 y가 어떻게 되는가? Momentum의 다른 관례를 혼합하면 무엇이 바뀌는가?
9. PDF가 1보다 높아도 되는 이유는 무엇인가? N(2,.5)의 표준편차는 얼마인가?
10. axis를 잘못 선택한 Softmax도 합이 1일 수 있는 이유는 무엇인가?
11. 고정 분산 Gaussian NLL과 MSE는 같은 수치인가, 같은 최적해인가? 분산이 θ에 의존하면 무엇을 버릴 수 없는가?
12. 불가능한 관측 사건의 CE를 유한하게 clipping하면 어떤 수학적 문제가 바뀌는가?
13. Adam의 같은 lr와 GD의 같은 lr는 같은 이동량인가? Newton 한 번과 GD 여러 번의 계산 시간은 무엇을 더 확인해야 하는가?
14. KL의 비대칭, 근사 정규화의 작은 음수, +∞ 경계는 각각 어떻게 설명하는가?
15. 내부 검사 성공·저장소 파일 존재·외부 평가 PASS를 각각 어떤 증거로 확인하는가?
