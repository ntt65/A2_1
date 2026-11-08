# 역전파와 확률적 손실의 수식 유도

이 문서는 기존 노트북의 유도를 일반 Markdown으로도 읽을 수 있게 정리한 [이슈 #25](https://github.com/ntt65/A2_1/issues/25)의 제출 증거다. 수식과 고정 입력은 기존 구현을 유지한다. 기준은 [수용 기준](ACCEPTANCE.md), 참고 수치의 정정 근거는 [결정 기록 D12·D13](DECISIONS.md)이다.

## 원본 노트북과 재현 위치

노트북 셀은 표시 순서 대신 고정 셀 ID로 식별한다. 아래 파일을 열어 해당 셀 ID 또는 제목을 찾으면 유도와 실행 코드를 함께 확인할 수 있다.

| 내용 | 원본 파일 | 셀 ID |
|---|---|---|
| 고정 입력과 shape | [역전파 노트북](../notebooks/backprop_derivation.ipynb) | `backprop-fixed-shapes`, `backprop-fixed-input` |
| 순전파·독립 손계산 상수 | [역전파 노트북](../notebooks/backprop_derivation.ipynb) | `backprop-forward-derivation`, `backprop-hand-reference` |
| 필수 6개 미분·편향 | [역전파 노트북](../notebooks/backprop_derivation.ipynb) | `backprop-chain-rule`, `backprop-backward-numpy` |
| 네 자리 비교·9개 매개변수 중심차분 | [역전파 노트북](../notebooks/backprop_derivation.ipynb) | `backprop-compare`, `backprop-gradient-check` |
| Gaussian MLE와 MSE | [확률·손실 노트북](../notebooks/probability_loss.ipynb) | `gaussian-mle-derivation`, `gaussian-mle-check` |
| Bernoulli MLE와 BCE | [확률·손실 노트북](../notebooks/probability_loss.ipynb) | `bernoulli-mle-derivation`, `bce-functions`, `bce-check` |
| Categorical MLE와 Cross-Entropy | [확률·손실 노트북](../notebooks/probability_loss.ipynb) | `categorical-mle-derivation`, `categorical-check` |

프로젝트 Python 3.11 환경에서 `python -m scripts.run_notebooks`를 실행하면 두 노트북을 각각 새 커널에서 처음부터 검증한다. Windows의 직접 실행 명령은 `.\.venv\Scripts\python.exe -m scripts.run_notebooks`다. 기본 실행은 노트북 원본을 덮어쓰지 않는다. 측정값은 [역전파 JSON](../outputs/backprop_metrics.json)과 [확률·손실 JSON](../outputs/probability_metrics.json)에 있다.

## 1. 고정 신경망의 입력과 shape

수식에서 $x$와 은닉 벡터는 열벡터로 본다. NumPy의 1차원 배열 `(2,)`에는 별도의 행·열 축이 없으므로 아래 곱 방향을 명시적으로 사용한다.

| 기호 | 고정값 | NumPy shape |
|---|---|---|
| $x$ | $[1,0]^T$ | `(2,)` |
| $W_1$ | $[[0.1,0.2],[0.3,0.4]]$ | `(2,2)` |
| $b_1$ | $[0,0]^T$ | `(2,)` |
| $W_2$ | $[0.5,0.6]$ | `(2,)` |
| $b_2$ | $0$ | scalar `()` |
| $y$ | $1$ | scalar `()` |

첫 층은 `W1 @ x + b1`, 출력층은 `W2 @ a1 + b2`다. 계산은 `float64`이며 이 예제는 단일 표본이므로 배치 평균 계수를 추가하지 않는다.

## 2. 순전파를 직접 계산하기

Sigmoid와 그 미분은 다음과 같다.

$$
\sigma(z)=\frac{1}{1+e^{-z}},\qquad
\sigma'(z)=\sigma(z)(1-\sigma(z)).
$$

1. 은닉 logit: $z_1=W_1x+b_1=[0.1,0.3]^T$, shape `(2,)`.
2. 은닉 활성값: $a_1=\sigma(z_1)=[0.524979187478940,0.574442516811659]^T$, shape `(2,)`.
3. 출력 logit: $z_2=W_2a_1+b_2=0.5a_{1,1}+0.6a_{1,2}=0.607155103826465$, shape `()`.
4. 예측 확률: $\hat y=\sigma(z_2)=0.647291569994618$, shape `()`.
5. BCE: $L=-[y\ln\hat y+(1-y)\ln(1-\hat y)]=-\ln\hat y=0.434958436851461$, shape `()`.

모든 로그는 자연로그다. 이 입력에서는 $0<\hat y<1$이므로 확률 표현을 그대로 미분할 수 있다. 중간값은 반올림하지 않고 다음 단계에 전달한다.

## 3. 출력층에서 입력층까지 연쇄 법칙

아래 $\odot$는 원소별 곱이다. $W_2$는 수식에서 행벡터이지만 NumPy 표현과 결과 기울기는 `(2,)`이다.

### 3.1 예측값에 대한 손실 미분

$$
\frac{\partial L}{\partial\hat y}
=-\frac{y}{\hat y}+\frac{1-y}{1-\hat y}.
$$

이 예제에서 $y=1$이므로 $\partial L/\partial\hat y=-1/\hat y=-1.544898846756671$, shape `()`다.

### 3.2 출력 logit에 대한 미분

$$
\begin{aligned}
\delta_2=\frac{\partial L}{\partial z_2}
&=\frac{\partial L}{\partial\hat y}\frac{\partial\hat y}{\partial z_2}\\
&=\left(-\frac{y}{\hat y}+\frac{1-y}{1-\hat y}\right)\hat y(1-\hat y)\\
&=-y(1-\hat y)+(1-y)\hat y=\hat y-y.
\end{aligned}
$$

따라서 $\delta_2=-0.352708430005382$, shape `()`다.

### 3.3 출력 가중치에 대한 미분

$z_2=\sum_jW_{2,j}a_{1,j}+b_2$에서 $\partial z_2/\partial W_{2,j}=a_{1,j}$이므로

$$
\frac{\partial L}{\partial W_2}=\delta_2a_1
=[-0.185164585001198,-0.202610718232981],
$$

shape은 `(2,)`다.

### 3.4 은닉 활성값에 대한 미분

$\partial z_2/\partial a_{1,j}=W_{2,j}$이므로

$$
\frac{\partial L}{\partial a_1}=\delta_2W_2
=[-0.176354215002691,-0.211625058003229],
$$

shape은 `(2,)`다.

### 3.5 은닉 logit에 대한 미분

은닉 Sigmoid는 각 원소에 독립적으로 작용한다. 따라서 그 Jacobian은 $a_1\odot(1-a_1)$을 대각 원소로 갖는다.

$$
\delta_1=\frac{\partial L}{\partial z_1}
=(\delta_2W_2)\odot a_1\odot(1-a_1)
=[-0.043978515808697,-0.051733504390926],
$$

shape은 `(2,)`다.

### 3.6 첫 층 가중치에 대한 미분

$z_{1,i}=\sum_jW_{1,ij}x_j+b_{1,i}$이므로

$$
\frac{\partial L}{\partial W_{1,ij}}=\delta_{1,i}x_j,\qquad
\frac{\partial L}{\partial W_1}=\delta_1x^T
=\begin{bmatrix}
-0.043978515808697&0\\
-0.051733504390926&0
\end{bmatrix}.
$$

NumPy의 대응식은 `np.outer(delta1, x)`이며 shape은 `(2,2)`다. 입력의 두 번째 성분이 0이므로 기울기 행렬의 두 번째 열도 0이다.

### 3.7 편향의 미분

편향은 덧셈으로 들어가므로 해당 logit에 대한 미분이 그대로 전달된다.

$$
\frac{\partial L}{\partial b_2}=\delta_2\quad\text{shape }(),\qquad
\frac{\partial L}{\partial b_1}=\delta_1\quad\text{shape }(2,).
$$

## 4. 독립 손계산 기준과 네 자리 표시

다음 기준은 노트북의 `backprop-hand-reference`에 고정한 독립 스칼라식 계산값이다. 문서의 짧은 표시는 계산에 재사용하지 않는다.

| 체크포인트 | 독립 기준값(일부 자리 표시) | 소수점 네 자리 표시 |
|---|---|---|
| $z_1$ | `[0.1, 0.3]` | `[0.1000, 0.3000]` |
| $a_1$ | `[0.524979187479, 0.574442516812]` | `[0.5250, 0.5744]` |
| $z_2$ | `0.607155103826` | `0.6072` |
| $\hat y$ | `0.647291569995` | `0.6473` |
| $L$ | `0.434958436851` | `0.4350` |
| $\partial L/\partial\hat y$ | `-1.544898846757` | `-1.5449` |
| $\delta_2$ | `-0.352708430005` | `-0.3527` |
| $\partial L/\partial W_2$ | `[-0.185164585001, -0.202610718233]` | `[-0.1852, -0.2026]` |
| $\partial L/\partial a_1$ | `[-0.176354215003, -0.211625058003]` | `[-0.1764, -0.2116]` |
| $\delta_1$ | `[-0.043978515809, -0.051733504391]` | `[-0.0440, -0.0517]` |
| $\partial L/\partial W_1$ | `[[-0.043978515809, 0], [-0.051733504391, 0]]` | `[[-0.0440, 0.0000], [-0.0517, 0.0000]]` |
| $\partial L/\partial b_1$ | `[-0.043978515809, -0.051733504391]` | `[-0.0440, -0.0517]` |
| $\partial L/\partial b_2$ | `-0.352708430005` | `-0.3527` |

원문 참고 예시의 `z2=0.6071` 및 두 번째 은닉 기울기 `-0.0513`은 고정 입력의 수식에서 재계산하면 각각 `0.6072`, `-0.0517`이다. 원문 자체는 보존하고 [D12](DECISIONS.md#d12--역전파는-고정-예제를-유지하고-참고-수치를-재계산한다)에 정정 근거를 남겼다.

검증은 13개 체크포인트의 shape·유한성·원소별 절대오차 `<=5e-5`와 네 자리 표시를 확인한다(`rtol=0`). 실제 gradient의 `-0.0`을 수정하지 않으며, 표시에만 반올림한 0의 부호를 통일한다. 별도로 `(9,)` 매개변수 벡터의 각 성분에 대해 순전파 손실만 사용하는 중심차분

$$
\frac{L(\theta+he_i)-L(\theta-he_i)}{2h},\qquad h=10^{-5}
$$

와 역전파 gradient를 비교한다(`atol=1e-6`, `rtol=0`). [저장된 측정 JSON](../outputs/backprop_metrics.json)의 손계산 최대 절대오차는 `2.220446049250313e-16`, 수치 gradient 최대오차는 `7.842670957103337e-12`다. 이는 고정 예제의 검증 결과이며 모든 입력의 오차 보장이 아니다.

## 5. Sigmoid와 BCE의 안정적인 logit 표현

$p=\sigma(z)=e^z/(1+e^z)$이면 $\ln p=z-\ln(1+e^z)$, $\ln(1-p)=-\ln(1+e^z)$다. 따라서

$$
\begin{aligned}
\ell(z,y)&=-[y\ln p+(1-y)\ln(1-p)]\\
&=\ln(1+e^z)-yz
=\operatorname{logaddexp}(0,z)-yz,\\
\frac{\partial\ell}{\partial z}&=\sigma(z)-y.
\end{aligned}
$$

`logaddexp`는 큰 양의 $z$에서 `exp(z)`의 overflow를 피한다. 다만 $y=1$, 큰 양의 $z$에서 마지막 뺄셈은 작은 양의 손실을 0으로 상쇄할 수 있다. **이진 레이블 $y\in\{0,1\}$**에 한해서 수학적으로 같은

$$\ell(z,y)=\operatorname{logaddexp}(0,(1-2y)z)$$

를 사용하면 이 뺄셈을 피할 수 있다. 확률·손실 노트북은 이 표현을 사용한다. 이 식을 임의의 soft target에 그대로 적용하지 않는다. float64의 극소값 underflow나 Sigmoid 포화까지 제거되는 것은 아니다. 손실과 미분의 검증은 표시용 clipping 없이 같은 수학적 함수를 기준으로 한다.

## 6. Gaussian likelihood에서 MSE까지

관측은 $y_i=f_\theta(x_i)+\epsilon_i$, $i=1,\ldots,n$이며 $n\ge1$이라고 하자. 오차가 서로 독립이고 $\epsilon_i\sim N(0,\sigma^2)$이며, 모든 표본에 같은 **알려진 양의 상수 분산** $\sigma^2$를 사용한다고 가정한다.

$$
p(y_i\mid x_i,\theta)=\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left[-\frac{(y_i-f_\theta(x_i))^2}{2\sigma^2}\right].
$$

독립성으로 likelihood는 곱이고 자연로그는 이를 합으로 바꾼다.

$$
\begin{aligned}
\mathcal L(\theta)&=\prod_{i=1}^{n}p(y_i\mid x_i,\theta),\\
\ln\mathcal L(\theta)&=-\frac n2\ln(2\pi\sigma^2)
-\frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i-f_\theta(x_i))^2,\\
\operatorname{NLL}(\theta)&=-\ln\mathcal L(\theta)
=\frac n2\ln(2\pi\sigma^2)+\frac{\operatorname{SSE}(\theta)}{2\sigma^2}\\
&=\frac n2\ln(2\pi\sigma^2)+\frac{n}{2\sigma^2}\operatorname{MSE}(\theta).
\end{aligned}
$$

여기서 $\operatorname{SSE}=\sum_i(y_i-f_\theta(x_i))^2$, $\operatorname{MSE}=\operatorname{SSE}/n$이다. 로그는 단조증가하므로 likelihood 최대화는 NLL 최소화와 같다. NLL의 첫 항은 $\theta$와 무관하고 SSE/MSE의 계수는 양수이므로 같은 매개변수 집합에서 최적해가 존재할 때

$$
\operatorname*{argmax}_\theta\mathcal L(\theta)
=\operatorname*{argmin}_\theta\operatorname{NLL}(\theta)
=\operatorname*{argmin}_\theta\operatorname{SSE}(\theta)
=\operatorname*{argmin}_\theta\operatorname{MSE}(\theta).
$$

평균 NLL은 $\operatorname{NLL}/n=\tfrac12\ln(2\pi\sigma^2)+\operatorname{MSE}/(2\sigma^2)$다. **최적해가 같다는 것이 NLL과 MSE의 수치가 같다는 뜻은 아니다.** 분산이 입력이나 $\theta$에 따라 바뀌면 로그 분산 항과 오차별 가중치를 버릴 수 없다. 분산 자체를 함께 추정하는 문제도 위의 고정분산 유도와 구분한다. 연속 밀도는 1보다 클 수 있어 Gaussian NLL은 음수일 수 있다.

## 7. Bernoulli likelihood에서 BCE까지

독립 표본의 레이블은 $y_i\in\{0,1\}$, 성공확률은 $p_i$다. 한 표본의 PMF는 $p_i^{y_i}(1-p_i)^{1-y_i}$이므로

$$
\begin{aligned}
\mathcal L&=\prod_{i=1}^{n}p_i^{y_i}(1-p_i)^{1-y_i},\\
\operatorname{NLL}=L_{\mathrm{sum}}
&=-\sum_{i=1}^{n}\left[y_i\ln p_i+(1-y_i)\ln(1-p_i)\right],\\
L_{\mathrm{mean}}&=L_{\mathrm{sum}}/n.
\end{aligned}
$$

즉 Bernoulli의 NLL이 BCE의 합이다. $p_i=\sigma(z_i)$일 때 합 손실의 각 logit 미분은 $p_i-y_i$, 평균 손실의 미분은 $(p_i-y_i)/n$이다. 고정 역전파 예제는 $n=1$이어서 추가 평균 계수가 없다.

확률 벡터와 레이블 벡터의 shape은 `(n,)`, 표본별 손실은 `(n,)`, 합·평균 손실은 scalar `()`다. 확률이 범위 밖이거나 비유한 값이면 유효한 분포 입력이 아니다.

경계에서 $p=0,y=0$ 또는 $p=1,y=1$의 손실은 0이다. $p=0,y=1$ 또는 $p=1,y=0$은 관측 사건에 확률 0을 부여하므로 likelihood 0, NLL $+\infty$다. 0 계수의 로그 항은 0으로 정의하여 제외한다. `0*log(0)`의 NaN을 만든 뒤 숨기거나 clipping으로 불가능한 사건을 유한 손실로 바꾸지 않는다.

## 8. Categorical likelihood에서 Cross-Entropy까지

표본 $i$의 실제 클래스가 $c_i$이고 클래스 수가 $K$일 때 $y_{ij}=1$ if $j=c_i$, 그렇지 않으면 0인 one-hot 레이블을 쓴다. 각 행에서 $p_{ij}\ge0$, $\sum_jp_{ij}=1$, $\sum_jy_{ij}=1$이다.

$$
\begin{aligned}
\mathcal L&=\prod_{i=1}^{n}\prod_{j=1}^{K}p_{ij}^{y_{ij}},\\
L_{\mathrm{sum}}=-\ln\mathcal L
&=-\sum_{i=1}^{n}\sum_{j=1}^{K}y_{ij}\ln p_{ij}
=-\sum_{i=1}^{n}\ln p_{i,c_i},\\
L_{\mathrm{mean}}&=L_{\mathrm{sum}}/n.
\end{aligned}
$$

이 식이 one-hot categorical Cross-Entropy다. logits·확률·one-hot 레이블의 shape은 `(n,K)`, 표본별 손실은 `(n,)`, 합·평균은 `()`다. Bernoulli BCE의 두 결과에 대한 유도와 다중 클래스 categorical 유도를 구분한다.

Softmax 확률 $p_j=e^{z_j}/\sum_ke^{z_k}$를 대입하면 한 표본의 손실은 $\ell(z,c)=\ln\sum_je^{z_j}-z_c$다. $m=\max_jz_j$, $s_j=z_j-m$으로 이동하면

$$
\ell(z,c)=\ln\sum_je^{s_j}-s_c,\qquad
\frac{\partial\ell}{\partial z_j}=p_j-y_j.
$$

최댓값을 뺀 뒤 지수화하면 overflow를 피하고 큰 공통 offset의 마지막 뺄셈에 따른 취소 오차도 줄인다. 평균 손실이면 logit gradient 전체를 $n$으로 나눈다.

실제 클래스 확률 $p_{i,c_i}=0$이면 손실은 $+\infty$다. target가 0인 클래스의 항은 0으로 제외하므로 그 클래스의 예측확률이 0이어도 해당 항에서 NaN을 만들지 않는다. 임의 clipping은 이 likelihood와 다른 목적함수를 만들 수 있다. 저장 JSON에는 비표준 무한 수를 쓰지 않으며 기존 노트북은 경계 검증의 통과 여부를 별도로 기록한다.

## 9. 검증 범위

이 문서는 기존 수식 유도와 실행 증거의 접근 경로를 보완한다. 추가 알고리즘이나 보너스 정보 이론 구현을 이 문서만으로 완료했다고 보지 않는다. 원본 노트북의 assertion, 독립 손계산 기준, 중심차분과 분포·손실 검증이 수치 증거이며, 문서에 수식을 적었다는 사실은 새 실행 결과를 뜻하지 않는다.
