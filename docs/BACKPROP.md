# Step 5 역전파 유도와 검증

[실행된 노트북](../notebooks/backprop_derivation.ipynb)이 수식·손계산·NumPy·검증의 본문이다. 작업 [이슈 #8](https://github.com/ntt65/A2_1/issues/8).

## 재현

저장소 루트에서 Python3.11 가상환경을 사용한다. Linux/macOS는 실행 파일을 `.venv/bin/python`으로 바꾼다.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_notebooks notebooks/backprop_derivation.ipynb
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\python.exe scripts/update_registry.py --check
```

노트북은 현재 가상환경의 새 커널에서 처음부터 실행한다. 기본 명령은 저장된 노트북을 변경하지 않는다.
`--write`로 실행 출력까지 저장할 때에는 이후 `python scripts/update_registry.py --update`로 원시 파일 해시를 갱신한다.
`--output-dir`로 측정 JSON의 위치를 지정할 수 있다. 기본 결과는 `outputs/backprop_metrics.json`이다.

## 고정 예제와 수식

`x=[1,0]`, `W1=[[.1,.2],[.3,.4]]`, `b1=[0,0]`, `W2=[.5,.6]`, `b2=0`, `y=1`을 유지한다.
`z1=W1@x+b1`, `a1=sigmoid(z1)`, `z2=W2@a1+b2`, `p=sigmoid(z2)`다.
단일 표본 BCE `L=−y log(p)−(1−y)log(1−p)`는 같은 logit에서 `logaddexp(0,z2)−y*z2`로 안정적으로 계산한다.

| 단계 | 식 | shape |
|---|---|---|
| dL/dp | `(p−y)/(p(1−p))` | scalar() |
| δ2=dL/dz2 | `p−y` | scalar() |
| dL/dW2 | `δ2*a1` | (2,) |
| dL/da1 | `δ2*W2` | (2,) |
| δ1=dL/dz1 | `(δ2*W2)*a1*(1−a1)` | (2,) |
| dL/dW1 | `outer(δ1,x)` | (2,2) |
| dL/db1, dL/db2 | `δ1`, `δ2` | (2,),scalar() |

독립 손계산 상수는 유도한 식을 충분한 정밀도로 평가해 고정했다. NumPy 구현의 출력을 기준값으로 다시 저장하지 않는다.
중간값을 네 자리로 자르지 않으며 최종 반올림 표시만 비교한다. 반올림0의 음수 부호는 표시에서만 통일한다.
원문 참고값 중 z2는 `.6072`, 두 번째 은닉 기울기는 `−.0517`이 맞으며 [D12](DECISIONS.md)에 근거를 기록했다.

## 검증 범위

- 모든 순전파 중간값, 필수6개 역전파 항과 편향 기울기: 독립 상수 대비 `atol=5e-5,rtol=0`, 네 자리 표시 일치.
- W1두번째열은 x의 두번째성분0 때문에0임을 확인.
- W1의4개·b1의2개·W2의2개·b2의1개, 총9개 매개변수에 중심차분을 적용하고 해석 gradient와 최대절대오차≤1e-6 비교.
- 새 커널에서 모든 셀을 순서대로 실행하고 assertion 실패가 없는지 확인.

실측 오차와 환경·보완 사항은 [작업 기록](worklogs/step-5.md), 원시 값은 [측정 JSON](../outputs/backprop_metrics.json)에 남긴다.
