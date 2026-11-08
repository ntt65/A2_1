# A2_1 발표자료

이 자료는 “AI가 어떻게 학습하는지 수학으로 직접 풀어보기” 미션의 구현 발표와 평가 질의응답을 나누어 준비한 것이다. 수식, 고정 입력, 실행한 결과와 그 의미를 슬라이드 본문에서 설명하고, 발표자 노트에는 계산 과정·예상 꼬리 질문·재현 명령·근거 파일을 덧붙인다. 저장소의 상세 보고서는 [최상위 README](../../README.md), 질문별 긴 답변은 [평가 Q&A 사전](../EVALUATION_QA.md)에서 읽을 수 있다.

## 두 자료의 역할

| 자료 | 발표할 내용 | 편집본 | 열람본 |
|---|---|---|---|
| 미션 설명·구현 | 선형대수 → 미적분 → 역전파 → 최적화 → 확률·손실의 연결, 실제 결과, 보너스 구현 | [PowerPoint](a2_1_mission_implementation.pptx) | [PDF](a2_1_mission_implementation.pdf) |
| 평가 Q&A | 기본 15항목의 질문·답·유도·실측, 보너스와 반례, iMac 시연 | [PowerPoint](a2_1_evaluation_qa.pptx) | [PDF](a2_1_evaluation_qa.pdf) |

미션 발표에서는 입력 벡터에 선형변환을 적용하는 일부터 시작해, 미분으로 변화 방향을 계산하고, 연쇄 법칙으로 신경망의 가중치 기울기를 얻은 다음, 최적화로 손실을 줄이는 순서로 설명한다. 마지막에는 손실이 임의로 정한 식이 아니라 Gaussian·Bernoulli·categorical 확률모형의 음의 로그우도와 연결됨을 유도한다. Adam·Newton·정보 이론은 기본 미션 뒤의 보너스로 구분한다.

평가 Q&A에서는 “구현되어 있다”는 답변에 그치지 않는다. 예를 들어 SVD 질문에는 실제 rank 10·50·100의 오차와 저장비율을, 역전파 질문에는 고정 입력의 순전파 값과 여섯 필수 미분을, 최적화 질문에는 학습률·업데이트 수·최종 반경을 제시한다. 각 질문은 [사전의 해당 항목](../EVALUATION_QA.md#qa-01)과 연결되어 있다. 아래 슬라이드 매핑에서 바로 찾아갈 수 있다.

## 발표에서 설명할 핵심 결과

발표는 “입력에 행렬을 곱하면 어떤 변화가 생기는가?”에서 시작한다. 이어 “손실을 줄이는 방향을 어떻게 구하는가?”를 미분과 역전파로 설명하고, “그 방향으로 얼마나 움직일 것인가?”를 GD·Momentum 실험으로 연결한다. 마지막에는 “줄이려는 손실을 왜 이 식으로 정의하는가?”에 확률모형과 MLE 유도로 답한다. 보너스는 이 흐름을 이해한 뒤 다른 갱신 방법과 정보 이론으로 확장한다.

각 장에서는 식을 그대로 읽기보다 먼저 기호와 입력을 짚고, 실제 수치 한 가지를 설명한 다음 그 결과가 성립하는 조건을 말한다. 예를 들어 Momentum 비교는 “이 설정에서는 먼저 반경에 진입하지만 진동 때문에 유지 시점은 늦다”라고 설명하고 최초 진입·유지 열을 나란히 보여 준다. Q&A에서는 질문에 대한 결론을 먼저 말하고 식·실측·반례 순서로 보충한다. 그림을 가리킨 설명만으로 끝내지 않고 본문의 표와 숫자로 확인할 수 있게 했다.

시연에 앞서 가상환경과 노트북 커널을 확인하고, 시연 중에는 새 실행의 출력과 슬라이드의 반올림된 수치를 대조한다. 플랫폼에 따라 미분의 마지막 자리가 달라지면 오차·허용치로 설명한다. 검증 실패를 숨기거나 이전 실행 화면을 새 실행처럼 소개하지 않는다. 보너스의 Adam 미도달 사례와 외부 40% 결과도 해당 조건과 시점을 붙여 그대로 설명한다.

- **선형대수:** 면적비는 `abs(det(A))`와 비교한다. Power Iteration의 고유값은 약 `4.61803398875`이며 NumPy 기준과 상대오차 약 `1.923e-16`로 일치한다. 64×64 입력에는 rank 100을 적용할 수 없으므로 128×128 보충 실험을 별도로 제시한다. rank 100의 복원 오차는 약 `0.0121830`이지만 저장비율 `1.56860`으로 원본보다 크다는 한계도 설명한다.
- **미적분·역전파:** 고정 입력의 중심차분 오차와 허용치를 함께 말한다. `h` 탐색의 일부 실패 표본을 숨기지 않는다. 역전파는 `x=[1,0]`인 예제의 출력 약 `0.64729157`, BCE 약 `0.43495844`에서 시작해 미분과 shape를 차례로 설명한다. 손계산 13개 체크포인트는 소수 네 자리에서 일치하며, 별도의 9개 매개변수 중심차분 검사 최대오차는 약 `7.843e-12`다.
- **최적화:** 원형 함수의 GD는 `lr=0.1`, 100회 후 반경 약 `1.4404e-9`다. `lr=0.5`는 한 번에 수렴하고, `1.0`은 진동, `1.1`은 발산하므로 “0.5 이상이면 전부 발산”으로 해석하지 않는다. Momentum은 이 설정에서 최초 반경 진입이 빠르더라도 이후 진동으로 안정적으로 머무는 시점이 늦을 수 있다.
- **확률·손실:** 두 정규 PDF와 두 Bernoulli PMF의 조건 및 수치 검사를 제시한다. Softmax의 행 합 최대오차는 약 `1.110e-16`이다. 고정 공통 분산의 Gaussian NLL과 MSE는 같은 최적해를 가지며 값 자체가 같다는 뜻은 아니다. BCE와 categorical CE의 유도는 각각 별도로 설명한다.
- **보너스:** Adam이 항상 가장 빠르다고 결론 내리지 않는다. 제시된 타원 실험에서는 200회 후 반경 약 `0.110114`로 반경 0.1에 도달하지 못했다. Newton의 한 단계 수렴은 해당 양의 정부호 이차함수의 결과다. 정보 이론은 자연로그를 사용하며 `CE=H+KL`, KL의 비대칭성, 경계의 무한대를 함께 다룬다.

위 숫자는 기존 `outputs/*metrics.json` 등의 실제 산출물에서 가져온 표시용 반올림이다. 계산은 반올림한 화면 값으로 다시 수행하지 않는다. 중심차분의 끝자리는 운영체제·NumPy 환경에 따라 달라질 수 있으며, 재현 시에는 명시한 허용오차로 확인한다.

## iMac의 VS Code에서 시연하기

1. `main`을 최신 상태로 받고 저장소 폴더를 VS Code로 연다. [README의 macOS 설치 절차](../../README.md#macosimac--linux)로 Python 3.11 가상환경과 고정 의존성을 준비한다.
2. Python 인터프리터와 노트북 커널을 이 저장소의 `.venv/bin/python`으로 선택한다. 다른 프로젝트의 커널을 사용하지 않는다.
3. 터미널에서 아래 명령을 실행한다. 새 실행을 통해 보고서가 재현되는지 확인하는 마지막 명령은 노트북도 실행하므로 앞의 명령보다 시간이 더 걸릴 수 있다.

```sh
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/update_registry.py --check
.venv/bin/python -m scripts.run_submission_reports --output-dir .cache/imac-demo
```

4. 슬라이드의 수식·측정표를 설명한 뒤 해당 노트북의 실행 셀이나 원본 출력 파일을 함께 보여준다. 역전파는 `notebooks/backprop_derivation.ipynb`, 확률·손실은 `notebooks/probability_loss.ipynb`를 사용한다.
5. 질의응답은 Q&A 발표자료의 질문 번호로 이동한다. 표에 없는 전체 정밀도 값이나 전체 실험 경로가 필요하면 발표자 노트의 출처를 따라 원본 JSON을 연다.

PPTX는 텍스트와 표를 편집할 수 있는 발표용 파일이고, PDF는 완성된 슬라이드를 같은 순서로 읽는 열람용 파일이다. PDF에는 PowerPoint 발표자 노트가 포함되지 않는다. 본문 텍스트와 표는 PDF에서도 검색·복사할 수 있다. 상세 답변은 PPTX 노트 및 Q&A 사전에 보존한다. PPTX의 한글 글꼴은 `Malgun Gothic`이며 iMac에서 설치된 글꼴에 따라 대체될 수 있으므로, 배치를 그대로 보여줄 때는 PDF를 사용할 수 있다. 사용자 iMac의 PowerPoint·Keynote에서 직접 실행한 것으로 주장하지 않는다.

## 결과의 범위

구현의 자체 검증과 외부 평가는 구분한다. 선행 main `2edda38`의 CI에서 Windows·Linux·macOS 각각 210개 검사가 통과했다. 현재 발표자료의 수치는 검증된 기존 실험 산출물에 근거한다. 외부 Naito 사전평가의 마지막 결과는 **40%(6/15)**이며, 소진된 사전평가를 새로 실행하지 않았다. 상세 경위와 제출 증거의 보강은 [평가 기록](../../Naito_AI_eval_result.md)에 있다.

## 원고와 재생성

두 원고는 `scripts/presentation/mission_content.mjs`, `scripts/presentation/qa_content.mjs`, 공통 작성 도구는 `scripts/presentation/build_presentations.mjs`다. 표와 수치 문장은 원본 JSON을 읽어 생성하며 기존 실험 파일을 덮어쓰지 않는다. PPTX와 PDF를 만드는 도구의 의존성은 프로젝트의 수학 구현용 `requirements-lock.txt`와 별개다.

작성 도구의 실행에는 번들 Node·모듈 경로·발표 작성 도구 경로·번들 Python을 명시한다. 실험 시연용 Python 환경에 이 작성 도구를 설치할 필요는 없다. 아래 명령은 저장소 루트에서 실행하며, 결과 폴더와 검토 폴더는 서로 다른 새 경로여야 한다.


```sh
node scripts/presentation/build_presentations.mjs \
  --modules "<번들 Node 모듈 경로>" \
  --skill-dir "<발표 작성 도구 경로>" \
  --python "<번들 Python 실행 파일>" \
  --output-dir .cache/presentation-revision-output \
  --build-dir .cache/presentation-revision-build \
  --skip-pdf
```

PPTX의 전체 슬라이드를 검토한 다음 같은 경로로 다시 실행하되 `--skip-pdf`를 `--pdf-only --pdf-backend powerpoint`로 바꾸면 최종 PPTX와 저장된 해시를 대조한 뒤 설치된 Windows PowerPoint로 PDF만 생성한다. 기존 완성 파일을 덮어쓰지 않으므로 재수정 시에는 새 revision 폴더를 사용한다. 실제 작성·검증 환경은 Windows의 번들 런타임·설치된 맑은 고딕·PowerPoint 16.0이며, 사용자 iMac에서 이 작성 도구를 검증한 것은 아니다. iMac 평가는 위 Python 시연 절차와 완성된 발표파일을 사용한다.

## 미션 설명·구현 — 30장

| 슬라이드 | 제목 | 원고 ID / 질문 |
|---:|---|---|
| 1 | AI 학습을 수학으로 구현하기 | `m01` |
| 2 | 구현 범위와 재현 조건 | `m02` |
| 3 | 행렬이 공간을 바꾸는 방식 | `m03` |
| 4 | 행렬식과 면적 변화율 | `m04` |
| 5 | Power Iteration의 반복과 검증 | `m05` |
| 6 | SVD로 이미지 성분 줄이기 | `m06` |
| 7 | rank와 복원 품질의 관계 | `m07` |
| 8 | 중심차분과 간격 h | `m08` |
| 9 | Gradient와 등고선의 수직 관계 | `m09` |
| 10 | 고정 신경망과 텐서 shape | `m10` |
| 11 | 순전파 손계산 | `m11` |
| 12 | 출력층의 연쇄 법칙 | `m12` |
| 13 | 은닉층과 첫 가중치의 미분 | `m13` |
| 14 | 순전파 손계산과 NumPy 비교 | `m14-forward` |
| 15 | 역전파 비교와 독립 중심차분 | `m14-backward` |
| 16 | GD와 Momentum의 원형 함수 비교 | `m15` |
| 17 | 학습률과 수렴의 안정 범위 | `m16` |
| 18 | 타원형 곡률과 Momentum | `m17` |
| 19 | 초기점이 달라지면 수렴 지표도 달라진다 | `m18` |
| 20 | 연속 밀도와 이산 확률 질량 | `m19` |
| 21 | 큰 logit에서도 안정적인 Softmax | `m20` |
| 22 | Gaussian MLE에서 MSE까지 | `m21` |
| 23 | Bernoulli MLE와 안정적인 BCE | `m22` |
| 24 | Categorical MLE와 Cross-Entropy | `m23` |
| 25 | Adam의 모멘트와 편향 보정 | `m24` |
| 26 | Adam의 원형 함수 실험 | `m25` |
| 27 | Adam의 타원형 함수 실험 | `m26` |
| 28 | Newton의 이차 근사와 선형계 | `m27` |
| 29 | 엔트로피·Cross-Entropy·KL | `m28` |
| 30 | iMac에서 재현하고 결과 확인하기 | `m29` |

## 평가 Q&A — 31장

| 슬라이드 | 제목 | 원고 ID / 질문 |
|---:|---|---|
| 1 | A2_1 평가 Q&A | `common-cover` |
| 2 | shape와 오차는 무엇을 뜻하나요? | `common-notation` |
| 3 | #1 행렬은 원을 어떻게 바꾸나요? | [qa-01](../EVALUATION_QA.md#qa-01) |
| 4 | #2 행렬식과 면적비는 일치하나요? | [qa-02](../EVALUATION_QA.md#qa-02) |
| 5 | #3 Power Iteration은 무엇을 찾나요? | [qa-03](../EVALUATION_QA.md#qa-03) |
| 6 | #4 SVD는 무엇을 남기나요? | [qa-04](../EVALUATION_QA.md#qa-04) |
| 7 | #4 rank를 높이면 압축도 좋아지나요? | [qa-04](../EVALUATION_QA.md#qa-04) |
| 8 | #5 중심차분은 어떻게 미분하나요? | [qa-05](../EVALUATION_QA.md#qa-05) |
| 9 | #5 h가 작을수록 정확한가요? | [qa-05](../EVALUATION_QA.md#qa-05) |
| 10 | #6 Gradient는 왜 등고선에 수직인가요? | [qa-06](../EVALUATION_QA.md#qa-06) |
| 11 | #7 고정 신경망의 순전파는 어떻게 되나요? | [qa-07](../EVALUATION_QA.md#qa-07) |
| 12 | #7 출력층 미분은 왜 p−y가 되나요? | [qa-07](../EVALUATION_QA.md#qa-07) |
| 13 | #7 은닉층과 첫 가중치는 어떻게 미분하나요? | [qa-07](../EVALUATION_QA.md#qa-07) |
| 14 | #8 순전파와 손계산은 일치하나요? | [qa-08](../EVALUATION_QA.md#qa-08) |
| 15 | #8 역전파 gradient도 독립적으로 확인했나요? | [qa-08](../EVALUATION_QA.md#qa-08) |
| 16 | #9 GD는 반경 0.1에 도달하나요? | [qa-09](../EVALUATION_QA.md#qa-09) |
| 17 | #10 lr=0.5는 정말 발산하나요? | [qa-10](../EVALUATION_QA.md#qa-10) |
| 18 | #11 Momentum은 항상 더 빠른가요? | [qa-11](../EVALUATION_QA.md#qa-11) |
| 19 | #12 곡률 차이는 경로를 어떻게 바꾸나요? | [qa-12](../EVALUATION_QA.md#qa-12) |
| 20 | #13 N(2,0.5)의 0.5는 무엇인가요? | [qa-13](../EVALUATION_QA.md#qa-13) |
| 21 | #13 PMF와 PDF는 무엇이 다른가요? | [qa-13](../EVALUATION_QA.md#qa-13) |
| 22 | #14 Softmax는 어느 축의 합을 1로 만드나요? | [qa-14](../EVALUATION_QA.md#qa-14) |
| 23 | #15 Gaussian MLE와 MSE는 왜 연결되나요? | [qa-15](../EVALUATION_QA.md#qa-15) |
| 24 | #15 Bernoulli의 음의 로그는 왜 BCE인가요? | [qa-15](../EVALUATION_QA.md#qa-15) |
| 25 | #15 다중 클래스 MLE는 어떻게 CE가 되나요? | [qa-15](../EVALUATION_QA.md#qa-15) |
| 26 | 보너스 Adam은 무엇을 보정하나요? | `bonus-adam` |
| 27 | 보너스 Newton은 왜 한 번에 도달하나요? | `bonus-newton` |
| 28 | 보너스 H·CE·KL은 어떻게 연결되나요? | `bonus-information` |
| 29 | 같은 결과는 어떻게 재현하나요? | `common-reproduction` |
| 30 | 내부 검증과 외부 점수는 같은 사실인가요? | `common-evaluation` |
| 31 | 반례에도 같은 답을 할 수 있나요? | `common-followup` |

검토·원본 보존·커밋·PR·CI 기록은 [이슈 #33 작업 기록](../worklogs/issue-33.md)을 따른다.
