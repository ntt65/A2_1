# A2_1 — AI가 어떻게 학습하는지 수학으로 직접 풀어보기

NumPy로 선형대수·미적분·역전파·최적화·확률을 직접 구현하고 수식과 수치 검증을 연결하는 프로젝트입니다.
현재는 **Step 4 최적화 구현·검증 완료** 상태입니다. 사용자 지시에 따라 단계별 보고 후 Step5~7을 순차 진행합니다.

## 계획과 작업 규칙

- [전체 계획과 단계별 체크리스트](docs/PROJECT_PLAN.md) / [전체 추적 이슈 #2](https://github.com/ntt65/A2_1/issues/2)
- [에이전트 운영 규칙](AGENTS.md) / [재사용할 작업 시작 프롬프트](docs/AGENT_PROMPT.md)
- [수학 검증·허용오차 기준](docs/ACCEPTANCE.md) / [원문 충돌과 설계 결정](docs/DECISIONS.md)
- [공통 기반 API와 레지스트리 사용법](docs/UTILITIES.md)
- [선형대수 API·수식·그림·실측 결과](docs/LINEAR_ALGEBRA.md)
- [미적분 API·수식·그림·실측 결과](docs/CALCULUS.md)
- [미션 원문](A2_1_mission.md) / [원문 PDF](docs/A2_1_mission.pdf)

모든 작업은 **이슈 → 전용 브랜치 → 기초·상세·실행계획 → 구현 → 점검·보완 → push·PR → 이슈 기록** 순서로 진행합니다.
한 단계의 코드·검증·기록을 마친 뒤 다음 단계로 넘어갑니다. 단계별 이슈는 전체 계획에서 확인할 수 있습니다.

## 개발환경 설치

기준은 Python 3.11입니다. 프로젝트 내부 가상환경을 사용하며, 검증한 전체 의존성은 `requirements-lock.txt`로 고정합니다.
`requirements.txt`는 구현·시각화 직접 의존성, `requirements-dev.txt`는 노트북 실행 도구를 선언합니다.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/update_registry.py --check
```

Linux/macOS:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/update_registry.py --check
```

테스트는 최적화 옵션(`-O`) 없이 실행합니다. 가상환경 활성화는 필수가 아닙니다.
의존성 갱신은 별도 이슈에서 선언 파일과 잠금 파일을 함께 변경하고 재검증합니다.

## 검증 상태와 산출물

환경 준비의 실행 결과는 [Step 0 작업 기록](docs/worklogs/step-0.md), 공통 기반 검증은 [Step 1 작업 기록](docs/worklogs/step-1.md), 선형대수는 [Step 2 작업 기록](docs/worklogs/step-2.md)에 기록합니다.
미적분 검증은 [Step 3 작업 기록](docs/worklogs/step-3.md)에 기록합니다. 수학 검증 기준은 [ACCEPTANCE.md](docs/ACCEPTANCE.md)에 정의되어 있습니다. 공통 유틸·레지스트리·선형대수·미적분은 구현했으며 Step 5~6은 미구현·미검증입니다.
`src/utils/`와 `.agent/registry.json`을 제공하며, `notebooks/`는 해당 구현 단계에서 생성합니다.
결과 PNG는 `outputs/`, 재계산 가능한 캐시는 `.cache/`에 저장합니다.

| 영역 | 현재 상태 |
|---|---|
| 개발환경·운영 규칙 | 로컬 환경 검사 3개·pip check 통과, PR 검토 대기 |
| 공통 유틸·캐시·레지스트리 | Step 1 구현·보완 완료, 로컬 전체 검사 48개 통과 |
| 선형대수 | Step 2 구현·보완 완료, 로컬 전체 검사 87개 통과 |
| 미적분 | Step 3 구현·검증 완료, 로컬 전체 검사 105개 통과 |
| 최적화 | Step4 구현·검증 완료, 전체125개 테스트 통과 |
| 역전파·확률/손실 노트북 | Step 5~6 예정 |
| 전체 재현 및 최종 결과 요약 | Step 7 예정 |

기존 [초안 PR #1](https://github.com/ntt65/A2_1/pull/1)은 후속 단계에서 재사용 여부를 검토합니다.
후속 단계도 검증이 끝나면 이 README에 실측 오차·그림·실행 예시를 추가합니다.

## 선형대수 결과

```powershell
.\.venv\Scripts\python.exe -m scripts.run_linear_algebra
```

- 행렬식 절댓값 대비 면적비 최대 상대오차: **5.55e-16** (기준 1%).
- Power Iteration: **λ=4.618033988750**, 32회, 상대오차 **1.92e-16** (기준 5%), 상대잔차 **6.40e-11**.
- 완전 rank SVD 복원 상대오차: 64×64 **2.36e-15**, 128×128 **1.10e-15** (기준 1e-10).
- 원본 측정값: [JSON](outputs/linear_algebra_metrics.json). rank·저장량 해석 및 예외는 [상세 문서](docs/LINEAR_ALGEBRA.md)를 참고합니다.

![회전·스케일·전단과 면적비](outputs/linear_transforms.png)

![SVD 보충 실험: k=10,50,100](outputs/svd_128.png)

## 미적분 결과

```powershell
.\.venv\Scripts\python.exe -m scripts.run_calculus
```

- x² at3 중심차분 절대오차 **3.93e-11 ≤1e-4**.
- 8개 원 표본+원점의 numerical gradient 최대오차 **3.30e-11 ≤1e-8**.
- 비원점 gradient·접선의 정규화 내적 **0**. 원점은 방향 없는 영벡터로 표시합니다.
- x²의 절단오차는 이론상0이며, sin 보충 실험으로 h별 반올림·절단 오차를 비교합니다.
- [상세 수식·오차 그림](docs/CALCULUS.md) / [측정 JSON](outputs/calculus_metrics.json).

![등고선과 Gradient·접선](outputs/gradient_geometry.png)

## 최적화 결과

`python -m scripts.run_optimizer`로 PNG3개와 JSON을 재생성합니다. 원형 GD lr=.1,100회 후 norm **1.44e-9 ≤.1**입니다.
지정 조건에서는 GD의 최종 오차가 Momentum보다 작습니다. lr=.5는 즉시수렴,1은진동,1.1은발산으로 구분했습니다.
[수식·API·비교 그림](docs/OPTIMIZATION.md) / [작업 기록](docs/worklogs/step-4.md) / [이슈 #7](https://github.com/ntt65/A2_1/issues/7).
