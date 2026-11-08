# A2_1 — AI가 어떻게 학습하는지 수학으로 직접 풀어보기

NumPy로 선형대수·미적분·역전파·최적화·확률을 직접 구현하고 수식과 수치 검증을 연결하는 프로젝트입니다.
현재는 **Step 1: 공통 수학·캐시·레지스트리** 단계입니다. 미션별 수학 알고리즘과 노트북은 이후 작업에서 추가합니다.

## 계획과 작업 규칙

- [전체 계획과 단계별 체크리스트](docs/PROJECT_PLAN.md) / [전체 추적 이슈 #2](https://github.com/ntt65/A2_1/issues/2)
- [에이전트 운영 규칙](AGENTS.md) / [재사용할 작업 시작 프롬프트](docs/AGENT_PROMPT.md)
- [수학 검증·허용오차 기준](docs/ACCEPTANCE.md) / [원문 충돌과 설계 결정](docs/DECISIONS.md)
- [공통 기반 API와 레지스트리 사용법](docs/UTILITIES.md)
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

환경 준비의 실행 결과는 [Step 0 작업 기록](docs/worklogs/step-0.md), 공통 기반 검증은 [Step 1 작업 기록](docs/worklogs/step-1.md)에 기록합니다.
수학 검증 기준은 [ACCEPTANCE.md](docs/ACCEPTANCE.md)에 정의되어 있습니다. 공통 유틸과 레지스트리는 구현했으며 Step 2~6은 미구현·미검증입니다.
`src/utils/`와 `.agent/registry.json`을 제공하며, `notebooks/`는 해당 구현 단계에서 생성합니다.
결과 PNG는 `outputs/`, 재계산 가능한 캐시는 `.cache/`에 저장합니다.

| 영역 | 현재 상태 |
|---|---|
| 개발환경·운영 규칙 | 로컬 환경 검사 3개·pip check 통과, PR 검토 대기 |
| 공통 유틸·캐시·레지스트리 | Step 1 구현·보완 완료, 로컬 전체 검사 48개 통과 |
| 선형대수·미적분·최적화 | Step 2~4 예정 |
| 역전파·확률/손실 노트북 | Step 5~6 예정 |
| 전체 재현 및 최종 결과 요약 | Step 7 예정 |

기존 [초안 PR #1](https://github.com/ntt65/A2_1/pull/1)은 후속 단계에서 재사용 여부를 검토합니다.
수치 결과와 시각화가 검증되면 이 README에 실측 오차·그림·실행 예시를 추가합니다.
