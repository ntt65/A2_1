# 이슈 #23 — Naito_AI 평가와 eig 비교 증거

기준일: 2026-11-08 (Asia/Seoul). [작업 이슈 #23](https://github.com/ntt65/A2_1/issues/23)의 기초·상세·실행계획을 따른다. 기존 브랜치 `codex/23-naito-eig-evidence`를 재사용했으며 기준 main은 `3f03488b60096ca06f791d902737c3fbf0b0c3c7`이다.

## 재개 지점과 범위

사용자가 이전 채팅의 변경을 보존하고 완료된 검증 확인 후 커밋·PR·기록부터 이어가도록 지시했다. 재개 시 기존 수정 8개·미추적 파일 5개가 있었고 이슈 #23의 PR과 완료 댓글은 없었다. 이전 채팅의 실행 기록과 `.cache/issue-23-tests.log`를 확인했다. 기존 소스·테스트·산출물을 다시 구현하거나 덮어쓰지 않고 누락된 작업 기록·전체 계획·완료 상태를 정리했다. 수용 기준의 출력 설명은 전용 CLI의 벡터 출력과 기본 실험의 요약 출력을 정확히 구분했다.

`Naito_AI_eval_result.md`의 원문 블록이 이전 채팅에 제공된 평가 원문과 일치함을 확인했다(줄바꿈 형식 정규화 및 양끝 공백 제외). 평가 원문은 그대로 유지하며 15개 항목 분석과 대응을 별도로 기록한다. 기존 `np.linalg.eig` 비교 테스트는 존재했으나 제출용 기본 실험에는 닫힌 해 기준만 보였던 문제를 보완했다.

## 수식·shape·검증 설계

- 고정 실수 대칭 행렬 `A=[[4,1],[1,3]]`: `(2,2)`, 명시적 초기 벡터 `[1,1]`: `(2,)`, dtype `float64`. 실행 진입점에서 공통 seed 42를 설정한다.
- 수동 반복식 `v_next=A@v/||A@v||₂`와 Rayleigh quotient `lambda=v.T@A@v/(v.T@v)`를 기존 `src/linear_algebra.py`에서 재사용한다. 알고리즘 본체는 변경하지 않았다.
- 검증 전용 `scripts/verify_power_iteration.py`에서 `np.linalg.eig`의 `argmax(abs(eigenvalues))`를 선택하고 고유벡터 부호를 정렬한다. 비교 고유값은 scalar, 벡터는 `(2,)`다.
- 상대오차 `abs(lambda_manual-lambda_numpy)/abs(lambda_numpy) <= 0.05`, 독립 상대 잔차 `||Av-lambda*v||₂/(||A||F*||v||₂) <= 1e-6`, 부호 정렬 벡터 L2 오차 `<=1e-8`를 검사한다. 닫힌 해 `(7+sqrt(5))/2`와도 교차 검증한다.
- 이 고정 행렬은 절댓값 지배 고유값이 분리되어 있다. 임의 행렬·초기 방향에 대해 작은 잔차만으로 지배 고유값 수렴을 보장하지 않는다. 잘못된 기준값은 `AssertionError`로 실패한다.

## 실제 실행과 결과

환경: Windows `10.0.26200`, Python `3.11.4`, NumPy `2.2.6`, Matplotlib `3.10.3`, 기존 `.venv`와 `requirements-lock.txt`. 아래 `python`은 프로젝트 `.venv/Scripts/python.exe`를 뜻한다. 최적화 모드 `-O`는 사용하지 않았다.

| 시점·명령 | 실제 결과 |
|---|---|
| 이전 채팅: `python -m unittest discover -s tests -p test_linear_algebra_experiment.py -q` (구현 전) | 2개 중 1개 오류: `numpy_eig_comparison` 누락 `KeyError` 확인 |
| 이전 채팅: `python -m unittest discover -s tests -p test_power_iteration_verification.py -q` | 3개 통과, 0.198초 |
| 이전 채팅: `python -m unittest discover -s tests -p test_linear_algebra_experiment.py -q` (연동 후) | 2개 통과, 1.302초 |
| 이전 채팅: `python -m scripts.verify_power_iteration` | 두 고유값·고유벡터·오차·허용치·PASS 출력, JSON·Markdown 저장 |
| 이전 채팅: `python -m scripts.run_linear_algebra --no-cache` | eig 비교·기존 변환/SVD assertion 통과, PNG·JSON·Markdown 생성 |
| 이전 채팅: `python scripts/update_registry.py --update` | 신규 검증 코드와 테스트 포함 색인 갱신 |
| 이전 채팅: `python -m unittest discover -s tests -q` | 전체 132개 통과, 26.959초, 종료코드 0. 새 커널 노트북 검증 포함 |
| 이전 채팅 및 재개 시: `python -m pip check` | `No broken requirements found.` |
| 이전 채팅 및 재개 시: `python scripts/update_registry.py --check` | `Registry check passed.` |
| 이전 채팅 및 재개 시: `git diff --check` | 오류 없음 |

이전 로컬 전체 실행은 `-q`였으며 `-v` 실행으로 바꾸어 기록하지 않았다. 소스와 테스트가 그대로이고 현재 색인이 일치하므로 같은 로컬 검사를 반복하지 않았다. PR의 기존 Windows/Linux CI는 지정 명령 `python -m unittest discover -s tests -v`, `python -m pip check`, 레지스트리 검사를 실행하며 최종 결과를 이슈에 기록한다.

| 수치 항목 | 실측값 |
|---|---:|
| 수동 Power Iteration 고유값 | 4.618033988749896 |
| `np.linalg.eig` 고유값 | 4.618033988749895 |
| 고유값 상대오차 | 1.923282552410481e-16 |
| 업데이트 횟수 | 32 |
| 독립 상대 잔차 | 6.399563856969149e-11 |
| 부호 정렬 고유벡터 L2 오차 | 1.4871237088596054e-10 |
| NumPy 기준값과 닫힌 해의 절대오차 | 0.0 |

[비교 보고서](../../outputs/power_iteration_eig_comparison.md)와 [원시 JSON](../../outputs/power_iteration_eig_comparison.json)에 실제 입력·두 계산·허용치·판정을 함께 기록했다.

## 발견 문제·보완과 독립 검토

- 실패하는 실험 회귀 검증을 먼저 확인한 뒤 기본 실험을 검증 전용 함수에 연결하여 누락된 비교 필드와 산출물을 추가했다. 고유벡터 부호 반전은 통과하고 잘못된 기준값은 실패하는 검증을 포함한다.
- 기존 전체 실행 로그의 Matplotlib/pyparsing 사용 중단 예정 경고와 Windows 커널 이벤트 루프 경고는 있었지만 테스트 실패는 없었다.
- 재개 시 별도 읽기 전용 리뷰에서 수학·shape·실패 검증·eig 격리·실험 연동에 차단 결함이 없음을 확인했다. README가 미작성 작업 기록을 링크하던 문제는 이 문서를 추가하여 해결했다.
- 기존 PNG 3개는 재생성 뒤 Git 변경이 없었다. 이번 보완은 비교 출력과 근거 문서이며 새로운 그림·노트북 편집은 없다.
- `np.linalg.eig`의 실제 호출은 전용 검증 스크립트와 기존 비교 테스트에만 있다. 기존 미션 원문·PDF·수학 알고리즘·잠금 의존성을 변경하지 않았다.

## 변경 파일과 레지스트리

- 코드·검증: `scripts/verify_power_iteration.py`, `scripts/run_linear_algebra.py`, `tests/test_power_iteration_verification.py`, `tests/test_linear_algebra_experiment.py`.
- 결과: `outputs/power_iteration_eig_comparison.json`, `outputs/power_iteration_eig_comparison.md`, `outputs/linear_algebra_metrics.json`.
- 문서: `Naito_AI_eval_result.md`, `AGENTS.md`, `README.md`, `docs/ACCEPTANCE.md`, `docs/LINEAR_ALGEBRA.md`, `docs/PROJECT_PLAN.md`, 본 작업 기록.
- 색인: `.agent/registry.json`은 35파일·56개념. 코드·검증 4개 파일의 정방향 항목과 역방향 연결을 갱신했고 `Power_Iteration_Eig_Verification` 개념을 추가했다. 신규 심볼·의존성·해시·관련 테스트가 현재 파일과 일치한다. 재개 시 문서 보완은 색인 범위 밖이므로 불필요한 색인 재생성을 하지 않았다.

## 게시·검토와 남은 사항

이 기록의 커밋 이후 확정되는 커밋 SHA·PR 링크·CI 결과는 [이슈 #23](https://github.com/ntt65/A2_1/issues/23)에 누적한다. 기존 브랜치에서 게시하고 PR에 `Closes #23`을 연결하며 자동 병합하지 않는다. 구현·로컬 검증 완료와 main 병합 완료를 구분한다.

Naito_AI 외부 재평가는 실행하지 않았다. 현재 외부 점수는 기존 93%(14/15)이며 자체 assertion 통과를 새 외부 점수로 표현하지 않는다. 나머지 PASS 권고와 보너스 구현은 후속 범위로 남긴다.
