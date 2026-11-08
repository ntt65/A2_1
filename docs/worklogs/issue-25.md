# 이슈 #25 — 남은 평가 권고의 실행 증거

2026-11-08 (Asia/Seoul). [이슈 #25](https://github.com/ntt65/A2_1/issues/25)의 3단계 계획에 따라 `codex/25-evaluation-evidence`에서 수행했다. 기준은 검증 완료된 #23 커밋 `f435dd4`, 선행 PR은 [#24](https://github.com/ntt65/A2_1/pull/24)다. 사용자의 보너스까지 완료 지시에 따라 이 단계 보고 후 보너스 이슈를 진행한다.

## 구현과 해석

- `scripts/run_evaluation_evidence.py`는 기본 PNG 10개의 저장된 width/height/pHYs DPI·CRC를 읽고, 별도 재생성한 SVD 패널 8개의 Axes bbox를 기록한다. 입력 64×64/128×128·Figure·패널 크기를 구분한다. 기본 PNG는 바이트 단위로 보존했다.
- `x²`, x=3, float64의 61개 h 중 46개가 오차 `<=1e-4`를 만족했다. 통과 샘플 그룹은 `5.6234132519034904e-12`~`1.0`, 최소 관측 오차 0의 표본은 h=1이다. 연속 실수 구간이나 다른 함수의 최적 h를 보장하지 않는다.
- Softmax shape `(3,)`, `(2,3)`, `(2,3,4)`와 axis -1/0/1의 6개 실제 계산에서 최대 합 오차는 `2.220446049250313e-16 <=1e-6`이다. 평균 오차·전체 입력·결과 shape도 기록했다.
- 면적비 관측에 +0.02, 고정 역전파 `dW1[0,0]`에 +0.001을 의도적으로 주입한 두 사례는 각각 1%/5e-5 기준으로 실패했다. 정상 계산은 통과하며 원시 입력·차분 부호·허용치·판정을 보고한다.
- `scripts/run_optimizer_sensitivity.py`는 원형/타원형 × GD/Momentum × 고정 초기점 5개의 100회 경로 `(101,2)`와 원시 loss/norm을 기록한다. 원형 lr=.1, 타원형 lr=.05, beta=.9, seed42, float64이며 초기점은 `[[5,5],[-5,5],[1,-3],[-2,-1],[0,0]]`다.
- 각 그룹의 5/5가 예산 끝에 반경 .1 안에 있다. 끝까지 유지 시작의 평균은 원형 GD 14.0/Momentum 54.6, 타원형 GD 25.4/Momentum 56.8회다. 원점의 0회도 포함한다. 미도달은 null, 성공 표본 통계와 전체 최종반경 통계는 분리한다. 모든 조건에서 Momentum이 빠르다고 주장하지 않는다.
- 원형 GD의 닫힌 해 `theta[t]=(1-2lr)^t theta[0]`와 손실 배율 `(1-2lr)^2`를 직접 비교했다. lr=.5/1/1.1의 배율은 0/-1/-1.2이고 손실 배율은 0/1/1.44다. 0 분모 비율은 제외했다.
- `docs/MATH_DERIVATIONS.md`에 기존 필수 6개 역전파·편향·shape·13개 손계산표, Gaussian/Bernoulli/categorical likelihood→NLL 전개와 안정식·경계 조건을 정리했다. 셀 ID 15개와 로컬 링크 13개를 확인했고 독립 수치 기준이 atol=5e-15, rtol=0으로 일치했다.

## 실제 검증

환경은 Windows 10.0.26200, Python 3.11.4, NumPy 2.2.6, Matplotlib 3.10.3, 기존 잠금 의존성이다. 아래 python은 `.venv/Scripts/python.exe`다.

| 명령/검사 | 결과 |
|---|---|
| `python -m scripts.run_evaluation_evidence --no-cache` | JSON/Markdown 생성, 정상값·주입 실패·PNG 보존 assertion 통과 |
| `python -m unittest discover -s tests -p test_evaluation_evidence.py -v` | 6개 통과, 최종 0.473초 |
| `python -m scripts.run_optimizer_sensitivity --no-cache` | PNG/JSON/Markdown 생성 및 수치 assertion 통과 |
| `python -m unittest discover -s tests -p test_optimizer_sensitivity.py -v` | 6개 통과, 0.942초 |
| `python -m unittest discover -s tests -v` | 144개 통과, 24.250초, 종료코드 0; 새 커널 노트북 포함 |
| `python -m pip check` | No broken requirements found. |
| `python scripts/update_registry.py --update` 및 `--check` | 39파일·61개념, 현재 소스와 일치 |
| 시각 검사 | 1680×1400 최적화 PNG의 축·단위·조건·범례·경로 정상 |
| 독립 검토 | 수식·출처·실패 진단·통계 분모·캐시 키·미도달 처리에 차단 결함 없음 |

최초 제한된 실행에서는 새 노트북 커널 시작이 진행되지 않아 중단했다(통과로 기록하지 않음). 정상 로컬 실행 권한으로 전체 검사를 재실행하여 위 144개 성공을 얻었다. 테스트 삭제나 허용오차 완화는 없었다. 기존 Matplotlib/pyparsing·커널 이벤트 루프 경고는 있었으나 최종 실패는 없었다.

## 변경과 게시

신규 스크립트·테스트 4개, 수치/민감도 JSON·Markdown·PNG 5개, 유도 문서, 본 기록을 추가했다. README·평가 분석·수용 기준·D15·전체 계획·레지스트리도 갱신했다. 원문 평가 블록·미션·기존 알고리즘·기존 PNG·노트북·의존성은 보존했다. 색인에는 신규 4파일과 5개 개념을 추가하고 정방향/역방향 및 해시를 확인했다.

본 문서 커밋 이후 확정되는 커밋·PR·Windows/Linux CI는 [이슈 #25](https://github.com/ntt65/A2_1/issues/25)에 누적한다. 선행 브랜치 기준의 PR로 제출하며 자동 병합하지 않는다. 외부 Naito_AI 재평가는 미실행이고 기존 93%(14/15) 점수를 유지한다.
