# 이슈 #27 — 보너스 전체 구현과 main 제출

2026-11-08 (Asia/Seoul). [이슈 #27](https://github.com/ntt65/A2_1/issues/27)의 기초·상세·실행계획을 따라 [원문 5절](../../A2_1_mission.md#5-보너스-과제-선택)의 Adam, Newton Method, 엔트로피·KL·Cross-Entropy를 모두 구현했다. 선행 #25/PR #26의 로컬 144개 검사와 Windows/Linux CI 성공·단계 보고 후 `a01f58c`에서 `codex/27-bonus-missions`를 시작했다.

## 사용자 지시와 전달 기준

처음에는 보너스까지 이슈별 구현·검증·PR·기록을 순차 진행하도록 요청했다. 이어 사용자가 main에서 평가할 수 있도록 README 정리와 병합까지 승인했다. 따라서 선행 평가 보완 PR #24 → #26 → 보너스 PR 순서로 main에 병합하고 최종 main CI와 로컬 동기화를 확인한다. 본 문서 커밋 이후 확정되는 PR 번호·커밋 SHA·병합/CI 결과는 이슈 #27에 누적한다. 외부 Naito_AI 재평가는 실행하지 않았으며 점수는 이전 93%(14/15) 기록을 유지한다.

## 수식·shape와 구현

- `src/adam.py`: `m=beta1*m+(1-beta1)*g`, `v=beta2*v+(1-beta2)*g²`, `mhat=m/(1-beta1^t)`, `vhat=v/(1-beta2^t)`, `theta-=lr*mhat/(sqrt(vhat)+epsilon)`. 기본 beta1=.9/beta2=.999/epsilon=1e-8. theta/g/m/v `(n,)`, t 정수. dtype float64, reset·반환 복사·입력/설정·표현 불가 연산·원자적 상태 검증.
- `src/optimizer.py`: 기존 수치 갱신을 보존하면서 실제 공통 snapshot/restore API를 추가했다. 후보 step이나 이후 callback 평가 실패 시 Momentum/Adam 상태를 복원하고 기존 오류 계약을 유지한다.
- `src/newton.py`: 유한·대칭·양의 정부호 Hessian `(n,n)`에 대해 `H*d=g`, `theta_next=theta-lr*d`. NumPy solve와 Cholesky를 사용하고 inverse/eig를 호출하지 않는다. `(n,)` 입력·복사·고정 damping `0<lr<=1`·비유한/특이/음곡률·잘못된 callback을 검증한다. 원형 Hessian2I, 타원형 diag(2,20).
- `src/information_theory.py`: `H=-sum(p ln p)`, `CE=-sum(p ln q)`, `KL=sum(p(ln p-ln q))`. 같은 shape/축을 사용하고 해당 클래스 축을 제거한다. 합1(atol1e-12,rtol0)·[0,1]·유한성 검증, 자연로그 nats, zero-target mask, 양의 p/영 q의 +inf 보존. 임의 clipping/자동 정규화 없음.
- 실험 스크립트 2개는 실행 경계에서 공통 seed42를 설정한다. Adam/GD/Momentum은 초기(5,5), 200회, 원형 lr=.1/타원형 lr=.05, 원시 경로 `(201,2)`를 저장한다. Newton/GD는 초기(5,5), 최대200회와 gradient tolerance1e-8을 같게 한다. 반복 경로는 공통 캐시와 완전한 키를 재사용하고 작은 정보 합계는 캐시하지 않는다.

## 독립 검토에서 찾은 실패와 보완

1. Adam의 `g=1e-200`, beta1=beta2=0, epsilon=1e-308에서 g²가0으로 underflow되어 기대 갱신 -1 대신 -1e108을 수락하는 반례를 재현했다. 첫/둘째 모멘트 연산의 정밀도 손실을 감지해 `_NonfiniteResult`로 거부하고 m/v/t 미커밋·runner 종료 회귀를 추가했다. 정확히 표현 가능한 최소 subnormal 모멘트는 허용한다. 두 실패 회귀를 먼저 확인한 뒤 수정했고 허용오차를 완화하지 않았다.
2. Newton의 `0.5H+0.5H.T`가 정확히 대칭인 subnormal H를 바꾸는 반례를 찾았다. 최소 양수의3배인 scalar H에서 후보1 대신1.25를 반환했다. 정확대칭은 그대로 보존하고 accepted 비대칭은 부호별 안전한 midpoint를 한 번 계산해 양쪽 삼각부에 기록했다.
3. 추가 2D subnormal SPD 회귀에서는 NumPy solve의 비스케일 소거도 유한 오답을 만들었다. H 전체가 subnormal이면 H와 g를 같은 최대절댓값으로 나눈 동치 선형계를 풀도록 보완했다. 최소/홀수 subnormal·비대칭·1e308 행렬 반례를 기존 허용오차 그대로 통과했다.
4. 정보 이론의 KL 비음수 수용 기준은 정확히 정규화된 고정 표본의 유한 결과로 한정했다. 입력 합의1e-12 허용오차 때문에 허용 입력 전체에 KL>=-1e-14를 보장한다고 쓰지 않는다. 양의 p/영 q는 inf이므로 유한 오차 비교에서 제외하고 경계 의미를 별도로 검사한다.

수정 후 독립 검토에서 Adam의 원래 반례·상태 보존·정확 subnormal, Newton의 5개 극소/대형 반례·입력 보존을 재확인했고 추가 차단 결함은 없었다. 조건수가 큰 Hessian의 전방오차나 일반 비선형 함수의 전역 수렴은 보장하지 않으며 [보너스 문서](../BONUS.md)에 명시했다.

## 실제 실행·검증

환경: Windows 10.0.26200, Python3.11.4, NumPy2.2.6, Matplotlib3.10.3, 기존 requirements-lock.txt. 아래 python은 프로젝트 `.venv/Scripts/python.exe`이며 -O를 사용하지 않았다.

| 명령/검사 | 실제 결과 |
|---|---|
| `python -m unittest discover -s tests -p test_adam.py -v` | 보완 후12개, 0.177초 통과 |
| `python -m unittest discover -s tests -p test_newton.py -v` | 보완 후15개, 0.176초 통과 |
| 정보 이론 수학·실험 관련 unittest | 11개, 0.669초 통과; 구현 전 import 실패를 먼저 확인 |
| `python -m unittest discover -s tests -p test_bonus_optimizer_experiment.py -v` | 4개, 2.138초 통과 |
| `python -m scripts.run_bonus_optimizers --no-cache` | PNG3·JSON·Markdown 생성, 독립 GD 닫힌해·Newton 해·예산/상태 검사 통과 |
| `python -m scripts.run_information_theory` | PNG·JSON·Markdown 생성, 독립 상수·경계·501점 곡선·항등식 검사 통과 |
| `python -m unittest discover -s tests -v` | 최종186개, 32.927초, 종료코드0; 기존 노트북 새 커널 검증 포함 |
| `python -m pip check` | No broken requirements found. |
| `python scripts/update_registry.py --update` 및 `--check` | 49파일·72개념, 현재 소스 일치 |
| 캐시·재현 | 캐시 사용/미사용·키 분리·반환 복사 검증. 최적화 no-cache 재실행 PNG3·JSON·Markdown 모두 저장본과 바이트 일치 |
| 시각 검사 | 새 PNG4개(최적화 각2100×644, 정보1792×686)의 축·범례·단위·조건·경로·무한경계 표시 확인 |

기존 Matplotlib/pyparsing 및 Windows 커널 이벤트 루프 경고는 있었으나 전체 검사 실패는 없었다. 반복 실험은 고정 조건이며 실제 측정치로 평가한다. 미실행 외부 재평가나 일반적 우월성을 통과로 보고하지 않는다.

## 주요 실측

- 원형 200회 최종반경: GD `2.934151e-19`, Momentum `9.489598e-5`, Adam `1.549128e-4`. 반경 .1 유지 시작은 각각20/77/79회.
- 타원형 200회 최종반경: GD `3.527540e-9`, Momentum `8.684780e-5`, Adam `1.101141e-1`. Adam은 .1 기준에 미도달이며 JSON의 first/stay는 null이다.
- 타원형 gradient norm<=1e-8 조기종료: GD197회(`9.677749e-9`), Newton1회(0). Newton의 Hessian·선형계 비용과 반복 횟수를 구분했다.
- p=(.25,.75), q=(.5,.5): H `.5623351446188083`, CE `.6931471805599453`, KL `.13081203594113697`. 역방향 KL `.14384103622589045`로 비대칭.
- 정보 이론 곡선의 유한 CE−(H+KL) 최대오차 `6.661338147750939e-16`, 최소 양수 q의 CE/KL `744.4400719213812`. p>0,q=0은 +inf이며 표준 JSON 문자열로 저장했다.

## 파일·레지스트리와 종료 상태

새 Python10개: 모듈3개·실험2개·검증5개. 기존 공통 optimizer1개를 확장했다. PNG4·JSON2·Markdown보고서2, BONUS 문서와 본 기록을 추가하고 README·수용기준·결정D16/D17·최적화 안내·전체계획·평가 분석·색인을 갱신했다. 레지스트리에는 신규10파일과11개 개념, 기존 optimizer의 상태 API·위치·해시·관련 검증을 반영했다. 모든 새 정의에 docstring이 있고 eig 호출은 기존 검증 경로에만 있다.

원문 미션·PDF·평가 원문 블록·기존 노트북·기본 산출물·잠금 의존성은 보존했다. 전체 제출 출력은 PNG15·JSON10·Markdown5다. 최종 main 제출 URL은 [저장소 main](https://github.com/ntt65/A2_1/tree/main)이며, 사용자 승인에 따른 관련 PR 순차 병합·최종 main CI·동기화·이슈 종료는 이 문서 이후 [이슈 #27](https://github.com/ntt65/A2_1/issues/27)에 기록한다.
