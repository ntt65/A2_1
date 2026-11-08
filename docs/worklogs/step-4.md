# Step 4 작업 기록

2026-11-08, [이슈 #7](https://github.com/ntt65/A2_1/issues/7), 브랜치 `codex/7-optimizers`, 선행 PR #14의 `87e990a` 기반.
사용자 지시에 따라 Step4~7은 단계별 종료 보고 후 순차 자동 진행한다.

## 기초계획

GD·Momentum을 NumPy로 구현하고 동일 시작점/예산에서 원형·타원형 경로를 비교한다. 초기 상태·비유한 종료를 추적하고 D10/D11의 수학적 정정을 적용한다.

## 상세계획

`VanillaGD.step`와 `Momentum.step/reset`, 초기점을 포함한 `OptimizationResult`, 원형/타원형 고정 실험, 캐시를 재사용하는 실행 스크립트와 PNG3개·JSON을 제공한다.
수식은 [최적화 문서](../OPTIMIZATION.md), 계약은 source docstring에 기록했다. shape은100회 기준 path(101,2),loss/gradientnorm(101,)이다.

## 실행·점검·보완

1. 이슈의 3단계 계획과 전용 원격/로컬 브랜치·upstream을 만들었다.
2. core·실험을 분담하고 독립 Decimal 점화식 리뷰로 비교했다.
3. 수치·예외·상태 복원·입력 보존·캐시 동일성·PNG 검사를 수행했다.
4. 캐시 경로에서 NumPy 정수 steps가 core와 달리 처리되지 않는 문제를 발견해 정수 검증·정규화를 추가하고 회귀 테스트했다.
5. 문서의100회 비교와20회 학습률 실험을 구분하고 callback예외 전파/비유한 상태의 차이를 명확히 했다.
6. 전체 테스트 **125개 통과(8.485초)**, pip check, registry check 통과. 색인 **28파일·270심볼·37개념**.
7. 원격 게시·PR·최종 CI 및 커밋 링크는 이슈 #7에 기록한다. 사용자 검토·병합과 구현 완료를 구분한다.

## 실제 결과

| 조건 | GD 최종 norm/loss | Momentum 최종 norm/loss |
|---|---|---|
| 원형 lr=.1,100회 | 1.4404e-9 /2.0748e-18 | .0201625 /.000406527 |
| 타원형 lr=.05,100회 | .000132807 /1.7638e-8 | .0198751 /.000805109 |

원형 GD는 기준norm≤.1 충족. lr=.5는1회에원점,lr1은부호진동,lr1.1은20회내유한발산을 확인했다.
독립 Decimal 점화식과 Momentum 결과 차이는 원형3.99e-17/타원형4.77e-17, 비대각 Hessian GD 닫힌경로 오차8.33e-17이었다.
PNG에서 경로·loss·gradientnorm·범례·축을 확인했다. 고정 조건에서 GD의 최종 오차가 작으며 Momentum의 우위를 일반화하지 않는다.

## 재현

```powershell
.\.venv\Scripts\python.exe -m scripts.run_optimizer
.\.venv\Scripts\python.exe -m scripts.run_optimizer --no-cache
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/update_registry.py --check
```

환경 Python3.11.4/NumPy2.2.6/Matplotlib3.10.3. 기존 pyparsing deprecation/Windows notebook event-loop 경고가 있었으나 모든 검증은 통과했다.
다음 단계는 [Step5 #8](https://github.com/ntt65/A2_1/issues/8)이며 이 단계의 게시·검증 후 진행한다.
