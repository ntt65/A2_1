# 전체 계획과 체크리스트

기준일: 2026-11-08 (Asia/Seoul). 전체 추적: [#2](https://github.com/ntt65/A2_1/issues/2).
목적은 AI 학습의 수학을 직접 유도하고 NumPy로 검증하는 것이다. 최종 제출 진입점은 [main의 README](https://github.com/ntt65/A2_1/tree/main)다.

## 진행 순서

전체 계획 → 작업 이슈 → 작업 시작 시 브랜치 → 기초계획 → 상세계획 → 실행계획 → 구현 → 점검 → 보완·재검증 → commit/push → PR → 이슈 기록 → 사용자 검토·병합.

| 단계 | 이슈 | 산출물 | 선행 단계 | 현재 상태 |
|---|---|---|---|---|
| 0 환경·운영 준비 | [#3](https://github.com/ntt65/A2_1/issues/3) | 환경, AGENTS, 계획, 템플릿, CI | 없음 | 로컬 검증 완료, [PR #11](https://github.com/ntt65/A2_1/pull/11) main 병합 완료 |
| 1 공통 기반 | [#4](https://github.com/ntt65/A2_1/issues/4) | utils, cache, registry | 0 | 구현·로컬 48개 검증 완료, [PR #12](https://github.com/ntt65/A2_1/pull/12) main 병합 완료 |
| 2 선형대수 | [#5](https://github.com/ntt65/A2_1/issues/5) | linear_algebra.py, 검증, PNG | 1 | 구현·로컬 87개 검사 완료, [PR #13](https://github.com/ntt65/A2_1/pull/13) main 병합 완료 |
| 3 미적분 | [#6](https://github.com/ntt65/A2_1/issues/6) | calculus.py, 검증, PNG | 2 | 구현·로컬105개 검사 완료, [PR #14](https://github.com/ntt65/A2_1/pull/14) main 병합 완료 |
| 4 최적화 | [#7](https://github.com/ntt65/A2_1/issues/7) | optimizer.py, 검증, PNG | 3 | 구현·125개 검사·CI 완료, [PR #15](https://github.com/ntt65/A2_1/pull/15) main 병합 완료 |
| 5 역전파 | [#8](https://github.com/ntt65/A2_1/issues/8) | backprop_derivation.ipynb | 4 | 구현·128개 검사·CI 완료, [PR #16](https://github.com/ntt65/A2_1/pull/16) main 병합 완료 |
| 6 확률·손실 | [#9](https://github.com/ntt65/A2_1/issues/9) | probability_loss.ipynb | 5 | 구현·129개 검사·CI 완료, [PR #17](https://github.com/ntt65/A2_1/pull/17) main 병합 완료 |
| 7 통합·제출 | [#10](https://github.com/ntt65/A2_1/issues/10) | 전체 검증, 결과, README | 1~6 | 새 환경 129개 검사·산출물 재현·README 완료, [PR #18](https://github.com/ntt65/A2_1/pull/18) main 병합 완료 |

후속 이슈는 미리 발행하되 브랜치는 시작할 때 최신 기준 브랜치에서 만든다. 제출 기준은 `main`이며, 병합 후 제출 안내를 마무리하는 [이슈 #19](https://github.com/ntt65/A2_1/issues/19)의 작업 브랜치는 `codex/19-main-submission`이다.
2026-11-08 사용자 지시에 따라 Step 4~7은 단계별 종료 보고 후 자동으로 다음 이슈를 진행했다. 이슈·브랜치·계획·검증·PR·작업 기록은 단계별로 유지했다.
중단된 동일 이슈를 재개할 때는 기존 브랜치·PR과 기록을 재사용한다.
사용자의 순차 진행 지시에 따라 Step 1은 당시 검증 완료·검토 대기였던 환경 PR #11의 변경 위에서 구현했다. Step 2는 검증 완료된 Step 1 위에서, Step 3은 Step 2 커밋 `112f833` 위에서 진행했다. 이후 단계도 선행 구현을 포함한 누적 브랜치로 개발하고 단계별 PR로 변경을 분리했다.
사용자가 평가 제출을 위한 병합을 승인한 뒤 PR #11~18을 순서대로 `main`에 병합했고, 후속 PR의 기준 브랜치도 선행 병합 후 `main`으로 전환했다. 병합 완료 커밋은 [`9d50875781b52e7b56c5911553da80bd6b15bda0`](https://github.com/ntt65/A2_1/commit/9d50875781b52e7b56c5911553da80bd6b15bda0)이며, 해당 트리는 최종 검증한 누적 구현의 트리와 일치한다. 병합 후 `main` 검사와 #19의 최종 CI·종료 상태는 [이슈 #19](https://github.com/ntt65/A2_1/issues/19)에 별도로 기록한다.
기존 [PR #1](https://github.com/ntt65/A2_1/pull/1)은 닫힌 미병합 초안이다. 검증한 단계별 PR로 대체했으며 최종 제출 브랜치에는 해당 초안을 병합하지 않았다.

## 기초계획

- 필수 범위: 원문 4.1~4.5와 최종 결과물 전체, 수식 유도·코드·assert·PNG·재현 방법.
- 제외: 별도 요청 없는 Adam/Newton/정보이론 보너스, 웹 서비스, 외부 데이터 서비스, Redis 서버.
- 수학 기준: [ACCEPTANCE.md](ACCEPTANCE.md), 충돌 처리: [DECISIONS.md](DECISIONS.md).
- 작업 로그: 각 이슈의 본문·댓글. 버전 관리할 설계와 재현 정보는 문서에도 반영한다.
- 최종 완료: 필수 검증 통과, 결과물 확인, README 재현 명령 확인, PR 검토·병합, 이슈 연결 종료.

## 상세계획

| 단계 | 핵심 설계 및 점검 |
|---|---|
| 0 | Python 3.11 프로젝트 가상환경, 고정 의존성, 환경 점검, Windows/Linux CI, 이슈·PR 양식 |
| 1 | 안정적 sigmoid/softmax, 중심차분, 진입점 seed, 공통 플롯, 로컬 캐시, 양방향 레지스트리·갱신 검사 |
| 2 | 면적과 abs(det), 수동 Power Iteration·수렴 실패, SVD rank·복원 오차·비교 이미지 |
| 3 | 중심차분 오차, gradient shape, 등고선/접선과 gradient 관계 |
| 4 | GD/Momentum 업데이트 관례, 학습률별 안정 범위, 동일 예산 비교, NaN/Inf 종료 |
| 5 | 원문 고정 입력과 W1@x 관례, 중간값·6개 필수 미분·shape, 독립 손계산/수치 gradient 비교 |
| 6 | 정규·베르누이 분포, 안정적 softmax 재사용, 가정별 NLL→MSE/BCE/CE 유도 |
| 7 | 새 환경에서 전체 테스트/노트북 재실행, 시각 검사, 레지스트리 정합성, README 결과표 |

완성된 주요 제출 구조:

```text
A2_1/
├── AGENTS.md
├── .agent/registry.json              # Step 1
├── .github/ISSUE_TEMPLATE/task.md
├── .github/pull_request_template.md
├── .github/workflows/ci.yml
├── docs/                            # 계획·검증·결정·작업별 기록
├── scripts/                         # 실험·새 커널 실행·레지스트리 관리
├── src/
│   ├── utils/                       # Step 1
│   ├── linear_algebra.py            # Step 2
│   ├── calculus.py                  # Step 3
│   └── optimizer.py                 # Step 4
├── tests/                           # 환경·수학·캐시·레지스트리·노트북 검증
├── notebooks/
│   ├── backprop_derivation.ipynb     # 실행된 역전파 유도·손계산 비교
│   └── probability_loss.ipynb        # 실행된 분포·MLE·손실 유도
├── outputs/                         # 제출 PNG 10개·측정 JSON 5개
├── requirements.txt
├── requirements-dev.txt
├── requirements-lock.txt
└── README.md
```

## 실행계획: Step 0

- [x] 저장소·원문·기존 PR·사용 가능한 Python 확인
- [x] 전체 계획 및 Step 0~7 이슈 발행
- [x] 현재 작업 전용 브랜치 생성
- [x] 운영 규칙·수용 기준·결정 기록·재사용 프롬프트 작성
- [x] 이슈·PR 템플릿과 단계별 계획 연결
- [x] 가상환경 설치 및 의존성 고정
- [x] 환경 테스트·노트북 도구·PNG 렌더링·pip check 실행
- [x] 검토에서 발견한 문제 보완 및 재검증
- [x] 커밋·원격 브랜치 게시 및 환경 PR #11 생성(로컬 Git 인증 입력 대기로 연결된 GitHub 앱 사용)
- [x] 이슈에 결과·실패/보완·커밋·PR 기록
- [x] 사용자 승인 후 PR #11 main 병합

## 전체 구현·검증 체크리스트

- [x] Step 0 환경과 운영 절차
- [x] Step 1 공통 유틸·캐시·레지스트리
- [x] Step 2 선형대수 수치·시각 검증
- [x] Step 3 수치 미분·Gradient 검증
- [x] Step 4 최적화 수렴·진동·발산 및 비교
- [x] Step 5 역전파 손계산·NumPy·수치 미분
- [x] Step 6 분포·MLE·노트북
- [x] Step 7 새 환경 전체 검사·산출물 재생성
- [x] Step 7 최종 README·PR #18 게시

Step 7의 검증·CI·종료 기록은 [이슈 #10](https://github.com/ntt65/A2_1/issues/10), 병합 후 제출 안내 정리와 최종 상태는 [이슈 #19](https://github.com/ntt65/A2_1/issues/19)에서 확인한다.

## 검토·병합 체크리스트

- [x] Step 0 검토·병합
- [x] Step 1 공통 유틸·캐시·레지스트리 검증·병합
- [x] Step 2 선형대수 수치·시각 검증·병합
- [x] Step 3 수치 미분·Gradient 검증·병합
- [x] Step 4 수렴/진동/발산과 Momentum 비교 검증·병합
- [x] Step 5 손계산·역전파·노트북 검증·병합
- [x] Step 6 분포·MLE·노트북 검증·병합
- [x] Step 7 전체 재현·제출 README 검증·병합

체크는 실제 완료 증거가 있을 때만 바꾼다. PR 생성만으로 검토·병합 완료를 표시하지 않는다.

## 실행계획: Step 1

- [x] 기존 이슈 #4의 범위·계획·선행 PR 상태 확인
- [x] `codex/4-shared-utils` 브랜치 생성 및 이슈 상세계획 갱신
- [x] 공통 수학·미분·시드·플롯 함수와 독립 기대값 검증
- [x] 메모리/디스크 캐시·무효화·손상 복구·복사 정책 검증
- [x] 레지스트리 생성/검사·심볼·개념·해시·테스트 연결 검증
- [x] 통합 테스트·환경·시각 검사·코드 리뷰와 보완
- [x] README·실측 작업 기록·인덱스·체크리스트 갱신
- [x] 커밋·원격 게시·단계별 PR #12·이슈 기록
- [x] 선행 PR 병합 확인 후 Step 1 검토·병합

## 실행계획: Step 2

- [x] 이슈 #5의 기초·상세·실행계획과 선행 상태 확인
- [x] 원격/로컬 `codex/5-linear-algebra` 브랜치·upstream 생성
- [x] 변환·면적, Power Iteration, SVD 및 독립 단위 테스트 구현
- [x] 합성 데이터·재현 스크립트·PNG/JSON 결과 생성
- [x] 독립 코드·수학·시각 검토와 회귀 보완
- [x] 전체 87개 테스트·환경·레지스트리 검사
- [x] API 문서·실측 결과·수용 기준·색인 갱신
- [x] 원격 커밋·PR #13 게시 및 구현·검증 이슈 기록(최종 CI 결과는 이슈 #5 참조)
- [x] 선행 PR 및 Step 2 검토·병합

## 실행계획: Step 3

- [x] 이슈 #6의 기초·상세·실행계획과 원문 4.2·선행 상태 확인
- [x] 원격/로컬 `codex/6-calculus` 브랜치·upstream 생성
- [x] 공통 중심차분 재사용, 수치 gradient·해석 gradient·독립 검증 구현
- [x] gradient/접선 그림·h별 오차 그림과 PNG/JSON 재현 명령 작성
- [x] 코드·수학·시각 검토 및 보완·전체 회귀 검증
- [x] 문서·수용 기준·색인·실측 작업 기록 갱신
- [x] 원격 게시·PR #14·구현/검증 이슈 기록(최종 CI 결과는 이슈 #6 참조)
- [x] 선행 PR 및 Step 3 검토·병합

## 실행계획: Step 4

- [x] 이슈 #7의 3단계 계획·D10/D11·선행 PR #14 확인
- [x] `codex/7-optimizers` 원격/로컬 브랜치와 upstream 생성
- [x] GD·Momentum·종료 상태·독립 수치 테스트 구현
- [x] 같은 예산의 실험·캐시·비교 PNG·원시 JSON 생성
- [x] 코드·수학·시각 리뷰와 보완, 전체 회귀 검사
- [x] 문서·수용 기준·레지스트리 갱신
- [x] 원격 게시·PR #15·Windows/Linux CI·이슈 기록
- [x] 사용자 승인 후 PR #15 main 병합

## 실행계획: Step 5

- [x] 이슈 #8 계획·원문·D12·독립 손계산 상수 확인
- [x] 전용 브랜치 codex/8-backprop-derivation 및 upstream
- [x] 수식·shape·손계산표·NumPy·9개 매개변수 중심차분 검증
- [x] 새 커널 실행 도구·노트북 회귀·전체 검사·독립 리뷰
- [x] 문서·수용 기준·레지스트리·실측 기록
- [x] 원격 게시·PR #16·Windows/Linux 보완 CI·이슈 기록
- [x] 사용자 승인 후 PR #16 main 병합

## 실행계획: Step 6

- [x] 이슈 #9 계획·원문·D13·독립 수치 기준 확인
- [x] 전용 브랜치 codex/9-probability-loss 및 upstream
- [x] 분포·Softmax·NLL/MSE/BCE/CE 유도·독립 검증
- [x] 새 커널 실행·PNG·전체 검사·독립 리뷰와 보완
- [x] 문서·수용 기준·레지스트리·실측 기록
- [x] 원격 게시·PR #17·Windows/Linux CI·이슈 기록
- [x] 사용자 승인 후 PR #17 main 병합

## 실행계획: Step 7

- [x] 이슈 #10의 기초·상세·실행계획 및 추가 README 요구 확인
- [x] codex/10-integration-delivery 브랜치와 upstream 생성
- [x] 새 Python 3.11 환경에 잠금 의존성 설치·전체 검사
- [x] 캐시 없이 모든 예제·노트북 재실행 및 새 산출물 대조
- [x] 미션 필수항목·수학 조건·정책·원문 보존 감사
- [x] 최종 문서 링크 감사(20개 문서의 130개 로컬 참조)
- [x] README 실제 명령·콘솔·결과표·그림·제출 URL 완성
- [x] 최종 문서·원격 게시·PR #18
- Step 7 검증·CI·이슈 기록·단계 보고: [이슈 #10](https://github.com/ntt65/A2_1/issues/10)
- [x] 사용자 승인 후 PR #18 main 병합

## 병합 후 제출 마무리: 이슈 #19

[이슈 #19](https://github.com/ntt65/A2_1/issues/19)에서 `main`의 평가 제출 안내와 작업 기록을 정리한다. 작업 브랜치는 `codex/19-main-submission`이며 기존 수학 구현·검증 수치는 유지한다.

- [x] PR #11~18의 main 병합 완료와 최종 검증 트리 일치 확인
- [x] README·전체 계획·작업 기록을 main 제출 기준으로 정리
문서 변경의 PR·최종 CI·병합 결과·제출 완료 보고는 이 문서 커밋 이후에 확정되므로 [이슈 #19](https://github.com/ntt65/A2_1/issues/19)의 최종 상태를 따른다.

## 평가 보완 1: 이슈 #23

[이슈 #23](https://github.com/ntt65/A2_1/issues/23)은 Naito_AI 평가 원문과 15개 항목 분석을 보존하고, FAIL #3의 Power Iteration / `np.linalg.eig` 비교 실행 증거를 보완한다. 기준은 main 커밋 `3f03488`, 기존 작업 브랜치는 `codex/23-naito-eig-evidence`다. 중단된 작업을 같은 브랜치에서 재개했으며 알고리즘 본체와 기존 변경을 보존했다.

- [x] 기존 이슈의 기초·상세·실행계획, 브랜치와 중복 PR 확인
- [x] 평가 원문 보존·15개 항목 분석·후속 권고 분리
- [x] 검증 전용 eig 비교, 콘솔·JSON·Markdown 보고서와 실패 회귀 검증
- [x] 전체 132개 테스트의 기존 성공 로그 확인, 환경·레지스트리·독립 리뷰
- [x] README·수용 기준·선형대수 문서·색인·[작업 기록](worklogs/issue-23.md) 정리

게시 커밋·PR·최종 Windows/Linux CI는 이 문서 커밋 이후 확정되므로 이슈 #23에 누적 기록한다. 구현·로컬 검증 완료와 PR 검토·main 병합을 구분하며 자동 병합하지 않는다. 나머지 평가 권고와 보너스는 후속 작업으로 남긴다. 외부 재평가 결과는 아직 없으며 기존 점수 93%(14/15)를 유지한다.

## 평가 보완과 보너스 완료 요청 (2026-11-08)

사용자가 보너스 미션까지 완료하도록 지시했다. 남은 평가 권고 → 보너스 Adam·Newton Method·정보 이론 순서로 진행하며 이슈별 구현·검증·게시·기록을 마친 뒤 중간 승인 없이 이어간다. 자동 병합과 외부 Naito_AI 재평가는 포함하지 않는다.

추가 지시: 같은 날 사용자가 “미션 평가는 main에서 받아야 하니 README.md까지 정리하고 머지까지 끝내”라고 승인했다. 이에 평가 보완·보너스의 관련 PR을 선행 순서대로 main에 병합하고 최종 main CI 및 로컬 동기화까지 확인한다. 앞선 병합 대기 문구는 당시 상태이며, 최종 결과는 이슈 #27의 기록을 따른다.

[이슈 #25](https://github.com/ntt65/A2_1/issues/25)의 `codex/25-evaluation-evidence`는 검증 완료된 #23 커밋 `f435dd4`에서 시작했다. 시작 당시 PR #24가 미병합이어서 후속 PR은 선행 작업 브랜치를 기준으로 구성했다. 사용자 병합 승인 후 PR #24와 #26을 순서대로 main에 병합했다.

- [x] 기존 이슈·PR·레지스트리·원문 확인, 3단계 계획 기록
- [x] 출력 메타데이터·수치 요약·실패 진단
- [x] 여러 초기점 민감도와 수렴/발산 근거
- [x] 일반 Markdown 수식 유도 연결
- [x] 전체 144개 검사·시각 검토·문서·색인·[작업 기록](worklogs/issue-25.md)
- [x] 이 단계 보고 후 보너스 이슈 시작

이슈 #25의 게시 커밋·PR·최종 CI는 문서 커밋 이후 해당 이슈에 기록한다. 로컬 검증 완료와 검토·병합 완료는 구분한다.

이슈 #25는 커밋 `a01f58c`, [PR #26](https://github.com/ntt65/A2_1/pull/26) 및 Windows/Linux CI 성공으로 완료 보고했다. 이후 [이슈 #27](https://github.com/ntt65/A2_1/issues/27)의 3단계 계획을 기록하고 같은 커밋에서 `codex/27-bonus-missions`를 시작했다. 선행 PR #24(`d238018`)와 #26(`fd7718a`)을 main에 병합한 뒤 보너스 PR은 main을 기준으로 제출한다.

- [x] 원문 5절 세 보너스의 범위·수식·shape·실패조건·검증계획 기록
- [x] Adam 구현·편향보정·상태복원 및 독립 검증
- [x] Newton 구현·Hessian 계약·GD 비교 및 독립 검증
- [x] 엔트로피·KL·Cross-Entropy 구현·경계·항등식 검증
- [x] 재현 CLI·캐시 동일성·PNG/JSON·수식 문서
- [x] 전체 회귀·새 커널·시각·독립 검토·레지스트리
로컬 전체186개 검사·환경·49파일/72개념 레지스트리·시각/독립 검토는 완료했다. 명시 파일 커밋·push·PR·CI·사용자 승인 main 병합·최종 보고는 [이슈 #27](https://github.com/ntt65/A2_1/issues/27)에 누적하고 [작업 기록](worklogs/issue-27.md)에 재현 증거를 보관한다.
