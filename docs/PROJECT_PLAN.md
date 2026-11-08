# 전체 계획과 체크리스트

기준일: 2026-11-08 (Asia/Seoul). 전체 추적: [#2](https://github.com/ntt65/A2_1/issues/2).
목적은 AI 학습의 수학을 직접 유도하고 NumPy로 검증하는 것이다. 최종 제출 진입점은 README다.

## 진행 순서

전체 계획 → 작업 이슈 → 작업 시작 시 브랜치 → 기초계획 → 상세계획 → 실행계획 → 구현 → 점검 → 보완·재검증 → commit/push → PR → 이슈 기록 → 사용자 검토·병합.

| 단계 | 이슈 | 산출물 | 선행 단계 | 현재 상태 |
|---|---|---|---|---|
| 0 환경·운영 준비 | [#3](https://github.com/ntt65/A2_1/issues/3) | 환경, AGENTS, 계획, 템플릿, CI | 없음 | 로컬 검증 완료, [PR #11](https://github.com/ntt65/A2_1/pull/11) 검토 대기 |
| 1 공통 기반 | [#4](https://github.com/ntt65/A2_1/issues/4) | utils, cache, registry | 0 | 구현·로컬 48개 검증 완료, [PR #12](https://github.com/ntt65/A2_1/pull/12) 검토 대기(선행 #11) |
| 2 선형대수 | [#5](https://github.com/ntt65/A2_1/issues/5) | linear_algebra.py, 검증, PNG | 1 | 구현·로컬 87개 검사 완료(선행 PR #12) |
| 3 미적분 | [#6](https://github.com/ntt65/A2_1/issues/6) | calculus.py, 검증, PNG | 2 | 계획됨 |
| 4 최적화 | [#7](https://github.com/ntt65/A2_1/issues/7) | optimizer.py, 검증, PNG | 3 | 계획됨 |
| 5 역전파 | [#8](https://github.com/ntt65/A2_1/issues/8) | backprop_derivation.ipynb | 4 | 계획됨 |
| 6 확률·손실 | [#9](https://github.com/ntt65/A2_1/issues/9) | probability_loss.ipynb | 5 | 계획됨 |
| 7 통합·제출 | [#10](https://github.com/ntt65/A2_1/issues/10) | 전체 검증, 결과, README | 1~6 | 계획됨 |

후속 이슈는 미리 발행하되 브랜치는 시작할 때 최신 기준 브랜치에서 만든다. 현재 작업 브랜치는 `codex/5-linear-algebra`이다.
중단된 동일 이슈를 재개할 때는 기존 브랜치·PR과 기록을 재사용한다.
사용자의 순차 진행 지시에 따라 Step 1을 시작했다. 선행 환경 PR #11은 검증 완료·검토 대기이므로,
해당 브랜치의 완료된 변경 위에서 Step 1을 구현한다. 단계별 변경을 분리한 PR로 검토하고 #11 병합 후 Step 1의 기준 브랜치를 main으로 전환한다.
이후 사용자 시작 지시에 따라 검증 완료된 Step 1 위에서 Step 2를 진행한다. 검토 순서는 #11→#12→Step 2 PR이며 선행 병합 후 각 PR의 기준 브랜치를 main으로 전환한다.
기존 [PR #1](https://github.com/ntt65/A2_1/pull/1)은 전체 구현 초안이며 후속 작업에서 필요한 내용을 검토한다. 존재만으로 검증·완료로 판정하지 않는다.

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

최종 목표 구조(현재 생성 여부와 다름):

```text
A2_1/
├── AGENTS.md
├── .agent/registry.json              # Step 1
├── .github/ISSUE_TEMPLATE/task.md
├── .github/pull_request_template.md
├── .github/workflows/ci.yml
├── docs/                            # 계획·검증·결정·작업별 기록
├── src/
│   ├── utils/                       # Step 1
│   ├── linear_algebra.py            # Step 2
│   ├── calculus.py                  # Step 3
│   └── optimizer.py                 # Step 4
├── tests/                           # 현재 환경 점검, 이후 수학 테스트
├── notebooks/                       # Step 5~6
├── outputs/                         # PNG, 실행 시 생성
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
- [ ] 사용자 검토 및 병합

## 전체 완료 체크리스트

- [ ] Step 0 검토·병합
- [ ] Step 1 공통 유틸·캐시·레지스트리 검증·병합
- [ ] Step 2 선형대수 수치·시각 검증·병합
- [ ] Step 3 수치 미분·Gradient 검증·병합
- [ ] Step 4 수렴/진동/발산과 Momentum 비교 검증·병합
- [ ] Step 5 손계산·역전파·노트북 검증·병합
- [ ] Step 6 분포·MLE·노트북 검증·병합
- [ ] Step 7 전체 재현·제출 README 검증·병합

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
- [ ] 선행 PR 병합 확인 후 Step 1 검토·병합

## 실행계획: Step 2

- [x] 이슈 #5의 기초·상세·실행계획과 선행 상태 확인
- [x] 원격/로컬 `codex/5-linear-algebra` 브랜치·upstream 생성
- [x] 변환·면적, Power Iteration, SVD 및 독립 단위 테스트 구현
- [x] 합성 데이터·재현 스크립트·PNG/JSON 결과 생성
- [x] 독립 코드·수학·시각 검토와 회귀 보완
- [x] 전체 87개 테스트·환경·레지스트리 검사
- [x] API 문서·실측 결과·수용 기준·색인 갱신
- [ ] 원격 커밋·PR 게시 및 CI·이슈 기록 완료
- [ ] 선행 PR 및 Step 2 검토·병합
