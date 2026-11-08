# Step 0 작업 기록

작업일: 2026-11-08 (Asia/Seoul).
전체 계획: [#2](https://github.com/ntt65/A2_1/issues/2). 현재 작업: [#3](https://github.com/ntt65/A2_1/issues/3).
브랜치: `codex/3-project-foundation`.
검토 요청: [PR #11](https://github.com/ntt65/A2_1/pull/11). 사용자 검토·병합 대기.

## 기초계획

알고리즘 구현에 앞서 개발환경과 작업 절차를 준비한다. 운영 규칙, 전체 계획, 수용 기준, 결정 기록,
이슈·PR 양식, 환경 의존성, CI와 환경 점검을 포함한다. Step 1~6 알고리즘 구현은 포함하지 않는다.

## 상세계획

- 설치된 Python 중 3.11을 기준으로 저장소 내부 `.venv`를 사용한다.
- 구현 의존성과 노트북 도구를 분리하고 설치·점검한 전체 버전을 고정한다.
- 수학 원문을 보존하고 모호한 조건과 잘못된 예시를 결정 기록으로 정정한다.
- 전체 이슈와 Step 0~7 이슈를 발행하고, 현재 작업만 전용 브랜치를 만든다.
- 환경 import, 화면 없는 PNG 렌더링, 현재 Python으로 노트북 커널 실행을 확인한다.
- Windows/Linux에서 같은 설치·점검 명령을 실행하는 CI를 구성한다.

## 실행 및 변경 기록

1. 기존 파일과 Git 상태를 확인했다. `main`의 `b1d2a32`는 사용자의 미션 원문 커밋이며 원격보다 한 커밋 앞서 있었다.
   원문 Markdown과 PDF를 보존하여 작업 브랜치에 포함한다. 원격 `main`에 직접 push하지 않는다.
2. 기존 전체 구현 초안 PR #1을 확인했다. 후속 단계에서 필요한 내용을 검토하도록 기록했다.
3. 전체 계획 #2, 환경 준비 #3, 구현·통합 #4~#10을 발행했다. 각 이슈에 3단계 계획·체크리스트·수용 기준을 넣었다.
4. 브랜치를 만들고 운영 규칙, 계획, 프롬프트, 수학 수용 기준, 결정 기록, 템플릿을 작성했다.
5. 독립 문서 검토에서 현재 단계의 영구 고정과 재개 시 브랜치 중복 생성 가능성을 발견하여 수정했다.
6. 원문의 GD 학습률, SVD 크기/rank, 역전파 참고 수치, Gaussian 표기 등을 확인하고 근거를 기록했다.

## 검증 결과

수학 테스트는 현재 미구현·미실행이다. 아래는 Step 0 환경 검사의 실제 결과다.

| 검사 | 결과 |
|---|---|
| Python/의존성 설치 | Python 3.11.4, NumPy 2.2.6, Matplotlib 3.10.3, Seaborn 0.13.2 |
| 환경 import·PNG·노트북 실행 | unittest 3개 통과(4.195초), 현재 가상환경 Python 커널 사용 |
| 노트북 도구 | nbformat 5.10.4, nbclient 0.10.2, ipykernel 6.29.5 |
| 의존성 일관성 | pip check 통과, 잠금 파일 기준 --no-index 재설치 확인 |
| 문서 링크·변경 내용 검토 | 문서 7개, 로컬 링크 20개 유효; 독립 운영 규칙 검토 및 보완 완료 |
| 원격 CI | 최초 게시 커밋의 Windows/Linux 검사 모두 통과([실행 기록](https://github.com/ntt65/A2_1/actions/runs/42430541456)); 최종 커밋 상태는 이슈 #3과 PR #11의 Checks에 기록 |

실행 명령:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
git diff --check
```

## 보완 기록

- 브랜치 생성 시 로컬 Git 메타데이터 쓰기 제한으로 첫 시도가 실패했다. 승인된 실행으로 작업 브랜치 생성을 완료했다.
- 영구 규칙의 현재 단계 고정을 제거하고 지정 이슈·전체 계획을 기준으로 판정하도록 수정했다.
- 동일 이슈 재개 시 기존 브랜치·PR을 재사용하도록 규칙과 프롬프트를 보완했다.
- 원문의 수학적 충돌은 `docs/DECISIONS.md`에 D07~D13으로 설명했다.
- 첫 노트북 검사는 제한된 실행 환경에서 커널 시작이 지연되어 중단했다. 승인된 일반 실행 환경에서 같은 명령을 재실행하여 3개 테스트 모두 통과했다. 테스트를 생략하거나 약화하지 않았다.
- 새로 추가한 설정 파일의 마지막 빈 줄 7곳을 staged diff 검사에서 발견해 정리하고 재검사했다.

## 커밋·원격 게시·upstream

- 로컬 커밋 `cf988c2`를 만든 뒤 Git push를 시도했으나 인증 입력을 요청하며 지연되었다.
  비대화형 연결 검사에서 원격 읽기는 성공했고 push는 `unable to get password from user`로 실패했다.
- 연결된 GitHub 앱의 Git 데이터 API로 같은 파일을 게시했다. 최초 원격 커밋은 `c7bae819d94e074feb61727aba78b08f8430f025`다.
- 로컬/최초 원격의 tree SHA는 모두 `3a06d34d4389d5d5498b88a5b2ff3ccd0af3812b`로 같았다.
  원문 Markdown/PDF를 포함해 파일 내용 전체가 일치하며 커밋 메타데이터·이력만 다르다.
- 원래 로컬 작업 이력은 `codex/3-project-foundation-local-preserved`에 보존했다. 사용자의 `main`과 원문 커밋 `b1d2a32`도 유지했다.
- 현재 작업 브랜치를 원격에서 다시 연결하여 upstream을 `origin/codex/3-project-foundation`으로 설정했다.
- 이 기록을 반영하는 후속 문서 커밋 역시 GitHub 앱으로 게시하며, 최종 커밋·CI 결과는 이슈 #3에 기록한다.
- upstream 연결과 Git push 인증은 별개다. 로컬 Git 인증은 이번 작업에서 계정 설정을 변경하지 않았으며, 후속 원격 게시에도 연결된 GitHub 앱을 사용할 수 있다.
- Git Credential Manager가 이미 시스템/사용자 Git 설정에 등록되어 있음을 확인했다. 따라서 인증 도구 미설치로 단정하지 않으며, 이번 에이전트 실행에서 대화형 인증을 완료하지 못한 것으로 기록한다. 토큰을 채팅이나 프로젝트 파일로 전달받지 않는다. 기존 브라우저 로그인과 운영체제 자격 증명 저장을 사용한다([GitHub 안내](https://docs.github.com/en/get-started/git-basics/caching-your-github-credentials-in-git)).

## 인덱스와 전달 상태

Step 0에는 수학 모듈이 없으므로 수식/텐서 shape와 레지스트리 구현은 해당하지 않는다.
`.agent/registry.json`의 스키마·생성·갱신·검사는 Step 1(#4)에 포함했다.
커밋·push·PR 및 원격 검증의 최신 링크는 이슈 #3에 누적 기록한다.
사용자 검토·병합 후 별도 지시에 따라 Step 1을 시작한다.
