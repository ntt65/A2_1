# Step 7 통합·제출 작업 기록

2026-11-08, [이슈 #10](https://github.com/ntt65/A2_1/issues/10), 브랜치 `codex/10-integration-delivery`, 선행 PR #17의 검증 완료 커밋 `669a61b` 기반.

## 기초계획

미션의 필수 결과물을 모두 포함한 누적 브랜치에서 설치·실험·노트북·검사를 다시 수행하고, README를 평가자가 그대로 실행할 수 있는 제출 진입점으로 정리한다.
사용자는 마지막까지 자동 진행하되 단계·세션 종료 보고를 요구했고, 특히 최종 README의 실제 실행 예제를 철저히 재현하도록 요청했다.
선택 보너스는 범위에 포함하지 않으며 구현 완료와 PR 검토·병합 대기를 구분한다.

## 상세·실행계획

1. #10의 계획을 갱신하고 전용 브랜치와 upstream을 만든다.
2. 기존 환경과 분리한 새 Python 3.11 가상환경에 잠금 의존성을 설치한다.
3. 전체 테스트와 색인을 검사하고, 캐시 없이 세 실험 및 두 노트북을 빈 결과 폴더에 재실행한다.
4. PNG·JSON 결과와 미션의 수식·shape·오차 기준을 대조한다. 금지 라이브러리·eig·seed·docstring·원문 보존을 독립 감사한다.
5. README의 실제 명령·출력·결과표·그림·조건 정정·제출 URL을 검토하고 문서 링크·체크리스트를 갱신한다.
6. 원격 커밋·PR·Windows/Linux CI와 이슈 기록을 마친 후 단계 종료를 보고한다. 자동 병합하지 않는다.

## 새 환경 재현 결과

새 경로 `.cache/integration-python311-20261108`와 `.cache/integration-artifacts-20261108`가 존재하지 않는 것을 확인한 뒤 생성했다.
Python **3.11.4**, NumPy **2.2.6**, Matplotlib **3.10.3**, Seaborn **0.13.2**, nbformat **5.10.4**, nbclient **0.10.2**, ipykernel **6.29.5**.
잠금 파일 설치와 `pip check` 성공. 패키지 다운로드 캐시는 사용할 수 있지만 계산 캐시는 우회했다.

아래는 저장소 루트에서 실제 실행한 명령이다. `$verificationPython`은 새 환경의 Python 경로다.

```powershell
.\.venv\Scripts\python.exe -m venv .cache/integration-python311-20261108
$verificationPython = '.\.cache\integration-python311-20261108\Scripts\python.exe'
& $verificationPython -m pip install --disable-pip-version-check -r requirements-lock.txt
& $verificationPython -m pip check
& $verificationPython -m unittest discover -s tests -v
& $verificationPython scripts/update_registry.py --check
& $verificationPython -m scripts.run_linear_algebra --no-cache --output-dir .cache/integration-artifacts-20261108
& $verificationPython -m scripts.run_calculus --output-dir .cache/integration-artifacts-20261108
& $verificationPython -m scripts.run_optimizer --no-cache --output-dir .cache/integration-artifacts-20261108
& $verificationPython -m scripts.run_notebooks --output-dir .cache/integration-artifacts-20261108
```

- **129개 테스트, 20.417초, OK**. 기존 단계의 실패를 숨기는 허용오차 완화나 테스트 삭제는 없었다.
- 역전파 **8개**, 확률 **9개** 코드 셀을 각각 새 커널에서 실행했다. 원본 노트북은 변경하지 않았다.
- 출력 **PNG 10개 + JSON 5개** 모두 기존 `outputs/`와 바이트 단위로 일치했다. PNG 서명·크기도 확인했다.
- 같은 OS·고정 라이브러리 환경에서의 바이트 일치 결과이며 다른 OS의 글꼴 렌더링까지 동일하다고 보장하는 것은 아니다. 수학 검증은 명시한 오차 기준으로 수행한다.
- [산출물 SHA-256·환경 증거](../verification/step-7-artifacts.json). 각 단계에서 직접 확인한 PNG와 바이트가 같으므로 시각 검토 결과도 그대로 적용된다.

실제 콘솔 수치 요약:

```text
Power Iteration: lambda=4.618033988750, relative error=1.923e-16, residual=6.400e-11
Derivative error: 3.931e-11; gradient error: 3.296e-11; orthogonality: 0.000e+00
circle: GD radius=1.440402e-09, loss=2.074758e-18; Momentum radius=2.016252e-02, loss=4.065273e-04
ellipse: GD radius=1.328070e-04, loss=1.763770e-08; Momentum radius=1.987508e-02, loss=8.051093e-04
Executed backprop_derivation.ipynb: 8 code cells passed; source unchanged
Executed probability_loss.ipynb: 9 code cells passed; source unchanged
```

각 실험은 위 요약 다음에 실제 저장 경로도 출력한다. 이 실행에서는 `.cache\integration-artifacts-20261108`이었다.

## 독립 감사와 보완

- 레지스트리 **33파일·289정의·55개념**과 모든 원시 소스 해시가 일치한다. 두 노트북의 셀 ID·제목·개념과 순차 실행 횟수·오류 출력 없음도 확인했다.
- 모든 함수·클래스에 docstring이 있다. 금지 프레임워크 import가 없다. `np.linalg.eig` 호출은 `tests/test_power_iteration.py`의 비교 검증 한 곳뿐이다.
- 세 실험 스크립트와 두 노트북에서 공통 seed 42를 진입점에 설정한다. 수학 알고리즘은 NumPy, 그림은 Matplotlib, 파일·검사는 표준 라이브러리와 개발 도구를 사용한다.
- 미션의 필수 3개 모듈·2개 노트북과 10개 PNG·5개 측정 JSON이 모두 있다. 지정 수식·shape·고정 입력·오차 기준은 단계별 검증과 연결했다.
- 원문 Markdown의 Git blob은 최초 사용자 커밋 `b1d2a32`와 같은 `6d09517395eb9fec871a754f2a4120afc663fee0`이다. Windows 작업 파일의 CRLF를 제외한 내용은 같다.
- 원문 PDF의 blob은 `a700621c0b5e3b10af82ff7c71b25e0427572191`이며 722631바이트 원본도 동일하다.
- README의 가상환경 실행 명령 누락, 닫힌 초안 PR #1을 재사용 후보로 설명하던 문구, 다음 단계 승인 대기 문구를 현재 지시에 맞게 정리했다.
- README에 SVD 크기/rank 충돌, 학습률 안정 범위, Momentum의 설정별 차이, 역전파 참고 수치 정정을 근거와 함께 연결한다.
- 새 환경 검증에서 새로운 실행 실패는 없었다. 기존 Matplotlib/pyparsing 사용 중단 예정 경고와 Windows 커널 이벤트 루프 경고가 있지만 모든 검증 종료코드는 0이다.

최종 README 독립 검토에서 수치·명령·고정 입력·조건 정정이 실제 소스/JSON과 일치함을 확인했다. 20개 문서의 로컬 링크 130개에 누락이 없고 33개 소스의 원시 해시가 색인과 일치했다.

## 전달 상태

최종 PR·커밋·원격 CI·문서 링크 검사 결과는 이슈 #10과 전체 추적 #2에 누적 기록한다.
구현 결과 전체는 [최종 누적 브랜치](https://github.com/ntt65/A2_1/tree/codex/10-integration-delivery)에 있다.
기존 초안 PR #1은 닫힌 미병합 상태이며, 단계별 검증 PR로 대체했다. PR #11부터 순서대로 검토·병합해야 기본 브랜치에 전체 결과가 반영된다.
