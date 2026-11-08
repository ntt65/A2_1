# Step 5 작업 기록

2026-11-08, [이슈 #8](https://github.com/ntt65/A2_1/issues/8), 브랜치 `codex/8-backprop-derivation`, 선행 PR #15의 `c653d59` 기반.

## 기초계획

원문 고정 2→2→1 신경망의 순전파·역전파를 손계산으로 유도하고 NumPy와 독립 수치 미분으로 교차 검증한다.

## 상세계획

고정 입력·W1@x 곱 방향·shape, 5개 순전파와 필수6개 역전파+bias 총13개 체크포인트를 노트북에 기록한다.
독립 고정 상수와 atol5e-5/rtol0 및 네자리 표시를 비교하고 전체9개 매개변수를 중심차분atol1e-6으로 검사한다.
공통sigmoid/수치gradient를 재사용하며, 새커널 실행 도구와 source 보존·실패시저장방지 검사를 추가한다.

## 실행·검토·보완

1. 이슈3단계 계획과 전용브랜치/upstream을 만들었다.
2. 노트북·실행 도구를 분담하고 독립 고정밀 수학 리뷰를 수행했다.
3. 네자리 표시에서 반올림0의 부호만 정규화하고 원래 gradient를 보존했다.
4. 기존출력/실행횟수를 지운 새커널, 현재Python, 타이밍metadata제거, 성공한 명시적 --write에서만 저장하도록 했다.
5. 노트북8개코드셀을 순서대로 실행해 저장했고 실행도구3개 회귀 검사를 통과했다.
6. 전체 **128개 테스트 통과(14.098초)**, pip check·registry check 통과. 색인 **31파일·279심볼·46개념**.
7. 최종 커밋·PR·Windows/Linux CI는 이슈 #8에 기록한다. 다음 Step6으로 자동 진행하되 검토·병합 상태는 구분한다.

## 실측

- 독립 손계산13개 체크포인트 최대오차 **2.220446049250313e-16**, 네자리 표시 모두일치.
- 전체9개 파라미터 수치gradient 최대오차 **7.842670957103337e-12 ≤1e-6**.
- 별도 Decimal70자리 중심차분 검토와 해석gradient 차이 **5.55e-17**.
- 원문의 z2=.6071/은닉기울기-.0513 참고값은 유도에 따라 .6072/-.0517로 표시했다.
- 새로운 차단 결함은 없었다. Windows Proactor경고와 한 추가실행의 libzmq 종료메시지가 있었으나 assertion과 실행 종료코드는 성공이었다. 기존환경을 임의로변경하지 않았다.

## 재현

```powershell
.\.venv\Scripts\python.exe -m scripts.run_notebooks notebooks/backprop_derivation.ipynb
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/update_registry.py --check
```

환경 Python3.11.4/NumPy2.2.6. [수식·실행 안내](../BACKPROP.md), [실행 노트북](../../notebooks/backprop_derivation.ipynb), [원시 측정값](../../outputs/backprop_metrics.json).

## 원격 Windows 검사 보완

최초 커밋 `7d7e4b9`의 Ubuntu 검사는 성공했으나 Windows push/PR 검사에서 실행 도구의 출력 폴더 검사1개가 실패했다.
GitHub Windows의 `RUNNER~1` 축약 경로가 `resolve()`에서 긴 경로로 바뀌면서, 같은 폴더를 문자열 비교한 assertion이 실패했다. 수학 노트북 검사는 통과했다.
로컬에서도 `alias/../artifacts` 경로를 통해 동일한 문자열 비교 실패를 재현한 후 `Path.samefile()`로 실제 폴더 동일성을 검사하도록 고쳤다.
해당 별칭 경로를 두 번의 새 커널 실행에 사용하는 회귀 사례를 유지했고, 실행 도구 본체는 변경하지 않았다. 수정 후 전체128개 검사·색인 검사를 다시 통과했다.
최종 수정 커밋과 Windows/Linux CI 결과는 이슈 #8 및 PR #16에 기록한다.
