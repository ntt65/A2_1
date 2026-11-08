# Step 6 작업 기록

2026-11-08, [이슈 #9](https://github.com/ntt65/A2_1/issues/9), 브랜치 `codex/9-probability-loss`, 선행 PR #16의 `043ffcd` 기반.

## 기초·상세·실행계획

정규·Bernoulli 분포와 Gaussian MLE→MSE, Bernoulli→BCE, categorical→CE를 가정·수식·독립 기대값으로 검증한다.
분산 표기, 합/평균, 자연로그, 경계 확률, 안정적인 logit 계산을 명시한다. 새 커널 실행 도구와 공통 Softmax/미분/seed를 재사용한다.
노트북 → 독립 리뷰 → 새 커널 실행 → 그림 점검 → 전체 검사 → 문서·색인 → 원격 게시·PR·CI 순서로 진행했다.

## 구현·검증 결과

- 노트북 19개 셀 중 코드 9개를 새 커널에서 실행해 저장했다. 함수 8개에 docstring, 고정 셀 ID·제목·개념을 제공한다.
- 정규분포 grid `(10001,)`, Bernoulli PMF `(2,)`, Softmax `(2,3)` 및 축별 정규화, Gaussian 잔차 `(3,)`, BCE `(2,)`, categorical `(3,)`를 검증했다.
- 전체 **129개 테스트 통과(18.540초)**. pip check와 레지스트리 검사 통과. 색인 **33파일·55개념**.
- 정규분포 적분 오차: `1.1102230246251565e-15`, `9.992007221626409e-16` (기준 `1e-4`).
- Softmax 합 오차 `1.1102230246251565e-16` (기준 `1e-6`). Gaussian NLL 항등식 오차 `4.440892098500626e-16`.
- BCE 중심차분 기울기 오차 `8.626654945942391e-12` (기준 `1e-6`). 독립 검토의 categorical 기울기 오차 `3.28e-12`.
- PNG 2개의 축·범례·분산 표기·확률 질량을 직접 확인했다. 본문에 이미지를 중복 내장하지 않고 저장 PNG를 연결한다.
- 큰 양수에서 작은 BCE가 사라지는 뺄셈을 줄이기 위해 이진 표적의 동치식 `logaddexp(0,(1-2y)*z)`를 사용한다. `z=40,y=1`에서 `4.248e-18`을 보존한다.
- 수학·구현의 차단 오류는 발견하지 않았다. 색인 갱신 명령에서 필수 `--update`를 누락한 호출은 실패했고 올바른 명령으로 갱신·검사를 완료했다.
- 기존 Matplotlib 사용 중단 예정 경고와 Windows 커널 이벤트 루프 경고는 있었으나 검사 종료코드는 0이었다.

## 재현

```powershell
.\.venv\Scripts\python.exe -m scripts.run_notebooks notebooks/probability_loss.ipynb
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/update_registry.py --check
```

환경 Python 3.11.4 / NumPy 2.2.6. [수식·그림](../PROBABILITY_LOSS.md), [원시 측정값](../../outputs/probability_metrics.json).
최종 커밋·PR·Windows/Linux CI는 이슈 #9에 기록한다. 구현 완료와 검토·병합 대기는 구분한다.
