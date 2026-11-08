# 이슈 #31 — README 상세 보고서·실제 실행 증거·평가 Q&A

작업일 2026-11-09(KST). [이슈 #31](https://github.com/ntt65/A2_1/issues/31), 브랜치 `codex/31-readable-experiment-results`, 기준 main `58921cafc5a285027cc4cf7a5ee36628d262c2e6`. 구현·로컬 검증 완료 후 커밋·PR·CI·사용자 승인 main 병합 결과를 이슈에 누적한다. 원격 게시 전 이 문서가 PR/병합 성공을 미리 주장하지 않는다.

## 문제와 범위

외부 평가는 14/15 → 9/15 → 6/15로 하락했고, 최신 2026-11-09 06:05:14 결과는 실제 실험 결과와 유도가 제출되지 않았다고 지적했다. 화면에서 PNG·PDF·Office 제외와 Markdown 권장을 확인했다. 알고리즘·그림이 존재한다는 사실만으로 평가기에 실행 증거가 전달됐다고 판단할 수 없었다. 실제 입력 목록·대상 SHA·길이 제한은 확인하지 못했다.

사용자는 README가 논문처럼 독립적으로 내용을 설명해야 하며, 실제 실행 화면도 가능한 한 텍스트로 옮기고 그 뒤에 링크를 붙이라고 요청했다. 따라서 링크 위주의 안내에서 목적·수식·실험 조건·실측·검증·그림 해석이 연결된 상세 본문으로 보강했다. 기본 미션을 앞에, 보너스를 마지막에 유지한다. 외부 기회 세 번이 소진되어 사전평가나 정식 제출 버튼은 누르지 않았다.

## 변경

- 순수 보고서 렌더러 3개와 재실행 도구 1개, 관련 테스트 4개를 추가했다. 기존 수학 알고리즘·잠금 의존성·원문은 변경하지 않았다.
- 세 실험과 두 노트북을 새 커널로 임시 출력에 실행해, 기존 JSON과 구조·수치(atol=rtol=1e-12)가 일치한 다음 다섯 Markdown 보고서와 README의 지정 본문을 생성한다. 기존 실험의 독립 기대값 assertion과 수학 허용치는 그대로 사용한다.
- 실제 원시 경로 `(101,2)`·학습률 경로 `(21,2)`, 역전파 13개 비교·9매개변수 중심차분, 분포·합오차·MLE 결과를 표로 기록했다. 수식은 원본 노트북의 안정된 셀 ID를 사용해 가져온다.
- 역전파 2개·확률 6개 선택 셀의 실제 stdout을 그대로 기록하고 선택 범위를 밝힌다. HTML·오류·잘못된 출력 구조는 거부한다. 기존 PNG 10개를 본문에서 볼 수 있게 하고 보너스·민감도까지 PNG 15개의 설명·경로를 유지했다.
- README의 보너스에는 Adam 갱신식·200회 비교, Newton 유도·197/1회 비교, 정보 이론 식·독립 값·경계·곡선 측정을 직접 기록했다.
- `docs/EVALUATION_QA.md`에 기본15항목·보너스·공통 용어의 사전형 설명, 수식 전개, 실측, 검증, 오해·반례와 꼬리질문을 수록했다.
- `Naito_AI_eval_result.md`에 최신40%와 세 원문을 보존하고, A1_1을 읽기 전용 비교한 `a1-comparison.md`에 18→17→19 과정과 평가 전 README 보완·만점 후 후속 구현을 구분했다.

## 실제 실행·검증

환경은 Windows, Python 3.11.4, NumPy 2.2.6, Matplotlib 3.10.3이며 기준 잠금 의존성을 사용했다. 각 독립 실험은 기존 공통 함수로 seed42를 한 번 설정하고 float64를 사용한다.

```text
python -m scripts.run_submission_reports --update-readme
Published 5 verified Markdown reports; original PNG/JSON/notebooks unchanged.

python -m unittest discover -s tests -v
Ran 209 tests in 36.917s
OK

python -m unittest tests.test_linear_calculus_reports -v
Ran 6 tests in 0.243s
OK

python -m pip check
No broken requirements found.

python scripts/update_registry.py --update
Registry updated: 57 files, 78 concepts.

python scripts/update_registry.py --check
Registry check passed.
```

전체209개는 새 실험·새커널, 측정값 변조·shape·비유한 수치·마커 실패·게시 중단·원본 보존까지 검사했다. 이후 변경한 Power Iteration 설명 문구는 관련6개 검사와 실제 보고서 재생성으로 확인했다. 최종 원격 커밋은 Windows/Linux CI에서 다시 전체 검증한다. Windows 노트북 실행의 Proactor/ZMQ 호환 경고와 기존 Matplotlib/Pyparsing 경고는 실행 실패가 아니며 성공 결과와 분리했다.

실행 화면 요청에 따라 아래 CLI도 별도 폴더에서 실제 실행하고 stdout을 README에 그대로 옮겼다. 노트북은 8셀·9셀을 각각 새 커널에서 완료했다. `.cache/`의 임시 결과·로그는 제출 산출물과 분리한다.

```text
python -m scripts.run_linear_algebra --no-cache --output-dir .cache/issue31-cli-results
python -m scripts.run_calculus --output-dir .cache/issue31-cli-results
python -m scripts.run_optimizer --no-cache --output-dir .cache/issue31-cli-results
python -m scripts.run_notebooks --output-dir .cache/issue31-cli-results
```

## 발견한 문제와 보완

1. 최초 통합 테스트가 한국어 본문의 네 자리 비교를 영문 `four`로 찾는 오류로 실패했다. 실제 본문의 의미 있는 표 제목을 검사하도록 수정했다. 실측·허용치 검사는 유지했다.
2. 초기 보고서의 줄 수 상한 검사가 사용자의 상세 보고서 요구와 충돌했다. 상한을 수식·학습 설명·미션 번호·기존 그림 링크와 인라인의 존재 검사로 대체했다. 수학적 실패를 숨기기 위해 수치 검사를 삭제한 것이 아니다.
3. 최초 `__covers__`에 개념명, 이어 심볼 식별자를 넣어 레지스트리가 거부했다. 문서화된 파일 경로 규격으로 바로잡고 정방향·역방향을 재생성했다.
4. README를 다른 경로에 생성할 때 링크 기준이 루트로 고정되는 문제를 검토에서 발견했다. 지정 README의 부모 경로를 사용하도록 보완하고 통합 검증했다.
5. 독립 문서 검토에서 Power Iteration 반복식과 Newton의 Taylor 근사가 부족함을 찾아 본문에 추가했다. 표의 norm 구분 기호, 정보 이론 JSON 파일명과 유한499개 표본 범위를 정리했다.

## 보존·문서 감사와 남은 상태

### 원격 Linux 검증에서 발견한 재현 비교 문제와 수정

첫 게시 커밋 `0c23e35`의 Linux CI는209개 중 새 통합 검사1개에서 실패했다. `calculus.gradient.numerical[1][0]`의 Linux값2.8284271247347443과 저장된 Windows값2.8284271247791533의 차이는 약4.44e-11이었다. 실제 해석값 대비1e-8 기준은 만족했지만, 새 도구가 모든 수치를1e-12로 저장값과 비교한 것이 잘못된 재현 조건이었다.

전역 허용오차를 확대하지 않고 검증 대상을 분리했다. 구조·입력·h·해석값·직접 계산값·메타데이터는 기존 엄격한 비교를 유지한다. 명시한 중심차분 값과 그 오차 요약만 저장값의 끝자리 비교에서 제외하고, 독립 해석값과 원래1e-4/1e-8/1e-6/1e-8 정확도 기준으로 다시 검사한다. shape·유한성 및 요약의 배열 일관성도 검사한다. 단순 비교 제외가 되지 않도록 61개 h 탐색에는 정확 유한차분식 `6`, `cos(3)*sin(h)/h`와 고정 함수의 반올림 예산 `8*eps*scale/h + 8*eps*abs(exact)`을 추가했다. scale은 x²에서 `max(3,(3+h)²)`, sin에서3으로 입력과 함수값의 크기를 덮는다. 매우 작은 h에서 큰 예산이 필요한 것은 그 근사가 부정확할 수 있음을 뜻하며 통과 표본 수를 고정하지 않는다.

관측된 Linux 차이를 그대로 넣은 재현 사례, gradient에1e-6 주입, h 탐색에.001 주입, 요약·h·해석gradient 변조, NaN·shape 오류 검출을 추가했다. Windows 결과의 숫자 문자열을 요구하던 통합 검사는 해당 실측표의 존재와 독립 수치 기준 검사로 바꿨다. README·보고서 헤더도 실제 검증 범위를 정확히 설명하도록 수정했다.

```text
python -m unittest discover -s tests -v
Ran 210 tests in 30.826s
OK

# h 탐색 독립 검산을 추가한 뒤 관련 새 커널 통합 검사
python -m unittest tests.test_submission_reports -v
Ran 6 tests in 8.358s
OK
```

첫 CI 실패를 숨기지 않으며 수정한 최종 head의 Windows/Linux CI를 모두 확인한 뒤 병합한다. 위209개 기록은 첫 로컬 검증,210개 기록은 플랫폼 보완 후 검증이다.

시작 시 SHA-256을 기록한 기존 outputs30개(PNG15·JSON10·Markdown5)와 노트북2개의 최종 해시가 모두 같았다. README의 기존 고유 링크93개가 모두 남고 PNG15개 경로도 존재함을 확인했다. README·Q&A·보고서·작업기록 등13문서의 로컬 링크와 코드 블록 경계 검사에서 누락이 없었다. 수식·조건·실측은 서로 독립적인 담당자가 검토했다. 새 레지스트리는 기존49파일/72개념에서57파일/78개념으로 갱신했다.

완료된 내부 검증은 외부40%를 바꾸지 않는다. 최신 외부 결과와 원문은 평가 문서에 남기고, 커밋·PR·정확한 최종 head CI·main 병합·동기화 결과는 이슈 #31에 기록한다. 사용자 요청의 발표자료 보강은 README 마무리 뒤 대상 확인을 거쳐 별도 작업으로 진행한다.
