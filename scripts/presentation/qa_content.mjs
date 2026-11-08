/**
 * 저장소의 측정 JSON을 읽어 평가 질문·답변 발표 원고를 구성한다.
 * 파일·난수 상태·실험 배열을 변경하지 않으며 모든 장의 원본 근거를 보존한다.
 */

/** 유한한 스칼라 value를 digits 유효숫자로 표시하고 null은 미도달로 구분한다. */
function number(value, digits = 6) {
  if (value === null) return '미도달';	// 미도달과 실제 영 회차를 구분한다.
  if (value === '+inf') return '+∞';	// 표준 JSON의 무한대 표기를 수학 기호로 바꾼다.
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError('표시할 측정값은 유한 수치, null 또는 +inf여야 합니다.');	// 잘못된 측정값을 근거처럼 표시하지 않는다.
  }
  return Number(value.toPrecision(digits)).toString();	// 본문 표시만 반올림하고 입력값은 보존한다.
}

/** 스칼라 또는 중첩 배열 value의 구조를 유지하며 각 원소를 digits 자리로 표시한다. */
function vector(value, digits = 6) {
  return Array.isArray(value)
    ? '[' + value.map(item => vector(item, digits)).join(', ') + ']'
    : number(value, digits);	// 중첩 배열에서도 성분 순서와 괄호를 보존한다.
}

/** 노트에서 측정값 value의 원래 JSON 정밀도와 배열 구조를 보존한다. */
function exact(value) {
  return JSON.stringify(value);	// 전체 정밀도 값은 발표자 노트에 기록한다.
}

/** 역전파 원본에서 비교한 소수점 네 자리 표시를 생성하고 표시용 음의 영만 정리한다. */
function four(value) {
  if (Array.isArray(value)) return '[' + value.map(four).join(', ') + ']';	// 배열 성분을 원래 순서대로 표시한다.
  return (Math.abs(value) < 0.00005 ? 0 : value).toFixed(4);	// 계산값을 수정하지 않고 마지막 표시만 반올림한다.
}

/** 질문의 anchor와 원본 paths를 모아 슬라이드의 근거 위치 목록을 반환한다. */
function sources(anchor, ...paths) {
  return ['docs/EVALUATION_QA.md#' + anchor, ...paths];	// 문서 절과 원본 자료를 함께 연결한다.
}

/** 답변·유도·실측·꼬리질문·재현 명령·출처를 이어 상세 발표자 노트를 구성한다. */
function notes(answer, derivation, measured, followup, references, command) {
  return [
    '질문과 답변\n' + answer,
    '유도와 해석\n' + derivation,
    '실측과 검증\n' + measured,
    '꼬리 질문과 한계\n' + followup,
    '재현\n' + (command || '.venv/bin/python -m scripts.run_submission_reports --output-dir .cache/imac-demo'),
    '원본 근거\n' + references.join('\n'),
  ].join('\n\n');	// 설명 단락을 구분하고 출처를 노트 본문에도 남긴다.
}

/**
 * 여덟 원본 보고서 사전에서 편집 가능한 슬라이드 원고 31개를 반환한다.
 *
 * data의 키는 linear, calculus, backprop, optimizer, probability, bonus,
 * information, sensitivity다. 실측과 실험 설정은 이 사전에서 읽고 수학 정의와
 * 문서에 명시된 수용 기준은 구분한다. 누락·비유한 표시값은 예외로 거부한다.
 * 반환한 노트는 원본 정밀도를 유지하며 본문 수치는 표시용으로 반올림한다.
 * data의 값이나 배열을 수정하지 않고 외부 파일 및 난수 상태에도 접근하지 않는다.
 */
export function qaSlides(data) {
  for (const key of ['linear', 'calculus', 'backprop', 'optimizer', 'probability',
    'bonus', 'information', 'sensitivity']) {
    if (!data[key] || typeof data[key] !== 'object') throw new TypeError('보고서가 없습니다: ' + key);	// 필수 측정 사전 누락을 먼저 검출한다.
  }
  const { linear: la, calculus: ca, backprop: bp, optimizer: op,
    probability: pr, bonus: bo, information: it, sensitivity: se } = data;	// 전달받은 사전을 분야별 이름에 연결한다.
  const d = ca.derivative;	// 고정 중심차분의 입력과 결과다.
  const gradient = ca.gradient;	// 점·해석값·수치값·수직성 검증을 읽는다.
  const eigen = la.power_iteration;	// 수동 고유쌍과 종료 기록을 읽는다.
  const eig = eigen.numpy_eig_comparison;	// 검증 전용 eig 비교 결과를 읽는다.
  const circle = op.comparisons.circle;	// 원형 함수의 두 최적화 경로다.
  const ellipse = op.comparisons.ellipse;	// 타원형 함수의 두 최적화 경로다.
  const forward = bp.forward;	// 고정 신경망의 순전파 측정값이다.
  const backward = bp.backward;	// 고정 신경망의 역전파 측정값이다.
  const parameters = bp.parameters;	// 입력과 가중치를 같은 원본에서 읽는다.
  const hStudy = ca.step_size_study;	// 모든 간격 표본과 오차를 읽는다.
  const passing = hStudy.h.filter((_, index) => hStudy.quadratic_error[index] <= d.tolerance);	// 통과한 실제 표본만 센다.
  const smallestError = Math.min(...hStudy.quadratic_error);	// 저장된 표본의 관측 최소오차다.
  const minimizingH = hStudy.h.filter((_, index) => hStudy.quadratic_error[index] === smallestError);	// 같은 최소오차를 갖는 간격을 모두 남긴다.
  const rational = it.examples.find(item => item.name === 'rational_forward');	// 유리수 분포의 정방향 예제다.
  const reverse = it.examples.find(item => item.name === 'rational_reverse');	// 비대칭을 보여주는 역방향 예제다.
  const tiny = it.examples.find(item => item.name === 'smallest_positive_q');	// 극소 양의 확률 사례다.
  const slides = [];	// 선언된 순서대로 31장 원고를 모은다.
  let refs;	// 각 장의 원본 근거 목록을 보관한다.

  refs = sources('qa-common', 'A2_1_mission.md', 'docs/ACCEPTANCE.md', 'docs/DECISIONS.md');
  slides.push({
    id: 'common-cover', title: 'A2_1 평가 Q&A', kind: 'cover',
    lead: '기본 15개 질문과 보너스 3개를 수식·실측·반례로 설명합니다',
    lines: ['선형대수 · 미적분 · 역전파 · 최적화 · 확률',
      '고정 입력과 독립 기준을 먼저 확인합니다',
      '결과의 적용 범위와 실패 조건까지 답합니다'],
    interpretation: '본문의 수치는 읽기 좋게 반올림했고 발표자 노트에는 원본 정밀도를 남겼습니다.',
    sources: refs,
    notes: notes(
      '이 자료는 계산을 실제로 설명하기 위한 질문과 답변이다. 기본 미션의 열다섯 평가 항목을 빠짐없이 다루고, 선택 보너스인 Adam, Newton, 정보 이론을 뒤에서 구분한다. 파일이 있다는 설명에서 멈추지 않고 어떤 입력에 어떤 식을 적용했으며 결과를 어떤 독립 기준으로 확인했는지 답한다.',
      '설명 순서는 정의와 가정, 고정 입력과 shape, 계산식, 실제 측정값, 허용치, 반례다. 원문의 참고 예시는 정답으로 복사하지 않는다. 수학적으로 충돌하는 SVD 크기와 rank, 원형 GD의 학습률, 역전파 반올림 값은 원문을 보존한 상태로 결정 기록의 근거에 따라 설명한다.',
      '이 원고의 수학 측정은 주입된 기존 JSON에서 직접 읽는다. 공통 dtype은 ' + la.dtype + ', 독립 실험의 seed는 ' + la.seed + ', 저장 결과의 NumPy 버전은 ' + la.numpy_version + '이다. 자료의 원본 코드 기준은 main 2edda38이며 보너스 결과와 기본 결과의 반복 예산은 서로 구분한다.',
      '꼬리 질문은 “다른 입력에서도 항상 맞는가?”이다. 고정 예제의 검사 성공은 모든 입력의 보장이 아니다. 이 자료는 구현의 수학과 관측을 설명하며 외부 평가의 점수를 자체 판정으로 바꾸지 않는다. 후속 장의 반례를 답할 수 있어야 조건을 이해한 것이다.',
      refs),
  });

  refs = sources('qa-common', 'docs/ACCEPTANCE.md', 'src/utils/math.py', 'src/utils/random.py');
  slides.push({
    id: 'common-notation', title: 'shape와 오차는 무엇을 뜻하나요?', kind: 'formula',
    lead: '곱 방향·배열 크기·오차 기준을 함께 적어야 같은 계산을 비교할 수 있습니다',
    formula: ['scalar: ()   벡터: (2,)   행렬: (2,2)', 'P′ = A @ P : (2,2) @ (2,n) → (2,n)',
      '절대오차 = |관측 − 기대|', '근접 비교: |관측 − 기대| ≤ atol + rtol·|기대|'],
    lines: ['float64도 정확한 실수 연산은 아닙니다', '0 근처에서는 절대오차 기준이 필요합니다'],
    interpretation: '본문의 반올림 표시와 검증에 사용한 원래 수치를 구분합니다.',
    sources: refs,
    notes: notes(
      'shape는 숫자의 개수만이 아니라 연산의 의미를 지정한다. NumPy의 길이 이 벡터는 행 또는 열 축이 별도로 없는 일차원 배열이므로 행렬 곱의 방향을 명시한다. scalar는 빈 shape이며 단일 원소 벡터와 다르다. 모든 표의 허용치는 해당 측정의 절대 또는 상대오차 정의와 함께 읽는다.',
      '절대오차는 관측값과 기대값의 차이 크기다. 상대오차는 기준 크기로 나눈 값이므로 기준이 영이면 따로 처리해야 한다. atol과 rtol을 함께 쓰는 비교는 atol+rtol·|기대|를 한계로 삼는다. 행렬 Frobenius norm과 벡터 유클리드 norm을 구별하고, 여러 성분을 비교했다면 어떤 최대값을 보고했는지도 밝힌다.',
      '미적분 고정 예제의 dtype은 ' + ca.dtype + ', 수치미분 허용치는 ' + exact(d.tolerance) +
      '이다. gradient 허용치 ' + exact(gradient.tolerance) + '와 수직성 허용치 ' +
      exact(gradient.orthogonality_tolerance) + '는 서로 다른 양을 검사한다. 통합 재현의 값 대조 기준은 각 기능의 수학 수용 기준을 대체하지 않는다.',
      '꼬리 질문은 “두 값이 화면에서 같으면 검증도 통과하는가?”이다. 반올림한 표시는 작은 차이를 숨길 수 있다. 반대로 음의 영은 수학적으로 영이므로 표시의 부호만 다른 것을 계산 실패로 보지 않는다. 실패를 숨기기 위해 허용오차를 바꾸지 않고 먼저 입력과 비교 정의를 확인한다.',
      refs),
  });

  refs = sources('qa-01', 'src/linear_algebra.py', 'scripts/run_linear_algebra.py',
    'outputs/linear_transforms.png', 'outputs/linear_algebra_metrics.json');
  slides.push({
    id: 'qa-01-transform', title: '#1 행렬은 원을 어떻게 바꾸나요?', kind: 'image',
    lead: '열벡터 묶음에 왼쪽에서 행렬을 곱하면 각 점의 위치가 바뀝니다',
    formula: ['P′ = A @ P', '회전 R(π/4), 스케일 diag(2, 0.5), 전단 Sh(1)'],
    image: 'outputs/linear_transforms.png',
    lines: ['회전은 방향, 스케일은 축별 크기, 전단은 기울기를 바꿉니다',
      '세 변환 모두 이 설정에서는 면적을 유지합니다'],
    interpretation: '가로 2배와 세로 0.5배의 곱은 1이므로 타원도 원과 면적이 같습니다.',
    sources: refs,
    notes: notes(
      '단위 원 위 점을 열벡터로 묶은 P에 A를 왼쪽에서 곱한다. 기본 구현의 단위 원은 끝점을 중복하지 않은 400점이며 P의 shape은 (2,400), A는 (2,2)다. 그림은 회전, 스케일링, 전단의 원본과 결과를 각 패널에서 비교한다.',
      '회전 행렬은 [[cosθ,−sinθ],[sinθ,cosθ]]이고 θ=π/4다. 스케일 행렬은 [[2,0],[0,.5]], 수평 전단은 [[1,1],[0,1]]이다. 전단에서는 x′=x+y, y′=y가 된다. 이 행렬들은 실행 스크립트의 고정 입력이며 JSON에는 변환 행렬 자체 대신 측정 결과가 저장돼 있다.',
      '저장된 면적비는 회전 ' + exact(la.transforms.rotation.area_ratio) + ', 스케일링 ' +
      exact(la.transforms.scaling.area_ratio) + ', 전단 ' + exact(la.transforms.shear.area_ratio) +
      '이다. 행렬식과 독립 면적 계산의 검증은 다음 장에서 수치로 대조한다. 그림의 좌표축과 equal aspect를 함께 읽어 모양의 왜곡을 판단한다.',
      '꼬리 질문은 “전단도 면적을 유지하니 회전과 같은가?”이다. 면적이 같아도 길이와 각도가 보존되는 것은 아니다. 오른쪽 곱이나 전치 행렬로 바꾸면 다른 변환이 될 수 있다. 가로 배율만 보고 면적이 두 배라고 말하면 세로 배율을 누락한 것이다.',
      refs, '.venv/bin/python -m scripts.run_linear_algebra'),
  });

  refs = sources('qa-02', 'src/linear_algebra.py', 'outputs/linear_algebra_metrics.json',
    'outputs/linear_algebra_report.md', 'outputs/evaluation_evidence.json');
  slides.push({
    id: 'qa-02-area', title: '#2 행렬식과 면적비는 일치하나요?', kind: 'table',
    lead: '좌표로 계산한 면적비를 |det(A)|와 독립적으로 비교합니다',
    table: { headers: ['변환', 'det(A)', '면적비', '오차', '기준'],
      rows: Object.entries(la.transforms).map(([name, row]) => [
        name, number(row.determinant), number(row.area_ratio), number(row.error),
        (row.error_kind === 'relative' ? '상대 ≤' : '절대 ≤') + number(row.tolerance)]),
      weights: [1.1, 0.8, 1, 1.1, 1.5] },
    formula: ['면적비 r = Area(AP) / Area(P)', 'det≠0: |r−|det||/|det|   det=0: |r|'],
    interpretation: 'det가 음수이면 방향이 뒤집힙니다. 면적은 음수가 되지 않습니다.',
    sources: refs,
    notes: notes(
      '행렬식은 부호 있는 면적 배율이다. 실제 다각형의 면적은 신발끈 공식으로 계산하고, 전후 면적의 비율을 행렬식의 절댓값과 비교한다. 같은 determinant를 두 계산에 재사용하면 독립 검증이 아니므로 좌표 기반 면적을 따로 구한다.',
      '신발끈 공식은 1/2·|Σ(xᵢyᵢ₊₁−yᵢxᵢ₊₁)|이다. det≠0에서는 |r−|det||/|det|를 상대오차로 사용한다. det=0에서는 분모가 영이므로 면적비 자체의 절대오차를 사용한다. 유한 다각형의 원본 면적이 π와 정확히 같을 필요는 없다.',
      '원본 면적은 ' + exact(la.transforms.scaling.area_before) + '이다. 반사의 det=' +
      exact(la.transforms.reflection.determinant) + ', 면적비=' + exact(la.transforms.reflection.area_ratio) +
      ', 상대오차=' + exact(la.transforms.reflection.error) + '이다. 특이 행렬은 면적비 ' +
      exact(la.transforms.singular.area_ratio) + ', 절대오차 한계 ' + exact(la.transforms.singular.tolerance) +
      '를 사용한다. 저장된 통과 여부는 ' + exact(Object.fromEntries(Object.entries(la.transforms).map(([key, row]) => [key, row.passed]))) + '이다.',
      '꼬리 질문은 “det=−6이면 면적도 −6배인가?”이다. 방향이 반전되고 양의 면적이 여섯 배라는 뜻이다. 특이 행렬에 상대오차를 강제하면 영으로 나누게 된다. 추가 진단의 injected 값은 검출기를 시험하려고 고의로 바꾼 값이며 실제 알고리즘 실패와 구분한다.',
      refs, '.venv/bin/python -m scripts.run_linear_algebra'),
  });

  refs = sources('qa-03', 'src/linear_algebra.py', 'scripts/verify_power_iteration.py',
    'outputs/power_iteration_eig_comparison.json', 'outputs/power_iteration_eig_comparison.md');
  slides.push({
    id: 'qa-03-power', title: '#3 Power Iteration은 무엇을 찾나요?', kind: 'formula',
    lead: '절댓값이 가장 큰 고유값을 목표로 반복하고 eig와 독립 기준으로 확인합니다',
    formula: ['vₜ₊₁ = Avₜ / ‖Avₜ‖₂', 'λ = vᵀAv / (vᵀv)',
      'A = ' + vector(eigen.matrix) + '   v₀ = ' + vector(eigen.initial),
      '독립 기준 λ = (7 + √5) / 2'],
    table: { headers: ['검사', '측정', '기준'], rows: [
      ['수동 λ / eig λ', number(eigen.eigenvalue, 12) + ' / ' + number(eig.numpy_eigenvalue, 12), '비교값'],
      ['고유값 상대오차', number(eig.relative_eigenvalue_error), '≤ ' + number(eig.relative_eigenvalue_tolerance)],
      ['상대 잔차', number(eigen.relative_residual), '≤ ' + number(eig.relative_residual_tolerance)],
      ['반복 횟수', String(eigen.iterations), eigen.reason]], weights: [1.3, 2.4, 1.4] },
    interpretation: '작은 잔차만으로 임의 입력의 지배 고유값에 도달했다고 단정할 수 없습니다.',
    sources: refs,
    notes: notes(
      'Av=λv인 방향에 A를 반복 적용하면 λ의 거듭제곱만큼 크기가 변한다. 지배 고유값의 절댓값이 다른 값과 분리되고 초기 벡터에 그 방향 성분이 있을 때 해당 방향이 상대적으로 우세해진다. 이는 대수적으로 가장 큰 고유값을 항상 찾는다는 뜻이 아니다.',
      '정규화 반복 뒤 Rayleigh quotient로 부호를 가진 λ를 구한다. 이 고정 대칭 행렬의 특성방정식은 λ²−7λ+11=0이며 독립 기준은 (7+√5)/2다. eig는 검증 코드에서만 호출하고 알고리즘 본체에는 없다. 고유벡터는 부호를 바꾸어도 같은 고유공간이므로 비교 전 부호를 정렬한다.',
      '수동 λ=' + exact(eigen.eigenvalue) + ', eig λ=' + exact(eig.numpy_eigenvalue) +
      ', v=' + exact(eigen.eigenvector) + '이다. 상대오차=' + exact(eig.relative_eigenvalue_error) +
      ', 잔차=' + exact(eigen.relative_residual) + ', 벡터 차이=' + exact(eig.aligned_eigenvector_l2_error) +
      '를 각각 독립 기준과 대조했다. converged=' + exact(eigen.converged) + '이다.',
      '꼬리 질문은 “잔차가 작으면 지배값인가?”이다. 초기 벡터가 지배 고유공간에 직교하면 다른 고유쌍의 잔차만 작을 수 있다. diag(−5,2)의 지배값은 −5이며 음수 부호를 버리면 안 된다. 참고 예시의 반복 횟수를 정답으로 맞추지 않고 종료 조건과 실제 횟수를 보고한다.',
      refs, '.venv/bin/python -m scripts.verify_power_iteration'),
  });

  refs = sources('qa-04', 'src/linear_algebra.py', 'outputs/svd_128.png',
    'outputs/linear_algebra_metrics.json', 'docs/DECISIONS.md');
  slides.push({
    id: 'qa-04-svd', title: '#4 SVD는 무엇을 남기나요?', kind: 'image',
    lead: '가장 큰 k개 특잇값과 대응 방향을 남겨 행렬을 근사합니다',
    formula: ['Aₖ = U[:,:k] diag(s[:k]) Vᵀ[:k,:]', 'shape: (m,k) @ (k,k) @ (k,n) → (m,n)'],
    image: 'outputs/svd_128.png',
    lines: ['보충 입력 ' + la.svd[1].shape.join('×') + ', 비교 k=' +
      la.svd[1].comparisons.map(row => row.k).join('/'),
      '오차는 화면 표시를 위한 clipping 전에 계산합니다'],
    interpretation: '64×64 입력에는 rank 100을 적용할 수 없어 128×128 보충 실험을 분리했습니다.',
    sources: refs,
    notes: notes(
      'SVD는 행렬의 정보를 직교 방향과 방향별 크기로 분해한다. A=UΣVᵀ에서 큰 특잇값부터 k개를 남기면 Aₖ를 얻는다. 낮은 rank에서 제거된 세부 정보가 보이고 k가 커지면 원본에 가까워지는 양상을 같은 Figure에서 비교한다.',
      '유효 rank는 정수이며 1≤k≤min(m,n)이다. 원문의 작은 grayscale 이미지 가이드와 rank 백 비교는 한 입력에서 동시에 만족할 수 없다. 따라서 작은 입력 본 실험과 별도의 큰 보충 실험을 구분한다. 잘못된 rank를 조용히 min 값으로 잘라내지 않는다.',
      '보충 입력 shape=' + exact(la.svd[1].shape) + ', 수치 rank=' + exact(la.svd[1].numerical_rank) +
      '이며 k와 상대 Frobenius 오차는 ' + exact(la.svd[1].comparisons.map(row => [row.k, row.relative_error])) +
      '이다. 합성 입력의 seed=' + la.seed + ', dtype=' + la.dtype +
      '이다. PNG의 전체 캔버스 크기는 이 입력 배열의 가로·세로 크기와 별개다.',
      '꼬리 질문은 “화면에서 비슷하면 검증이 끝났나?”이다. 실제 오차는 원본과 clipping 전 복원 배열의 차이로 계산한다. 시각화에서 값 범위를 잘라 보여준 결과를 수치 검증에 다시 사용하면 오차의 의미가 바뀐다. 다음 장에서는 복원 정확도와 저장 원소 수를 분리한다.',
      refs, '.venv/bin/python -m scripts.run_linear_algebra --no-cache'),
  });

  refs = sources('qa-04', 'outputs/linear_algebra_metrics.json', 'outputs/linear_algebra_report.md',
    'outputs/svd_64.png', 'outputs/svd_128.png', 'docs/DECISIONS.md');
  slides.push({
    id: 'qa-04-storage', title: '#4 rank를 높이면 압축도 좋아지나요?', kind: 'table',
    lead: '복원 오차는 줄지만 저장량은 늘어날 수 있습니다',
    table: { headers: ['입력', 'k', '상대 복원 오차', '저장/원본'], rows:
      la.svd.flatMap(image => image.comparisons.map(row => [
        image.shape.join('×'), String(row.k), number(row.relative_error), number(row.stored_over_original)])),
      weights: [1, 0.6, 1.6, 1.3] },
    formula: ['상대오차 = ‖A−Aₖ‖F / ‖A‖F', '저장 원소 비율 = k(m+n+1) / (mn)'],
    interpretation: '저장 비율이 1을 넘으면 원소 수 기준의 저장 공간 절감이 아닙니다.',
    sources: refs,
    notes: notes(
      '저장할 SVD 인자의 원소 수는 Uₖ의 mk개, sₖ의 k개, Vₖᵀ의 kn개를 합친 k(m+n+1)개다. 원본의 mn개와 비교해야 하며 PNG 파일의 압축된 바이트 크기와 비교하는 식이 아니다. 정확도가 좋아지는 것과 저장량이 줄어드는 것은 서로 다른 지표다.',
      'Frobenius norm은 모든 행렬 성분의 제곱합의 제곱근이다. 원본 norm으로 나누어 입력 크기에 대한 상대 오차를 얻는다. rank가 증가할 때 복원 오차가 수치 허용치 내에서 증가하지 않는지 확인하고 완전 rank의 복원은 별도로 작은 오차를 요구한다.',
      '완전 rank 복원 오차는 입력별로 ' + exact(la.svd.map(image => ({
        shape: image.shape, full_rank_relative_error: image.full_rank_relative_error
      }))) + '이며 기준은 1e-10이다. 원본 두 입력의 각 k·오차·저장 비율은 표에 JSON값을 직접 배치했다. 표의 짧은 표시는 반올림이며 원본 보고서에는 전체 정밀도가 남아 있다.',
      '꼬리 질문은 “오차가 작으니 저장 공간도 절약했나?”이다. 표에서 저장 비율이 일을 넘는 사례가 반례다. 완전 rank로 모든 정보를 복원해도 인자 방식 저장이 원본보다 커질 수 있다. 임의의 낮은 rank에 고유값 검증의 오 퍼센트 기준을 가져다 적용하지 않는다.',
      refs, '.venv/bin/python -m scripts.run_linear_algebra --no-cache'),
  });

  refs = sources('qa-05', 'src/utils/math.py', 'src/calculus.py',
    'outputs/calculus_metrics.json', 'outputs/calculus_report.md');
  slides.push({
    id: 'qa-05-fixed', title: '#5 중심차분은 어떻게 미분하나요?', kind: 'formula',
    lead: '양쪽 함수값의 차이로 기울기를 구하고 해석 미분과 비교합니다',
    formula: ['Dₕf(x) = [f(x+h) − f(x−h)] / (2h)', 'f(x)=x² → f′(x)=2x',
      'x=' + number(d.x) + ', h=' + number(d.h) + ', 해석값=' + number(d.analytic)],
    table: { headers: ['수치 미분', '절대오차', '허용치'],
      rows: [[number(d.numerical, 16), number(d.absolute_error), '≤ ' + number(d.tolerance)]],
      weights: [2, 1.5, 1] },
    interpretation: 'x²의 중심차분은 정확 산술에서 절단오차가 0이지만 반올림 오차는 남습니다.',
    sources: refs,
    notes: notes(
      '중심차분은 한쪽 점만 보는 전진차분과 달리 x+h와 x−h를 대칭으로 사용한다. 함수값의 차이를 두 점 사이 거리인 2h로 나누어 기울기를 근사한다. 여기서는 고정된 이차함수와 점에서 해석적으로 아는 정답을 사용한다.',
      'Taylor 전개에서 대칭인 두 식을 빼면 짝수 차수 항이 소거되어 일반적으로 Dₕf=f′+h²f‴/6+O(h⁴)다. 그러나 이차함수의 삼차 미분은 영이다. 직접 전개해도 ((x+h)²−(x−h)²)/(2h)=2x가 되어 정확 산술의 절단오차가 없다.',
      '실제 입력은 ' + exact(d) + '이다. 이 고정 입력의 절대오차를 명시한 허용치와 비교한다. 중심차분 공통 함수를 재사용하며 dtype은 ' + ca.dtype +
      '이다. 데이터의 작은 차이는 잘못된 미분식이 아니라 유한 정밀도의 덧셈과 뺄셈에서 생길 수 있는 반올림을 포함한다.',
      '꼬리 질문은 “절단오차가 없는데 왜 결과가 정확히 해석값과 같지 않은가?”이다. 컴퓨터는 x+h와 제곱값을 유한 비트로 표현한다. 가까운 두 값의 차이를 작은 수로 나누며 상대 오차가 커질 수 있다. h는 양의 유한 값이어야 하고, 다음 장의 탐색 전체가 통과한 것으로 이 고정 검증을 확대하지 않는다.',
      refs, '.venv/bin/python -m scripts.run_calculus'),
  });

  refs = sources('qa-05', 'outputs/derivative_error.png', 'outputs/calculus_metrics.json',
    'outputs/evaluation_evidence.md', 'docs/DECISIONS.md');
  slides.push({
    id: 'qa-05-h', title: '#5 h가 작을수록 정확한가요?', kind: 'image',
    lead: '작은 h의 반올림 오차와 함수별 절단오차를 함께 봐야 합니다',
    image: 'outputs/derivative_error.png',
    lines: ['x² 표본 ' + hStudy.h.length + '개: 통과 ' + passing.length + ', 실패 ' +
      (hStudy.h.length - passing.length) + ' (오차 ≤ ' + number(d.tolerance) + ')',
      '전체 최대오차 ' + number(Math.max(...hStudy.quadratic_error)),
      '관측 최소오차 ' + number(smallestError) + ', 해당 h=' + vector(minimizingH)],
    interpretation: '시험한 표본의 결과이며 그 사이 연속 구간이나 보편적인 최적 h를 보장하지 않습니다.',
    sources: refs,
    notes: notes(
      '고정된 h 하나의 성공과 여러 h를 탐색한 결과는 구분한다. 그림은 같은 점에서 이차함수와 sine의 중심차분 오차가 간격에 따라 달라지는 모습을 보여준다. 작은 간격의 오차가 크다는 사실과 큰 간격의 절단오차가 생길 수 있다는 사실을 동시에 설명해야 한다.',
      '이차함수는 정확 산술에서 절단오차가 없지만 sine은 일반적인 O(h²) 절단오차를 갖는다. 너무 작은 h에서는 가까운 함수값끼리 빼는 과정의 반올림이 지배할 수 있다. 따라서 h를 계속 줄이는 것이 항상 개선은 아니며 최소값을 찾았더라도 함수와 dtype에 의존하는 관측이다.',
      '격자 h의 범위는 ' + exact(hStudy.h[0]) + '부터 ' + exact(hStudy.h.at(-1)) + '이며 ' +
      hStudy.h.length + '개다. 통과한 표본은 ' + passing.length + '개, 첫 통과 h=' +
      exact(passing[0]) + ', 마지막 통과 h=' + exact(passing.at(-1)) + '이다. 전체 최대오차=' +
      exact(Math.max(...hStudy.quadratic_error)) + ', 최소오차=' + exact(smallestError) +
      ', 최소값의 h=' + exact(minimizingH) + '이다. 원본 JSON에는 모든 표본이 있다.',
      '꼬리 질문은 “이차함수에서 큰 h가 잘되니 다른 함수에도 그대로 쓰면 되나?”이다. sine 곡선이 반례다. 인접한 통과 표본을 묶었다는 말은 그 사이 모든 실수 h를 검증했다는 뜻이 아니다. 탐색 실패 표본은 의도적으로 드러내며 기본 고정 입력 검증의 실패와 혼동하지 않는다.',
      refs, '.venv/bin/python -m scripts.run_calculus'),
  });

  refs = sources('qa-06', 'src/calculus.py', 'outputs/gradient_geometry.png',
    'outputs/calculus_metrics.json', 'outputs/calculus_report.md');
  slides.push({
    id: 'qa-06-gradient', title: '#6 Gradient는 왜 등고선에 수직인가요?', kind: 'image',
    lead: '등고선을 따라 함수값이 변하지 않으므로 gradient와 접선의 내적은 0입니다',
    image: 'outputs/gradient_geometry.png',
    formula: ['∇f=(2x,2y), 접선=(-y,x)', '∇f·접선 = -2xy+2xy = 0'],
    lines: ['검사 shape=(' + gradient.points.length + ',' + gradient.points[0].length + ')',
      '최대 gradient 오차 ' + number(gradient.maximum_absolute_error),
      '정규화 내적 최대 ' + number(gradient.maximum_normalized_dot)],
    interpretation: '원점은 gradient가 영벡터이므로 특정 방향을 부여하지 않습니다.',
    sources: refs,
    notes: notes(
      '등고선은 함수값이 일정한 점들의 집합이다. 그 위의 곡선을 r(t)라 하면 f(r(t))가 상수이므로 미분은 영이다. 연쇄법칙으로 ∇f(r(t))·r′(t)=0이 되어 비영 gradient가 접선과 수직임을 알 수 있다.',
      '이차 원형 함수에서는 gradient가 (2x,2y)이고 접선 방향은 (−y,x)다. 두 벡터의 내적을 전개하면 −2xy+2xy로 소거된다. 수치 검증은 gradient와 접선을 각각 길이로 나눈 뒤 내적을 비교하며 원점은 이 나눗셈과 방향 해석에서 제외한다.',
      '측정 점 수=' + gradient.points.length + ', 첫 점=' + exact(gradient.points[0]) +
      ', 해당 해석 gradient=' + exact(gradient.analytic[0]) + ', 수치 gradient=' +
      exact(gradient.numerical[0]) + '이다. 최대 절대오차=' + exact(gradient.maximum_absolute_error) +
      '≤' + exact(gradient.tolerance) + ', 최대 정규화 내적=' + exact(gradient.maximum_normalized_dot) +
      '≤' + exact(gradient.orthogonality_tolerance) + '이다. 원점 값=' + exact(gradient.origin_gradient) +
      ', 방향=' + exact(gradient.origin_direction) + '이다.',
      '꼬리 질문은 “내적이 영이면 원점에도 수직 화살표를 그릴 수 있나?”이다. 영벡터에는 방향이 없으므로 그렇지 않다. 그림에서는 화살표 길이를 가독성에 맞춰 정규화하지만 검증에 쓰는 원래 gradient의 크기는 바꾸지 않는다. 시각 검토와 수치 검증은 서로 보완한다.',
      refs, '.venv/bin/python -m scripts.run_calculus'),
  });
  refs = sources('qa-07', 'notebooks/backprop_derivation.ipynb#backprop-fixed-input',
    'notebooks/backprop_derivation.ipynb#backprop-forward-derivation',
    'outputs/backprop_metrics.json', 'docs/MATH_DERIVATIONS.md');
  slides.push({
    id: 'qa-07-forward', title: '#7 고정 신경망의 순전파는 어떻게 되나요?', kind: 'table',
    lead: '입력 2개 → 은닉 2개 → 출력 1개, 단일 표본 BCE를 사용합니다',
    lines: ['x=' + vector(parameters.x) + ', W₁=' + vector(parameters.W1),
      'W₂=' + vector(parameters.W2) + ', b₁=' + vector(parameters.b1) +
      ', b₂=' + number(parameters.b2) + ', y=' + number(parameters.y_true)],
    table: { headers: ['계산', '실제 결과', 'shape'], rows: [
      ['z₁=W₁x+b₁', vector(forward.z1), '(2,)'],
      ['a₁=σ(z₁)', vector(forward.a1, 9), '(2,)'],
      ['z₂=W₂a₁+b₂', number(forward.z2, 10), '()'],
      ['p=σ(z₂)', number(forward.y_pred, 10), '()'],
      ['L=−y ln p−(1−y)ln(1−p)', number(forward.L, 10), '()']],
      weights: [2, 2.3, 0.7] },
    interpretation: '계산 중간값을 네 자리로 잘라 재사용하지 않고 마지막 표시에서만 반올림합니다.',
    sources: refs,
    notes: notes(
      '역전파를 설명하려면 먼저 같은 순전파와 손실을 확정해야 한다. 첫 층에서는 W₁이 x의 왼쪽에 곱해지고, 두 번째 층은 두 은닉 활성값의 가중합을 scalar로 만든다. 활성화는 두 층 모두 Sigmoid이고 정답은 고정된 단일 이진 레이블이다.',
      'σ(z)=1/(1+exp(−z))를 원소별로 적용한다. z₁의 두 성분을 계산한 뒤 a₁을 얻고, z₂=W₂,₁a₁,₁+W₂,₂a₁,₂+b₂를 계산한다. 마지막 확률 p에 단일 표본 BCE를 적용한다. 한 표본이므로 배치 평균 계수를 추가하지 않으며 수식에서는 자연로그를 사용한다.',
      '원본 parameters=' + exact(parameters) + '이다. forward=' + exact(forward) +
      '를 그대로 읽어 본문에 반올림 표시했다. z₁과 a₁은 길이 이 벡터, z₂와 예측값과 손실은 scalar다. 이후 미분의 기대값은 이 순전파의 반올림 전 값에 대응한다. 출력 shape과 곱 방향이 다른 예제로 바뀌지 않았는지 함께 확인한다.',
      '꼬리 질문은 “출력 logit을 네 자리로 반올림한 뒤 Sigmoid를 계산해도 같은 검증인가?”이다. 작은 차이가 뒷단으로 전파되므로 그렇지 않다. 원문 참고 숫자와 다른 부분은 입력을 바꾸는 대신 정확한 식으로 재계산하고 결정 기록에 이유를 남긴다.',
      refs, '.venv/bin/python -m scripts.run_notebooks'),
  });

  refs = sources('qa-07', 'notebooks/backprop_derivation.ipynb#backprop-chain-rule',
    'outputs/backprop_metrics.json', 'docs/MATH_DERIVATIONS.md');
  slides.push({
    id: 'qa-07-output', title: '#7 출력층 미분은 왜 p−y가 되나요?', kind: 'formula',
    lead: 'BCE의 미분과 Sigmoid의 미분을 연쇄법칙으로 곱하면 항이 소거됩니다',
    formula: ['∂L/∂p = −y/p + (1−y)/(1−p)',
      'δ₂ = (∂L/∂p)·p(1−p) = p−y',
      '∂L/∂W₂ = δ₂ a₁', '∂L/∂b₂ = δ₂'],
    table: { headers: ['미분', '실제 결과', 'shape'], rows: [
      ['∂L/∂p', number(backward.dL_dy_pred, 10), '()'],
      ['δ₂', number(backward.dL_dz2, 10), '()'],
      ['∂L/∂W₂', vector(backward.dL_dW2, 8), '(2,)']], weights: [1.2, 2.4, 0.7] },
    interpretation: 'p−y는 BCE와 Sigmoid를 합성한 결과이며 임의의 손실·활성화에 그대로 쓰지 않습니다.',
    sources: refs,
    notes: notes(
      '필수 여섯 미분 중 앞의 세 개는 손실에서 출력 확률, 출력 logit, 출력 가중치로 이동한다. BCE를 확률 p로 미분하면 −y/p+(1−y)/(1−p)다. 이 고정 예제는 확률이 영과 일 사이에 있으므로 분모가 정의된다.',
      'Sigmoid의 미분은 p(1−p)다. 두 식을 곱하면 −y(1−p)+(1−y)p가 되어 −y+yp+p−yp=p−y로 정리된다. z₂=ΣW₂,ⱼa₁,ⱼ+b₂에서 W₂,ⱼ에 대한 미분은 a₁,ⱼ이므로 가중치 미분은 δ₂a₁이다. 편향은 계수가 일이므로 δ₂를 그대로 받는다.',
      '저장값은 ∂L/∂p=' + exact(backward.dL_dy_pred) + ', δ₂=' + exact(backward.dL_dz2) +
      ', ∂L/∂W₂=' + exact(backward.dL_dW2) + ', ∂L/∂b₂=' + exact(backward.dL_db2) +
      '이다. δ₂는 scalar이고 W₂와 그 gradient는 같은 (2,) shape이다. 모든 미분은 앞 장의 고정 parameters와 forward값에 대응하며 독립 기준의 대조는 다음 비교 장에서 따로 제시한다.',
      '꼬리 질문은 “정답이 일인 경우에만 p−y인가?”이다. 이진 BCE와 Sigmoid의 합성식으로 양쪽 레이블에서 성립한다. 다만 확률 경계의 분수식을 그대로 계산하면 정의되지 않는 항이 생길 수 있어 안정적인 logit 손실과 같은 함수를 검증해야 한다. 임의 clipping 뒤 원래 gradient를 붙이지 않는다.',
      refs, '.venv/bin/python -m scripts.run_notebooks'),
  });

  refs = sources('qa-07', 'notebooks/backprop_derivation.ipynb#backprop-chain-rule',
    'notebooks/backprop_derivation.ipynb#backprop-backward-numpy',
    'outputs/backprop_metrics.json', 'docs/MATH_DERIVATIONS.md');
  slides.push({
    id: 'qa-07-hidden', title: '#7 은닉층과 첫 가중치는 어떻게 미분하나요?', kind: 'formula',
    lead: '연쇄법칙을 은닉 활성값·logit·가중치 순서로 적용합니다',
    formula: ['∂L/∂a₁ = δ₂ W₂',
      'δ₁ = (δ₂ W₂) ⊙ a₁ ⊙ (1−a₁)',
      '∂L/∂W₁ = outer(δ₁, x)', '∂L/∂b₁ = δ₁'],
    table: { headers: ['미분', '실제 결과', 'shape'], rows: [
      ['∂L/∂a₁', vector(backward.dL_da1, 8), '(2,)'],
      ['δ₁', vector(backward.dL_dz1, 8), '(2,)'],
      ['∂L/∂W₁', vector(backward.dL_dW1, 6), '(2,2)']], weights: [1.2, 2.7, 0.7] },
    interpretation: 'x의 두 번째 성분이 0이므로 dL/dW₁의 두 번째 열도 0입니다.',
    sources: refs,
    notes: notes(
      '나머지 세 필수 미분은 은닉 활성값 a₁, 은닉 logit z₁, 첫 층 가중치 W₁에 관한 것이다. 출력의 가중합을 a₁로 미분하면 W₂를 얻으므로 ∂L/∂a₁=δ₂W₂다. 은닉 Sigmoid는 성분마다 독립적으로 작용한다.',
      '은닉 Sigmoid Jacobian의 대각 성분은 a₁,ᵢ(1−a₁,ᵢ)이고 비대각 성분은 영이다. 따라서 원소별 곱으로 δ₁=(δ₂W₂)⊙a₁⊙(1−a₁)을 얻는다. z₁,ᵢ=ΣⱼW₁,ᵢⱼxⱼ+b₁,ᵢ이므로 ∂L/∂W₁,ᵢⱼ=δ₁,ᵢxⱼ다. 이를 행렬로 쓰면 δ₁xᵀ이고 NumPy에서는 outer를 사용한다.',
      '실제 값은 ∂L/∂a₁=' + exact(backward.dL_da1) + ', δ₁=' + exact(backward.dL_dz1) +
      ', ∂L/∂W₁=' + exact(backward.dL_dW1) + ', ∂L/∂b₁=' + exact(backward.dL_db1) +
      '이다. first-layer gradient의 shape은 ' + exact(bp.shapes.dL_dW1) +
      '이다. 편향은 각 해당 logit의 미분을 그대로 받는다.',
      '꼬리 질문은 “두 번째 열이 영이면 두 번째 은닉 뉴런은 학습하지 않나?”이다. 영인 것은 입력의 두 번째 좌표와 연결된 가중치 열이다. 두 번째 은닉 뉴런의 첫 가중치와 편향 미분은 영이 아니다. 행과 열의 의미를 뒤집거나 원소별 곱을 행렬 곱으로 바꾸지 않는다.',
      refs, '.venv/bin/python -m scripts.run_notebooks'),
  });

  const checkpointNames = [
    ['z1', 'z₁'], ['a1', 'a₁'], ['z2', 'z₂'], ['y_pred', 'p'], ['L', 'L'],
    ['dL_dy_pred', '∂L/∂p'], ['dL_dz2', 'δ₂'], ['dL_dW2', '∂L/∂W₂'],
    ['dL_da1', '∂L/∂a₁'], ['dL_dz1', 'δ₁'], ['dL_dW1', '∂L/∂W₁'],
    ['dL_db1', '∂L/∂b₁'], ['dL_db2', '∂L/∂b₂'],
  ];
  refs = sources('qa-08', 'notebooks/backprop_derivation.ipynb#backprop-hand-reference',
    'notebooks/backprop_derivation.ipynb#backprop-compare',
    'notebooks/backprop_derivation.ipynb#backprop-gradient-check', 'outputs/backprop_metrics.json');
  slides.push({
    id: 'qa-08-forward', title: '#8 순전파와 손계산은 일치하나요?', kind: 'table',
    lead: '순전파 5개 체크포인트를 독립 상수와 비교하고 마지막에 네 자리로 표시합니다',
    table: { headers: ['체크포인트', '독립 기준의 네 자리 표시', 'shape'],
      rows: checkpointNames.slice(0, 5).map(([key, label]) => [
        label, four(bp.hand_reference[key]),
        bp.shapes[key].length ? '(' + bp.shapes[key].join(',') + (bp.shapes[key].length === 1 ? ',' : '') + ')' : '()']),
      weights: [1.2, 3, 0.8] },
    interpretation: '손계산 최대오차 ' + number(bp.max_hand_absolute_error) +
      ', 중심차분 최대오차 ' + number(bp.max_numerical_gradient_error) + '입니다.',
    sources: refs,
    notes: notes(
      '같은 구현을 두 번 호출해 같은 값이 나온 것은 독립 검증이 아니다. 손계산으로 평가한 상수를 hand_reference에 고정하고 NumPy의 forward와 backward를 각 성분에서 비교했다. 충분한 정밀도의 값을 검증에 사용하고 네 자리 표시는 마지막 단계에서만 만든다.',
      '손계산 비교는 atol=5e-5, rtol=0이며 표시 문자열의 네 자리 일치도 별도로 확인한다. 추가로 매개변수 θ의 각 성분에 대해 [L(θ+heᵢ)−L(θ−heᵢ)]/(2h)를 계산해 해석 gradient와 비교한다. 이는 역전파와 다른 계산 경로이므로 같은 잘못된 식을 두 곳에 복제하는 위험을 줄인다.',
      '최대 손계산 오차=' + exact(bp.max_hand_absolute_error) + ', four_decimal_match=' +
      exact(bp.four_decimal_match) + '이다. 매개변수 수=' + bp.parameter_count + ', 중심차분 h=' +
      exact(bp.derivative_h) + ', 최대 수치 gradient 오차=' + exact(bp.max_numerical_gradient_error) +
      '이며 기준은 atol=1e-6, rtol=0이다. 매개변수 순서는 ' + bp.parameter_order.join(', ') + '이다.',
      '꼬리 질문은 “원문 참고값과 다르면 실패인가?”이다. z₂와 두 번째 은닉 기울기의 올바른 네 자리 표시는 ' +
      four(bp.hand_reference.z2) + ', ' + four(bp.hand_reference.dL_dz1[1]) +
      '이다. 원문 예시를 맞추기 위해 입력을 바꾸지 않는다. 음의 영은 표시에서만 정리하고 원래 배열을 보존하며, 고정 예제의 성공을 모든 입력에 대한 보장으로 일반화하지 않는다.',
      refs, '.venv/bin/python -m scripts.run_notebooks'),
  });

  slides.push({
    id: 'qa-08-backward', title: '#8 역전파 gradient도 독립적으로 확인했나요?', kind: 'table',
    lead: '역전파 8개 체크포인트와 별도 중심차분의 9개 매개변수 gradient를 확인합니다',
    table: { headers: ['체크포인트', '독립 기준의 네 자리 표시', 'shape'],
      rows: checkpointNames.slice(5).map(([key, label]) => [
        label, four(bp.hand_reference[key]),
        bp.shapes[key].length ? '(' + bp.shapes[key].join(',') + (bp.shapes[key].length === 1 ? ',' : '') + ')' : '()']),
      weights: [1.2, 3, 0.8] },
    interpretation: '네 자리 일치=' + String(bp.four_decimal_match) + '; ' + bp.parameter_count +
      '개 매개변수 중심차분 오차 ' + number(bp.max_numerical_gradient_error) + ' ≤ 1e-6입니다.',
    sources: refs,
    notes: notes(
      '역전파의 여섯 필수 미분과 두 편향 미분을 독립 손계산 기준과 비교한다. 출력 확률에 대한 미분과 각 중간값에 대한 미분까지 확인해야 마지막 가중치 gradient만 우연히 맞은 경우를 놓치지 않는다. 본문은 표시용 네 자리이며 검증에는 원본 정밀도를 사용한다.',
      '첫 네 매개변수는 W₁의 행 우선 순서, 다음 두 개는 b₁, 다음 두 개는 W₂, 마지막은 b₂다. 각 좌표에 작은 양·음의 변화 h를 준 손실 차이를 2h로 나누면 중심차분 gradient를 얻는다. 이를 해석 gradient와 비교하므로 구현의 마지막 손실과 역전파를 서로 다른 경로로 확인한다.',
      '중심차분 h=' + exact(bp.derivative_h) + ', 해석 gradient=' + exact(bp.analytical_gradient) +
      ', 최대 gradient 오차=' + exact(bp.max_numerical_gradient_error) +
      '다. 중심차분 허용치는 atol=1e-6, rtol=0이고 손계산 기준은 atol=5e-5, rtol=0이다. 모든 비교의 네 자리 일치 기록=' +
      exact(bp.four_decimal_match) + ', 원본 노트북의 assertion 기록=' + exact(bp.all_assertions_passed) + '다.',
      '꼬리 질문은 “수치 미분과 맞으면 유도가 필요 없나?”이다. 수치 검증은 특정 입력의 일관성을 확인하고 연쇄법칙 유도는 어떤 식과 shape을 계산하는지 설명한다. 너무 작은 h나 다른 함수에서는 중심차분 오차가 달라질 수 있다. 부호 있는 영은 표시만 정리하고 원래 배열이나 독립 기준을 바꾸지 않는다.',
      refs, '.venv/bin/python -m scripts.run_notebooks'),
  });

  refs = sources('qa-09', 'src/optimizer.py', 'outputs/optimizer_metrics.json',
    'outputs/optimizer_report.md', 'outputs/optimizer_sensitivity.json');
  slides.push({
    id: 'qa-09-gd', title: '#9 GD는 반경 0.1에 도달하나요?', kind: 'formula',
    lead: '원형 함수의 반복식은 닫힌 해로도 확인할 수 있습니다',
    formula: ['∇f(θ)=2θ,  θₜ₊₁=(1−2lr)θₜ',
      'lr=' + number(circle.GD.lr) + ', θ₀=' + vector(circle.GD.initial),
      'θₜ = (1−2lr)ᵗ θ₀'],
    table: { headers: ['검사', '실제 결과'], rows: [
      ['업데이트 / 경로 shape', circle.GD.iterations + ' / (' + circle.GD.path.length + ',2)'],
      ['최종 좌표', vector(circle.GD.path.at(-1))],
      ['최종 반경', number(circle.GD.final_norm)],
      ['반경 .1 첫 진입 / 유지 시작', circle.GD.first_radius_0_1 + ' / ' +
        circle.GD.remaining_inside_radius_0_1_through_budget]], weights: [1.6, 2.4] },
    interpretation: '초기점을 포함한 경로 길이와 실제 업데이트 횟수는 하나 차이 납니다.',
    sources: refs,
    notes: notes(
      'Vanilla GD는 현재 gradient의 반대 방향으로 학습률만큼 이동한다. 원형 함수 f=x²+y²의 gradient는 2θ이므로 다음 좌표가 현재 좌표의 일정한 배율이 된다. 이 단순한 예제에서는 반복 구현뿐 아니라 닫힌 식을 독립 기준으로 사용할 수 있다.',
      'θₜ₊₁=(1−2lr)θₜ를 반복하면 θₜ=(1−2lr)ᵗθ₀다. 반경에는 배율의 절댓값, 손실에는 배율의 제곱이 적용된다. 따라서 시작점과 학습률과 반복 예산이 같아야 비교가 가능하다. 정지 조건을 사용하지 않은 고정 예산 실험과 조기 종료 실험을 구분한다.',
      '시작=' + exact(circle.GD.initial) + ', lr=' + exact(circle.GD.lr) + ', 업데이트=' +
      circle.GD.iterations + '이다. 최종 반경=' + exact(circle.GD.final_norm) + ', 손실=' +
      exact(circle.GD.losses.at(-1)) + ', gradient norm=' + exact(circle.GD.gradient_norms.at(-1)) +
      '이다. 첫 진입=' + exact(circle.GD.first_radius_0_1) + ', 유지 시작=' +
      exact(circle.GD.remaining_inside_radius_0_1_through_budget) + ', status=' + circle.GD.status + '이다.',
      '꼬리 질문은 “budget_exhausted는 수렴 실패인가?”이다. 이 설정에서는 지정 업데이트 예산을 모두 사용했다는 상태이며 별도 반경 기준의 실패를 뜻하지 않는다. 반경 기준은 원문의 0.1을 유지한다. 참고 예시 좌표를 복사하거나 초기점을 업데이트 횟수에 더하지 않는다.',
      refs, '.venv/bin/python -m scripts.run_optimizer --no-cache'),
  });

  refs = sources('qa-10', 'outputs/optimizer_learning_rates.png', 'outputs/optimizer_metrics.json',
    'outputs/optimizer_sensitivity.json', 'docs/DECISIONS.md');
  slides.push({
    id: 'qa-10-rate', title: '#10 lr=0.5는 정말 발산하나요?', kind: 'image',
    lead: '원형 GD의 수렴 조건은 |1−2lr|<1, 즉 0<lr<1입니다',
    image: 'outputs/optimizer_learning_rates.png',
    table: { headers: ['lr', '좌표 배율', '결과', '최종 반경'],
      rows: se.stability.map(row => [number(row.lr), number(row.coordinate_multiplier),
        ({ one_step_convergence: '1회 수렴', constant_radius_sign_oscillation: '진동',
          divergence_for_nonzero_initial: '발산' })[row.classification] || row.classification,
        number(row.final_radius)]), weights: [0.6, 0.8, 1.2, 1.2] },
    interpretation: '원점이 아닌 초기점에서 lr=.5는 1회 수렴, lr=1은 진동, lr>1은 발산합니다.',
    sources: refs,
    notes: notes(
      '원문에 있는 “학습률이 0.5 이상이면 발산”은 이 함수와 GD 식에 맞지 않는다. θₜ₊₁=(1−2lr)θₜ에서 배율의 절댓값을 직접 확인해야 한다. 원문을 보존하고 잘못된 조건을 수식에서 정정한다.',
      '배율의 절댓값이 일보다 작으면 비영 초기점의 반경은 줄어든다. lr가 반이면 배율이 영이어서 한 번에 원점에 도달한다. lr가 일이면 배율이 음의 일이라 좌표 부호만 바뀐다. 일보다 크면 절댓값이 일을 넘어 반경과 손실이 증가한다. 손실 배율은 좌표 배율의 제곱이다.',
      '저장된 안정성 사례는 ' + exact(se.stability.map(row => ({
        lr: row.lr, steps: row.steps, multiplier: row.coordinate_multiplier,
        final_radius: row.final_radius, classification: row.classification
      }))) + '이다. 같은 시작점과 유한한 관측 예산으로 비교하며 그림은 원형 GD의 세 경로와 최종 반경을 보여준다.',
      '꼬리 질문은 “이미 원점에서 시작해도 발산하나?”이다. gradient가 영이므로 이 예제의 원점은 그대로 원점이다. 발산 설명에는 비영 초기점 조건이 필요하다. 함수에 임의의 이분의 일을 붙여 gradient를 바꾸면 임계 학습률도 바뀌므로 원래 조건의 정정과 혼동하지 않는다.',
      refs, '.venv/bin/python -m scripts.run_optimizer --no-cache'),
  });

  refs = sources('qa-11', 'src/optimizer.py', 'outputs/optimizer_circle.png',
    'outputs/optimizer_metrics.json', 'outputs/optimizer_sensitivity.json');
  slides.push({
    id: 'qa-11-momentum', title: '#11 Momentum은 항상 더 빠른가요?', kind: 'image',
    lead: '첫 도달이 빠른 것과 그 뒤 계속 머무는 것은 다른 성질입니다',
    image: 'outputs/optimizer_circle.png',
    table: { headers: ['방법', '최종 반경', '첫 진입', '유지 시작'],
      rows: ['GD', 'Momentum'].map(name => [name, number(circle[name].final_norm),
        number(circle[name].first_radius_0_1),
        number(circle[name].remaining_inside_radius_0_1_through_budget)]),
      weights: [1.1, 1.3, 0.8, 0.9] },
    interpretation: 'v←βv+g, θ←θ−lr·v를 사용합니다. 첫 진입 뒤 다시 반경 밖으로 나갈 수 있습니다.',
    sources: refs,
    notes: notes(
      'Momentum은 이전 gradient 정보를 속도에 누적한다. 같은 방향의 gradient가 이어지면 이동이 가속될 수 있지만 최소점을 지나치는 진동도 가능하다. 따라서 첫 진입 횟수만으로 모든 의미의 수렴 속도를 비교할 수 없다.',
      '프로젝트의 관례는 v_next=βv+g, θ_next=θ−lr·v_next, 초기 속도 영이다. gradient에 1−β를 곱하는 다른 관례와 섞지 않는다. 첫 진입은 최초로 반경 이하가 된 인덱스이고 유지 시작은 그 이후 기록 끝까지 모두 반경 이하인 최초 인덱스다. 유한 예산 끝까지의 유지가 무한 시간 보장은 아니다.',
      'lr=' + exact(circle.Momentum.lr) + ', β=' + exact(circle.Momentum.beta) + ', 시작=' +
      exact(circle.Momentum.initial) + ', 업데이트=' + circle.Momentum.iterations + '다. Momentum 최종 좌표=' +
      exact(circle.Momentum.path.at(-1)) + ', 반경=' + exact(circle.Momentum.final_norm) +
      ', 첫 진입=' + exact(circle.Momentum.first_radius_0_1) + ', 유지 시작=' +
      exact(circle.Momentum.remaining_inside_radius_0_1_through_budget) + '다. 별도 민감도 실험은 ' +
      se.initial_points.length + '개의 고정 초기점으로 확장한다.',
      '꼬리 질문은 “첫 진입이 더 빠르니 항상 우수한가?”이다. 이 고정 실험의 최종 반경은 GD가 더 작으며 Momentum은 진입 후 이탈한다. 미도달 null을 영 회차로 바꾸어 평균을 계산하면 통계가 왜곡된다. 원점을 초기값으로 썼을 때의 진짜 영 회차와 결측을 구분한다.',
      refs, '.venv/bin/python -m scripts.run_optimizer --no-cache'),
  });

  refs = sources('qa-12', 'outputs/optimizer_ellipse.png', 'outputs/optimizer_metrics.json',
    'outputs/optimizer_report.md', 'docs/DECISIONS.md');
  slides.push({
    id: 'qa-12-ellipse', title: '#12 곡률 차이는 경로를 어떻게 바꾸나요?', kind: 'image',
    lead: 'x²+10y²의 gradient는 (2x,20y), Hessian은 diag(2,20)입니다',
    image: 'outputs/optimizer_ellipse.png',
    table: { headers: ['방법', '최종 반경', '첫 진입', '유지 시작'],
      rows: ['GD', 'Momentum'].map(name => [name, number(ellipse[name].final_norm),
        number(ellipse[name].first_radius_0_1),
        number(ellipse[name].remaining_inside_radius_0_1_through_budget)]),
      weights: [1.1, 1.3, 0.8, 0.9] },
    interpretation: 'lr=' + number(ellipse.GD.lr) + '의 GD는 y를 한 번에 0으로 만듭니다. 이 설정에서는 GD가 더 유리합니다.',
    sources: refs,
    notes: notes(
      '타원형 함수는 좌표마다 곡률이 달라 같은 학습률의 효과가 다르다. gradient가 (2x,20y)이므로 GD의 x 배율은 1−2lr, y 배율은 1−20lr다. 두 방향 모두 줄어야 하므로 양의 학습률 수렴 범위는 0<lr<0.1이다.',
      '원문의 고정 lr에서는 y의 배율이 영이다. 따라서 GD가 y축에서 계속 지그재그를 그린다고 가정하면 안 된다. Momentum을 각 Hessian 고유방향에서 보면 θₜ₊₁=(1+β−lrλ)θₜ−βθₜ₋₁이고 특성방정식은 r²−(1+β−lrλ)r+β=0이다. 이 근의 크기와 위상으로 감쇠와 진동을 해석한다.',
      'GD의 lr=' + exact(ellipse.GD.lr) + ', 최종 좌표=' + exact(ellipse.GD.path.at(-1)) +
      ', 손실=' + exact(ellipse.GD.losses.at(-1)) + '이다. Momentum β=' + exact(ellipse.Momentum.beta) +
      ', 최종 좌표=' + exact(ellipse.Momentum.path.at(-1)) + ', 손실=' +
      exact(ellipse.Momentum.losses.at(-1)) + '이다. 두 방법 모두 같은 시작점과 ' +
      ellipse.GD.iterations + '회 예산으로 비교한다.',
      '꼬리 질문은 “원문이 Momentum의 이점을 요구하니 유리한 그림만 골라야 하나?”이다. 관측된 불리함도 보존하고 해당 설정의 한계를 설명한다. 다른 학습률의 이점은 별도 실험으로 기록해야 한다. 뒤의 Adam 비교는 반복 예산이 다르므로 기본 실험의 최종값과 섞지 않는다.',
      refs, '.venv/bin/python -m scripts.run_optimizer --no-cache'),
  });
  refs = sources('qa-13', 'notebooks/probability_loss.ipynb#normal-density-check',
    'outputs/probability_normal.png', 'outputs/probability_metrics.json', 'docs/DECISIONS.md');
  slides.push({
    id: 'qa-13-normal', title: '#13 N(2,0.5)의 0.5는 무엇인가요?', kind: 'image',
    lead: '정규분포의 두 번째 인자는 분산입니다. 표준편차는 그 제곱근입니다',
    image: 'outputs/probability_normal.png',
    table: { headers: ['평균 μ', '분산 σ²', '표준편차 σ', '적분 오차'],
      rows: pr.normals.map(row => [number(row.mu), number(row.variance),
        number(row.sigma), number(row.integral_absolute_error)]),
      weights: [0.8, 0.9, 1.2, 1.4] },
    interpretation: 'PDF는 점의 확률이 아니라 밀도입니다. 확률은 구간에서 적분해 구합니다.',
    sources: refs,
    notes: notes(
      '정규분포를 N(μ,σ²)로 표기하므로 두 번째 인자는 분산이다. 이를 표준편차로 읽으면 다른 분포를 계산한다. 그림에서 평균은 중심 위치를 정하고 분산은 퍼짐과 봉우리 높이에 영향을 준다.',
      'PDF 식은 1/√(2πσ²)·exp(−(x−μ)²/(2σ²))다. 연속 변수의 한 점 자체의 확률과 PDF 높이는 다른 양이다. 확률은 구간 적분으로 얻으며 전체 실수 구간의 적분은 일이다. 유한한 구간의 수치 적분은 이론식의 증명을 대신하지 않고 구현의 정규화와 비음수성을 확인한다.',
      '저장된 두 분포의 측정은 ' + exact(pr.normals.map(row => ({
        mu: row.mu, variance: row.variance, sigma: row.sigma,
        grid_points: row.grid_points, integral: row.integral,
        integral_absolute_error: row.integral_absolute_error
      }))) + '이다. 각 μ±8σ를 균등 격자로 나누어 사다리꼴 적분했고 합의 일 대비 오차 기준은 1e-4다.',
      '꼬리 질문은 “PDF가 일보다 높으면 잘못된 확률인가?”이다. 충분히 좁은 분포에서는 밀도 봉우리가 일을 넘을 수 있다. 구간 확률이 일 이하여야 한다는 사실을 밀도 높이의 상한으로 오해하지 않는다. 다음 장의 이산 PMF는 점별 확률이므로 합과 범위를 직접 확인한다.',
      refs, '.venv/bin/python -m scripts.run_notebooks'),
  });

  refs = sources('qa-13', 'notebooks/probability_loss.ipynb#bernoulli-check',
    'outputs/probability_bernoulli.png', 'outputs/probability_metrics.json',
    'outputs/probability_report.md');
  slides.push({
    id: 'qa-13-bernoulli', title: '#13 PMF와 PDF는 무엇이 다른가요?', kind: 'image',
    lead: 'Bernoulli는 0과 1 두 결과의 확률을 직접 더해 1이 되는지 확인합니다',
    image: 'outputs/probability_bernoulli.png',
    formula: ['P(X=k)=pᵏ(1−p)¹⁻ᵏ,  k∈{0,1}'],
    table: { headers: ['p', 'P(0)', 'P(1)', '합 오차'], rows: pr.bernoulli.map(row => [
      number(row.p), number(row.pmf[0]), number(row.pmf[1]), number(row.mass_error)]),
      weights: [0.7, 1, 1, 1.1] },
    interpretation: 'PMF는 지지집합의 합, PDF는 구간 적분으로 확률을 확인합니다.',
    sources: refs,
    notes: notes(
      'Bernoulli 분포는 성공 또는 실패의 두 결과를 표현한다. 지지집합을 영과 일로 정하고 성공확률을 p라고 하면 P(1)=p, P(0)=1−p다. 그림의 막대 높이는 각 점에 대한 확률이며 연속 PDF의 밀도 높이와 의미가 다르다.',
      'P(X=k)=pᵏ(1−p)¹⁻ᵏ에서 k가 영이면 1−p, 일이면 p가 남는다. 두 확률은 각각 영과 일 사이여야 하고 합은 일이다. 이 식을 여러 독립 레이블의 관측 확률로 곱한 뒤 음의 로그를 취하면 뒤에서 다룰 BCE가 나온다.',
      '저장된 사례는 ' + exact(pr.bernoulli) + '이다. 합 오차 허용치는 1e-12이다. 두 경우의 차이는 같은 지지집합에서 성공과 실패에 배정한 질량이 바뀐 것이다. 화면은 소수 자릿수를 정리하지만 원본 JSON의 1−p 표현은 그대로 보존한다.',
      '꼬리 질문은 “보완 확률의 끝에 작은 자릿수가 붙으면 질량 합이 잘못된 것인가?”이다. 이진 부동소수점의 표현과 합 오차 검증을 구분해야 한다. 단지 표시가 길다는 이유로 값을 임의 정규화하지 않는다. PMF에서는 각 점의 확률을 더하지만 PDF의 여러 높이를 그대로 더하는 것은 적분이 아니다.',
      refs, '.venv/bin/python -m scripts.run_notebooks'),
  });

  refs = sources('qa-14', 'src/utils/math.py', 'notebooks/probability_loss.ipynb#softmax-check',
    'outputs/probability_metrics.json', 'outputs/evaluation_evidence.md');
  slides.push({
    id: 'qa-14-softmax', title: '#14 Softmax는 어느 축의 합을 1로 만드나요?', kind: 'formula',
    lead: '최댓값을 뺀 뒤 선택한 클래스 축을 따라 정규화합니다',
    formula: ['pⱼ = exp(zⱼ−m) / Σₖexp(zₖ−m)', 'm=maxₖ zₖ,  axis=' + pr.softmax.axis],
    table: { headers: ['입력 logit 행', '출력 확률 행', '행 합'], rows:
      pr.softmax.logits.map((row, index) => [vector(row), vector(pr.softmax.probabilities[index]),
        number(pr.softmax.row_sums[index], 16)]), weights: [1.7, 2.6, 1] },
    lines: ['행 합 최대오차 ' + number(pr.softmax.max_sum_error) + ' ≤ 1e-6',
      'axis=0의 별도 열 합 검증 오차 ' + number(pr.softmax.column_sum_error)],
    interpretation: '행 정규화 결과의 열 합도 1이라는 뜻은 아닙니다. 작은 확률은 underflow로 0이 될 수 있습니다.',
    sources: refs,
    notes: notes(
      'Softmax는 여러 logit을 음이 아닌 확률로 바꾸고 지정한 축에서 합을 일로 만든다. 어떤 축이 클래스를 뜻하는지 입력 shape과 함께 정해야 한다. 잘못된 축으로 계산해도 합 자체는 일일 수 있으므로 합 검증만으로 의미가 맞다고 결론 내리지 않는다.',
      '모든 logit에서 같은 m을 빼면 분자와 분모의 exp(m)이 약분되어 수학적으로 같은 확률이다. m을 최댓값으로 선택하면 지수의 양의 overflow를 피한다. keepdims를 사용해 축 길이를 유지한 중간값으로 broadcasting하고 최종 출력의 shape은 입력과 같다.',
      '실제 logits=' + exact(pr.softmax.logits) + ', probabilities=' +
      exact(pr.softmax.probabilities) + ', row_sums=' + exact(pr.softmax.row_sums) +
      '이다. 행 합 최대오차=' + exact(pr.softmax.max_sum_error) + ', 별도의 axis=0 검증 오차=' +
      exact(pr.softmax.column_sum_error) + '다. 허용치는 1e-6이다. 입력은 비어 있지 않은 유한 실수 배열이고 axis는 유효한 정수여야 한다.',
      '꼬리 질문은 “수학적으로 양수인 확률이 영이면 clipping했나?”이다. 매우 작은 지수값은 float64에서 underflow될 수 있다. 이 구현은 그런 값을 임의의 양수로 바꾸지 않는다. scalar와 빈 배열, 잘못된 축, bool과 complex 입력을 거부하며 일반 분포의 합·shape 계약과 구분한다.',
      refs, '.venv/bin/python -m scripts.run_notebooks'),
  });

  refs = sources('qa-15', 'notebooks/probability_loss.ipynb#gaussian-mle-derivation',
    'notebooks/probability_loss.ipynb#gaussian-mle-check',
    'outputs/probability_metrics.json', 'docs/MATH_DERIVATIONS.md');
  slides.push({
    id: 'qa-15-gaussian', title: '#15 Gaussian MLE와 MSE는 왜 연결되나요?', kind: 'formula',
    lead: '독립 오차와 고정된 양의 공통 분산이면 두 목적함수의 최적해가 같습니다',
    formula: ['yᵢ=fθ(xᵢ)+εᵢ,  εᵢ~N(0,σ²)',
      'L(θ)=∏ᵢp(yᵢ|xᵢ,θ)',
      'NLL = n/2·ln(2πσ²) + SSE/(2σ²)',
      'MSE=SSE/n → argmin NLL = argmin MSE'],
    table: { headers: ['n / 분산', 'SSE', 'MSE', 'NLL'], rows: [[
      pr.gaussian.n + ' / ' + number(pr.gaussian.variance), number(pr.gaussian.sse),
      number(pr.gaussian.mse), number(pr.gaussian.nll, 10)]], weights: [1, 0.8, 0.8, 1.5] },
    interpretation: 'NLL과 MSE는 같은 수치가 아닙니다. 분산이 θ에 의존하면 생략할 수 없는 항이 생깁니다.',
    sources: refs,
    notes: notes(
      'MLE는 관측 데이터를 고정한 뒤 그 데이터의 likelihood를 가장 크게 하는 매개변수를 찾는다. Gaussian 오차의 독립성과 모든 관측에 같은 알려진 양의 상수 분산을 가정하면 MSE 최소화와 연결할 수 있다.',
      '한 관측의 밀도는 1/√(2πσ²)·exp(−잔차²/(2σ²))다. 독립 표본의 곱을 로그합으로 바꾸고 음수를 붙이면 n/2·ln(2πσ²)+SSE/(2σ²)가 된다. 첫 항은 θ와 무관하고 SSE의 계수는 양수다. SSE=n·MSE이므로 같은 매개변수 집합에서 최적해가 존재할 때 최소점이 같다.',
      '저장된 Gaussian 측정은 ' + exact(pr.gaussian) +
      '이다. 닫힌 식과 PDF의 직접 음의 로그합을 대조한 identity_error의 기준은 1e-14다. 원본 노트북의 gaussian-mle-check 셀에는 관측·예측 벡터와 별도의 독립 NLL 상수가 있다. 표의 NLL과 MSE는 값이 다르다는 사실을 함께 보여준다.',
      '꼬리 질문은 “분산도 학습하면 로그 분산 항을 버려도 되나?”이다. 분산이 입력이나 매개변수에 의존하면 로그 분산 항과 오차별 가중치를 유지해야 한다. 연속 밀도는 일을 넘을 수 있어 Gaussian NLL이 음수일 수도 있다. 합 손실과 평균 손실은 같은 최적해를 갖지만 크기와 gradient는 다르다.',
      refs, '.venv/bin/python -m scripts.run_notebooks'),
  });

  refs = sources('qa-15', 'notebooks/probability_loss.ipynb#bernoulli-mle-derivation',
    'notebooks/probability_loss.ipynb#bce-check',
    'outputs/probability_metrics.json', 'docs/MATH_DERIVATIONS.md');
  slides.push({
    id: 'qa-15-bernoulli', title: '#15 Bernoulli의 음의 로그는 왜 BCE인가요?', kind: 'formula',
    lead: '관측 레이블의 확률을 곱하고 음의 로그를 취하면 BCE의 합이 됩니다',
    formula: ['L=∏ᵢpᵢʸⁱ(1−pᵢ)¹⁻ʸⁱ',
      'NLL=−Σᵢ[yᵢln pᵢ+(1−yᵢ)ln(1−pᵢ)]',
      'pᵢ=σ(zᵢ) → ∂Lsum/∂zᵢ=pᵢ−yᵢ',
      '평균 손실 Lmean=Lsum/n'],
    table: { headers: ['y / p', '합 BCE', '평균 BCE'], rows: [[
      vector(pr.bce.y) + ' / ' + vector(pr.bce.probabilities),
      number(pr.bce.sum_loss, 10), number(pr.bce.mean_loss, 10)]], weights: [2, 1.5, 1.5] },
    interpretation: '불가능한 관측 사건은 손실 +∞입니다. 확률을 clipping하면 다른 목적함수가 됩니다.',
    sources: refs,
    notes: notes(
      'Bernoulli의 한 표본 확률은 pʸ(1−p)¹⁻ʸ이고 레이블은 영 또는 일이다. 독립 표본의 확률을 곱한 likelihood에 음의 로그를 취하면 BCE 합이 된다. 확률과 레이블은 같은 길이의 벡터이고 표본별 손실은 같은 shape, 합과 평균은 scalar다.',
      'p=σ(z)를 대입하면 log(1+exp(z))−yz가 된다. 미분은 σ(z)−y이며 평균 손실에서는 n으로 나눈다. 이진 레이블에 한해 동치인 logaddexp(0,(1−2y)z)를 쓰면 큰 양의 z에서 마지막 뺄셈의 취소를 피할 수 있다. 이 식을 soft target에 그대로 적용하면 원래 BCE와 다른 함수가 된다.',
      '합=' + exact(pr.bce.sum_loss) + ', 평균=' + exact(pr.bce.mean_loss) +
      ', 안정 logit 합=' + exact(pr.bce.logit_sum_loss) + ', gradient 최대오차=' +
      exact(pr.bce.max_gradient_error) + '다. y=1,z=40에서 양의 손실=' +
      exact(pr.bce.positive_tail_at_40) + ', 직접 뺄셈 값=' +
      exact(pr.bce.subtractive_formula_at_40) + '이다. 손실 비교 atol=1e-15, gradient 비교 atol=1e-8이다.',
      '꼬리 질문은 “영 확률에 로그를 취하면 모두 실패인가?”이다. 관측한 사건에 영 확률을 주면 +∞지만 계수가 영인 로그 항은 영으로 제외한다. p=0,y=0과 p=1,y=1의 손실은 영이다. 무한 손실을 임의의 유한 수로 바꾼 뒤 원래 미분으로 검증하지 않는다.',
      refs, '.venv/bin/python -m scripts.run_notebooks'),
  });

  refs = sources('qa-15', 'notebooks/probability_loss.ipynb#categorical-mle-derivation',
    'notebooks/probability_loss.ipynb#categorical-check',
    'outputs/probability_metrics.json', 'docs/MATH_DERIVATIONS.md');
  slides.push({
    id: 'qa-15-categorical', title: '#15 다중 클래스 MLE는 어떻게 CE가 되나요?', kind: 'formula',
    lead: 'one-hot 레이블에서는 관측 클래스에 배정한 확률의 음의 로그만 남습니다',
    formula: ['L=∏ᵢ∏ⱼpᵢⱼʸⁱʲ', 'CEsum=−ΣᵢΣⱼyᵢⱼln pᵢⱼ=−Σᵢln pᵢ,cᵢ',
      'ℓ(z,c)=ln Σⱼexp(zⱼ−m)−(z_c−m)', '∂ℓ/∂zⱼ=pⱼ−yⱼ'],
    table: { headers: ['확률 / 클래스(0부터)', '확률 CE', 'logit CE'],
      rows: [[vector(pr.categorical.probabilities) + ' / ' + pr.categorical.target,
        number(pr.categorical.probability_ce, 10), number(pr.categorical.logit_ce, 10)]],
      weights: [2.1, 1.4, 1.4] },
    interpretation: '평균 손실의 gradient에는 1/n이 붙습니다. 실제 클래스의 확률 0은 손실 +∞입니다.',
    sources: refs,
    notes: notes(
      'Categorical은 여러 클래스 중 하나가 관측되는 분포다. one-hot 레이블은 실제 클래스만 일이고 나머지는 영이므로 likelihood의 곱에서 관측 클래스의 확률만 남는다. 음의 로그를 취하면 다중 클래스 Cross-Entropy를 얻는다.',
      '일반 배치 유도에서 확률·logit·one-hot은 (n,K), 표본별 손실은 (n,), 합·평균은 scalar다. 원본 노트북의 _class_vector와 CE 구현은 단일 표본 (K,)와 클래스 인덱스를 받아 scalar를 반환하므로 배치 API와 구분한다. Softmax를 대입하면 lnΣexp(z)−z_c가 된다. 공통 최댓값을 빼서 overflow와 큰 offset의 취소를 줄인다. logit 미분은 p−y이고 평균이면 n으로 나눈다.',
      '확률=' + exact(pr.categorical.probabilities) + ', 정답 클래스=' + pr.categorical.target +
      ', 확률 CE=' + exact(pr.categorical.probability_ce) + ', logit CE=' +
      exact(pr.categorical.logit_ce) + '다. 비교 허용치는 1e-15이며 극단 logit 사례의 손실=' +
      exact(pr.categorical.extreme_ce) + ', 경계 정책 검사=' + exact(pr.categorical.boundary_policy_verified) +
      '이다. 클래스 인덱스는 영부터 센다.',
      '꼬리 질문은 “정답이 아닌 클래스의 확률도 영이면 무한대인가?”이다. one-hot 계수가 영인 항은 손실에 기여하지 않는다. 실제 클래스가 영 확률이면 likelihood가 영이므로 +∞다. Bernoulli BCE와 관련되지만 K개 클래스 중 하나라는 가정과 shape을 구분해야 한다.',
      refs, '.venv/bin/python -m scripts.run_notebooks'),
  });
  const adam = bo.adam_comparisons;	// 기본 실험과 구분되는 보너스 200회 비교를 읽는다.
  refs = sources('qa-adam', 'src/adam.py', 'tests/test_adam.py',
    'outputs/bonus_optimizer_metrics.json', 'outputs/bonus_optimizer_report.md', 'docs/BONUS.md');
  slides.push({
    id: 'bonus-adam', title: '보너스 Adam은 무엇을 보정하나요?', kind: 'formula',
    lead: '시작 ' + vector(adam.circle.Adam.initial) + ', 원형 lr=' + number(adam.circle.Adam.lr) +
      ' / 타원 lr=' + number(adam.ellipse.Adam.lr) + ', ' + adam.circle.Adam.steps_budget + '회 비교',
    formula: ['mₜ=β₁mₜ₋₁+(1−β₁)gₜ,  vₜ=β₂vₜ₋₁+(1−β₂)gₜ²',
      'm̂ₜ=mₜ/(1−β₁ᵗ),  v̂ₜ=vₜ/(1−β₂ᵗ)',
      'θₜ=θₜ₋₁−lr·m̂ₜ/(√v̂ₜ+ε)',
      'β₁=' + number(adam.circle.Adam.settings.beta1) + ', β₂=' +
      number(adam.circle.Adam.settings.beta2) + ', ε=' + number(adam.circle.Adam.settings.epsilon)],
    table: { headers: ['방법', '원형 최종 반경', '타원형 최종 반경'],
      rows: ['GD', 'Momentum', 'Adam'].map(name => [name,
        number(adam.circle[name].final_radius), number(adam.ellipse[name].final_radius)]),
      weights: [1, 1.5, 1.5] },
    interpretation: '편향 보정한 이동평균으로 좌표별 갱신합니다. 같은 lr 숫자는 같은 이동량을 뜻하지 않으며 Adam이 항상 빠르지는 않습니다.',
    sources: refs,
    notes: notes(
      'Adam은 gradient의 일차 모멘트와 제곱 gradient의 이차 모멘트를 지수 이동평균으로 추적한다. 시작 상태가 영이므로 초기 평균은 영 쪽으로 치우치며 각각 1−β₁ᵗ와 1−β₂ᵗ로 나누어 보정한다. 일정한 gradient라면 보정 후의 모멘트는 각각 g와 g²가 된다.',
      '곱과 제곱과 나눗셈은 원소별이다. epsilon은 제곱근 바깥에 더하며 sqrt(vhat+epsilon)과 다른 식이다. 기존 Momentum의 v=βv+g와 Adam의 (1−β₁)g 관례를 혼합하지 않는다. 학습률이 같아도 Adam의 정규화 때문에 실제 좌표 이동량은 GD와 달라진다.',
      '시작=' + exact(adam.circle.Adam.initial) + ', 원형 lr=' + exact(adam.circle.Adam.lr) +
      ', 타원형 lr=' + exact(adam.ellipse.Adam.lr) + ', 예산=' +
      adam.circle.Adam.steps_budget + '이다. Adam 원형 최종 반경=' + exact(adam.circle.Adam.final_radius) +
      ', 타원형=' + exact(adam.ellipse.Adam.final_radius) + ', 타원형 첫 진입=' +
      exact(adam.ellipse.Adam.first_radius_0_1) + '이다. 독립 첫 두 단계 기대값은 atol=1e-14, rtol=0으로 검증한다.',
      '꼬리 질문은 “적응적 방법이면 반드시 더 빠른가?”이다. 이 설정의 타원형 Adam은 예산 안에서 반경 기준에 도달하지 못했다. 상태 shape을 바꾸려면 reset이 필요하며 잘못된 후보는 상태를 복원한다. 모멘트 underflow로 정밀도를 잃는 경우를 거부하되 정확한 subnormal을 모두 무조건 거부하는 것은 아니다.',
      refs, '.venv/bin/python -m scripts.run_bonus_optimizers --no-cache'),
  });

  const newton = bo.newton_comparison;	// 같은 gradient 종료 기준의 조기 종료 비교다.
  refs = sources('qa-newton', 'src/newton.py', 'tests/test_newton.py',
    'outputs/bonus_optimizer_metrics.json', 'outputs/bonus_newton_ellipse.png', 'docs/BONUS.md');
  slides.push({
    id: 'bonus-newton', title: '보너스 Newton은 왜 한 번에 도달하나요?', kind: 'formula',
    lead: '정확한 이차함수에서는 Hessian으로 곡률을 보정한 방향이 최소점을 가리킵니다',
    formula: ['f(θ+d)≈f(θ)+gᵀd+½dᵀHd',
      'H·direction=g,  θnext=θ−lr·direction',
      'H=diag(2,20),  g=(2x,20y) → direction=(x,y)',
      '유한·대칭·양의 정부호 Hessian을 요구합니다'],
    table: { headers: ['방법', 'lr', '수락 업데이트', '최종 ‖g‖'],
      rows: ['GD', 'Newton'].map(name => [name, number(newton[name].lr),
        String(newton[name].iterations), number(newton[name].gradient_norms.at(-1))]),
      weights: [1, 0.7, 1.2, 1.5] },
    interpretation: '종료 기준 ‖g‖≤' + number(newton.Newton.gradient_tolerance) +
      '은 같습니다. 적은 반복 수가 더 짧은 실행 시간을 보장하지는 않습니다.',
    sources: refs,
    notes: notes(
      'Newton은 현재점의 이차 근사를 최소화하는 방향을 사용한다. 근사식을 이동량 d로 미분하면 g+Hd이며 영으로 두면 d=−H⁻¹g가 된다. 구현은 역행렬을 직접 만들지 않고 선형계 H·direction=g를 푼 뒤 그 방향을 뺀다.',
      '타원형 이차함수의 Hessian은 diag(2,20), gradient는 (2x,20y)다. 선형계를 풀면 direction=(x,y)이므로 lr가 일이면 바로 원점이다. 이 함수는 이차 근사가 정확한 특별한 예제다. lr가 일보다 작은 양수이면 고정 damping이며 일반 비선형 함수의 전역 수렴은 보장하지 않는다.',
      '최대 예산=' + newton.Newton.steps_budget + ', gradient 허용치=' +
      exact(newton.Newton.gradient_tolerance) + '이다. GD는 ' + newton.GD.iterations +
      '회 후 norm=' + exact(newton.GD.gradient_norms.at(-1)) + ', Newton은 ' +
      newton.Newton.iterations + '회 후 norm=' + exact(newton.Newton.gradient_norms.at(-1)) +
      '이며 두 종료 상태는 ' + newton.GD.status + ', ' + newton.Newton.status + '다.',
      '꼬리 질문은 “Hessian이 있으면 항상 안전하고 빠른가?”이다. 밀집 행렬 저장에는 O(n²), 선형계 직접 풀이에는 보통 O(n³) 비용이 든다. 대칭성과 양의 정부호 조건을 요구하고 특이·부정부호 입력을 조용히 보정하지 않는다. Cholesky 성공만으로 조건수가 큰 문제의 전방오차가 작다는 보장도 없다.',
      refs, '.venv/bin/python -m scripts.run_bonus_optimizers --no-cache'),
  });

  refs = sources('qa-information', 'src/information_theory.py',
    'outputs/information_theory.json', 'outputs/information_theory.md',
    'outputs/information_theory.png', 'docs/BONUS.md');
  slides.push({
    id: 'bonus-information', title: '보너스 H·CE·KL은 어떻게 연결되나요?', kind: 'formula',
    lead: '자연로그 단위는 ' + it.units + '이며 유한한 경우 CE=H+KL입니다',
    formula: ['H(p)=−Σpᵢln pᵢ', 'CE(p,q)=−Σpᵢln qᵢ',
      'KL(p∥q)=Σpᵢ(ln pᵢ−ln qᵢ)', 'pᵢ=0 항은 0,  pᵢ>0·qᵢ=0이면 CE/KL=+∞'],
    table: { headers: ['p → q', 'H', 'CE', 'KL'],
      rows: [rational, reverse].map(row => [vector(row.p) + ' → ' + vector(row.q),
        number(row.entropy_nats), number(row.cross_entropy_nats), number(row.kl_nats)]),
      weights: [2.2, 1, 1, 1] },
    interpretation: 'KL은 비대칭입니다. 같은 shape·명시한 클래스 축을 사용하며 clipping과 자동 정규화를 하지 않습니다.',
    sources: refs,
    notes: notes(
      '엔트로피는 자기 분포의 평균 정보량이고 Cross-Entropy는 실제 분포 p 아래에서 예측 q의 로그손실 평균이다. KL은 두 양의 차이로 유도된다. 각 식의 p ln p 항을 더하고 빼면 CE=H+KL을 얻고 고정 p에서 두 목적의 최소점이 같음을 알 수 있다.',
      '분포는 정확히 같은 shape이어야 하고 암묵적 broadcasting을 하지 않는다. 클래스 축의 확률 합을 atol=' +
      exact(it.normalization_atol) + ', rtol=' + exact(it.normalization_rtol) +
      '으로 검사한다. 클래스 축이 비면 거부하지만 빈 배치 축은 허용된다. 계산은 p/q를 먼저 만들지 않고 로그 차이를 써서 비율 overflow를 피한다.',
      '정방향 H/CE/KL=' + exact([rational.entropy_nats, rational.cross_entropy_nats, rational.kl_nats]) +
      ', 역방향 KL=' + exact(reverse.kl_nats) + '이다. 극소 q 사례의 CE=' +
      exact(tiny.cross_entropy_nats) + ', 유한 곡선의 항등식 최대오차=' +
      exact(it.curve_checks.maximum_identity_error) + '다. JSON의 무한 결과는 문자열 +inf로 기록하고 설명을 함께 남긴다.',
      '꼬리 질문은 “KL은 거리이며 모든 허용 입력의 계산값이 비음수인가?”이다. 일반적으로 비대칭이므로 거리 metric이 아니다. 정확히 정규화된 분포의 수학적 비음수성과 합 오차를 허용한 입력은 구분한다. 허용한 질량 차이로 작은 음수가 나올 수 있으며 임의로 영으로 바꾸지 않는다. 무한 결과에 ∞−∞ 검사를 적용하지 않는다.',
      refs, '.venv/bin/python -m scripts.run_information_theory'),
  });

  refs = sources('qa-common', 'requirements-lock.txt', 'src/utils/random.py',
    'src/utils/cache_manager.py', 'scripts/run_notebooks.py', 'scripts/run_submission_reports.py');
  slides.push({
    id: 'common-reproduction', title: '같은 결과는 어떻게 재현하나요?', kind: 'text',
    lead: 'iMac의 VSCode에서 프로젝트 가상환경과 같은 입력·실행 순서를 사용합니다',
    lines: ['Python 3.11 잠금 환경 · dtype=' + la.dtype + ' · NumPy ' + la.numpy_version,
      '독립 실험 시작 시 seed ' + la.seed + '를 한 번 설정합니다',
      '새 커널에서 노트북을 처음부터 실행합니다',
      '캐시 사용/미사용의 값과 기존 JSON을 대조합니다'],
    formula: ['.venv/bin/python -m scripts.run_submission_reports',
      '--output-dir .cache/imac-demo',
      '.venv/bin/python scripts/update_registry.py --check'],
    interpretation: '첫 두 줄은 하나의 명령입니다. 시연 출력은 .cache/imac-demo에 저장합니다.',
    sources: refs,
    notes: notes(
      '재현은 seed 하나를 맞추는 것보다 넓은 조건이다. Python과 잠금 의존성, 입력 배열, dtype, 알고리즘 설정, 난수 호출 순서를 함께 기록한다. iMac의 VSCode 터미널에서는 저장소 루트에서 .venv/bin/python을 명시해 다른 인터프리터가 선택되지 않도록 한다.',
      'seed는 각 독립 실험의 진입점에서 공통 함수로 한 번만 설정한다. import나 개별 수학 함수마다 다시 설정하면 호출자의 난수 흐름이 바뀐다. 노트북은 기존 출력과 실행 순서를 초기화한 새 커널에서 실행한다. 캐시는 전체 입력과 shape/dtype, 버전, 매개변수, 관련 seed를 SHA-256 키에 반영한다.',
      '주입된 기본 측정 환경은 seed=' + exact(la.seed) + ', dtype=' + exact(la.dtype) +
      ', NumPy=' + exact(la.numpy_version) + '다. 통합 스크립트는 세 실험과 두 노트북을 임시 위치에서 실행하고 기존 측정값의 구조·값 및 독립 정확도를 대조한다. source와 JSON 해시는 생성 보고서에 남는다. PNG와 노트북 원본을 시연 출력으로 덮어쓰지 않는다.',
      '꼬리 질문은 “캐시는 무엇을 절약하나?”이다. 비용이 큰 SVD·meshgrid·경로 계산을 재사용하며 Redis 서버가 아니다. --no-cache를 제공하는 개별 실험에서는 캐시 읽기와 쓰기를 모두 생략한다. 작은 정보 이론 합은 무조건 캐시하지 않는다. python -O는 assert를 제거할 수 있어 검증에 쓰지 않는다.',
      refs, '.venv/bin/python -m scripts.run_submission_reports --output-dir .cache/imac-demo\n' +
      '.venv/bin/python -m unittest discover -s tests -v\n' +
      '.venv/bin/python -m pip check\n' +
      '.venv/bin/python scripts/update_registry.py --check'),
  });

  refs = sources('qa-evaluation', 'docs/worklogs/issue-31.md', 'Naito_AI_eval_result.md',
    'MISSION_EVIDENCE.md', 'docs/ACCEPTANCE.md');
  slides.push({
    id: 'common-evaluation', title: '내부 검증과 외부 점수는 같은 사실인가요?', kind: 'table',
    lead: '계산의 정확도, 원격 제출 상태, 평가기가 읽은 입력과 판정은 구분합니다',
    table: { headers: ['기록', '확인한 사실', '뜻하지 않는 것'], rows: [
      ['내부 회귀', '확정 PR에서 210개 검사 통과', '모든 입력·모든 조건의 보장'],
      ['현재 근거 원문', 'main 2edda38', '평가기의 전체 파일 수집 확인'],
      ['기존 3차 평가', '40% · 6/15', '현재 자료의 재평가 점수'],
      ['원인 분석', '수집 범위·판단 변화 조사 필요', '캐시·길이 제한 원인 확정']],
      weights: [1, 1.9, 2] },
    interpretation: '40%는 기존 3차 외부 평가 결과입니다. 현재 자료의 재평가나 점수 회복을 주장하지 않습니다.',
    sources: refs,
    notes: notes(
      '내부 검증은 특정 입력과 수식과 허용치에서 구현이 맞게 동작했는지를 확인한다. 외부 평가는 실제로 전달된 입력과 평가 절차의 판정이다. 저장소에 파일이 존재한다는 사실과 평가기가 그 파일을 읽었다는 사실도 서로 다르다. 두 종류의 결과를 한 숫자로 합치지 않는다.',
      '이 장의 배포 기록은 실험 JSON의 관측값과 별도로 확정된 작업 기록에서 온 메타데이터다. 기준 소스는 main 2edda38이고 최종 검증 PR head는 828415d다. 해당 head의 회귀 210개가 Windows 20.776초, macOS 36.276초, Linux 19.377초에 통과했다. 운영체제별 시간은 이 실행 기록이며 성능 우열 실험이 아니다.',
      '외부 최신 판정은 기존 3차 평가의 40%, 열다섯 항목 중 여섯 통과이며 세 차례 평가 기회가 소진됐다. 현재 근거 원문 main 2edda38을 새로 평가한 점수가 아니다. 이전 원문은 시점별로 보존한다. 앞 장들에 사용한 계산값은 기존 JSON에서 읽지만 이 장의 평가 점수와 CI 시간은 평가 원문과 이슈 기록에서 구분해 인용한다.',
      '꼬리 질문은 “평가가 파일이 없다고 했으니 버그인가?”이다. 실제 수집 대상 SHA, 파일 목록, 추출 오류가 없으면 원인을 확정할 수 없다. 인용이 없다는 사실도 미수집의 증명은 아니다. README 길이, 확장자 필터, 부분 캐시, 판단 변동은 확인 자료가 없으면 가설로 유지한다. 원문 FAIL을 자체 PASS로 수정하지 않는다.',
      refs, '.venv/bin/python -m unittest discover -s tests -v'),
  });

  refs = sources('qa-evaluation', 'docs/EVALUATION_QA.md', 'docs/DECISIONS.md',
    'outputs/linear_algebra_metrics.json', 'outputs/optimizer_metrics.json',
    'outputs/probability_metrics.json');
  slides.push({
    id: 'common-followup', title: '반례에도 같은 답을 할 수 있나요?', kind: 'table',
    lead: '공식의 이름보다 가정이 바뀌었을 때의 결과를 설명하는 것이 중요합니다',
    table: { headers: ['꼬리 질문', '핵심 답변'], rows: [
      ['lr=.5이면 발산하나요?', '원형 GD에서는 1회에 원점 도달'],
      ['64×64에 rank 100을 쓰나요?', '불가능하므로 128×128 실험을 분리'],
      ['Momentum은 항상 더 빠른가요?', '첫 진입·유지·최종오차를 따로 비교'],
      ['NLL과 MSE는 같은 값인가요?', '고정 분산 가정에서 최적해가 같음'],
      ['작은 잔차는 지배값의 증명인가요?', '다른 고유쌍일 수도 있음']], weights: [1.8, 2.4] },
    interpretation: '답변은 가정 → 식 → 실측 → 한계 순서로 연결합니다. 자세한 근거는 각 항목의 노트에 있습니다.',
    sources: refs,
    notes: notes(
      '꼬리 질문은 다음처럼 짧게 답한 뒤 해당 장의 수식과 실측으로 연결한다. 행렬 곱 방향을 바꾸면 다른 변환이다. 음의 determinant는 방향 반전이지 음의 면적이 아니다. 작은 고유쌍 잔차가 지배값을 보장하지 않는다. SVD rank를 높이면 복원은 좋아져도 저장량이 늘 수 있다.',
      '중심차분의 작은 h는 반올림 오차를 키울 수 있고 이차함수와 sine의 절단오차는 다르다. 원점 gradient에는 방향이 없다. BCE와 Sigmoid를 합성하면 δ₂=p−y이며 첫 층 gradient의 두 번째 열은 입력 두 번째 성분이 영이어서 영이다. 네 자리 표시는 독립 상수와 중심차분 검증을 대체하지 않는다.',
      '원형 lr=.5 사례의 최종 반경은 ' + exact(op.learning_rates['0.5'].final_norm) +
      '이다. 기본 Momentum의 첫 진입=' + exact(circle.Momentum.first_radius_0_1) +
      ', 유지 시작=' + exact(circle.Momentum.remaining_inside_radius_0_1_through_budget) +
      '를 구별한다. Gaussian NLL=' + exact(pr.gaussian.nll) + ', MSE=' + exact(pr.gaussian.mse) +
      '는 같은 숫자가 아니다. 원본 수치와 기준을 함께 설명한다.',
      '남은 답변도 조건을 붙인다. PDF 높이와 PMF 질량은 다르다. Softmax는 축이 틀려도 합이 일일 수 있다. 불가능 사건의 CE를 clipping하면 목적함수가 바뀐다. Adam의 같은 lr가 같은 이동량은 아니며 Newton 반복 수는 시간 비교가 아니다. KL은 비대칭이고 근사 정규화의 작은 음수와 무한 손실을 구분한다. 내부 성공과 외부 PASS는 각각의 증거로 확인한다.',
      refs),
  });
  return slides;	// 입력을 변경하지 않고 전체 평가 질문 원고를 반환한다.
}
