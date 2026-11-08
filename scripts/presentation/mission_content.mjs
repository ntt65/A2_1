/**
 * A2_1 미션 발표의 한국어 본문과 발표자 노트를 생성한다.
 * 기존 JSON의 실측값과 원본 PNG 경로를 연결하며 실험 실행이나 파일 쓰기는 하지 않는다.
 */

/** 유한 실측 scalar를 표시하며 null은 미도달, +inf 문자열은 수학적 무한대로 유지한다. */
function number(value, digits = 6) {
  if (value === null) return '미도달';	// 미도달을 임의의 반복 횟수로 바꾸지 않는다.
  if (value === '+inf') return '+∞';	// 원본 JSON의 무한대 표기를 보존한다.
  if (typeof value !== 'number' || !Number.isFinite(value)) {	// 잘못된 자료형과 비유한 측정값을 거부한다.
    throw new TypeError('실측값은 유한 숫자, null 또는 +inf 문자열이어야 합니다.');	// 누락값을 그럴듯한 숫자로 대체하지 않는다.
  }
  if (value === 0) return '0';	// 표시에서만 부호 있는 영을 정리한다.
  if (Math.abs(value) < 1e-4 || Math.abs(value) >= 1e6) return value.toExponential(3);	// 작은 오차와 큰 값은 지수 표기로 읽기 쉽게 만든다.
  return Number(value.toPrecision(digits)).toString();	// 실측값을 지정한 유효자리로 표시한다.
}

/** 손계산 비교용 scalar·중첩 배열을 네 자리로 표시하며 비유한 값은 TypeError로 거부한다. */
function four(value) {
  if (Array.isArray(value)) return `[${value.map(four).join(', ')}]`;	// 배열의 행·열 구조를 유지한다.
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new TypeError('실측값이 유한한 숫자가 아닙니다.');	// 잘못된 비교 자료를 거부한다.
  const text = value.toFixed(4);	// 원시 값은 바꾸지 않고 표시만 반올림한다.
  return Number(text) === 0 ? '0.0000' : text;	// 네 자리 표시에서만 음의 영을 없앤다.
}

/** 고정 숫자 입력의 중첩 배열 구조를 보존해 문자열로 표시한다. */
function vector(value) {
  return Array.isArray(value) ? `[${value.map(vector).join(', ')}]` : number(value);	// scalar와 벡터·행렬을 같은 규칙으로 표시한다.
}

/** 설명 순서와 문단 구분을 유지하며 발표자 노트 문자열을 만든다. */
function notes(...paragraphs) {
  return paragraphs.join('\n\n');	// 문단 사이에 빈 줄을 두어 발표자가 읽기 쉽게 한다.
}

/** 기존 기본·보너스 경로에서 반경·손실·첫 진입·유지 시작의 문자열 표 행을 반환한다. */
function optimizerRows(runs, bonus = false) {
  return Object.entries(runs).map(([method, run]) => [	// 기존 경로 요약값을 재계산 없이 가져온다.
    method,
    number(bonus ? run.final_radius : run.final_norm),
    number(run.losses.at(-1)),
    number(run.first_radius_0_1),
    number(run.remaining_inside_radius_0_1_through_budget),
  ]);
}

/**
 * 기존 여덟 JSON 객체에서 한국어 미션 발표 30장의 배열을 만든다.
 * 입력 키는 linear, calculus, backprop, optimizer, probability, bonus,
 * information, sensitivity다. 반환 원소에는 제목·본문·표·식·그림 경로·노트·출처가 있다.
 * 입출력 실행이나 실험 재계산, 입력 변경을 하지 않는 순수 함수다.
 * 필수 필드 누락이나 잘못된 실측값은 예외로 알리며 대체 수치를 만들지 않는다.
 * 식과 고정 입력은 직접 설명하고 실측값은 data에서 읽는다.
 * 마지막 CI 정보는 명시적으로 확인한 PR #32의 head 828415d 기록이다.
 */
export function missionSlides(data) {
  const {linear, calculus, backprop, optimizer, probability, bonus, information, sensitivity} = data;	// 실험별 기존 JSON 객체를 연결한다.
  const power = linear.power_iteration;	// 직접 반복한 고유쌍의 측정값이다.
  const eig = power.numpy_eig_comparison;	// 검증 전용 eig 비교값이다.
  const small = linear.svd.find(row => row.shape[0] === 64);	// 작은 이미지 본 실험을 고른다.
  const large = linear.svd.find(row => row.shape[0] === 128);	// 지정 rank를 위한 보충 실험을 고른다.
  const derivative = calculus.derivative;	// 고정 간격 중심차분 결과다.
  const gradient = calculus.gradient;	// 기울기·접선 수치 검증 결과다.
  const study = calculus.step_size_study;	// 간격별 원시 오차 표본이다.
  const passingH = study.quadratic_error.filter(error => error <= derivative.tolerance).length;	// 측정 오차에서 통과 표본 수만 집계한다.
  const params = backprop.parameters;	// 신경망 고정 입력이다.
  const forward = backprop.forward;	// 순전파 실측값이다.
  const backward = backprop.backward;	// 역전파 실측값이다.
  const checkpoints = {...forward, ...backward};	// 입력 객체를 바꾸지 않고 비교 항목을 모은다.
  const circle = optimizer.comparisons.circle;	// 원형 기본 최적화 결과다.
  const ellipse = optimizer.comparisons.ellipse;	// 타원형 기본 최적화 결과다.
  const gaussian = probability.gaussian;	// Gaussian NLL 수치 예제다.
  const bce = probability.bce;	// Bernoulli BCE 수치 예제다.
  const categorical = probability.categorical;	// categorical CE 수치 예제다.
  const softmax = probability.softmax;	// 축별 확률 정규화 결과다.
  const adam = bonus.adam_comparisons;	// 보너스 세 방법의 경로 비교다.
  const adamSettings = adam.circle.Adam.settings;	// 실제 사용한 모멘트 계수와 epsilon이다.
  const newton = bonus.newton_comparison;	// 같은 종료 기준의 Newton·GD 비교다.
  const rational = information.examples.find(row => row.name === 'rational_forward');	// 독립 유리확률 기준 예제다.
  const reversed = information.examples.find(row => row.name === 'rational_reverse');	// KL 비대칭을 확인한 역순 예제다.
  const curve = information.curve_checks;	// 정보 이론 곡선의 측정 요약이다.
  const radiusHeaders = ['방법', '최종 반경', '최종 손실', '첫 진입', '유지 시작'];	// 기본·보너스 비교 표의 같은 기준을 명시한다.

  return [	// 각 장의 본문과 상세 발표자 노트를 순서대로 반환한다.
    {
      id: 'm01', title: 'AI 학습을 수학으로 구현하기',
      lead: 'A2_1 미션 설명과 NumPy 구현', kind: 'cover',
      lines: ['선형대수 · 미적분 · 역전파 · 최적화 · 확률', '기본 미션과 Adam·Newton·정보 이론'],
      notes: notes(
        '이 발표의 목적은 AI가 학습한다는 말을 실제 계산으로 설명하는 것이다. 데이터가 행렬 연산을 거쳐 예측으로 바뀌고, 예측의 손실을 미분한 뒤 각 가중치를 갱신하는 과정을 작은 예제로 끝까지 추적한다. 자동 미분 엔진을 호출하지 않고 수식과 NumPy 구현을 대응시켰다는 점이 핵심이다. 먼저 기본 미션의 선형대수, 미적분, 역전파, 최적화, 확률을 순서대로 살펴보고, 그 뒤 Adam과 Newton, 정보 이론으로 연결한다.',
        '그림은 저장소에서 실제 실험으로 만든 PNG를 사용한다. 표의 값은 같은 실험의 JSON에서 가져오며, 발표용 숫자를 별도로 만들어 성공처럼 보이게 하지 않는다. 순전파 손계산과 실제 코드의 중간값을 비교하고, 최적화에서는 최초 도달과 일정 예산 끝까지 머무는 시점을 구분한다. 확률에서는 가정과 경계 확률을 먼저 확인한 뒤 손실 함수의 의미를 설명한다.',
        '발표 중 질문을 받으면 각 수식의 입력과 shape, 검증의 독립 기준, 결론이 성립하는 조건을 함께 답한다. 예를 들어 Momentum이 언제나 빠르다거나 Newton이 항상 한 번에 끝난다고 일반화하지 않는다. 구현의 내부 검증 결과와 외부 평가 점수는 서로 다른 사실이다. 평가 수집과 항목별 대응은 별도 Q&A 자료에서 다루고, 여기서는 구현한 수학과 실측 결과를 중심으로 설명한다. 기준 본문은 main 2edda38의 README와 미션 원문이다.',
      ),
      sources: ['A2_1_mission.md', 'README.md', 'docs/ACCEPTANCE.md'],
    },
    {
      id: 'm02', title: '구현 범위와 재현 조건',
      lead: `NumPy ${linear.numpy_version}, ${linear.dtype}, 독립 실험 시작 시 seed ${linear.seed}`,
      kind: 'table',
      table: {headers: ['영역', '구현·유도', '미션'], rows: [
        ['선형대수', '행렬 변환, Power Iteration, SVD', '#1~#4'],
        ['미적분', '중심차분, Gradient 기하', '#5~#6'],
        ['역전파', '고정 신경망, 연쇄 법칙, 독립 검증', '#7~#8'],
        ['최적화', 'GD, Momentum, 학습률 실험', '#9~#12'],
        ['확률·손실', '분포, Softmax, MLE 유도', '#13~#15'],
      ], weights: [1.1, 3.2, 0.9]},
      interpretation: '수학 계산은 NumPy, 시각화는 Matplotlib, 파일·검증 도구는 표준 라이브러리를 사용한다.',
      notes: notes(
        '기본 미션의 요구를 다섯 영역으로 묶은 지도다. 선형대수와 미적분은 이후 신경망 계산을 이해하는 토대이며, 역전파는 손실의 변화율을 각 매개변수에 전달하고 최적화는 그 기울기를 실제 이동으로 바꾼다. 확률 영역은 왜 MSE나 Cross-Entropy를 선택하는지 설명한다. 각 영역은 코드만 존재하는 것으로 완료 처리하지 않고 수식, 독립 기대값, 실행 결과, 해석을 함께 제공한다.',
        `공통 계산 조건은 ${linear.dtype}이며 각 독립 실험의 진입점에서 seed ${linear.seed}를 한 번 설정한다. 함수 호출이나 import마다 전역 시드를 다시 설정하지 않는다. 선형대수·미적분·최적화는 src 모듈과 실행 스크립트로 나누었고, 역전파와 확률은 안정된 셀 ID를 가진 노트북으로 유도와 검증을 함께 보존했다. np.linalg.eig는 비교 검증에만 사용하며 Power Iteration 본체는 행렬·벡터 곱을 직접 반복한다. SVD에는 허용된 np.linalg.svd를 사용한다.`,
        '중심차분과 안정적인 Sigmoid·Softmax는 공통 유틸리티를 재사용한다. 비용이 큰 SVD나 실험 경로에는 로컬 캐시를 적용하되 입력, shape, dtype, 설정, 버전, seed가 키에 들어간다. 캐시를 끈 계산과 같은 결과인지 별도로 확인한다. 재현 명령은 마지막 장에서 iMac의 VSCode 터미널 기준으로 안내한다. 표의 미션 번호는 요구사항을 찾기 위한 대응이며 외부 평가의 PASS 표시를 대신하지 않는다.',
      ),
      sources: ['A2_1_mission.md', 'README.md', 'src/utils/math.py', 'src/utils/cache_manager.py', '.agent/registry.json'],
    },
    {
      id: 'm03', title: '행렬이 공간을 바꾸는 방식',
      lead: '미션 #1  같은 단위 원에 회전·스케일링·전단 적용', kind: 'image',
      image: 'outputs/linear_transforms.png',
      lines: ['열벡터 묶음 P:(2,400), 변환 행렬 A:(2,2)', '출력 A@P:(2,400), 각 점에 같은 선형 변환 적용'],
      interpretation: '회전한 원의 윤곽은 겹친다. 기준점의 화살표가 움직인 방향으로 회전을 확인한다.',
      notes: notes(
        '신경망의 선형층은 좌표를 섞고 늘리는 행렬 연산이다. 이 그림은 같은 단위 원의 표본점을 열벡터로 묶어 각 변환을 적용한다. 왼쪽은 45도 회전, 가운데는 가로 2배·세로 0.5배 스케일링, 오른쪽은 k=1인 수평 전단이다. 회색 점선은 원본이고 파란 실선은 변환한 도형이다. 기준점의 회색·주황색 화살표를 함께 보면 원의 외곽선만으로 드러나지 않는 회전을 확인할 수 있다.',
        '점을 행으로 쌓는 관례와 열로 쌓는 관례를 혼동하지 않아야 한다. 여기서는 P가 (2,400)이므로 A@P를 사용하고 결과도 같은 shape다. 회전은 길이와 각도를 보존한다. 스케일링은 축마다 다른 배율을 주어 원을 타원으로 바꾼다. 전단은 y에 비례해 x를 이동시켜 기울어진 형태를 만든다. 세 그림이 모두 같은 면적을 갖는 것은 이번에 선택한 행렬의 행렬식 절댓값이 같기 때문이다.',
        `실제 면적비는 회전 ${number(linear.transforms.rotation.area_ratio, 16)}, 스케일링 ${number(linear.transforms.scaling.area_ratio, 16)}, 전단 ${number(linear.transforms.shear.area_ratio, 16)}다. 다음 장에서는 도형이 비슷해 보인다는 시각적 판단을 넘어 좌표로 계산한 면적과 행렬식을 독립적으로 비교한다. 가능한 질문은 원을 회전했는데 왜 변화가 없어 보이는가이다. 원의 회전 대칭성 때문에 점 집합의 윤곽이 같을 뿐, 개별 점은 이동한다는 점을 화살표로 답한다.`,
      ),
      sources: ['src/linear_algebra.py#transformation_figure', 'scripts/run_linear_algebra.py', 'outputs/linear_algebra_metrics.json'],
    },
    {
      id: 'm04', title: '행렬식과 면적 변화율',
      lead: '미션 #2  좌표로 구한 다각형 면적과 det(A)를 독립 비교', kind: 'table',
      table: {headers: ['변환', 'det(A)', '면적비', '오차', '허용치'], rows:
        Object.entries(linear.transforms).map(([name, row]) => [name, number(row.determinant), number(row.area_ratio), number(row.error), number(row.tolerance)]),
      weights: [1.4, 0.8, 0.9, 1.2, 0.9]},
      formula: ['면적비 = area_after / area_before', '비특이: |면적비−|det(A)|| / |det(A)|', '특이: |면적비|, 분모 0인 상대오차는 사용하지 않음'],
      interpretation: '음의 행렬식은 방향 반전을 뜻한다. 도형의 면적 크기는 |det(A)|배가 된다.',
      notes: notes(
        '행렬식은 부호를 포함한 부피 변화율이며 2차원에서는 면적과 연결된다. 변환 전후의 표본 좌표에 다각형 면적 공식을 적용하고 두 면적의 비를 구한다. 이 계산을 행렬식 값에서 다시 만드는 방식으로 구현하면 자기 자신을 검증하는 문제가 생기므로, 좌표 기반 면적과 행렬식 계산을 분리했다. 단위 원을 유한 개의 점으로 근사했기 때문에 원본 면적은 정확한 π와 조금 다르지만 같은 다각형의 변환 전후 비를 비교한다.',
        `반사 사례는 det=${number(linear.transforms.reflection.determinant)}, 면적비=${number(linear.transforms.reflection.area_ratio, 16)}, 상대오차=${number(linear.transforms.reflection.error)}다. 행렬식의 음수 부호를 면적이 음수라는 의미로 읽지 않는다. 스케일링 (2,0.5)은 가로와 세로 모양을 바꾸지만 면적 배율의 곱은 1이다. 특이 사례는 한 축을 0으로 눌러 선분으로 만들며 기대 면적비도 0이다. 이때 0으로 나누는 상대오차를 쓰지 않고 절대오차를 검사한다.`,
        '표에서 비특이 행렬의 허용치는 1%에 해당하는 상대오차 0.01이며 특이 행렬의 수치 잔여 면적은 별도의 작은 절대 기준으로 판단한다. 절대오차와 상대오차는 단위와 기준 크기에 따라 해석이 달라지므로 이름을 생략하지 않는다. 예상 질문은 행렬식이 1이면 항상 도형이 같으냐는 것이다. 면적이 같다는 뜻이며 각도나 모양이 같다는 뜻은 아니다. 회전, 스케일링, 전단을 함께 비교한 이유가 여기에 있다.',
      ),
      sources: ['src/linear_algebra.py#determinant_area_check', 'outputs/linear_algebra_metrics.json', 'docs/ACCEPTANCE.md'],
    },
    {
      id: 'm05', title: 'Power Iteration의 반복과 검증',
      lead: `미션 #3  A=${vector(power.matrix)}, 초기 v=${vector(power.initial)}`, kind: 'formula',
      formula: ['w = A v,  v_next = w / ‖w‖₂', 'λ = (vᵀAv) / (vᵀv)', '상대 잔차 = ‖Av−λv‖₂ / (‖A‖F ‖v‖₂)'],
      table: {headers: ['검사', '실측', '기준'], rows: [
        ['Power / eig', `${number(power.eigenvalue, 15)} / ${number(eig.numpy_eigenvalue, 15)}`, '(7+√5)/2'],
        ['고유값 상대오차', number(eig.relative_eigenvalue_error), `≤${number(eig.relative_eigenvalue_tolerance)}`],
        ['상대 잔차', number(power.relative_residual), `≤${number(eig.relative_residual_tolerance)}`],
        ['반복·종료', `${power.iterations}회`, power.reason],
      ], weights: [1.3, 2.5, 1.3]},
      interpretation: '지배 고유값은 절댓값 기준이다. 작은 잔차만으로 임의 입력의 지배성까지 증명되지는 않는다.',
      notes: notes(
        '벡터를 고유벡터들의 성분으로 생각하면 행렬을 반복해서 곱할수록 절댓값이 가장 큰 고유값의 성분이 상대적으로 우세해진다. 매 단계 길이를 정규화하는 이유는 크기 자체의 증가를 억제하고 방향을 비교하기 위해서다. Rayleigh quotient로 현재 방향의 고유값을 추정하고 Av와 λv의 차이인 잔차로 고유방정식의 만족 정도를 확인한다. 여기서 행렬 Frobenius norm과 벡터 2-norm은 서로 다른 기호와 역할을 갖는다.',
        `실험은 대칭 행렬 ${vector(power.matrix)}와 초기 벡터 ${vector(power.initial)}를 사용한다. 분석적으로 구한 기준은 (7+√5)/2이며 별도 검증 코드에서 np.linalg.eig의 결과와도 비교했다. ${power.iterations}회의 정규화 갱신 후 고유값은 ${number(power.eigenvalue, 16)}, 고유벡터는 ${vector(power.eigenvector)}, 상대 잔차는 ${number(power.relative_residual)}다. 원문의 예시 반복 수에 억지로 맞추지 않고 실제 종료 조건을 따른다.`,
        '지배 고유값의 절댓값이 다른 고유값과 분리되고 초기 벡터에 해당 방향의 성분이 있어야 일반적인 수렴 설명을 적용할 수 있다. 음의 지배 고유값은 벡터 부호를 번갈아 바꿀 수 있어 부호를 고려해야 한다. 고유벡터 v와 −v는 같은 방향을 나타내므로 비교 때 부호를 정렬한다. 작은 잔차는 어떤 고유쌍에 가까움을 뜻하고, 초기 벡터가 다른 고유공간에 있으면 지배 고유값이라는 보장은 없다. 알고리즘 본체에는 eig를 호출하지 않는다는 구현 제한도 함께 답한다.',
      ),
      sources: ['src/linear_algebra.py#power_iteration', 'scripts/verify_power_iteration.py', 'outputs/power_iteration_eig_comparison.json', 'docs/DECISIONS.md#d08'],
    },
    {
      id: 'm06', title: 'SVD로 이미지 성분 줄이기',
      lead: `미션 #4  본 실험 입력 ${small.shape.join('×')}, 유효 rank만 사용`, kind: 'image',
      image: 'outputs/svd_64.png',
      table: {headers: ['k', '상대 복원오차', '저장/원본 원소'], rows: small.comparisons.map(row => [number(row.k), number(row.relative_error), number(row.stored_over_original)]), weights: [0.7, 1.5, 1.8]},
      interpretation: '큰 특잇값의 성분부터 복원한다. 선명도와 저장 원소 수를 함께 비교해야 압축 효과를 판단할 수 있다.',
      notes: notes(
        'SVD는 행렬 A를 UΣVᵀ로 분해하고 앞의 k개 특잇값 성분만 남겨 A_k를 만든다. 입력 기저 V와 출력 기저 U는 서로 다를 수 있으며 직사각형 행렬에도 적용된다. 따라서 앞 장의 정방행렬 고유방정식 Av=λv와 동일한 연산으로 설명하지 않는다. 구현은 허용된 np.linalg.svd를 사용하고 full_matrices=False의 얇은 인자를 저장한다. 복원은 U의 앞 k열, 대각 특잇값, Vᵀ의 앞 k행을 곱한다.',
        `그림은 seed ${linear.seed}로 만든 ${small.shape.join('×')} 합성 grayscale 이미지다. 풍경, 원판, 굴곡 질감과 잡음을 포함해 높은 rank에서도 남는 세부가 있다. 왼쪽은 원본이고 나머지는 k=${small.comparisons.map(row => row.k).join(', ')} 복원이다. 모든 패널은 같은 밝기 매핑을 사용한다. 완전 rank의 상대 Frobenius 오차는 ${number(small.full_rank_relative_error)}이며 허용치 1e−10 이내다. 표의 오차는 화면 표시용 clipping 이전 배열에서 계산했다.`,
        '저장 비용은 압축 PNG의 파일 크기가 아니라 U_k, 특잇값 k개, Vᵀ_k의 수치 원소 수 k(m+n+1)를 원본 mn과 비교한다. k가 커지면 근사 오차는 줄지만 인자 저장량이 원본보다 커질 수 있다. 저장비가 1보다 큰 경우에도 복원 검증으로는 의미가 있으나 공간 절감이라고 부르지 않는다. 왜 k=100이 없는지 질문하면 입력의 최소 차원이 64라 허용 rank를 넘는다고 설명하고, 다음 장의 별도 보충 실험으로 원문의 지정 rank를 확인한다.',
      ),
      sources: ['src/linear_algebra.py#svd_decompose', 'src/linear_algebra.py#svd_comparison_figure', 'scripts/run_linear_algebra.py#generate_grayscale_image', 'outputs/linear_algebra_metrics.json'],
    },
    {
      id: 'm07', title: 'rank와 복원 품질의 관계',
      lead: `미션 #4  ${large.shape.join('×')} 보충 실험으로 k=10·50·100 비교`, kind: 'image',
      image: 'outputs/svd_128.png',
      table: {headers: ['k', '상대 Frobenius 오차', '저장/원본 원소'], rows: large.comparisons.map(row => [number(row.k), number(row.relative_error), number(row.stored_over_original)]), weights: [0.7, 1.8, 1.7]},
      interpretation: `k=${large.comparisons.at(-1).k}의 오차는 ${number(large.comparisons.at(-1).relative_error)}, 저장비는 ${number(large.comparisons.at(-1).stored_over_original)}배다.`,
      notes: notes(
        '원문에는 64×64 이하 이미지 가이드와 k=10,50,100 비교 요구가 함께 있다. 두 조건을 한 입력에 강제로 적용하면 k=100이 유효한 rank라는 잘못된 설명이 된다. 따라서 앞 장의 작은 본 실험을 유지하고 이 장에서 별도의 128×128 보충 실험을 수행했다. rank 범위를 조용히 잘라내거나 64를 100으로 표시하지 않는다. 이 구분은 결정 기록 D09와 수용 기준에 명시했다.',
        `실제 상대 복원오차는 ${large.comparisons.map(row => `k=${row.k}에서 ${number(row.relative_error)}`).join(', ')}다. 큰 k에서 높은 주파수의 질감과 작은 세부가 더 복원되는지 그림을 본 뒤 표의 오차를 확인한다. 마지막 저장비 ${number(large.comparisons.at(-1).stored_over_original)}는 1보다 크므로 원본 수치 배열보다 저장 원소가 많다. 완전 rank ${large.numerical_rank}로 별도 확인한 복원오차는 ${number(large.full_rank_relative_error)}다.`,
        '그림 제목의 원본 크기는 계산 입력 배열의 shape이고, 슬라이드에 표시한 PNG 캔버스의 픽셀 수와 다르다. 축과 제목이 포함된 큰 그림 전체를 64px로 축소해야 한다는 뜻도 아니다. 두 크기의 합성 이미지는 서로 다른 입력이므로 64 실험과 128 실험 사이의 시각적 차이를 rank 효과 하나로 설명하지 않는다. 비교의 핵심은 각 그림 안에서 같은 원본과 밝기 기준을 놓고 k만 바꾸었다는 점이다. 수치 오차와 저장량을 함께 제시하면 복원 품질과 압축 비용의 교환 관계를 설명할 수 있다.',
      ),
      sources: ['outputs/linear_algebra_metrics.json', 'src/linear_algebra.py#reconstruct_svd', 'docs/DECISIONS.md#d09', 'A2_1_mission.md'],
    },
    {
      id: 'm08', title: '중심차분과 간격 h',
      lead: `미션 #5  f(x)=x², x=${number(derivative.x)}, h=${number(derivative.h)}`, kind: 'image',
      image: 'outputs/derivative_error.png',
      table: {headers: ['검사', '실측', '해석·기준'], rows: [
        ['고정 h의 미분', number(derivative.numerical, 12), number(derivative.analytic)],
        ['고정 h의 절대오차', number(derivative.absolute_error), `≤${number(derivative.tolerance)}`],
        ['통과한 h 표본', `${passingH}/${study.h.length}`, '탐색 전체 통과와 구분'],
      ], weights: [1.3, 1.5, 2.0]},
      interpretation: 'x²의 중심차분은 정확 연산에서 절단오차가 0이다. sin(x)로 일반적인 큰 h의 근사오차도 확인한다.',
      notes: notes(
        '중심차분은 f(x+h)와 f(x−h)의 차이를 2h로 나누어 한 점의 변화율을 근사한다. 충분히 매끄러운 함수의 Taylor 전개에서 두 식을 빼면 짝수차 항이 상쇄되고, 2h로 나눈 뒤 주도 절단오차가 보통 h²에 비례한다. 단, x²는 3차 이상 미분이 0이므로 정확한 산술에서는 이 절단오차가 아예 사라지는 특별한 함수다. x²에서 보이는 최소오차 간격을 모든 함수의 최적 h로 일반화하면 안 된다.',
        `고정 조건의 계산값은 ${number(derivative.numerical, 16)}, 해석값은 ${number(derivative.analytic)}이고 절대오차는 ${number(derivative.absolute_error)}다. 이 값은 허용치 ${number(derivative.tolerance)}를 만족한다. 별도 탐색은 h=${number(study.h[0])}부터 ${number(study.h.at(-1))}까지 ${study.h.length}개 표본이며 그중 ${passingH}개가 같은 오차 기준을 만족한다. 탐색에 실패하는 간격이 있다는 사실을 고정 h의 성공과 섞어 숨기지 않는다.`,
        '그림 가로축은 로그 간격의 h이고 세로축은 절대오차다. sin(x)의 점선은 h² 절단오차의 주도항이며 큰 h에서는 근사 오차가 커지고 너무 작은 h에서는 비슷한 두 함수값의 뺄셈과 반올림 영향이 커질 수 있다. 오차가 정확히 0인 관측값을 임의의 양수로 바꾸지 않도록 세로축은 원점 부근을 선형으로 표시한다. 중심차분은 이후 역전파 검증에서 손실 함수만으로 기울기를 확인하는 독립 도구로 재사용한다. 검증 간격과 허용오차를 함께 공개해야 다른 사람이 같은 계산을 재현할 수 있다.',
      ),
      sources: ['src/calculus.py#numerical_derivative', 'src/utils/numerical.py', 'outputs/calculus_metrics.json', 'scripts/run_calculus.py'],
    },
    {
      id: 'm09', title: 'Gradient와 등고선의 수직 관계',
      lead: '미션 #6  f(x,y)=x²+y², ∇f=(2x,2y)', kind: 'image',
      image: 'outputs/gradient_geometry.png',
      table: {headers: ['검사', '최대오차', '허용치'], rows: [
        ['수치·해석 gradient', number(gradient.maximum_absolute_error), number(gradient.tolerance)],
        ['정규화 내적', number(gradient.maximum_normalized_dot), number(gradient.orthogonality_tolerance)],
        ['원점 gradient', vector(gradient.origin_gradient), '방향 없음'],
      ], weights: [2, 1.3, 1.3]},
      interpretation: '주황색은 증가 방향, 파란색은 등고선의 접선이다. 그림의 화살표 길이는 방향 표시를 위해 정규화했다.',
      notes: notes(
        '등고선은 함수값이 같은 점들을 잇는다. 등고선을 따라 아주 조금 움직일 때 함수값의 변화율은 0이므로 그 접선과 gradient의 내적이 0이다. 이 예제의 gradient는 (2x,2y)이며 접선 방향을 (−y,x)로 고르면 내적이 −2xy+2xy=0임을 바로 확인할 수 있다. 경사하강법은 증가가 가장 빠른 gradient의 반대 방향으로 이동한다. 다만 유한한 학습률의 한 번 이동이 언제나 손실을 줄이는지는 별도의 안정 조건이 필요하다.',
        `반경 2의 ${gradient.nonzero_tangents.length}개 점과 원점, 총 ${gradient.points.length}개 표본에서 검증했다. 수치 gradient 배열의 shape는 (${gradient.points.length},2)이며 해석값과의 최대오차는 ${number(gradient.maximum_absolute_error)}다. 원점 이외의 정규화 내적 최대 절댓값은 ${number(gradient.maximum_normalized_dot)}이다. 원점의 수치·해석 gradient는 ${vector(gradient.origin_gradient)}이고 방향은 null로 기록했다. 영벡터를 정규화해 방향을 만들어내지 않는다.`,
        '그림의 주황색과 파란색 화살표는 각각 길이를 일정하게 맞춰 방향을 읽기 쉽게 만들었다. 따라서 화면에서 길이가 비슷하다고 실제 gradient와 접선 벡터의 norm이 같다는 결론을 내리면 안 된다. 실제 크기는 원시 수치로 따로 검증한다. 원점을 x 표시만 하고 화살표를 생략한 이유도 이 구분 때문이다. 예상 질문은 눈으로 수직처럼 보이는 것으로 충분하냐는 것이다. 시각화는 직관을 제공하고 내적의 수치 검증이 그 관계를 별도 기준으로 확인한다고 답한다.',
      ),
      sources: ['src/calculus.py#gradient_figure', 'outputs/calculus_metrics.json', 'scripts/run_calculus.py'],
    },
    {
      id: 'm10', title: '고정 신경망과 텐서 shape',
      lead: '미션 #7  입력 2개, 은닉 2개, 출력 1개의 단일 표본', kind: 'table',
      table: {headers: ['기호', '고정 입력', 'shape'], rows: [
        ['x', vector(params.x), '(2,)'],
        ['W1', vector(params.W1), '(2,2)'],
        ['b1', vector(params.b1), '(2,)'],
        ['W2', vector(params.W2), '(2,)'],
        ['b2', number(params.b2), 'scalar ()'],
        ['정답 y', number(params.y_true), 'scalar ()'],
      ], weights: [1, 3.1, 1.1]},
      lines: ['은닉층과 출력층 활성화는 Sigmoid', '손실은 단일 표본 BCE, 배치 평균 계수 없음'],
      interpretation: '행렬 곱·내적·원소별 곱·외적을 각 단계의 shape와 함께 구분한다.',
      notes: notes(
        '이 고정 예제는 자동 미분 없이 신경망의 모든 계산을 추적하기 위한 최소한의 구조다. 입력을 열벡터로 해석하고 W1@x+b1을 계산하면 은닉층 선형 출력 z1의 shape는 (2,)다. Sigmoid를 원소별 적용한 a1도 같은 shape다. W2와 a1의 내적에 scalar b2를 더하면 출력 logit z2가 scalar이고 마지막 Sigmoid의 예측도 scalar다. 손실은 정답이 1인 단일 표본 BCE다.',
        `사용한 값은 x=${vector(params.x)}, W1=${vector(params.W1)}, b1=${vector(params.b1)}, W2=${vector(params.W2)}, b2=${number(params.b2)}, y=${number(params.y_true)}다. 숫자는 노트북의 고정 입력과 JSON을 그대로 따른다. W1의 각 행은 한 은닉 뉴런으로 들어오는 입력 가중치이며 출력 W2는 두 은닉 활성화의 가중합을 만든다. bias는 입력에 의존하지 않는 이동량이다. 매개변수는 W1의 네 원소, b1의 두 원소, W2의 두 원소, b2 하나로 총 ${backprop.parameter_count}개다.`,
        'shape 표기는 구현 오류를 예방하는 핵심 근거다. 같은 기호를 열벡터와 행벡터로 오가면서 전치나 곱을 생략하면 결과의 차원이 맞지 않거나 NumPy broadcasting으로 의도하지 않은 계산이 성공할 수 있다. 이 예제는 배치 차원을 추가하지 않았으므로 평균 손실의 1/n 계수가 없다. 다음 장의 순전파 값을 먼저 확인한 뒤 출력층에서 입력층 방향으로 연쇄 법칙을 적용한다. 수식을 외우는 것보다 각 미분의 입력과 출력 shape가 무엇인지 말할 수 있어야 한다.',
      ),
      sources: ['notebooks/backprop_derivation.ipynb#backprop-fixed-shapes', 'notebooks/backprop_derivation.ipynb#backprop-fixed-input', 'outputs/backprop_metrics.json'],
    },
    {
      id: 'm11', title: '순전파 손계산',
      lead: '미션 #7  입력에서 예측과 BCE 손실까지', kind: 'table',
      table: {headers: ['단계', '계산식', '실측값', 'shape'], rows: [
        ['z1', 'W1@x+b1', vector(forward.z1), '(2,)'],
        ['a1', 'σ(z1)', vector(forward.a1), '(2,)'],
        ['z2', 'W2@a1+b2', number(forward.z2, 12), '()'],
        ['p', 'σ(z2)', number(forward.y_pred, 12), '()'],
        ['L', '−ln(p), y=1', number(forward.L, 12), '()'],
      ], weights: [0.65, 1.55, 2.3, 0.7]},
      interpretation: `예측 ${number(forward.y_pred)}는 정답 ${number(params.y_true)}보다 작다. logit 증가 방향이 이 지점의 손실을 줄인다.`,
      notes: notes(
        'Sigmoid는 σ(z)=1/(1+exp(−z))로 정의하고 각 은닉 성분에 독립적으로 적용한다. W1의 첫 행과 x의 내적으로 첫 은닉 logit을, 두 번째 행과 x의 내적으로 둘째 logit을 계산한다. 은닉 활성화 두 값을 W2의 가중합으로 합친 뒤 출력 Sigmoid를 적용한다. 정답이 1이므로 BCE의 두 항 중 −ln(p)만 남는다. 이 순서가 뒤의 역전파에서 거꾸로 따라갈 계산 그래프다.',
        `실제 z1=${vector(forward.z1)}, a1=${vector(forward.a1)}, z2=${number(forward.z2, 16)}, p=${number(forward.y_pred, 16)}, L=${number(forward.L, 16)}다. 표의 자릿수는 읽기 위한 표시이며 다음 계산에는 반올림한 화면 값을 다시 넣지 않는다. 특히 z2는 소수점 네 자리에서 ${four(forward.z2)}다. 원문 참고 예시에 있던 끝자리와 차이가 나더라도 고정 입력의 수식을 재계산한 값을 사용하고 원문 자체는 보존했다.`,
        '가능한 질문은 출력이 확률이면 손실의 범위가 0과 1 사이냐는 것이다. BCE는 확률의 음의 로그이므로 정답 사건의 확률이 작아질수록 제한 없이 커질 수 있다. 예측 p가 1에 가까워지면 이 표본의 손실은 0에 가까워진다. 현재 p는 포화된 극단값이 아니므로 확률 표현의 BCE와 안정적인 logit 표현을 직접 비교할 수 있다. 극단 logit에서는 작은 확률을 먼저 만들었다가 로그를 취하는 방식이 표현 한계를 겪을 수 있으며 뒤의 확률 장에서 안정식을 별도로 다룬다.',
      ),
      sources: ['notebooks/backprop_derivation.ipynb#backprop-forward-derivation', 'notebooks/backprop_derivation.ipynb#backprop-forward-numpy', 'outputs/backprop_metrics.json', 'docs/DECISIONS.md#d12'],
    },
    {
      id: 'm12', title: '출력층의 연쇄 법칙',
      lead: '미션 #7  손실을 예측, logit, 출력 가중치 순으로 미분', kind: 'formula',
      formula: [
        '∂L/∂p = −1/p                         scalar',
        'δ2 = (∂L/∂p) p(1−p) = p−1          scalar',
        '∂L/∂W2 = δ2 a1                       (2,)',
        '∂L/∂b2 = δ2                          scalar',
      ],
      lines: [`∂L/∂p=${number(backward.dL_dy_pred)}, δ2=${number(backward.dL_dz2)}`,
        `∂L/∂W2=${vector(backward.dL_dW2)}`],
      interpretation: '음수 기울기를 빼는 작은 갱신은 이 예제의 출력 logit을 높이는 방향으로 작용한다.',
      notes: notes(
        '먼저 정답 y=1인 BCE L=−ln(p)를 p로 미분하면 −1/p다. 출력 Sigmoid의 도함수는 p(1−p)이므로 두 값을 곱하면 logit z2에 대한 손실 기울기 δ2=p−1이 된다. 일반적인 Bernoulli 정답 y에 대해서는 p−y이며, 이번 슬라이드는 고정 예제의 y=1을 대입했다. Sigmoid와 BCE를 각각 미분한 항이 연쇄 법칙으로 결합하면서 식이 단순해지는 과정을 생략하지 않는다.',
        `실측 ∂L/∂p는 ${number(backward.dL_dy_pred, 14)}, δ2는 ${number(backward.dL_dz2, 14)}다. z2=W2·a1+b2에서 W2의 각 원소에 대한 미분은 해당 은닉 활성화이므로 ∂L/∂W2=${vector(backward.dL_dW2)}가 된다. b2의 미분은 1이라 ∂L/∂b2=δ2=${number(backward.dL_db2)}다. δ2는 scalar이고 a1은 (2,)이므로 scalar와 벡터의 곱으로 (2,) 기울기를 얻는다.`,
        '현재 예측이 정답보다 작으므로 δ2가 음수다. 손실을 줄이려는 국소 방향은 출력 logit을 늘리는 쪽이다. 이 예제에서 두 은닉 활성화가 양수이므로 출력 가중치 기울기도 음수이고, 충분히 작은 경사하강 갱신은 그 가중치를 늘린다. 여기서 충분히 작다는 조건을 빼고 임의의 큰 학습률에서 손실이 반드시 감소한다고 말하지 않는다. 예상 질문은 BCE와 Sigmoid에서 gradient가 단순해졌으니 중간 미분을 생략해도 되느냐이다. 구현 최적화로 결합할 수 있지만 이번 학습 목표는 연쇄 법칙의 경로와 shape를 설명하는 것이다.',
      ),
      sources: ['notebooks/backprop_derivation.ipynb#backprop-chain-rule', 'notebooks/backprop_derivation.ipynb#backprop-backward-numpy', 'docs/MATH_DERIVATIONS.md', 'outputs/backprop_metrics.json'],
    },
    {
      id: 'm13', title: '은닉층과 첫 가중치의 미분',
      lead: '미션 #7  원소별 곱과 외적을 구분해 입력층까지 전달', kind: 'formula',
      formula: [
        '∂L/∂a1 = δ2 W2                           (2,)',
        'δ1 = (δ2 W2) ⊙ a1 ⊙ (1−a1)             (2,)',
        '∂L/∂W1 = outer(δ1,x)                    (2,2)',
        '∂L/∂b1 = δ1                              (2,)',
      ],
      lines: [`δ1=${vector(backward.dL_dz1)}`, `∂L/∂W1=${four(backward.dL_dW1)}`],
      interpretation: 'x의 두 번째 성분이 0이므로 W1의 두 번째 열은 이 표본의 예측에 영향을 주지 않는다.',
      notes: notes(
        '출력 logit의 은닉 활성화에 대한 미분은 W2이므로 ∂L/∂a1=δ2W2다. 이어 은닉 Sigmoid의 도함수를 각 성분에 곱한다. 서로 다른 은닉 뉴런의 Sigmoid는 해당 logit에만 의존하므로 이 단계는 행렬 내적이 아닌 원소별 곱이다. 이렇게 얻은 δ1을 첫 선형층으로 전달한다. z1_i=Σ_j W1_ij x_j+b1_i에서 W1_ij에 대한 미분은 x_j이므로 가중치 미분의 i,j 원소는 δ1_i x_j다.',
        `따라서 ∂L/∂a1=${vector(backward.dL_da1)}, δ1=${vector(backward.dL_dz1)}, ∂L/∂W1=${vector(backward.dL_dW1)}다. 편향 미분은 ${vector(backward.dL_db1)}로 δ1과 같다. 입력 ${vector(params.x)}의 두 번째 성분이 0이라 외적의 두 번째 열도 0이다. NumPy 원시 배열에는 −0.0이 남을 수 있지만 표시할 때만 0.0000으로 정규화하며 원시 수치나 비교 기준을 변경하지 않는다.`,
        '가능한 질문은 두 번째 열의 가중치는 앞으로도 학습되지 않느냐이다. 이 표본에서만 그 입력 성분이 0인 것이므로 다른 표본에서 두 번째 입력이 활성화되면 그 열의 기울기도 생길 수 있다. 또 다른 질문은 왜 x와 δ1의 순서를 바꾸면 안 되느냐이다. W1의 행은 은닉 뉴런, 열은 입력 특징에 해당하므로 outer(δ1,x)의 축 의미가 맞아야 한다. 역전파의 필수 여섯 미분과 두 편향 기울기를 모두 구했으며 다음 장에서 손계산 상수와 독립 중심차분으로 검증한다.',
      ),
      sources: ['notebooks/backprop_derivation.ipynb#backprop-chain-rule', 'docs/MATH_DERIVATIONS.md', 'outputs/backprop_metrics.json'],
    },
    {
      id: 'm14-forward', title: '순전파 손계산과 NumPy 비교',
      lead: `미션 #8  순전파 ${Object.keys(forward).length}개 체크포인트의 네 자리 일치`, kind: 'table',
      table: {headers: ['항목', 'shape', '손계산·NumPy 공통 4자리', '최대 절대오차'], rows:
        Object.entries(forward).map(([key, value]) => [key, `(${backprop.shapes[key].join(',')}${backprop.shapes[key].length === 1 ? ',' : ''})`, four(value), number(backprop.hand_absolute_errors[key])]),
      weights: [1.35, 0.65, 3.2, 1.2]},
      interpretation: '기준값은 계산 결과를 복사하지 않은 독립 상수다. 반올림은 마지막 표시에만 적용한다.',
      notes: notes(
        '앞에서 구한 순전파를 손계산 기준 상수와 NumPy 실측으로 비교한다. 기준 상수는 NumPy 출력 배열을 다시 가져와 정답으로 삼은 값이 아니라 고정 입력의 닫힌 수식을 독립적으로 평가해 기록했다. 각 원소를 절대오차 5e−5, 상대 허용오차 0으로 확인하고 네 자리 반올림 표시가 실제로 같은지도 검사한다. 공통 표시 열은 두 값의 표시가 일치한다는 뜻이며, 오차 열은 표시 전에 원시 수치로 계산한 차이다.',
        `순전파 항목은 ${Object.keys(forward).join(', ')}다. z1은 ${vector(forward.z1)}, 은닉 활성화는 ${vector(forward.a1)}, 출력 logit은 ${number(forward.z2, 16)}, 출력 예측은 ${number(forward.y_pred, 16)}, BCE 손실은 ${number(forward.L, 16)}다. 이 예제에서는 표의 순전파 오차가 저장된 기준과 일치한다. 전체 순전파·역전파 ${Object.keys(checkpoints).length}개를 합친 손계산 최대오차는 ${number(backprop.max_hand_absolute_error)}이며 다음 장에서 역전파의 작은 차이까지 보여 준다.`,
        '중간값을 네 자리로 잘라 다음 계산에 다시 넣으면 뒤의 값이 달라질 수 있다. 그래서 계산은 float64와 충분한 정밀도의 독립 기준으로 수행하고 화면 표시만 반올림한다. 원문의 참고 예시는 정답으로 간주하지 않고 고정 수식으로 재계산했으며, 원문 자체는 보존하고 차이를 결정 기록에 남겼다. 가능한 질문은 같은 값이 나오면 코드가 완전히 맞다는 뜻이냐는 것이다. 이 표는 고정 예제의 순전파를 확인하며 임의 입력을 모두 증명하지 않는다. 역전파 중간값 비교와 순전파 손실만 사용하는 중심차분을 이어서 확인한다.',
      ),
      sources: ['notebooks/backprop_derivation.ipynb#backprop-hand-reference', 'notebooks/backprop_derivation.ipynb#backprop-compare', 'outputs/backprop_metrics.json', 'docs/DECISIONS.md#d12'],
    },
    {
      id: 'm14-backward', title: '역전파 비교와 독립 중심차분',
      lead: `미션 #8  역전파 ${Object.keys(backward).length}개 체크포인트와 ${backprop.parameter_count}개 매개변수 검증`, kind: 'table',
      table: {headers: ['항목', 'shape', '손계산·NumPy 공통 4자리', '최대 절대오차'], rows:
        Object.entries(backward).map(([key, value]) => [key, `(${backprop.shapes[key].join(',')}${backprop.shapes[key].length === 1 ? ',' : ''})`, four(value), number(backprop.hand_absolute_errors[key])]),
      weights: [1.35, 0.65, 3.2, 1.2]},
      interpretation: `손계산 최대오차 ${number(backprop.max_hand_absolute_error)}, ${backprop.parameter_count}개 매개변수 중심차분 최대오차 ${number(backprop.max_numerical_gradient_error)}`, 
      notes: notes(
        '직전 순전파 표와 이 표를 합쳐 앞에서 유도한 모든 체크포인트를 확인한다. 손계산 기준은 NumPy의 계산 결과를 다시 가져와 정답으로 삼은 값이 아니라 닫힌 수식을 독립적으로 평가해 기록한 상수다. 각 원소를 절대오차 5e−5, 상대 허용오차 0으로 비교하고, 소수점 네 자리 표시도 실제로 같음을 검사한다. 공통 네 자리 열은 두 표시가 일치한 값을 보여 주며 최대 절대오차 열은 반올림하기 전의 원시 값으로 계산한다.',
        `기록된 four_decimal_match는 ${backprop.four_decimal_match}이고 손계산 최대오차는 ${number(backprop.max_hand_absolute_error)}다. 별도 중심차분은 h=${number(backprop.derivative_h)}로 매개변수를 하나씩 교란해 순전파 손실만 다시 계산한다. ${backprop.parameter_count}개 전체 기울기의 최대오차는 ${number(backprop.max_numerical_gradient_error)}로 허용치 1e−6 이내다. 매개변수 순서는 ${backprop.parameter_order.join(', ')}다. 각 기울기를 같은 위치의 원소끼리 비교해야 한다.`,
        '두 검증의 역할이 다르다. 중간 체크포인트는 오류가 시작된 단계를 찾기 쉽게 하고, 손실만 사용하는 중심차분은 역전파 식을 코드와 기준에 똑같이 잘못 옮기는 오류를 줄인다. 중심차분 역시 h와 반올림 영향을 받는 근사이므로 정확한 기호 증명과 동일시하지 않는다. 고정 예제의 성공은 임의의 모든 신경망 입력에서 정확함을 보장하지 않는다. 실제 노트북은 새 커널에서 순서대로 실행하고 비교 출력과 assert 성공을 남겼으며, README에도 해당 stdout을 그대로 제공한다.',
      ),
      sources: ['notebooks/backprop_derivation.ipynb#backprop-hand-reference', 'notebooks/backprop_derivation.ipynb#backprop-compare', 'notebooks/backprop_derivation.ipynb#backprop-gradient-check', 'outputs/backprop_metrics.json'],
    },
    {
      id: 'm15', title: 'GD와 Momentum의 원형 함수 비교',
      lead: `미션 #9·#11  시작 ${vector(circle.GD.initial)}, lr=${number(circle.GD.lr)}, ${circle.GD.iterations}회`, kind: 'image',
      image: 'outputs/optimizer_circle.png',
      table: {headers: radiusHeaders, rows: optimizerRows(circle), weights: [1.1, 1.3, 1.3, 0.9, 1]},
      interpretation: '첫 진입과 예산 끝까지 유지한 시점은 다르다. 원점을 지나간 사실만으로 안정적인 수렴을 판단하지 않는다.',
      notes: notes(
        '기울기를 계산한 다음에는 실제 가중치 이동으로 바꾼다. 원형 함수 f=x²+y²의 gradient는 2θ이며 GD는 θ_next=θ−lr·g로 갱신한다. 이 구현의 Momentum은 v_next=βv+g, θ_next=θ−lr·v_next이고 초기 v=0이다. gradient에 (1−β)를 곱하는 다른 관례와 혼합하지 않는다. 두 방법은 같은 목적함수, 시작점, 학습률, 업데이트 예산을 사용한다.',
        `GD의 최종 반경은 ${number(circle.GD.final_norm)}, 손실은 ${number(circle.GD.losses.at(-1))}다. 반경 0.1 최초 진입은 GD ${number(circle.GD.first_radius_0_1)}회, Momentum ${number(circle.Momentum.first_radius_0_1)}회다. 그러나 기록한 ${circle.GD.iterations}회까지 계속 안에 머무는 시작점은 각각 ${number(circle.GD.remaining_inside_radius_0_1_through_budget)}회와 ${number(circle.Momentum.remaining_inside_radius_0_1_through_budget)}회다. Momentum은 먼저 들어갔다가 관성 때문에 다시 나오는 구간이 있다.`,
        '그림 왼쪽은 등고선 위 경로이고 가운데는 손실, 오른쪽은 gradient norm이다. 사각형은 시작점, x 표시는 마지막 점이며 경로의 점은 각 업데이트를 뜻한다. 손실과 gradient 그래프는 정확한 0을 보존하도록 원점 근처를 선형으로 표시한다. 초기점을 포함해 100회 갱신의 경로 길이는 101이다. 고정 예산 실험이라 중간에 반경 기준을 만족해도 계속 계산한다. 가장 먼저 반경 안에 들어온 방법과 최종 오차가 가장 작은 방법이 다를 수 있으며, 어떤 지표로 빠르다고 말하는지 먼저 정해야 한다.',
      ),
      sources: ['src/optimizer.py#VanillaGD', 'src/optimizer.py#Momentum', 'src/optimizer.py#optimizer_comparison_figure', 'outputs/optimizer_metrics.json'],
    },
    {
      id: 'm16', title: '학습률과 수렴의 안정 범위',
      lead: '미션 #10  원형 GD의 좌표 배율은 1−2lr', kind: 'image',
      image: 'outputs/optimizer_learning_rates.png',
      table: {headers: ['lr', '좌표 배율', '최종 반경', '관측'], rows: [
        ['0.5', '0', number(optimizer.learning_rates['0.5'].final_norm), '한 번에 원점'],
        ['1', '−1', number(optimizer.learning_rates['1.0'].final_norm), '크기 일정, 부호 진동'],
        ['1.1', '−1.2', number(optimizer.learning_rates['1.1'].final_norm), '반경 증가'],
      ], weights: [0.6, 1.0, 1.25, 2.0]},
      interpretation: '0<lr<1에서 수렴한다. 원문의 “lr≥0.5이면 발산”은 이 함수의 실제 반복식과 맞지 않는다.',
      notes: notes(
        'f=x²+y²에서 gradient가 2θ이므로 GD 갱신은 θ_next=(1−2lr)θ다. 시작점이 원점이 아니면 반복의 크기는 |1−2lr|의 거듭제곱으로 결정된다. 이 절댓값이 1보다 작을 때 수렴하므로 양의 학습률 범위는 0<lr<1이다. lr=0.5이면 배율이 0이라 한 번에 원점에 도착하고, lr=1이면 −1이라 부호만 바뀌며 같은 반경을 유지한다. lr>1이면 절댓값이 1을 넘어 반경이 커진다.',
        `그림은 시작 ${vector(optimizer.learning_rates['1.1'].initial)}에서 ${optimizer.learning_rates['1.1'].iterations}회 갱신한 실제 경로다. lr=0.5의 마지막 반경은 ${number(optimizer.learning_rates['0.5'].final_norm)}, lr=1은 ${number(optimizer.learning_rates['1.0'].final_norm)}, lr=1.1은 ${number(optimizer.learning_rates['1.1'].final_norm)}다. 원점에서 시작하면 모든 경우 0에 머무르므로 발산 설명에는 비영 초기점이라는 조건이 필요하다. 경로는 유한한 관측 구간이고 장기 분류는 위 반복식으로 뒷받침한다.`,
        '세 패널은 각각 자신의 좌표 범위를 사용한다. 같은 화면 길이로 보이는 선분이 같은 좌표 크기를 뜻하지 않으므로 축 눈금과 마지막 반경을 함께 읽는다. 손실은 좌표의 제곱합이므로 한 단계의 손실 배율은 (1−2lr)²다. 원문 참고 문구를 재현하기 위해 gradient를 바꾸거나 0.5의 수렴을 발산으로 표시하지 않았다. 이 정정은 요구를 무시한 것이 아니라 주어진 함수에서 직접 유도한 결과를 보존한 것이다. 다른 목적함수에서는 곡률이 달라 안정 범위도 달라진다는 점을 다음 타원형 실험으로 연결한다.',
      ),
      sources: ['src/optimizer.py#learning_rate_figure', 'outputs/optimizer_metrics.json', 'docs/DECISIONS.md#d10', 'A2_1_mission.md'],
    },
    {
      id: 'm17', title: '타원형 곡률과 Momentum',
      lead: `미션 #12  f=x²+10y², lr=${number(ellipse.GD.lr)}, ${ellipse.GD.iterations}회`, kind: 'image',
      image: 'outputs/optimizer_ellipse.png',
      table: {headers: radiusHeaders, rows: optimizerRows(ellipse), weights: [1.1, 1.3, 1.3, 0.9, 1]},
      interpretation: '이 lr에서는 GD의 y가 첫 갱신에 0이 된다. Momentum의 이점과 진동은 실제 설정별로 판단한다.',
      notes: notes(
        '타원형 함수 f=x²+10y²의 gradient는 (2x,20y)이고 Hessian은 diag(2,20)이다. y 방향 곡률이 x 방향보다 열 배 크므로 같은 학습률에서도 두 좌표의 갱신 배율이 다르다. GD의 x 배율은 1−2lr, y 배율은 1−20lr이다. 두 절댓값이 모두 1보다 작으려면 양의 학습률은 0<lr<0.1이어야 한다. 지정된 lr=0.05에서는 y 배율이 정확히 0이라 첫 갱신 뒤 x축을 따라 이동한다.',
        `반경 0.1 최초 진입은 GD ${number(ellipse.GD.first_radius_0_1)}회, Momentum ${number(ellipse.Momentum.first_radius_0_1)}회다. 유지 시작은 각각 ${number(ellipse.GD.remaining_inside_radius_0_1_through_budget)}회와 ${number(ellipse.Momentum.remaining_inside_radius_0_1_through_budget)}회다. 최종 반경은 GD ${number(ellipse.GD.final_norm)}, Momentum ${number(ellipse.Momentum.final_norm)}이고 손실은 표에 함께 기록했다. 이번 고정 조건에서 Momentum이 더 빠르거나 매 단계 손실이 더 작다고 주장할 수 없다.`,
        'Momentum은 같은 방향의 기울기를 누적해 이동을 가속할 수 있지만 방향이 바뀌어도 이전 속도가 남아 진동할 수 있다. 원문의 설명처럼 GD가 언제나 지그재그를 만들고 Momentum이 항상 부드러운 곡선을 만든다고 가정하지 않는다. 그림의 왼쪽 경로만으로 판단하지 않고 손실과 gradient norm의 시간 변화를 함께 읽는다. 공정한 비교는 설정을 숨기지 않는 데서 시작하며, 이번 실험은 각 알고리즘의 최적 학습률을 탐색한 벤치마크가 아니다. 같은 설정에서 발견한 결과와 일반적인 알고리즘의 가능성을 구분해 설명한다.',
      ),
      sources: ['src/optimizer.py#elliptic_gradient', 'outputs/optimizer_metrics.json', 'docs/OPTIMIZATION.md', 'docs/DECISIONS.md#d11'],
    },
    {
      id: 'm18', title: '초기점이 달라지면 수렴 지표도 달라진다',
      lead: `추가 검증  고정 초기점 ${sensitivity.initial_points.length}개, 각 ${sensitivity.steps}회`, kind: 'image',
      image: 'outputs/optimizer_sensitivity.png',
      table: {headers: ['조건', '성공/전체', '첫 진입 평균', '유지 시작 평균'], rows:
        Object.entries(sensitivity.experiments).flatMap(([objective, runs]) => Object.entries(runs).map(([method, run]) => [
          `${objective} ${method}`,
          `${run.summary.first_radius_0_1.success_count}/${run.summary.sample_count}`,
          number(run.summary.first_radius_0_1.mean_successful_updates),
          number(run.summary.remaining_inside_radius_0_1_through_budget.mean_successful_updates),
        ])), weights: [1.8, 0.9, 1.3, 1.5]},
      interpretation: '서로 다른 초기점의 모집단이다. 동일 입력의 반복 실행 평균이나 모든 초기점에 대한 보장이 아니다.',
      notes: notes(
        `기본 실험의 한 시작점에만 의존한 해석을 줄이기 위해 ${vector(sensitivity.initial_points)}를 추가로 사용했다. 각 조건은 같은 고정 학습률과 ${sensitivity.steps}회 예산으로 실행하고, 목적함수와 방법별 경로 및 반경을 기록한다. 그림은 여러 초기점의 경로와 업데이트에 따른 반경을 함께 보여 준다. 원점도 모집단에 포함되므로 처음부터 반경 안에 있는 사례의 도달 횟수는 0이다.`,
        '표의 성공 수는 반경 기준을 기록 구간 안에 만족한 초기점 수다. 평균은 성공 사례의 업데이트 횟수에 대한 조건부 통계이며, 미도달을 예산 끝의 숫자로 바꿔 평균에 넣지 않는다. 분산은 모집단 분산의 ddof=0 관례를 사용한다. 이번 모집단에서는 표시한 조건의 모든 사례가 도달했지만 이 결과를 임의의 초기점이나 무한 시간의 보장으로 확대하지 않는다. 유지 시작도 마지막 관측 시점까지 계속 기준을 만족한다는 뜻이다.',
        `원형 GD의 첫 진입·유지 평균은 각각 ${number(sensitivity.experiments.circle.GD.summary.first_radius_0_1.mean_successful_updates)}회와 ${number(sensitivity.experiments.circle.GD.summary.remaining_inside_radius_0_1_through_budget.mean_successful_updates)}회다. 원형 Momentum은 ${number(sensitivity.experiments.circle.Momentum.summary.first_radius_0_1.mean_successful_updates)}회와 ${number(sensitivity.experiments.circle.Momentum.summary.remaining_inside_radius_0_1_through_budget.mean_successful_updates)}회로 차이가 커진다. 다른 초기점에서 반복한 실험과 같은 입력을 여러 번 재실행한 재현성 검사는 서로 다르다. 캐시 사용 여부의 동일성은 동일 설정의 계산을 비교하는 별도 검증으로 확인했다.`,
      ),
      sources: ['scripts/run_optimizer_sensitivity.py', 'outputs/optimizer_sensitivity.json', 'outputs/optimizer_sensitivity.md', 'docs/ACCEPTANCE.md'],
    },
    {
      id: 'm19', title: '연속 밀도와 이산 확률 질량',
      lead: '미션 #13  정규분포 PDF와 Bernoulli PMF', kind: 'image',
      images: ['outputs/probability_normal.png', 'outputs/probability_bernoulli.png'],
      lines: [
        `정규 적분오차 ${probability.normals.map(row => number(row.integral_absolute_error)).join(' / ')}`,
        `Bernoulli 질량합 오차 ${probability.bernoulli.map(row => number(row.mass_error)).join(' / ')}`,
        '정규분포의 두 번째 매개변수는 분산',
        'PDF는 면적, PMF는 막대 높이로 확률을 읽음',
      ],
      interpretation: '분산이 작으면 정규곡선은 좁고 높아진다. 두 분포 모두 전체 확률은 1이다.',
      notes: notes(
        '정규분포는 N(μ,분산) 표기를 사용한다. N(2,0.5)의 표준편차는 0.5가 아니라 √0.5다. 평균은 중심을 옮기고 분산은 퍼짐을 바꾼다. 연속 분포의 PDF 높이는 한 점이 나올 확률이 아니며, 구간의 확률은 그 구간 아래 면적으로 구한다. 특정 PDF 값이 1보다 커질 수 있어도 전체 적분이 1이면 확률 모형과 모순되지 않는다. 이 그림은 두 정규곡선의 중심과 폭을 비교한다.',
        `각 분포의 μ±8σ 구간을 ${probability.normals[0].grid_points}점으로 나누어 수치 적분했다. 적분값은 ${probability.normals.map(row => number(row.integral, 16)).join(', ')}이고 1과의 오차는 ${probability.normals.map(row => number(row.integral_absolute_error)).join(', ')}다. 이 유한 구간의 수치 검증과 무한 구간 적분의 수학적 정규화를 구분한다. Bernoulli p=${probability.bernoulli.map(row => number(row.p)).join(', ')}의 PMF는 각각 ${probability.bernoulli.map(row => vector(row.pmf)).join(', ')}이며 두 결과의 질량합 오차는 모두 표기한 값이다.`,
        'Bernoulli는 0 또는 1의 이산 결과를 모델링하므로 두 막대의 높이가 각 결과의 확률이다. 같은 p에 속한 막대 둘을 더하면 1이고 p를 0.3에서 0.7로 바꾸면 두 결과의 확률이 서로 바뀐다. 정규 PDF와 Bernoulli PMF를 연결하는 이유는 이후 회귀와 분류의 손실을 어떤 관측 모형에서 유도하는지 이해하기 위해서다. 경계 p=0과 p=1도 유효한 분포이며, 어떤 사건을 정답으로 관측했는지에 따라 뒤에서 로그 손실이 0 또는 무한대가 된다.',
      ),
      sources: ['notebooks/probability_loss.ipynb#normal-density-check', 'notebooks/probability_loss.ipynb#bernoulli-check', 'outputs/probability_metrics.json'],
    },
    {
      id: 'm20', title: '큰 logit에서도 안정적인 Softmax',
      lead: `미션 #14  입력 (2,3), 클래스 축 axis=${softmax.axis}`, kind: 'formula',
      formula: ['pⱼ = exp(zⱼ) / Σₖexp(zₖ)', 'm = maxₖzₖ', 'pⱼ = exp(zⱼ−m) / Σₖexp(zₖ−m)'],
      table: {headers: ['logits', '출력 확률', '행 합'], rows: softmax.logits.map((row, index) => [
        vector(row), vector(softmax.probabilities[index]), number(softmax.row_sums[index], 16),
      ]), weights: [1.6, 2.7, 1.3]},
      interpretation: `행 합 최대오차 ${number(softmax.max_sum_error)}, 열 축 재검증 ${number(softmax.column_sum_error)}로 기준 1e−6 이내다.`,
      notes: notes(
        'Softmax는 여러 실수 점수인 logit을 합이 1인 클래스 확률로 바꾼다. 모든 logit에서 같은 상수 m을 빼면 분자와 분모에 같은 exp(−m)이 곱해져 비율은 변하지 않는다. m을 해당 클래스 축의 최댓값으로 고르면 지수에 들어가는 값이 0 이하가 되어 큰 양수 지수의 overflow를 피할 수 있다. 입력이 행마다 한 관측인 (2,3)이므로 axis=-1은 각 행의 세 클래스를 정규화한다.',
        `첫 입력은 ${vector(softmax.logits[0])}이고 결과는 ${vector(softmax.probabilities[0])}다. 둘째 입력 ${vector(softmax.logits[1])}는 float64에서 ${vector(softmax.probabilities[1])}로 표시된다. 실제 행 합은 ${softmax.row_sums.join(', ')}이며 최대 합 오차는 ${number(softmax.max_sum_error)}다. axis=0으로 정규화하는 별도 열 검증의 오차도 ${number(softmax.column_sum_error)}다. 축을 바꾸면 한 분포로 묶는 원소가 바뀌므로 같은 계산으로 생각하면 안 된다.`,
        '아주 작은 확률은 유한한 logit에서 수학적으로 양수여도 부동소수점에서 0으로 underflow할 수 있다. 출력 확률에 곧바로 로그를 취하면 원래 유한한 logit 손실을 잘못 무한대로 만들 수 있어, 뒤의 BCE와 CE는 log-sum-exp나 안정적인 softplus 표현을 사용한다. 확률합이 맞는다는 검사만으로 모든 출력값이 의미 있게 정확하다는 보장은 없다. 유한성, 비음수성, shape 보존, 큰 입력과 여러 축을 함께 검사하고 동일 공통 Softmax를 노트북에서도 재사용한다.',
      ),
      sources: ['src/utils/math.py#softmax', 'notebooks/probability_loss.ipynb#softmax-derivation', 'notebooks/probability_loss.ipynb#softmax-check', 'outputs/probability_metrics.json'],
    },
    {
      id: 'm21', title: 'Gaussian MLE에서 MSE까지',
      lead: '미션 #15  독립 오차와 고정된 양의 공통 분산 가정', kind: 'formula',
      formula: [
        'yᵢ = fθ(xᵢ)+εᵢ,  εᵢ 독립 N(0,σ²)',
        'L(θ) = Πᵢ p(yᵢ|xᵢ,θ)',
        'NLL = (n/2)ln(2πσ²) + SSE/(2σ²)',
        'argmin NLL = argmin SSE = argmin MSE',
      ],
      table: {headers: ['n', 'σ²', 'SSE', 'MSE', 'NLL'], rows: [[number(gaussian.n), number(gaussian.variance), number(gaussian.sse), number(gaussian.mse), number(gaussian.nll, 12)]], weights: [0.5, 0.6, 0.7, 0.7, 1.6]},
      interpretation: `값은 달라도 최소점은 같다. density/NLL 항등식 오차는 ${number(gaussian.identity_error)}다.`,
      notes: notes(
        '회귀 예측의 오차를 독립인 평균 0 Gaussian으로 가정한다. 각 관측의 조건부 PDF를 곱하면 우도이며, 로그를 취하면 곱이 합으로 바뀐다. 정규 PDF의 지수 부분은 제곱잔차를 만들고 정규화 계수는 log(2πσ²) 항을 만든다. 음의 로그를 취한 NLL은 n/2·ln(2πσ²)+SSE/(2σ²)다. 분산이 고정된 양수이면 첫 항은 θ와 무관하고 두 번째 배율은 양수이므로 NLL과 SSE의 최소점이 같다.',
        `관측 [0,1,2]와 예측 [1,0.5,1.5]의 예제는 n=${number(gaussian.n)}, σ²=${number(gaussian.variance)}, SSE=${number(gaussian.sse)}, MSE=${number(gaussian.mse)}다. NLL=${number(gaussian.nll, 16)}, 표본 평균 NLL=${number(gaussian.mean_nll, 16)}이며 직접 확률밀도에서 계산한 항등식 오차는 ${number(gaussian.identity_error)}다. NLL이 MSE와 같은 숫자라는 주장은 이 표 자체와 맞지 않는다. SSE를 n으로 나눈 MSE도 n이 고정이면 같은 최소점을 갖는다.`,
        '가능한 질문은 분산도 학습하면 어떠냐는 것이다. 그 경우 log σ² 항이 더 이상 최적화 변수와 무관한 상수가 아니므로 단순히 제거할 수 없다. 관측별 분산이 다르면 잔차마다 다른 가중치가 생긴다. 독립 가정이 깨져 공분산이 생기면 일반적인 제곱오차 대신 그 구조를 반영한 이차형식이 나온다. 이 슬라이드는 지정한 가정에서 MSE를 선택하는 이유를 설명한다. 로그는 자연로그이고 likelihood는 density들의 곱이므로 연속 관측의 한 점 확률과 혼동하지 않는다.',
      ),
      sources: ['notebooks/probability_loss.ipynb#gaussian-mle-derivation', 'notebooks/probability_loss.ipynb#gaussian-mle-check', 'docs/MATH_DERIVATIONS.md', 'outputs/probability_metrics.json'],
    },
    {
      id: 'm22', title: 'Bernoulli MLE와 안정적인 BCE',
      lead: '미션 #15  이진 관측의 우도에 음의 자연로그 적용', kind: 'formula',
      formula: [
        'L = Πᵢ pᵢ^yᵢ (1−pᵢ)^(1−yᵢ)',
        'BCE_sum = −Σᵢ[yᵢ ln pᵢ+(1−yᵢ)ln(1−pᵢ)]',
        'ℓ(z,y) = (1−y)softplus(z)+y softplus(−z)',
        '∂ℓ/∂z = sigmoid(z)−y',
      ],
      lines: [`y=${vector(bce.y)}, p=${vector(bce.probabilities)}`,
        `합 ${number(bce.sum_loss, 12)}, 평균 ${number(bce.mean_loss, 12)}`,
        `기울기 최대오차 ${number(bce.max_gradient_error)}`],
      interpretation: '평균은 합을 표본 수로 나눈 값이다. 불가능한 정답 사건에 확률 0을 주면 손실은 +∞다.',
      notes: notes(
        'Bernoulli는 정답 y가 0 또는 1인 관측을 설명한다. 한 관측의 likelihood p^y(1−p)^(1−y)는 y=1이면 p, y=0이면 1−p가 된다. 독립 관측들의 우도를 곱하고 음의 자연로그를 취하면 BCE 합을 얻는다. 평균 손실은 합을 표본 수로 나누므로 고정 표본 수에서 최소점은 같지만 gradient의 배율은 달라진다. 역전파 예제의 단일 표본에는 이 평균 계수를 추가하지 않았다.',
        `실측 합은 ${number(bce.sum_loss, 16)}, 평균은 ${number(bce.mean_loss, 16)}, logit 표현의 합은 ${number(bce.logit_sum_loss, 16)}다. 해석 gradient는 ${vector(bce.analytic_sum_gradient)}이고 중심차분 최대오차는 ${number(bce.max_gradient_error)}다. 극단 logit 사례의 손실은 ${vector(bce.extreme_losses)}로 유한하다. z=40,y=1에서 안정식은 ${number(bce.positive_tail_at_40)}의 양수 꼬리를 보존하지만 logaddexp(0,z)−z를 그대로 빼면 ${number(bce.subtractive_formula_at_40)}으로 상쇄될 수 있다.`,
        'softplus(z)는 log(1+exp(z))이며 구현은 큰 z에서 직접 exp를 계산하는 대신 안정적인 식을 사용한다. 슬라이드의 양수 가중합 형태는 y가 0 또는 1일 때 불필요한 큰 두 수의 뺄셈을 피한다. 확률 입력의 경계에서는 올바른 사건에 확률 1을 주면 손실 0, 정답 사건에 확률 0을 주면 +∞다. 확률을 epsilon으로 올려 유한하게 바꾼 뒤 원래 손실의 gradient를 붙이면 서로 다른 함수를 검증하게 된다. 이 구현은 경계 의미를 유지하며 clipping으로 불가능 사건을 숨기지 않는다.',
      ),
      sources: ['notebooks/probability_loss.ipynb#bernoulli-mle-derivation', 'notebooks/probability_loss.ipynb#bce-functions', 'notebooks/probability_loss.ipynb#bce-check', 'outputs/probability_metrics.json'],
    },
    {
      id: 'm23', title: 'Categorical MLE와 Cross-Entropy',
      lead: '미션 #15  여러 클래스 중 하나의 관측과 one-hot target', kind: 'formula',
      formula: [
        'L = Πᵢ Πₖ pᵢₖ^yᵢₖ,  Σₖyᵢₖ=1',
        'CE_sum = −Σᵢ Σₖ yᵢₖ ln pᵢₖ',
        '한 표본 CE = −ln p_target',
        'logit CE = logsumexp(z) − z_target',
      ],
      table: {headers: ['확률', '정답 index', '확률 CE', 'logit CE'], rows: [[
        vector(categorical.probabilities), number(categorical.target), number(categorical.probability_ce, 12), number(categorical.logit_ce, 12),
      ]], weights: [2, 0.9, 1.3, 1.3]},
      interpretation: `극단 logit 예제의 CE도 ${number(categorical.extreme_ce)}로 계산한다. 정답 클래스의 확률이 0이면 +∞다.`,
      notes: notes(
        'Categorical 모형은 한 관측이 여러 클래스 중 하나에 속할 때 사용한다. one-hot target의 실제 클래스 항만 지수가 1이고 나머지는 0이므로 한 관측의 likelihood는 실제 클래스의 확률이다. 독립 관측의 likelihood를 곱한 뒤 음의 로그를 취하면 −ΣᵢΣₖ yᵢₖ ln pᵢₖ로 Cross-Entropy 합을 얻는다. 표본 수로 나눈 평균도 고정 표본 수에서 같은 최소점을 갖지만 gradient의 배율은 달라진다.',
        `확률 ${vector(categorical.probabilities)}의 정답 index는 ${number(categorical.target)}다. 인덱스를 0부터 세므로 이는 세 번째 클래스에 해당한다. 확률에서 계산한 CE는 ${number(categorical.probability_ce, 16)}, logit에서 계산한 CE는 ${number(categorical.logit_ce, 16)}로 일치한다. logits=[−1000,0,1000]에서 첫 클래스를 정답으로 놓은 극단 예제의 손실은 ${number(categorical.extreme_ce)}다. Softmax에서 작은 확률이 0으로 표시되더라도 안정적인 logit 계산으로 유한한 손실을 유지한다.`,
        '확률 입력 자체가 p=[0,0,1]이라면 세 번째 클래스의 손실은 0이고 첫 번째 클래스를 관측하면 +∞다. 이것은 유한 logit의 계산 표현 한계와 구분해야 한다. one-hot에서 선택되지 않은 클래스의 0·ln0 항은 기여가 0인 것으로 다루며 무조건 모든 원소에 로그를 취해 NaN을 만들지 않는다. 자연로그 단위는 nats다. 뒤의 정보 이론 보너스에서는 one-hot을 넘어 일반적인 target 분포 p와 모델 q 사이의 CE를 구현하고, entropy와 KL의 관계로 이 손실의 의미를 확장한다.',
      ),
      sources: ['notebooks/probability_loss.ipynb#categorical-mle-derivation', 'notebooks/probability_loss.ipynb#categorical-check', 'outputs/probability_metrics.json', 'docs/MATH_DERIVATIONS.md'],
    },
    {
      id: 'm24', title: 'Adam의 모멘트와 편향 보정',
      lead: '보너스 1  방향의 누적과 좌표별 크기 보정', kind: 'formula',
      formula: [
        'mₜ = β₁mₜ₋₁ + (1−β₁)gₜ',
        'vₜ = β₂vₜ₋₁ + (1−β₂)gₜ²',
        'm̂ₜ = mₜ/(1−β₁ᵗ),  v̂ₜ = vₜ/(1−β₂ᵗ)',
        'θₜ = θₜ₋₁ − α m̂ₜ/(√v̂ₜ+ε)',
      ],
      lines: [
        `β₁=${number(adamSettings.beta1)}, β₂=${number(adamSettings.beta2)}, ε=${number(adamSettings.epsilon)}`,
        'm₀=v₀=0, 제곱·나눗셈은 원소별 연산',
        'ε은 제곱근 밖, 거부된 갱신은 상태를 되돌림',
      ],
      interpretation: '같은 숫자의 학습률이라도 GD와 Adam의 실제 이동량은 다르다. 속도 비교에는 설정과 종료 지표가 필요하다.',
      notes: notes(
        'Adam은 gradient의 지수 이동평균 m과 gradient 제곱의 지수 이동평균 v를 유지한다. m은 누적된 방향을, v는 좌표별 gradient 크기를 반영한다. 두 상태를 0에서 시작하면 초기 이동평균이 0 방향으로 편향되므로 각각 1−β₁ᵗ와 1−β₂ᵗ로 나누어 보정한다. 일정한 gradient를 넣으면 보정 뒤 m̂=g, v̂=g²가 되어 왜 그 분모가 필요한지 독립적으로 확인할 수 있다. 제곱과 나눗셈은 원소별이며 epsilon은 제곱근 밖이다.',
        `비교에 사용한 계수는 β1=${number(adamSettings.beta1)}, β2=${number(adamSettings.beta2)}, epsilon=${number(adamSettings.epsilon)}다. theta와 gradient, m, v의 shape는 같은 (n,)이고 실행 횟수 t는 수락된 갱신 수다. 각 실험을 새 상태에서 시작하며 shape를 바꿀 때는 reset이 필요하다. 공개 상태는 복사본으로 반환해 외부 코드가 내부 모멘트를 몰래 바꾸지 못하게 한다. 기존 Momentum의 v=βv+g 관례와 Adam의 (1−β1)g를 혼동하지 않는다.`,
        '수치 안정성에서는 단순히 최종 theta가 유한한지만 확인해서 충분하지 않다. gradient 제곱이 underflow로 0이 되면 분모가 잘못 작아져 과도한 이동이 발생할 수 있다. 구현은 표현 정밀도를 잃는 중간 계산을 거부하고, 정확히 표현되는 subnormal은 허용한다. 후보의 손실이나 gradient 평가가 실패하면 runner가 마지막 수락 상태의 m, v, t로 복원하며 Momentum의 속도에도 같은 정책을 적용한다. 다음 두 장은 동일 시작점과 고정 예산의 실제 결과로 적응성이 항상 빠른 도달을 뜻하는지 확인한다.',
      ),
      sources: ['src/adam.py', 'src/optimizer.py#run_optimization', 'docs/BONUS.md', 'outputs/bonus_optimizer_metrics.json', 'tests/test_adam.py'],
    },
    {
      id: 'm25', title: 'Adam의 원형 함수 실험',
      lead: `보너스 1  시작 ${vector(adam.circle.GD.initial)}, lr=${number(adam.circle.GD.lr)}, ${adam.circle.GD.iterations}회`, kind: 'image',
      image: 'outputs/bonus_adam_circle.png',
      table: {headers: radiusHeaders, rows: optimizerRows(adam.circle, true), weights: [1.0, 1.3, 1.3, 0.9, 1.0]},
      interpretation: '동일한 예산의 고정 설정 비교다. 각 알고리즘의 최적 학습률을 탐색한 보편적인 순위가 아니다.',
      notes: notes(
        '원형 함수는 방향별 곡률이 같아 알고리즘의 상태 누적과 정규화 효과를 구분해서 관찰하기 쉽다. GD, Momentum, Adam에 동일한 시작점과 같은 숫자의 학습률, 같은 업데이트 예산을 주었다. 기본 미션의 100회 비교와 달리 보너스는 200회이므로 최종값을 서로 다른 예산끼리 직접 비교하지 않는다. 모든 경로는 초기점을 포함하며 고정 예산 동안 반경 기준을 만족해도 중간 종료하지 않는다.',
        `실제 최종 반경은 GD ${number(adam.circle.GD.final_radius)}, Momentum ${number(adam.circle.Momentum.final_radius)}, Adam ${number(adam.circle.Adam.final_radius)}다. 반경 ${number(adam.circle.GD.radius_threshold)}의 최초 진입은 각각 ${number(adam.circle.GD.first_radius_0_1)}, ${number(adam.circle.Momentum.first_radius_0_1)}, ${number(adam.circle.Adam.first_radius_0_1)}회다. 유지 시작은 ${number(adam.circle.GD.remaining_inside_radius_0_1_through_budget)}, ${number(adam.circle.Momentum.remaining_inside_radius_0_1_through_budget)}, ${number(adam.circle.Adam.remaining_inside_radius_0_1_through_budget)}회다. 이 조건에서는 GD의 최종 손실이 가장 작다.`,
        '그림의 왼쪽은 경로, 가운데는 손실, 오른쪽은 gradient norm이다. Adam은 gradient 제곱의 이동평균으로 나누므로 gradient가 크다고 GD처럼 비례해서 큰 이동을 하는 것은 아니다. 같은 lr가 같은 실효 이동량을 의미하지 않는 이유다. Momentum의 빠른 첫 진입과 이후 진동도 유지 지표에서 확인한다. 어느 방법이 더 좋다는 질문에는 목적함수와 설정, 비교 예산, 원하는 지표를 먼저 밝힌다. 이 실험은 직접 구현한 갱신식이 어떻게 움직이는지 설명하는 고정 예제이며 시간 성능이나 실제 대규모 신경망의 순위를 결론 내리지 않는다.',
      ),
      sources: ['scripts/run_bonus_optimizers.py', 'outputs/bonus_optimizer_metrics.json', 'outputs/bonus_optimizer_report.md', 'docs/BONUS.md'],
    },
    {
      id: 'm26', title: 'Adam의 타원형 함수 실험',
      lead: `보너스 1  f=x²+10y², lr=${number(adam.ellipse.GD.lr)}, ${adam.ellipse.GD.iterations}회`, kind: 'image',
      image: 'outputs/bonus_adam_ellipse.png',
      table: {headers: radiusHeaders, rows: optimizerRows(adam.ellipse, true), weights: [1.0, 1.3, 1.3, 0.9, 1.0]},
      interpretation: `Adam의 최종 반경 ${number(adam.ellipse.Adam.final_radius)}는 기준 ${number(adam.ellipse.Adam.radius_threshold)} 밖이다. 미도달을 그대로 기록한다.`,
      notes: notes(
        '타원형 함수에서는 두 좌표의 gradient 배율이 다르다. 같은 시작 좌표에서 Adam의 gradient 크기 정규화가 두 방향의 초기 이동을 비슷하게 만들 수 있다. 그러나 이것이 주어진 예산 안에 원하는 반경에 반드시 도달한다는 의미는 아니다. 원형과 다른 학습률을 사용한 이유는 기본 비교 조건을 유지하기 위해서이며 각 방법의 최적 학습률을 따로 찾은 결과가 아니다. Momentum과 Adam의 상태는 각 실행마다 새로 초기화한다.',
        `GD의 마지막 반경은 ${number(adam.ellipse.GD.final_radius)}, Momentum은 ${number(adam.ellipse.Momentum.final_radius)}, Adam은 ${number(adam.ellipse.Adam.final_radius)}다. Adam의 최종 손실은 ${number(adam.ellipse.Adam.losses.at(-1))}이고 반경 기준 ${number(adam.ellipse.Adam.radius_threshold)}에 대한 최초 진입과 유지 시작은 모두 ${number(adam.ellipse.Adam.first_radius_0_1)}다. JSON의 null은 숫자 0이나 마지막 반복 수로 대체하지 않는다. 이 실패 사실이 있으므로 적응적 학습률은 언제나 더 빠르다는 설명을 할 수 없다.`,
        '가능한 질문은 Adam 구현이 틀려서 실패한 것이냐이다. 갱신식은 독립된 첫 단계 기대값과 편향 보정, 경계 입력, 상태 복원으로 따로 검증했다. 구현의 정확성과 특정 설정에서 성능 기준을 만족하는지는 다른 문제다. 그림에서는 이동 방향과 손실의 실제 변화를 읽고 표에서는 정한 반경 기준의 도달 여부를 확인한다. 같은 예산의 결과를 바탕으로 한계까지 기록하는 것이 실험의 목적이다. 더 긴 예산이나 다른 학습률에서의 결과는 새 조건의 실험으로 구분해야 하며 여기서 실행하지 않은 설정의 성공을 약속하지 않는다.',
      ),
      sources: ['outputs/bonus_optimizer_metrics.json', 'outputs/bonus_optimizer_report.md', 'tests/test_adam.py', 'tests/test_bonus_optimizer_experiment.py', 'docs/BONUS.md'],
    },
    {
      id: 'm27', title: 'Newton의 이차 근사와 선형계',
      lead: `보너스 2  같은 종료 기준 ‖gradient‖₂≤${number(newton.GD.gradient_tolerance)}`, kind: 'image',
      image: 'outputs/bonus_newton_ellipse.png',
      formula: ['q(s)=f(θ)+gᵀs+(1/2)sᵀHs', 'g+Hs=0,  H d=g,  θ_next=θ−αd'],
      table: {headers: ['방법', '수락 업데이트', '최종 gradient norm'], rows: Object.entries(newton).map(([method, run]) => [method, number(run.iterations), number(run.gradient_norms.at(-1))]), weights: [1.0, 1.5, 2.2]},
      interpretation: '정확한 SPD 이차함수의 반복 수 이점이다. Hessian 구성·선형계 비용을 포함한 실행 시간 비교와 구분한다.',
      notes: notes(
        '현재점에서 변위 s에 대한 Taylor 이차 근사를 q(s)=f(θ)+gᵀs+(1/2)sᵀHs로 둔다. 이를 s로 미분해 g+Hs=0을 풀면 Newton 방향을 얻는다. H가 양의 정부호이면 이 근사는 엄격한 볼록 이차함수여서 정지점이 최소점이다. 실제 구현은 역행렬을 만들지 않고 H·d=g를 np.linalg.solve로 풀어 θ−lr·d로 이동한다. 입력 theta와 g는 (n,), Hessian은 (n,n)이며 유한성, 대칭, 양의 정부호를 검사한다.',
        `타원형의 H=diag(2,20), g=(2x,20y)이므로 d=(x,y)다. lr=${number(newton.Newton.lr)}이면 한 번에 원점에 도착한다. 같은 시작점과 최대 ${newton.GD.steps_budget}회, 같은 gradient 종료 기준 ${number(newton.GD.gradient_tolerance)}에서 GD는 ${newton.GD.iterations}회, Newton은 ${newton.Newton.iterations}회를 수락했다. GD의 마지막 gradient norm은 ${number(newton.GD.gradient_norms.at(-1))}, Newton은 ${number(newton.Newton.gradient_norms.at(-1))}다. 각 경로 길이가 다른 것은 같은 종료 기준을 만족하면 멈췄기 때문이다.`,
        '반복 수가 적다는 사실을 실행 시간이 같은 비율로 빠르다는 뜻으로 말하지 않는다. 일반 밀집 문제는 Hessian 저장 O(n²)와 선형계 풀이 O(n³) 비용이 필요하다. 구현은 singular·indefinite Hessian을 거부하고 고정 damping만 제공한다. 정확 대칭 행렬은 그대로 보존하며 허용된 작은 비대칭만 안전하게 평균하고, 전체 subnormal Hessian은 동치 스케일링으로 해결한다. Cholesky 통과만으로 조건수가 큰 문제의 방향 정확도나 일반 비선형 함수의 전역 수렴을 보장하지 않는다. 이 한계를 보너스의 장단점으로 설명한다.',
      ),
      sources: ['src/newton.py', 'src/optimizer.py#run_optimization', 'outputs/bonus_optimizer_metrics.json', 'docs/BONUS.md', 'tests/test_newton.py'],
    },
    {
      id: 'm28', title: '엔트로피·Cross-Entropy·KL',
      lead: `보너스 3  자연로그, 단위 ${information.units}`, kind: 'image',
      image: 'outputs/information_theory.png',
      formula: ['H(p)=−Σpᵢ ln pᵢ', 'CE(p,q)=−Σpᵢ ln qᵢ', 'KL(p‖q)=Σpᵢ(ln pᵢ−ln qᵢ)', 'CE(p,q)=H(p)+KL(p‖q)'],
      interpretation: `항등식 최대오차 ${number(curve.maximum_identity_error)}. p=${vector(rational.p)}에서 H=${number(rational.entropy_nats)}, KL=${number(rational.kl_nats)}.`,
      notes: notes(
        'Entropy는 target 분포 p 자체의 불확실성, Cross-Entropy는 p의 사건을 모델 q로 설명할 때의 평균 음의 로그확률이다. KL의 합을 분리하면 Σp ln p−Σp ln q가 되어 CE=H+KL을 얻는다. 고정된 p에서 H는 상수이므로 q에 대한 CE 최소화와 KL 최소화는 같은 문제다. one-hot p라면 H=0이므로 앞의 실제 클래스 음의 로그확률과 연결된다. 로그는 자연로그이므로 단위는 bits가 아니라 nats다.',
        `독립 예제 p=${vector(rational.p)}, q=${vector(rational.q)}에서 H=${number(rational.entropy_nats, 16)}, CE=${number(rational.cross_entropy_nats, 16)}, KL=${number(rational.kl_nats, 16)}다. 순서를 뒤집은 KL은 ${number(reversed.kl_nats, 16)}로 달라 거리 metric이 아니다. Bernoulli entropy의 최대는 p=${number(curve.maximum_entropy_parameter)}에서 ${number(curve.maximum_entropy_nats)}이며 고정 target에 대한 KL 최소는 q=${number(curve.minimum_kl_parameter)}에서 ${number(curve.minimum_finite_kl_nats)}다. 유한한 ${curve.finite_model_count}개 표본의 항등식 최대오차는 ${number(curve.maximum_identity_error)}다.`,
        'p=0인 항은 0으로 처리하며 양의 p에 q=0을 주면 CE와 KL은 +∞다. 이를 작은 epsilon으로 바꾸지 않는다. 그림의 유한 곡선은 무한 경계 두 점을 제외하지만 JSON에는 +inf 문자열로 의미를 보존한다. 작은 q에서는 p/q를 먼저 만들어 overflow하지 않도록 로그 차이로 계산한다. 입력 분포는 같은 shape와 지정 축을 사용하고 합의 허용오차 안에서만 받으며 자동 정규화하지 않는다. 수학적 KL 비음수 성질과 허용오차로 인한 작은 질량 차이를 구분하고, 작은 음수를 무조건 0으로 바꿔 숨기지 않는다.',
      ),
      sources: ['src/information_theory.py', 'outputs/information_theory.json', 'outputs/information_theory.md', 'scripts/run_information_theory.py', 'docs/BONUS.md'],
    },
    {
      id: 'm29', title: 'iMac에서 재현하고 결과 확인하기',
      lead: 'VSCode 터미널에서 저장소 루트를 열고 실행', kind: 'table',
      lines: [
        '.venv/bin/python -m scripts.run_submission_reports --output-dir .cache/imac-demo',
        '.venv/bin/python scripts/update_registry.py --check',
        '.venv/bin/python -m unittest discover -s tests -v',
      ],
      table: {headers: ['검증 환경', '통과 검사', '실행 시간'], rows: [
        ['Windows', '210', '20.776 s'],
        ['macOS', '210', '36.276 s'],
        ['Linux', '210', '19.377 s'],
      ], weights: [1.3, 1.2, 1.4]},
      interpretation: '기준 main 2edda38, 검증한 PR head 828415d. 내부 검증 통과는 외부 평가 점수의 회복을 의미하지 않는다.',
      notes: notes(
        'iMac의 VSCode에서 저장소 폴더를 연 뒤 터미널이 프로젝트 루트를 가리키는지 확인한다. 기준 Python 3.11과 잠금 의존성으로 만든 .venv를 사용하고, 표시한 .venv/bin/python을 직접 호출하면 활성화 여부에 의존하지 않는다. 새 환경에서는 먼저 README의 가상환경 생성과 requirements-lock.txt 설치 절차를 따른다. python -O는 assert를 제거하므로 검증에 사용하지 않는다. 첫 명령은 기존 실험과 노트북을 실행해 측정값을 확인하고 .cache/imac-demo에 시연 보고서를 만든다. run_submission_reports에는 --check 옵션이 없으며, 색인 검사는 별도의 update_registry.py --check 명령으로 수행한다.',
        '보너스를 추가 시연하려면 .venv/bin/python -m scripts.run_bonus_optimizers --no-cache와 .venv/bin/python -m scripts.run_information_theory를 사용한다. --no-cache는 해당 최적화 CLI의 캐시 읽기와 쓰기를 모두 끄는 옵션이다. 원래 PNG, JSON, 노트북을 임의 수정하지 않고 보고서의 표, 실제 stdout, 원본 링크를 연결해 결과를 확인한다. 캐시는 계산 비용을 줄이고 레지스트리는 관련 소스와 검증 위치를 찾게 해 주며 두 역할을 혼동하지 않는다. 실행 시연에서는 생성한 보고서와 명령의 종료 결과를 함께 보여 준다.',
        '완료된 PR #32의 head 828415d에서 전체 210개 검사가 Windows 20.776초, macOS 36.276초, Linux 19.377초로 통과했다. main 기준 커밋은 2edda38이다. 이 숫자는 실제 확인한 검증 기록이며 시연 때의 실행 시간이나 이후 모든 변경의 성공을 보장하는 값이 아니다. 최신 외부 평가 40%는 내부 검사 통과와 구분해 보존하며 점수 회복을 주장하지 않는다. 학습 결과의 핵심은 수식의 조건을 말하고, 구현의 shape와 실측을 제시하며, 한계를 포함해 재현할 수 있는 상태라는 점이다.',
      ),
      sources: ['README.md', 'requirements-lock.txt', 'docs/worklogs/issue-31.md', 'docs/EVALUATION_QA.md', 'Naito_AI_eval_result.md'],
    },
  ];
}
