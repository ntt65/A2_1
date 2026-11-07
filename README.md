# AI가 어떻게 학습하는지 수학으로 직접 풀어보기

## 1) 프로젝트 개요
본 저장소는 AI 학습의 수학적 기반(선형대수, 미적분, 최적화, 확률/통계)을 **NumPy 구현 + 시각화 + 유도 노트북**으로 재현합니다.

## 2) 폴더 구조
```text
/home/runner/work/A2_1/A2_1
├── src
│   ├── linear_algebra.py
│   ├── calculus.py
│   └── optimizer.py
├── notebooks
│   ├── backprop_derivation.ipynb
│   └── probability_loss.ipynb
├── outputs
├── agents.md
├── requirements.txt
└── README.md
```

## 3) 기초 계획
- 선형대수: 변환 시각화, determinant-면적비 검증, Power Iteration, SVD 압축
- 미적분: 중심차분 수치미분, 2변수 Gradient 시각화
- 역전파: 2층 신경망 수식 유도 + 고정 예제 손계산 + NumPy 일치 검증
- 최적화: Vanilla GD/Momentum 경로 비교 + 발산 케이스 시각화
- 확률/통계: PDF/PMF 시각화 + Softmax 합 검증 + MLE 연결 수식

## 4) 실행 계획
1. 의존성 설치
2. 각 모듈 함수 실행하여 `outputs/`에 PNG 저장
3. 노트북 실행으로 수식/검증 결과 확인
4. 오차 조건(1e-4, 1%, 5%, 1e-6) 확인

## 5) 실행 체크표
- [x] `src/linear_algebra.py` 구현
- [x] `src/calculus.py` 구현
- [x] `src/optimizer.py` 구현
- [x] `notebooks/backprop_derivation.ipynb` 작성
- [x] `notebooks/probability_loss.ipynb` 작성
- [x] `agents.md` 작성
- [x] `requirements.txt` 작성

## 6) 설치 및 실행 방법
```bash
cd /home/runner/work/A2_1/A2_1
python -m pip install -r requirements.txt
```

### 선형대수/미적분/최적화 결과 생성 예시
```python
from src.linear_algebra import (
    rotation_matrix, scaling_matrix, shear_matrix,
    visualize_transformation, determinant_area_check,
    compare_power_iteration, create_sample_image, compare_svd_compression,
)
from src.calculus import verify_quadratic_derivative, visualize_gradient_contours
from src.optimizer import run_optimizer_experiments
import numpy as np

visualize_transformation(rotation_matrix(np.pi/4), "Rotation", "outputs/transform_rotation.png")
visualize_transformation(scaling_matrix(2, 0.5), "Scaling", "outputs/transform_scaling.png")
visualize_transformation(shear_matrix(1.0), "Shear", "outputs/transform_shear.png")

print(determinant_area_check(scaling_matrix(2, 0.5)))
print(compare_power_iteration(np.array([[4, 1], [1, 3]], dtype=float)))

img = create_sample_image(64)
compare_svd_compression(img, ks=(10, 50, 100), output_path="outputs/svd_comparison.png")

print(verify_quadratic_derivative())
visualize_gradient_contours("outputs/gradient_contour.png")

print(run_optimizer_experiments("outputs"))
```

### 노트북 실행
- `notebooks/backprop_derivation.ipynb`
- `notebooks/probability_loss.ipynb`

## 7) 주요 결과 요약
- 중심차분 수치미분: `f(x)=x^2`, `x=3`에서 해석해(6) 대비 오차 `<=1e-4`
- determinant와 면적비 상대오차: `<=1%` 확인 가능
- Power Iteration 최대 고유값 오차: `<=5%` 검증 가능
- Softmax 출력합: `|sum-1| <= 1e-6`
- Vanilla GD (`lr=0.1`, 100회) 원점 반경 0.1 이내 도달 여부 확인 가능
- Momentum vs Vanilla 경로를 동일 Figure에서 비교 가능

## 8) 제약사항 준수
- 구현/계산: NumPy
- 시각화: Matplotlib (Seaborn 호환)
- 검증 목적: `np.linalg.eig`만 비교에 사용
- 자동미분/고수준 최적화 라이브러리 미사용
