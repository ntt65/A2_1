# Executed Power Iteration versus np.linalg.eig verification

Reproduce from the repository root: `python -m scripts.verify_power_iteration`.
Use the Python 3.11 project environment described in README.md.

NumPy 2.2.6; float64; seed 42; matrix [[4.0, 1.0], [1.0, 3.0]]; explicit initial vector [1.0, 1.0].

```text
Power Iteration: lambda=4.618033988749896, vector=[0.8506508082738572, 0.5257311122456358]
np.linalg.eig: lambda=4.618033988749895, vector(sign aligned)=[0.85065080835204, 0.5257311121191336]
Relative eigenvalue error=1.923283e-16 <= 0.05 (5%): PASS; iterations=32, residual=6.399564e-11
```

Relative eigenvalue error = abs(lambda_manual - lambda_numpy) / abs(lambda_numpy).

Independent reference: (7 + sqrt(5)) / 2 = 4.618033988749895.
NumPy versus independent reference absolute error: 0.000000e+00.
Sign-aligned eigenvector L2 error: 1.487124e-10 <= 1.0e-08.

The reference eigenvalue is selected by argmax(abs(eigenvalues)). Eigenvectors have an arbitrary sign, so the NumPy vector is sign-aligned before comparison. np.linalg.eig is called only in this verification script and comparison tests; the algorithm in src/linear_algebra.py remains manual.

This fixed symmetric matrix has a separated dominant eigenvalue. A small residual alone does not prove dominance for arbitrary matrices or initial directions.
