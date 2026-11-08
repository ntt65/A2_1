# Supplementary evaluation evidence

Reproduce: `python -m scripts.run_evaluation_evidence --no-cache`.

## Stored PNG metadata

The 64x64 guide applies to the main SVD input, not entire plotted Figures.

| PNG | Canvas pixels | Actual DPI (x/y) |
|---|---:|---:|
| linear_transforms.png | 1890 x 616 | [140.0048, 140.0048] |
| svd_64.png | 1904 x 588 | [140.0048, 140.0048] |
| svd_128.png | 1904 x 588 | [140.0048, 140.0048] |
| gradient_geometry.png | 1176 x 1008 | [140.0048, 140.0048] |
| derivative_error.png | 1260 x 770 | [140.0048, 140.0048] |
| optimizer_circle.png | 2100 x 644 | [140.0048, 140.0048] |
| optimizer_ellipse.png | 2100 x 644 | [140.0048, 140.0048] |
| optimizer_learning_rates.png | 2016 x 672 | [140.0048, 140.0048] |
| probability_normal.png | 1148 x 672 | [140.0048, 140.0048] |
| probability_bernoulli.png | 1008 x 644 | [140.0048, 140.0048] |

## SVD input and rendered panels

Image Axes dimensions measured after recreating the original seeded experiment at 140 DPI.
These fractional pixel extents exclude titles and are not PNG crop metadata.

| File | Input shape | Image panel | Width x height (pixels) |
|---|---|---|---:|
| svd_64.png | [64, 64] | original | 464.332 x 464.332 |
| svd_64.png | [64, 64] | rank 10 | 464.332 x 464.332 |
| svd_64.png | [64, 64] | rank 50 | 464.332 x 464.332 |
| svd_64.png | [64, 64] | rank 64 | 464.332 x 464.332 |
| svd_128.png | [128, 128] | original | 464.332 x 464.332 |
| svd_128.png | [128, 128] | rank 10 | 464.332 x 464.332 |
| svd_128.png | [128, 128] | rank 50 | 464.332 x 464.332 |
| svd_128.png | [128, 128] | rank 100 | 464.332 x 464.332 |

## Sampled derivative h

x**2 at x=3, float64; 46/61 sampled h satisfy error <= 1e-4.
Passing consecutive sample groups: `[{'first_h': 5.6234132519034904e-12, 'last_h': 1.0, 'sample_count': 46}]`.
Minimum observed error: 0.000000e+00; sampled minimizers: `[1.0]`.
No continuous interval or universal optimal h is claimed. Quadratic central differences have zero truncation error in exact arithmetic.

## Softmax axes

| Input shape | Axis | Maximum sum error | Mean sum error |
|---|---:|---:|---:|
| [3] | -1 | 1.110223e-16 | 1.110223e-16 |
| [2, 3] | 0 | 2.220446e-16 | 2.220446e-16 |
| [2, 3] | 1 | 1.110223e-16 | 1.110223e-16 |
| [2, 3, 4] | -1 | 0.000000e+00 | 0.000000e+00 |
| [2, 3, 4] | 0 | 0.000000e+00 | 0.000000e+00 |
| [2, 3, 4] | 1 | 0.000000e+00 | 0.000000e+00 |

All six cases pass the 1e-6 tolerance; complete inputs are in the JSON.

## Intentional failure diagnostics

Both genuine fixed calculations pass. These are explicitly injected wrong observations:

- Area: diag(2,0.5), unit square; adding 0.02 to ratio 1 yields 2% error > 1%: REJECTED.
- Backprop: mission input, add 0.001 to dL/dW1[0,0]; max error > 5e-5: REJECTED.
- Signed gradient difference (observed minus independent reference): `[[0.0010000000000000078, -0.0], [6.938893903907228e-18, -0.0]]`.

The JSON records full reproduction inputs, expected/observed values, shapes, differences and tolerances.
