"""Reproduce the information-theory bonus with exact examples and Bernoulli curves."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.information_theory import cross_entropy, entropy, kl_divergence
from src.utils.random import set_seed

__concepts__ = {
    "Information_Theory_Experiment": ["example_report", "run_experiment", "main"],
    "Information_Theory_Visualization": ["information_figure"],
}


def _json_number(value: float) -> float | str:
    """Encode finite scalars normally and +inf as the explained string '+inf'; reject NaN/-inf."""
    value = float(value)
    if np.isposinf(value):
        return "+inf"
    if not np.isfinite(value):
        raise ValueError("only finite values or positive infinity belong in the report")
    return value


def example_report() -> list[dict]:
    """Check independent constants and return scalar example records in nats.

    Each p/q is a normalized vector (n,), never broadcast or clipped.
    The rational reference values are fixed independently of implementation;
    errors use atol=1e-14 (2e-13 for the subnormal example), rtol=0.
    Boundary infinities are serialized explicitly; assertion failures propagate.
    """
    cases = [
        ("uniform_four", [.25, .25, .25, .25], [.25, .25, .25, .25],
         (1.3862943611198906, 1.3862943611198906, 0.)),
        ("point_mass", [0., 1.], [0., 1.], (0., 0., 0.)),
        ("rational_forward", [.25, .75], [.5, .5],
         (.5623351446188083, .6931471805599453, .13081203594113697)),
        ("rational_reverse", [.5, .5], [.25, .75],
         (.6931471805599453, .8369882167858358, .14384103622589045)),
        ("support_mismatch", [1., 0.], [0., 1.], (0., np.inf, np.inf)),
        ("smallest_positive_q", [1., 0.], [np.nextafter(0., 1.), 1.],
         (0., 744.4400719213812, 744.4400719213812)),
    ]
    records = []
    for label, p, q, expected in cases:
        actual = [entropy(p), cross_entropy(p, q), kl_divergence(p, q)]
        tolerance = 2e-13 if label == "smallest_positive_q" else 1e-14
        errors = []
        for measured, reference in zip(actual, expected):
            if np.isposinf(reference):
                assert np.isposinf(measured)
                errors.append(None)
            else:
                error = abs(measured - reference)
                assert error <= tolerance
                errors.append(float(error))
        if all(np.isfinite(value) for value in actual):
            assert abs(actual[1] - actual[0] - actual[2]) <= 2e-13
            assert actual[2] >= -1e-14
        records.append({"name": label, "p": p, "q": q, "shape": [len(p)],
                        "entropy_nats": _json_number(actual[0]),
                        "cross_entropy_nats": _json_number(actual[1]),
                        "kl_nats": _json_number(actual[2]),
                        "independent_expected_H_CE_KL": [_json_number(value) for value in expected],
                        "absolute_errors_H_CE_KL": errors, "atol": tolerance, "rtol": 0., "passed": True})
    return records


def information_figure(parameters: np.ndarray, binary_entropy: np.ndarray,
                       model_parameters: np.ndarray, ce: np.ndarray, divergence: np.ndarray,
                       target_parameter: float, target_entropy: float):
    """Return Figure/Axes(2,) for entropy and finite Bernoulli CE/KL curves in nats.

    Input curves have shape (n,). The caller supplies validated probabilities
    and measures. Endpoint +inf values are excluded from drawn line segments
    and explicitly annotated, never clipped or presented as finite. Probability
    axes are unitless and information axes use natural-log nats. Caller closes.
    """
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), layout="constrained")
    try:
        axes[0].plot(parameters, binary_entropy, color="#2466b0", linewidth=2,
                     label="H(Bernoulli(p))")
        axes[0].scatter([0., .5, 1.], [0., np.log(2.), 0.], color="#c45100", zorder=4,
                        label="Endpoints and maximum")
        axes[0].axhline(np.log(2.), color="#777777", linestyle=":", label="ln(2) nats")
        axes[0].set(xlabel="Outcome-1 probability p (unitless)", ylabel="Entropy (nats)",
                    title="Binary entropy: maximum at p=0.5", xlim=(0, 1), ylim=(-.025, .76))
        finite = np.isfinite(ce) & np.isfinite(divergence)
        axes[1].plot(model_parameters[finite], ce[finite], color="#2466b0", linewidth=2,
                     label="CE(P,Q)")
        axes[1].plot(model_parameters[finite], divergence[finite], color="#c45100", linewidth=2,
                     label="KL(P || Q)")
        axes[1].axhline(target_entropy, color="#29864a", linestyle="--", label="H(P); CE = H + KL")
        axes[1].axvline(target_parameter, color="#777777", linestyle=":", label="q=p=0.3")
        axes[1].scatter([target_parameter, target_parameter], [target_entropy, 0.],
                        color=["#2466b0", "#c45100"], zorder=4)
        axes[1].text(.5, .97, "q=0 or 1: CE and KL are +infinity\nEndpoint infinities are not drawn",
                     transform=axes[1].transAxes, ha="center", va="top", fontsize=9,
                     bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
        axes[1].set(xlabel="Model outcome-1 probability q (unitless)", ylabel="Information (nats)",
                    title="Fixed P=Bernoulli(0.3), varying Q=Bernoulli(q)", xlim=(0, 1))
        for axis in axes:
            axis.grid(alpha=.25)
            axis.legend(fontsize=8, loc="lower center" if axis is axes[0] else "center")
        figure.suptitle("Entropy, cross-entropy and KL divergence — natural logarithms", fontsize=13)
        figure.supxlabel("501 grid values; float64; exact zero-probability conventions; no clipping or renormalization", fontsize=9)
        return figure, axes
    except Exception:
        plt.close(figure)
        raise


def report_markdown(report: dict) -> str:
    """Format examples, formulas, shapes, support rules and executed checks into Markdown."""
    lines = ["# Information theory bonus: executed evidence", "",
             "Reproduce: `python -m scripts.run_information_theory`.", "",
             "NumPy float64; seed 42 set once at the experiment entry; natural logs, units nats.", "",
             "## Formulas and shape contract", "",
             "- `H(p) = -sum_i p_i ln(p_i)`.",
             "- `CE(p,q) = -sum_i p_i ln(q_i)`.",
             "- `KL(p||q) = sum_i p_i (ln(p_i)-ln(q_i))`; subtract logs instead of forming a ratio.", "",
             "The selected class axis is removed. Input `(n,)` returns a scalar; `(batch,n)` on axis=-1 returns `(batch,)`.",
             "Every probability is finite and in [0,1], and each class sum must equal 1 within atol=1e-12, rtol=0.",
             "p/q shapes must match exactly. No broadcasting, clipping, renormalization, or function-level reseeding occurs.",
             "A zero target term is zero, including 0*ln(0). Positive p with q=0 yields +inf.",
             "The JSON string `\"+inf\"` encodes this mathematical infinity; it is not a finite cap. NaN is never serialized.", "",
             "## Independent examples", "",
             "| Example | H(p), nats | CE(p,q), nats | KL(p||q), nats | Result |", "|---|---:|---:|---:|---|"]
    for row in report["examples"]:
        lines.append(f"| {row['name']} | {row['entropy_nats']} | {row['cross_entropy_nats']} | {row['kl_nats']} | PASS |")
    checks = report["curve_checks"]
    lines += ["", "Rational reference: p=(1/4,3/4), q=(1/2,1/2). Reversing KL changes its value.",
              "For the smallest positive float64 q, -ln(q)=1074 ln(2); CE/KL remain finite without overflow.", "",
              "## Bernoulli curves", "",
              "The first curve uses 501 values p in [0,1]. The second fixes P=(0.7,0.3) and uses the same q grid for Q=(1-q,q).",
              f"- Maximum binary entropy: {checks['maximum_entropy_nats']:.15f} nats at p={checks['maximum_entropy_parameter']}.",
              f"- Minimum finite KL: {checks['minimum_finite_kl_nats']:.6e} at q={checks['minimum_kl_parameter']}.",
              f"- Maximum finite-support CE-(H+KL) absolute error: {checks['maximum_identity_error']:.6e} <= 1e-12.",
              f"- Maximum H(p)-H(1-p) absolute error: {checks['maximum_entropy_symmetry_error']:.6e} <= 1e-14.",
              "- q=0 and q=1 produce +inf for both CE and KL; finite line segments exclude these endpoints.", "",
              "![Information theory curves](information_theory.png)", "",
              "CE = H + KL follows by separating sum p ln(p) and -sum p ln(q). Gibbs' inequality gives KL>=0 for normalized distributions, with equality when p=q on support.",
              "Thus, for fixed P, minimizing cross-entropy over Q also minimizes KL. KL is not symmetric and is not a distance metric.",
              "The 1e-12 input sum tolerance permits floating-point roundoff, not arbitrary unnormalized weights. No universal continuous-distribution or sample-estimation claims are made.", ""]
    return "\n".join(lines)


def run_experiment(output_dir: Path) -> dict:
    """Write PNG/JSON/Markdown after exact examples and 501-point curve assertions.

    Input/output probabilities have shape (501,2); information curves have
    shape (501,). Model boundary infinities are encoded as '+inf' in JSON.
    The calculation is inexpensive and intentionally uncached. File and
    numerical assertion failures propagate; the generated Figure is closed.
    """
    set_seed(42)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = example_report()
    parameters = np.linspace(0., 1., 501, dtype=np.float64)
    binary_probabilities = np.column_stack((1. - parameters, parameters))
    binary_entropy = entropy(binary_probabilities)
    target_parameter = .3
    target = np.array([1. - target_parameter, target_parameter])
    targets = np.tile(target, (len(parameters), 1))
    model_probabilities = binary_probabilities.copy()
    ce = cross_entropy(targets, model_probabilities)
    divergence = kl_divergence(targets, model_probabilities)
    target_entropy = entropy(target)
    finite = np.isfinite(ce) & np.isfinite(divergence)
    identity_error = float(np.max(np.abs(ce[finite] - target_entropy - divergence[finite])))
    symmetry_error = float(np.max(np.abs(binary_entropy - binary_entropy[::-1])))
    assert binary_entropy.shape == ce.shape == divergence.shape == (501,)
    assert identity_error <= 1e-12 and symmetry_error <= 1e-14
    assert np.all(divergence[finite] >= -1e-14)
    assert np.all(np.isposinf(ce[[0, -1]])) and np.all(np.isposinf(divergence[[0, -1]]))
    assert abs(float(np.max(binary_entropy)) - .6931471805599453) <= 1e-14
    assert abs(target_entropy - .6108643020548935) <= 1e-14
    minimum_index = int(np.argmin(divergence))
    assert abs(parameters[minimum_index] - target_parameter) <= 1e-15
    assert abs(divergence[minimum_index]) <= 1e-14
    report = {
        "seed": 42, "dtype": "float64", "numpy_version": np.__version__, "units": "nats",
        "normalization_atol": 1e-12, "normalization_rtol": 0., "examples": examples,
        "binary_entropy": {"parameter": parameters.tolist(), "probabilities": binary_probabilities.tolist(),
                           "entropy_nats": binary_entropy.tolist()},
        "fixed_target": {"parameter": target_parameter, "probabilities": target.tolist(), "entropy_nats": target_entropy},
        "varying_model": {"parameter": parameters.tolist(), "probabilities": model_probabilities.tolist(),
                          "cross_entropy_nats": [_json_number(value) for value in ce],
                          "kl_nats": [_json_number(value) for value in divergence]},
        "curve_checks": {"maximum_entropy_nats": float(np.max(binary_entropy)),
                         "maximum_entropy_parameter": float(parameters[np.argmax(binary_entropy)]),
                         "minimum_finite_kl_nats": float(divergence[minimum_index]),
                         "minimum_kl_parameter": float(parameters[minimum_index]),
                         "maximum_identity_error": identity_error, "identity_atol": 1e-12,
                         "maximum_entropy_symmetry_error": symmetry_error, "symmetry_atol": 1e-14,
                         "finite_model_count": int(np.count_nonzero(finite)), "infinite_boundary_count": 2,
                         "passed": True},
        "infinity_encoding": "The string '+inf' denotes mathematical positive infinity when p>0 and q=0; no clipping is performed.",
        "figure": {"filename": "information_theory.png", "width_px": 1792, "height_px": 686,
                   "nominal_dpi": 140, "format": "PNG"},
    }
    figure, _ = information_figure(parameters, binary_entropy, parameters, ce, divergence,
                                   target_parameter, target_entropy)
    try:
        figure.savefig(output_dir / "information_theory.png", dpi=140)
    finally:
        plt.close(figure)
    (output_dir / "information_theory.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    (output_dir / "information_theory.md").write_text(report_markdown(report), encoding="utf-8", newline="\n")
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the information-theory experiment with an optional writable output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    arguments = parser.parse_args(argv)
    report = run_experiment(arguments.output_dir)
    print(f"Information theory: {len(report['examples'])} independent examples passed; "
          f"CE=H+KL maximum error={report['curve_checks']['maximum_identity_error']:.6e} nats")
    print(f"Entropy maximum=ln(2), KL minimum at q=0.3; boundary +inf retained. Saved in {arguments.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
