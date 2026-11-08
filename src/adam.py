"""Bias-corrected Adam using NumPy and the shared optimizer runner contract."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.optimizer import VanillaGD, _NonfiniteResult, _finite_result, _step_inputs
from src.utils.numerical import _finite_scalar

__concepts__ = {
    "Adam_Update": ["Adam"],
    "Optimizer_State_Rollback": ["Adam._snapshot_state", "Adam._restore_state"],
}


class Adam(VanillaGD):
    """Apply Algorithm 1 Adam with bias correction to finite vectors (n,).

    m_t=beta1*m_(t-1)+(1-beta1)*g_t,
    v_t=beta2*v_(t-1)+(1-beta2)*g_t**2,
    m_hat=m_t/(1-beta1**t), v_hat=v_t/(1-beta2**t),
    theta_next=theta-lr*m_hat/(sqrt(v_hat)+epsilon).

    epsilon is outside sqrt. lr/epsilon are positive finite scalars and
    beta1/beta2 are finite scalars in [0,1). All inputs use float64. Input
    theta and gradient must be nonempty matching (n,) vectors, preserving
    the established state shape until reset(). m/v properties return copies,
    t counts accepted step() calls, and a failed step commits no state.
    Invalid numeric types (including booleans) raise TypeError; invalid
    shape/range or nonfinite input raises ValueError. Nonrepresentable
    intermediate arithmetic raises FloatingPointError. Moment arithmetic
    that underflows with lost precision (zero or inexact subnormal result)
    is also rejected, including weighted terms: a lost squared gradient
    must not silently produce a finite but incorrect normalized update.
    This deliberately restricts extreme gradients/moment decay; no scaled
    or arbitrary-precision moment representation is provided. Exact
    representable subnormal moments remain allowed. The shared runner
    restores state on these failures and on rejected candidate callbacks.
    """

    def __init__(self, lr: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        """Validate scalar hyperparameters and initialize m/v=None, t=0."""
        super().__init__(lr)
        self.beta1 = _finite_scalar(beta1, "beta1")
        self.beta2 = _finite_scalar(beta2, "beta2")
        self.epsilon = _finite_scalar(epsilon, "epsilon")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("beta1 and beta2 must satisfy 0 <= beta < 1")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self._m = None
        self._v = None
        self._t = 0

    @property
    def m(self) -> NDArray[np.float64] | None:
        """Return an isolated first-moment vector (n,), or None before initialization."""
        return None if self._m is None else self._m.copy()

    @property
    def v(self) -> NDArray[np.float64] | None:
        """Return an isolated second-moment vector (n,), or None before initialization."""
        return None if self._v is None else self._v.copy()

    @property
    def t(self) -> int:
        """Return the count of accepted updates since reset, initially zero."""
        return self._t

    def reset(self) -> None:
        """Clear moments and update count so the next step may establish a new shape."""
        self._m = None
        self._v = None
        self._t = 0

    def _snapshot_state(self) -> tuple:
        """Return independent (m(n,), v(n,), t) state for candidate rollback."""
        return self.m, self.v, self._t

    def _restore_state(self, state: tuple) -> None:
        """Restore a snapshot made by this Adam instance, copying both moments."""
        first, second, updates = state
        self._m = None if first is None else first.copy()
        self._v = None if second is None else second.copy()
        self._t = updates

    def step(self, theta: ArrayLike, gradient: ArrayLike) -> NDArray[np.float64]:
        """Return a finite (n,) update; reject moment underflow and commit m/v/t atomically."""
        point, derivative = _step_inputs(theta, gradient)
        if self._m is not None and self._m.shape != point.shape:
            raise ValueError("reset Adam before changing the parameter shape")
        previous_m = np.zeros_like(point) if self._m is None else self._m
        previous_v = np.zeros_like(point) if self._v is None else self._v
        updates = self._t + 1
        try:
            with np.errstate(over="ignore", invalid="ignore", under="raise"):
                moment = self.beta1 * previous_m + (1.0 - self.beta1) * derivative
        except FloatingPointError as error:
            raise _NonfiniteResult("Adam first moment underflow loses float64 precision") from error
        try:
            with np.errstate(over="ignore", invalid="ignore", under="raise"):
                variance = self.beta2 * previous_v + (1.0 - self.beta2) * derivative**2
        except FloatingPointError as error:
            raise _NonfiniteResult("Adam second moment underflow loses float64 precision") from error
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            corrected_m = moment / (1.0 - self.beta1**updates)
            corrected_v = variance / (1.0 - self.beta2**updates)
            candidate = point - self.lr * (corrected_m / (np.sqrt(corrected_v) + self.epsilon))
        _finite_result(moment, "Adam first moment")
        _finite_result(variance, "Adam second moment")
        _finite_result(corrected_m, "Adam corrected first moment")
        _finite_result(corrected_v, "Adam corrected second moment")
        _finite_result(candidate, "Adam update")
        self._m = moment.copy()
        self._v = variance.copy()
        self._t = updates
        return candidate
