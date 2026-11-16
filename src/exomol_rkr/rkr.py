"""Dunham power-series approximation and RKR singular quadrature potential inversion."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DunhamExpansion:
    """Dunham rovibrational expansion G(v+1/2) and B(v+1/2) with Birge-Sponer baseline."""

    y_k0: np.ndarray
    y_k1: np.ndarray
    c_mu: float
    zpe_cm: float
    fit_rms_cm: float
    morse_we_cm: float
    morse_wexe_cm: float
    morse_de_cm: float
    morse_rms_cm: float

    @property
    def omega_e(self) -> float:
        return float(self.y_k0[0])

    @property
    def omega_e_xe(self) -> float:
        return float(-self.y_k0[1])

    @property
    def b_e(self) -> float:
        return float(self.y_k1[0])

    @property
    def alpha_e(self) -> float:
        return float(-self.y_k1[1])

    @property
    def re_a(self) -> float:
        return float(np.sqrt(self.c_mu / self.y_k1[0]))

    def eval_g(self, x: float | np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        val = np.zeros_like(x_arr)
        for k, coeff in enumerate(self.y_k0, start=1):
            val = val + coeff * (x_arr**k)
        return val

    def eval_divided_diff_g(self, x: float | np.ndarray, sin2_t: np.ndarray) -> np.ndarray:
        """Cancellation-free divided difference (G(x) - G(x sin^2 t)) / (x (1 - sin^2 t))."""
        x_arr = np.asarray(x, dtype=np.float64)
        val = np.zeros_like(x_arr * sin2_t, dtype=np.float64)
        s_sum = np.ones_like(sin2_t, dtype=np.float64)
        s_pow = np.ones_like(sin2_t, dtype=np.float64)
        for k, coeff in enumerate(self.y_k0, start=1):
            if k > 1:
                s_pow = s_pow * sin2_t
                s_sum = s_sum + s_pow
            val = val + coeff * (x_arr ** (k - 1)) * s_sum
        return val

    def eval_b(self, x: float | np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        val = np.zeros_like(x_arr)
        for k, coeff in enumerate(self.y_k1):
            val = val + coeff * (x_arr**k)
        return val


@dataclass(frozen=True)
class RKRCurve:
    """Classical turning points r_min(v), r_max(v) reconstructed via Klein integrals."""

    v_levels: np.ndarray
    g_v: np.ndarray
    f_v: np.ndarray
    g_inv_v: np.ndarray
    r_min: np.ndarray
    r_max: np.ndarray
    re_a: float
    c_mu: float
    dunham: DunhamExpansion


def _legendre_p_and_dp(n: int, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate Legendre polynomial P_n(x) and derivative P_n'(x) via 3-term recurrence."""
    if n == 0:
        return np.ones_like(x), np.zeros_like(x)
    p0, p1 = np.ones_like(x), x.copy()
    for k in range(2, n + 1):
        p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
    dp = n * (x * p1 - p0) / (x**2 - 1.0)
    return p1, dp


def gauss_legendre(n: int, a: float = -1.0, b: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute n-point Gauss-Legendre nodes and weights on [a, b] via Newton's method."""
    m = (n + 1) // 2
    x = np.cos(np.pi * (np.arange(1, m + 1, dtype=np.float64) - 0.25) / (n + 0.5))
    for _ in range(10):
        p_n, dp_n = _legendre_p_and_dp(n, x)
        dx = p_n / dp_n
        x -= dx
        if np.max(np.abs(dx)) < 1e-15:
            break
    _, dp_n = _legendre_p_and_dp(n, x)
    w = 2.0 / ((1.0 - x**2) * (dp_n**2))
    nodes = np.concatenate([-x, x[::-1][n % 2 :]])
    weights = np.concatenate([w, w[::-1][n % 2 :]])
    return 0.5 * (b - a) * nodes + 0.5 * (a + b), 0.5 * (b - a) * weights


def fit_dunham(
    v_levels: np.ndarray, e_v0: np.ndarray, b_v: np.ndarray, c_mu: float, deg_g: int = 8, deg_b: int = 5
) -> DunhamExpansion:
    """Fit Dunham expansion coefficients Y_k0, Y_k1 and 2-parameter Birge-Sponer baseline."""
    x = np.asarray(v_levels, dtype=np.float64) + 0.5
    deg_g_eff, deg_b_eff = min(deg_g, len(x) - 1), min(deg_b, len(x) - 1)

    a_g = np.column_stack([x**k - 0.5**k for k in range(1, deg_g_eff + 1)])
    y_k0, *_ = np.linalg.lstsq(a_g, e_v0, rcond=None)
    zpe = float(sum(y_k0[k - 1] * (0.5**k) for k in range(1, deg_g_eff + 1)))
    fit_rms = float(np.sqrt(np.mean((a_g @ y_k0 - e_v0) ** 2)))

    a_b = np.column_stack([x**k for k in range(0, deg_b_eff + 1)])
    y_k1, *_ = np.linalg.lstsq(a_b, b_v, rcond=None)

    v_fl = np.asarray(v_levels, dtype=np.float64)
    v_mid = 0.5 * (v_fl[:-1] + v_fl[1:] + 1.0)
    slope, intercept = np.polyfit(v_mid, np.diff(e_v0) / np.diff(v_fl), 1)
    morse_wexe, morse_we = float(-0.5 * slope), float(intercept)
    morse_de = float((morse_we**2) / (4.0 * morse_wexe))
    morse_rms = float(np.sqrt(np.mean((e_v0 - (morse_we * v_fl - morse_wexe * (v_fl**2 + v_fl))) ** 2)))

    return DunhamExpansion(y_k0, y_k1, float(c_mu), zpe, fit_rms, morse_we, morse_wexe, morse_de, morse_rms)


def rkr_turning_points(dunham: DunhamExpansion, v_levels: np.ndarray, n_quad: int = 48) -> RKRCurve:
    """Invert empirical turning points r_min(v), r_max(v) via vectorized singular Klein integrals."""
    theta, weights = gauss_legendre(n_quad, 0.0, 0.5 * np.pi)
    sin_t, sin2_t = np.sin(theta)[None, :], (np.sin(theta) ** 2)[None, :]
    w_row = weights[None, :]

    x_grid = np.asarray(v_levels, dtype=np.float64) + 0.5
    x_col = x_grid[:, None]
    g_vals = dunham.eval_g(x_grid)
    c_mu = dunham.c_mu

    diff_ratio = dunham.eval_divided_diff_g(x_col, sin2_t)
    kernel = w_row * sin_t / np.sqrt(diff_ratio)
    f_vals = 2.0 * np.sqrt(c_mu * x_grid) * np.sum(kernel, axis=1)
    g_inv_vals = 2.0 * np.sqrt(x_grid / c_mu) * np.sum(dunham.eval_b(x_col * sin2_t) * kernel, axis=1)
    rad = np.sqrt(f_vals / g_inv_vals + f_vals**2)

    return RKRCurve(
        np.asarray(v_levels, dtype=np.int64), g_vals, f_vals, g_inv_vals,
        rad - f_vals, rad + f_vals, dunham.re_a, c_mu, dunham,
    )
