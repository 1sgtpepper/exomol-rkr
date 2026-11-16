"""4th-order Numerov-Cooley Sturm-Liouville radial Schrodinger eigenvalue solver."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EigenState:
    """Converged radial eigenstate and quantum expectation invariants."""

    v: int
    j_rot: int
    energy: float
    nodes: int
    u: np.ndarray
    b_rot: float
    virial_lhs: float
    virial_rhs: float

    @property
    def virial_relative_error(self) -> float:
        return abs(self.virial_lhs - self.virial_rhs) / max(abs(self.virial_lhs), 1e-30)


def morse_potential(r: np.ndarray, de_cm: float, alpha_a: float, re_a: float) -> np.ndarray:
    """Analytical Morse potential V(r) = D_e * (1 - exp(-alpha * (r - r_e)))^2."""
    return de_cm * (1.0 - np.exp(-alpha_a * (r - re_a))) ** 2


def morse_exact_energy(v: int | np.ndarray, de_cm: float, alpha_a: float, c_mu: float) -> float | np.ndarray:
    """Closed-form rotationless (J=0) Morse eigenvalue E_v in [cm^-1]."""
    vp = np.asarray(v, dtype=np.float64) + 0.5
    omega_e = 2.0 * alpha_a * np.sqrt(c_mu * de_cm)
    val = omega_e * vp - (c_mu * (alpha_a**2)) * (vp**2)
    return float(val) if np.ndim(v) == 0 else val


def diff_central_o4(y: np.ndarray, h: float) -> np.ndarray:
    """5-point O(h^4) central finite-difference first derivative."""
    dy = np.empty_like(y, dtype=np.float64)
    dy[2:-2] = (-y[4:] + 8.0 * y[3:-1] - 8.0 * y[1:-3] + y[:-4]) / (12.0 * h)
    dy[0] = (-25.0 * y[0] + 48.0 * y[1] - 36.0 * y[2] + 16.0 * y[3] - 3.0 * y[4]) / (12.0 * h)
    dy[1] = (-3.0 * y[0] - 10.0 * y[1] + 18.0 * y[2] - 6.0 * y[3] + y[4]) / (12.0 * h)
    dy[-2] = (3.0 * y[-1] + 10.0 * y[-2] - 18.0 * y[-3] + 6.0 * y[-4] - y[-5]) / (12.0 * h)
    dy[-1] = (25.0 * y[-1] - 48.0 * y[-2] + 36.0 * y[-3] - 16.0 * y[-4] + 3.0 * y[-5]) / (12.0 * h)
    return dy


def simpson_integrate(y: np.ndarray, h: float) -> float:
    """Composite Simpson's 1/3 rule O(h^4) numerical integration."""
    n = len(y)
    if n < 3:
        return float(np.trapezoid(y, dx=h))
    if n % 2 == 1:
        return float((h / 3.0) * (y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-2:2])))
    s_main = (h / 3.0) * (y[0] + y[-2] + 4.0 * np.sum(y[1:-2:2]) + 2.0 * np.sum(y[2:-3:2]))
    return float(s_main + 0.5 * h * (y[-2] + y[-1]))


def _sweep_numerov(a_steps: list[float]) -> list[float]:
    """Integrate 3-point Numerov recurrence y_{k+1} = a_k * y_k - y_{k-1} with overflow rescaling."""
    y = [0.0] * (len(a_steps) + 2)
    y0, y1 = 0.0, 1e-12
    y[1] = y1
    for idx, a_val in enumerate(a_steps):
        y2 = a_val * y1 - y0
        if y2 > 1e100 or y2 < -1e100:
            y0 *= 1e-100
            y1 *= 1e-100
            y2 *= 1e-100
            for k in range(idx + 2):
                y[k] *= 1e-100
        y[idx + 2] = y2
        y0, y1 = y1, y2
    return y


def numerov_shoot(
    r: np.ndarray, v_eff: np.ndarray, energy: float, c_mu: float, c_coeff: float = 1.0 / 12.0
) -> tuple[int, float, np.ndarray]:
    """Bidirectional Numerov integration matched at the outer classical turning point."""
    n = len(r)
    h = float(r[1] - r[0])
    w_grid = (v_eff - energy) / c_mu
    c = (c_coeff * (h**2)) * w_grid

    allowed = np.where(w_grid < 0.0)[0]
    m = int(np.clip(int(allowed[-1]) if len(allowed) > 5 else int(np.argmin(v_eff)), 5, n - 6))
    one_m_c = 1.0 - c
    a = ((2.0 + (h**2) * w_grid - 2.0 * c) / one_m_c).tolist()

    y_out = _sweep_numerov(a[1:m])
    y_in = _sweep_numerov(a[n - 2 : m : -1])[::-1]

    y_full = np.empty(n, dtype=np.float64)
    y_full[:m] = np.asarray(y_out[:m], dtype=np.float64) * (float(one_m_c[m]) / y_out[m])
    y_full[m:] = np.asarray(y_in, dtype=np.float64) * (float(one_m_c[m]) / y_in[0])
    u = y_full / one_m_c

    nodes = int(np.sum(u[1:m] * u[2 : m + 1] < 0.0))
    mismatch = -y_full[m + 1] + a[m] * y_full[m] - y_full[m - 1]
    return nodes, float((c_mu / (h**2)) * mismatch / np.sum(u**2)), u


def solve_radial_schrodinger(
    r: np.ndarray,
    v_pot: np.ndarray,
    v: int,
    j_rot: int,
    c_mu: float,
    e_hint: float | None = None,
    tol: float = 1e-11,
    c_coeff: float = 1.0 / 12.0,
) -> EigenState:
    """Solve -c_mu u'' + V_eff(r) u = E u via safeguarded Sturm bisection and Cooley update."""
    h = float(r[1] - r[0])
    v_eff = v_pot + (c_mu * j_rot * (j_rot + 1)) / (r**2) if j_rot > 0 else v_pot
    low, high = float(np.min(v_eff)), float(v_eff[-1]) * 0.9995
    energy = float(e_hint) if (e_hint is not None and low < e_hint < high) else 0.5 * (low + high)

    for _ in range(48):
        nd, de_corr, u = numerov_shoot(r, v_eff, energy, c_mu, c_coeff=c_coeff)
        if nd == v and (abs(de_corr) < tol or 0.5 * (high - low) < tol):
            return _package_eigenstate(r, h, v_eff, v, j_rot, energy, nd, u, c_mu)
        if nd > v or (nd == v and de_corr < 0.0):
            high = energy
        else:
            low = energy
        e_next = energy + de_corr
        energy = e_next if (nd == v and low < e_next < high) else 0.5 * (low + high)

    nd, _, u = numerov_shoot(r, v_eff, energy, c_mu, c_coeff=c_coeff)
    return _package_eigenstate(r, h, v_eff, v, j_rot, energy, nd, u, c_mu)


def _package_eigenstate(
    r: np.ndarray, h: float, v_eff: np.ndarray, v: int, j_rot: int,
    energy: float, nodes: int, u: np.ndarray, c_mu: float,
) -> EigenState:
    u_norm = u / np.sqrt(simpson_integrate(u**2, h))
    prob = u_norm**2
    b_rot = simpson_integrate(prob * (c_mu / (r**2)), h)
    t_kin = energy - simpson_integrate(prob * v_eff, h)
    vir_rhs = simpson_integrate(prob * r * diff_central_o4(v_eff, h), h)
    return EigenState(v, j_rot, float(energy), int(nodes), u_norm, float(b_rot), float(2.0 * t_kin), float(vir_rhs))
