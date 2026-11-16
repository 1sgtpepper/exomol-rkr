"""Inverse Schrodinger potential refinement via Conjugate Gradient and Brent line search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

from .numerov import solve_radial_schrodinger
from .rkr import RKRCurve


@dataclass(frozen=True)
class EMOPotential:
    """Extended Morse Oscillator V(r) = D_e * (1 - exp(-beta(r) * (r - r_e)))^2."""

    de_cm: float
    re_a: float
    b_coeffs: np.ndarray

    def surkus_y(self, r: np.ndarray) -> np.ndarray:
        r2, re2 = np.asarray(r, dtype=np.float64) ** 2, self.re_a**2
        return (r2 - re2) / (r2 + re2)

    def evaluate(self, r: np.ndarray) -> np.ndarray:
        return self.evaluate_with_grad(r)[0]

    def evaluate_with_grad(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return V(r) and parameter sensitivities dV(r)/db_m of shape (M, N)."""
        r_arr = np.asarray(r, dtype=np.float64)
        y = self.surkus_y(r_arr)
        y_pows = np.vstack([y**m for m in range(len(self.b_coeffs))])
        dr = r_arr - self.re_a
        ex = np.exp(-np.clip((self.b_coeffs @ y_pows) * dr, -50.0, 50.0))
        one_m_ex = 1.0 - ex
        return self.de_cm * (one_m_ex**2), (2.0 * self.de_cm * one_m_ex * ex * dr) * y_pows


@dataclass(frozen=True)
class CGInversionResult:
    """Summary of RKR-initialized Conjugate Gradient quantum potential refinement."""

    emo_initial: EMOPotential
    emo_refined: EMOPotential
    e_initial_cm: np.ndarray
    e_refined_cm: np.ndarray
    rms_initial_cm: float
    rms_refined_cm: float
    history: list[dict[str, float]]


def fit_emo_from_rkr(rkr: RKRCurve, de_cm: float, degree: int = 5) -> EMOPotential:
    """Fit an Extended Morse Oscillator directly to RKR turning points (r_min, r_max, G_v)."""
    re_a = rkr.re_a
    r_tp = np.concatenate([rkr.r_min[::-1], [re_a], rkr.r_max])
    g_tp = np.concatenate([rkr.g_v[::-1], [0.0], rkr.g_v])
    s = np.sign(r_tp - re_a) * np.sqrt(np.clip(g_tp / de_cm, 0.0, 0.9999))
    dr = r_tp - re_a
    nz = np.abs(dr) > 1e-10
    beta_tp = np.empty_like(r_tp)
    beta_tp[nz] = -np.log(1.0 - s[nz]) / dr[nz]
    beta_tp[~nz] = rkr.dunham.omega_e / (2.0 * np.sqrt(rkr.c_mu * de_cm))
    y_tp = (r_tp**2 - re_a**2) / (r_tp**2 + re_a**2)
    return EMOPotential(float(de_cm), float(re_a), np.polyfit(y_tp, beta_tp, degree)[::-1])


def brent_minimize(
    func: Callable[[float], float], a: float, b: float, tol: float = 1e-3, max_iter: int = 10
) -> tuple[float, float]:
    """Brent's 1D minimization combining parabolic interpolation and golden-section search."""
    golden = 0.3819660112501051
    x = w = v = a + golden * (b - a)
    fx = fw = fv = float(func(x))
    d = e = b - a

    for _ in range(max_iter):
        xm = 0.5 * (a + b)
        tol1 = tol * abs(x) + 1e-7
        tol2 = 2.0 * tol1
        if abs(x - xm) <= (tol2 - 0.5 * (b - a)):
            break
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            etemp, e = e, d
            if abs(p) < abs(0.5 * q * etemp) and p > q * (a - x) and p < q * (b - x):
                d = p / q
                if (x + d - a) < tol2 or (b - (x + d)) < tol2:
                    d = tol1 if xm >= x else -tol1
            else:
                e = (b - x) if x < xm else (a - x)
                d = golden * e
        else:
            e = (b - x) if x < xm else (a - x)
            d = golden * e

        u = x + (d if abs(d) >= tol1 else (tol1 if d > 0.0 else -tol1))
        fu = float(func(u))
        if fu <= fx:
            if u >= x:
                a = x
            else:
                b = x
            v, fv = w, fw
            w, fw = x, fx
            x, fx = u, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv = w, fw
                w, fw = u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
    return float(x), float(fx)


def evaluate_emo_spectrum(
    r: np.ndarray, emo: EMOPotential, v_levels: np.ndarray, e_obs: np.ndarray, c_mu: float, e_hints: np.ndarray | None = None
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve all v_levels under EMO potential and compute Hellmann-Feynman Jacobian."""
    v_pot, dv_db = emo.evaluate_with_grad(r)
    energies = np.empty(len(v_levels), dtype=np.float64)
    u_mat = np.empty((len(v_levels), len(r)), dtype=np.float64)
    for idx, v in enumerate(v_levels):
        hint = float(e_hints[idx]) if e_hints is not None else None
        st = solve_radial_schrodinger(r, v_pot, int(v), 0, c_mu, e_hint=hint, tol=1e-10)
        energies[idx], u_mat[idx] = st.energy, st.u

    e_rel = energies - energies[0]
    residuals = e_rel - e_obs
    j_hf = np.trapezoid((u_mat[:, None, :] ** 2) * dv_db[None, :, :], r, axis=2)
    j_rel = j_hf - j_hf[0:1, :]
    return 0.5 * float(np.sum(residuals**2)), residuals @ j_rel, j_rel, energies, e_rel, j_hf


def refine_emo_conjugate_gradient(
    r: np.ndarray,
    emo_init: EMOPotential,
    v_levels: np.ndarray,
    e_obs: np.ndarray,
    c_mu: float,
    e_hints: np.ndarray | None = None,
    max_iter: int = 5,
    damping: float = 1e-4,
) -> CGInversionResult:
    """Refine EMO potential via Fletcher-Reeves conjugate gradient and Brent line search."""
    loss0, grad_b, j_curr, e_abs_c, e_init_rel, j_hf = evaluate_emo_spectrum(
        r, emo_init, v_levels, e_obs, c_mu, e_hints=e_hints
    )
    rms0 = float(np.sqrt(np.mean((e_init_rel - e_obs) ** 2)))

    h0 = j_curr.T @ j_curr
    h0_diag = np.diag(h0)
    l_chol = np.linalg.cholesky(h0 + damping * np.diag(h0_diag))
    l_inv = np.linalg.inv(l_chol.T)

    g_k = l_inv.T @ grad_b
    d_k = -g_k.copy()
    b_curr = emo_init.b_coeffs.copy()
    e_curr_rel = e_init_rel.copy()
    history = [{"iter": 0.0, "alpha": 0.0, "beta": 0.0, "loss": loss0, "rms_cm": rms0}]

    for k in range(1, max_iter + 1):
        if float(np.dot(g_k, d_k)) >= 0.0:
            d_k = -g_k.copy()
        db_dir = l_inv @ d_k
        jd = j_curr @ db_dir
        jd_abs = j_hf @ db_dir
        curv = float(np.dot(jd, jd) + 0.1 * damping * np.dot(h0_diag, db_dir**2))
        alpha_quad = max(1e-3, float(-np.dot(g_k, d_k) / max(curv, 1e-30)))
        ref_abs = e_abs_c.copy()

        def phi(alpha: float) -> float:
            v_cand = EMOPotential(emo_init.de_cm, emo_init.re_a, b_curr + alpha * db_dir).evaluate(r)
            e_cand = np.array(
                [
                    solve_radial_schrodinger(
                        r, v_cand, int(v), 0, c_mu, e_hint=float(ref_abs[i] + alpha * jd_abs[i]), tol=1e-10
                    ).energy
                    for i, v in enumerate(v_levels)
                ]
            )
            return 0.5 * float(np.sum(((e_cand - e_cand[0]) - e_obs) ** 2))

        alpha_k, _ = brent_minimize(phi, 0.0, 2.0 * alpha_quad, tol=1e-3, max_iter=8)
        b_curr = b_curr + alpha_k * db_dir
        emo_curr = EMOPotential(emo_init.de_cm, emo_init.re_a, b_curr)
        loss_next, grad_b_next, j_next, e_abs_next, e_rel_next, j_hf_next = evaluate_emo_spectrum(
            r, emo_curr, v_levels, e_obs, c_mu, e_hints=e_abs_c + alpha_k * jd_abs
        )
        g_next = l_inv.T @ grad_b_next
        beta_k = float(np.dot(g_next, g_next) / max(float(np.dot(g_k, g_k)), 1e-30))
        d_k = -g_next + beta_k * d_k
        g_k, j_curr, j_hf, e_abs_c, e_curr_rel = g_next, j_next, j_hf_next, e_abs_next, e_rel_next
        rms_k = float(np.sqrt(np.mean((e_curr_rel - e_obs) ** 2)))
        history.append({"iter": float(k), "alpha": alpha_k, "beta": beta_k, "loss": loss_next, "rms_cm": rms_k})

    emo_final = EMOPotential(emo_init.de_cm, emo_init.re_a, b_curr)
    rms_final = float(np.sqrt(np.mean((e_curr_rel - e_obs) ** 2)))
    return CGInversionResult(emo_init, emo_final, e_init_rel, e_curr_rel, rms0, rms_final, history)
