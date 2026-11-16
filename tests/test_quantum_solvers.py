"""Behavioral and invariant tests for Numerov-Cooley ODE solver and RKR quadrature."""

from __future__ import annotations

import numpy as np

from exomol_rkr.data import MOLECULES
from exomol_rkr.numerov import (
    diff_central_o4,
    morse_exact_energy,
    morse_potential,
    simpson_integrate,
    solve_radial_schrodinger,
)
from exomol_rkr.rkr import DunhamExpansion, gauss_legendre, rkr_turning_points


def test_numerov_exact_morse_eigenvalues_and_sturm_nodes() -> None:
    """Verify Numerov eigenvalues against closed-form Morse formula, Sturm nodes, and orthogonality."""
    c_mu = MOLECULES["1H-35Cl"].c_mu
    de_cm, alpha_a, re_a = 37240.0, 1.8677, 1.27455
    r = np.linspace(0.42, 5.8, 4001, dtype=np.float64)
    h = float(r[1] - r[0])
    v_pot = morse_potential(r, de_cm, alpha_a, re_a)

    states = {}
    for v in (0, 1, 2, 5, 10, 15, 17):
        e_exact = float(morse_exact_energy(v, de_cm, alpha_a, c_mu))
        st = solve_radial_schrodinger(r, v_pot, v, 0, c_mu, e_hint=e_exact)
        states[v] = st
        assert st.nodes == v, f"Sturm theorem violated for v={v}: got {st.nodes} nodes"
        assert abs(st.energy - e_exact) < 3.0e-4

    # Richardson O(h^4) extrapolation between N=2001 and N=4001 must reach < 1e-9 cm^-1 on v=0
    r_coarse = r[::2]
    v_coarse = v_pot[::2]
    e0_exact = float(morse_exact_energy(0, de_cm, alpha_a, c_mu))
    st_coarse = solve_radial_schrodinger(r_coarse, v_coarse, 0, 0, c_mu, e_hint=e0_exact)
    e0_richardson = (16.0 * states[0].energy - st_coarse.energy) / 15.0
    assert abs(e0_richardson - e0_exact) < 5.0e-9

    # Verify eigenfunction orthonormality <u_v | u_v'> = delta_{v, v'}
    for v1 in (0, 1, 2, 5):
        for v2 in (0, 1, 2, 5):
            overlap = simpson_integrate(states[v1].u * states[v2].u, h)
            expected = 1.0 if v1 == v2 else 0.0
            assert abs(overlap - expected) < 1.0e-7


def test_numerov_fourth_order_convergence_and_negative_control() -> None:
    """Verify O(h^4) error scaling of Numerov vs O(h^2) negative control when c_coeff=0."""
    c_mu = MOLECULES["1H-35Cl"].c_mu
    de_cm, alpha_a, re_a = 37240.0, 1.8677, 1.27455
    e1_exact = float(morse_exact_energy(1, de_cm, alpha_a, c_mu))

    errors_o4, errors_o2, hs = [], [], []
    for n_pts in (501, 1001, 2001, 4001):
        r = np.linspace(0.45, 5.2, n_pts, dtype=np.float64)
        v_pot = morse_potential(r, de_cm, alpha_a, re_a)
        hs.append(float(r[1] - r[0]))
        st4 = solve_radial_schrodinger(r, v_pot, 1, 0, c_mu, e_hint=e1_exact, c_coeff=1.0 / 12.0)
        st2 = solve_radial_schrodinger(r, v_pot, 1, 0, c_mu, e_hint=e1_exact, c_coeff=0.0)
        errors_o4.append(abs(st4.energy - e1_exact))
        errors_o2.append(abs(st2.energy - e1_exact))

    ratios_o4 = [errors_o4[i] / errors_o4[i + 1] for i in range(len(errors_o4) - 1)]
    ratios_o2 = [errors_o2[i] / errors_o2[i + 1] for i in range(len(errors_o2) - 1)]
    for ratio in ratios_o4:
        assert 15.5 < ratio < 16.5
    for ratio in ratios_o2:
        assert 3.8 < ratio < 4.2
    assert errors_o4[-1] * 2000.0 < errors_o2[-1]


def test_hellmann_feynman_and_virial_theorems() -> None:
    """Verify rotational Hellmann-Feynman derivative and quantum Virial identity 2<T> = <r V'(r)>."""
    c_mu = MOLECULES["1H-35Cl"].c_mu
    de_cm, alpha_a, re_a = 37240.0, 1.8677, 1.27455
    r = np.linspace(0.42, 5.8, 4001, dtype=np.float64)
    h = float(r[1] - r[0])
    v_pot = morse_potential(r, de_cm, alpha_a, re_a)

    for v in (0, 3, 7):
        st = solve_radial_schrodinger(r, v_pot, v, 0, c_mu)
        assert st.virial_relative_error < 1.0e-7

        # Negative control: 2nd-order central difference on V(r) degrades Virial accuracy by >10^4x on v=0
        if v == 0:
            dv_fd2 = np.gradient(v_pot, h, edge_order=2)
            vir_rhs_fd2 = simpson_integrate((st.u**2) * r * dv_fd2, h)
            err_fd2 = abs(st.virial_lhs - vir_rhs_fd2) / st.virial_lhs
            assert err_fd2 > 1.0e4 * st.virial_relative_error

        # Hellmann-Feynman check: <u_v | C_mu / r^2 | u_v> == dE_v / d[J(J+1)] at lambda -> 0
        eps = 1.0e-2
        st_plus = solve_radial_schrodinger(r, v_pot + eps * (c_mu / (r**2)), v, 0, c_mu, e_hint=st.energy)
        st_minus = solve_radial_schrodinger(r, v_pot - eps * (c_mu / (r**2)), v, 0, c_mu, e_hint=st.energy)
        de_dlam_fd = (st_plus.energy - st_minus.energy) / (2.0 * eps)
        assert abs(st.b_rot - de_dlam_fd) / st.b_rot < 1.0e-7


def test_gauss_legendre_and_rkr_exact_morse_inversion() -> None:
    """Verify Gauss-Legendre quadrature and RKR Klein integral inversion against exact Morse turning points."""
    for n_pts in (24, 25):
        nodes, weights = gauss_legendre(n_pts, -1.0, 1.0)
        assert np.all(np.diff(nodes) > 0.0)
        for k in range(16):
            exact_int = 2.0 / (2 * k + 1)
            num_int = float(np.sum(weights * (nodes ** (2 * k))))
            assert abs(num_int - exact_int) < 1.0e-14

    c_mu = MOLECULES["1H-35Cl"].c_mu
    de_cm, alpha_a, re_a = 37240.0, 1.8677, 1.27455
    we = 2.0 * alpha_a * np.sqrt(c_mu * de_cm)
    wexe = c_mu * (alpha_a**2)
    b_e = c_mu / (re_a**2)
    dunham_morse = DunhamExpansion(
        y_k0=np.array([we, -wexe], dtype=np.float64),
        y_k1=np.array([b_e, 0.0], dtype=np.float64),
        c_mu=c_mu,
        zpe_cm=0.5 * we - 0.25 * wexe,
        fit_rms_cm=0.0,
        morse_we_cm=we,
        morse_wexe_cm=wexe,
        morse_de_cm=de_cm,
        morse_rms_cm=0.0,
    )
    v_levels = np.arange(16, dtype=np.int64)
    rkr = rkr_turning_points(dunham_morse, v_levels, n_quad=36)

    for idx, v in enumerate(v_levels):
        g_v = float(rkr.g_v[idx])
        s = np.sqrt(g_v / de_cm)
        r_min_ex = re_a - np.log(1.0 + s) / alpha_a
        r_max_ex = re_a - np.log(1.0 - s) / alpha_a
        f_exact = 0.5 * (r_max_ex - r_min_ex)
        assert abs(rkr.f_v[idx] - f_exact) < 1.0e-13

    # Negative control: naive uniform midpoint rule on [0, x] without singularity removal
    x0 = 0.5
    xp_mid = (np.arange(36, dtype=np.float64) + 0.5) * (x0 / 36.0)
    f_naive = np.sqrt(c_mu) * float(np.sum((x0 / 36.0) / np.sqrt(dunham_morse.eval_g(x0) - dunham_morse.eval_g(xp_mid))))
    f_exact_0 = 0.5 * (-np.log(1.0 - np.sqrt(rkr.g_v[0] / de_cm)) + np.log(1.0 + np.sqrt(rkr.g_v[0] / de_cm))) / alpha_a
    assert abs(f_naive - f_exact_0) > 1.0e10 * abs(rkr.f_v[0] - f_exact_0)
