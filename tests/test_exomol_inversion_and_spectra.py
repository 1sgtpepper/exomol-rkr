"""Behavioral tests for ExoMol RKR/CG potential inversion, Boltzmann stats, and FTIR spectra."""

from __future__ import annotations

import numpy as np

from exomol_rkr.data import load_molecule
from exomol_rkr.invert import (
    EMOPotential,
    brent_minimize,
    evaluate_emo_spectrum,
    fit_emo_from_rkr,
    refine_emo_conjugate_gradient,
)
from exomol_rkr.numerov import solve_radial_schrodinger
from exomol_rkr.rkr import fit_dunham, rkr_turning_points
from exomol_rkr.spectra import (
    compute_stick_spectrum,
    partition_function,
    synthesize_ftir_interferogram,
    thermal_j_max,
)


def test_exomol_dunham_and_nist_constants() -> None:
    """Verify Dunham constants and RKR turning-point monotonicity on real 1H-35Cl and 12C-16O data."""
    hcl = load_molecule("1H-35Cl")
    co = load_molecule("12C-16O", max_v=20)

    assert len(hcl.state_id) == 710
    assert hcl.trans_up_id is not None and len(hcl.trans_up_id) == 8892
    assert len(co.state_id) == 6383

    dunham_hcl = fit_dunham(hcl.v_levels, hcl.e_v0, hcl.b_v, hcl.spec.c_mu, deg_g=8, deg_b=5)
    dunham_co = fit_dunham(co.v_levels, co.e_v0, co.b_v, co.spec.c_mu, deg_g=6, deg_b=4)

    # Compare extracted equilibrium constants against NIST Chemistry WebBook benchmarks
    assert abs(dunham_hcl.omega_e - 2990.95) < 1.0
    assert abs(dunham_hcl.re_a - 1.27455) < 6.0e-4
    assert dunham_hcl.morse_rms_cm > 100.0  # 2-param Morse fails near dissociation (106.31 cm^-1)

    assert abs(dunham_co.omega_e - 2169.8136) < 0.01
    assert abs(dunham_co.omega_e_xe - 13.2883) < 0.01
    assert abs(dunham_co.b_e - 1.93128) < 1.0e-4
    assert abs(dunham_co.re_a - 1.128323) < 1.0e-5

    rkr_hcl = rkr_turning_points(dunham_hcl, hcl.v_levels)
    rkr_co = rkr_turning_points(dunham_co, co.v_levels)
    for rkr in (rkr_hcl, rkr_co):
        assert np.all(np.diff(rkr.r_min) < 0.0)
        assert np.all(np.diff(rkr.r_max) > 0.0)
        assert np.all(rkr.r_min < rkr.re_a)
        assert np.all(rkr.r_max > rkr.re_a)


def test_rkr_and_conjugate_gradient_emo_inversion() -> None:
    """Verify Hellmann-Feynman parameter Jacobian and Conjugate Gradient + Brent EMO refinement."""
    x_min, f_min = brent_minimize(lambda x: (x - 0.73) ** 2 + 0.1 * (x - 0.73) ** 4, 0.0, 1.5, tol=1e-6, max_iter=25)
    assert abs(x_min - 0.73) < 1.0e-5
    assert abs(f_min) < 1.0e-10

    hcl = load_molecule("1H-35Cl")
    dunham = fit_dunham(hcl.v_levels, hcl.e_v0, hcl.b_v, hcl.spec.c_mu)
    rkr = rkr_turning_points(dunham, hcl.v_levels)
    emo_init = fit_emo_from_rkr(rkr, de_cm=hcl.spec.de_cm, degree=5)

    r_grid = np.linspace(0.5, 4.5, 2001, dtype=np.float64)
    _, _, j_rel, e_abs, _, _ = evaluate_emo_spectrum(
        r_grid, emo_init, hcl.v_levels, hcl.e_v0, hcl.spec.c_mu, e_hints=rkr.g_v
    )

    # Check Hellmann-Feynman relative Jacobian J_rel against central finite differences for m=0, 1
    eps = 1.0e-5
    for m in (0, 1):
        b_p = emo_init.b_coeffs.copy()
        b_m = emo_init.b_coeffs.copy()
        b_p[m] += eps
        b_m[m] -= eps
        v_p = EMOPotential(emo_init.de_cm, emo_init.re_a, b_p).evaluate(r_grid)
        v_m = EMOPotential(emo_init.de_cm, emo_init.re_a, b_m).evaluate(r_grid)
        e_p = np.array([solve_radial_schrodinger(r_grid, v_p, int(v), 0, hcl.spec.c_mu, e_hint=float(e_abs[i])).energy for i, v in enumerate(hcl.v_levels)])
        e_m = np.array([solve_radial_schrodinger(r_grid, v_m, int(v), 0, hcl.spec.c_mu, e_hint=float(e_abs[i])).energy for i, v in enumerate(hcl.v_levels)])
        fd_col = ((e_p - e_p[0]) - (e_m - e_m[0])) / (2.0 * eps)
        rel_err = np.max(np.abs(j_rel[:, m] - fd_col)) / np.max(np.abs(fd_col))
        assert rel_err < 1.0e-6

    cg_res = refine_emo_conjugate_gradient(
        r_grid,
        emo_init,
        hcl.v_levels,
        hcl.e_v0,
        hcl.spec.c_mu,
        e_hints=rkr.g_v,
        max_iter=5,
    )
    assert cg_res.rms_initial_cm < 3.2
    assert cg_res.rms_refined_cm < 1.0
    assert cg_res.history[-1]["loss"] < 0.12 * cg_res.history[0]["loss"]
    losses = [h["loss"] for h in cg_res.history]
    assert all(losses[i + 1] < losses[i] for i in range(len(losses) - 1))
    alphas = [round(h["alpha"], 6) for h in cg_res.history[1:]]
    assert len(set(alphas)) == len(alphas)


def test_boltzmann_partition_function_and_ftir_revivals() -> None:
    """Verify Boltzmann partition function against ExoMol .pf, Ritz combinations, and FTIR revivals."""
    hcl = load_molecule("1H-35Cl")
    co = load_molecule("12C-16O")

    temps = np.array([10.0, 100.0, 300.0, 1000.0, 3000.0])
    q_hcl = partition_function(hcl.energy_cm, hcl.g_tot, temps)
    q_hcl_ref = np.array([hcl.pf_q_ref[int(t) - 1] for t in temps])
    assert np.max(np.abs(q_hcl - q_hcl_ref) / q_hcl_ref) < 1.0e-6

    q_co = partition_function(co.energy_cm, co.g_tot, temps)
    q_co_ref = np.array([co.pf_q_ref[int(t) - 1] for t in temps])
    assert np.max(np.abs(q_co - q_co_ref) / q_co_ref) < 1.0e-5

    # Ritz combination check across all 8,892 transitions
    assert hcl.trans_up_id is not None and hcl.trans_lo_id is not None and hcl.trans_nu_cm is not None
    nu_ritz = hcl.energy_cm[hcl.trans_up_id - 1] - hcl.energy_cm[hcl.trans_lo_id - 1]
    assert np.max(np.abs(nu_ritz - hcl.trans_nu_cm)) <= 1.1e-3

    # Thermal shift of fundamental R-branch peak from 300 K to 1500 K
    nu_300, s_300, j_lo_300, dj_300 = compute_stick_spectrum(hcl, 300.0, v_up=1, v_lo=0)
    nu_1500, s_1500, j_lo_1500, dj_1500 = compute_stick_spectrum(hcl, 1500.0, v_up=1, v_lo=0)
    j_peak_300 = int(j_lo_300[dj_300 == 1][np.argmax(s_300[dj_300 == 1])])
    j_peak_1500 = int(j_lo_1500[dj_1500 == 1][np.argmax(s_1500[dj_1500 == 1])])
    assert j_peak_300 == 2
    assert j_peak_1500 == 6
    assert abs(j_peak_300 - thermal_j_max(300.0, float(hcl.b_v[0]))) < 1.0
    assert abs(j_peak_1500 - thermal_j_max(1500.0, float(hcl.b_v[0]))) < 1.0

    # FTIR synthesis and inversion invariants
    ftir = synthesize_ftir_interferogram(nu_300, s_300)
    assert abs(ftir.zpd_intensity - ftir.integrated_intensity) / ftir.integrated_intensity < 1.0e-12
    assert np.max(np.abs(ftir.spectrum_unapodized - ftir.spectrum_true)) / np.max(ftir.spectrum_true) < 1.0e-12

    peak_nu_true = float(ftir.nu_grid_cm[np.argmax(ftir.spectrum_true)])
    peak_nu_apod = float(ftir.nu_grid_cm[np.argmax(ftir.spectrum_apodized)])
    assert abs(peak_nu_true - peak_nu_apod) < 0.05

    expected_revival_cm = 1.0 / (2.0 * float(hcl.b_v[0]))
    assert abs(ftir.revival_opd_cm - expected_revival_cm) / expected_revival_cm < 0.05
