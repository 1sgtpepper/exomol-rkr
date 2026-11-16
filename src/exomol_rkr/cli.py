"""End-to-end ExoMol spectroscopy pipeline, verification summary, and figure generation."""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import load_molecule
from .invert import CGInversionResult, fit_emo_from_rkr, refine_emo_conjugate_gradient
from .numerov import EigenState, morse_exact_energy, morse_potential, solve_radial_schrodinger
from .rkr import DunhamExpansion, RKRCurve, fit_dunham, rkr_turning_points
from .spectra import FTIRResult, compute_stick_spectrum, partition_function, synthesize_ftir_interferogram, thermal_j_max


def run_pipeline(output_dir: Path | str | None = None) -> dict[str, object]:
    """Execute full ExoMol analysis for 1H-35Cl and 12C-16O and save outputs."""
    root = Path(__file__).resolve().parents[2]
    out_dir = Path(output_dir) if output_dir is not None else root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    hcl = load_molecule("1H-35Cl")
    co = load_molecule("12C-16O", max_v=20)
    dunham_hcl = fit_dunham(hcl.v_levels, hcl.e_v0, hcl.b_v, hcl.spec.c_mu, deg_g=8, deg_b=5)
    rkr_hcl = rkr_turning_points(dunham_hcl, hcl.v_levels, n_quad=48)
    dunham_co = fit_dunham(co.v_levels, co.e_v0, co.b_v, co.spec.c_mu, deg_g=6, deg_b=4)
    rkr_co = rkr_turning_points(dunham_co, co.v_levels, n_quad=48)

    r_grid = np.linspace(0.5, 4.5, 2001, dtype=np.float64)
    emo_init = fit_emo_from_rkr(rkr_hcl, de_cm=hcl.spec.de_cm, degree=5)
    cg_res = refine_emo_conjugate_gradient(
        r_grid, emo_init, hcl.v_levels, hcl.e_v0, hcl.spec.c_mu, e_hints=rkr_hcl.g_v, max_iter=5
    )

    v_refined = cg_res.emo_refined.evaluate(r_grid)
    sample_states = {
        int(v): solve_radial_schrodinger(r_grid, v_refined, int(v), 0, hcl.spec.c_mu, e_hint=float(rkr_hcl.g_v[int(v)]))
        for v in (0, 3, 7, 12, 17)
    }

    de_m, alpha_m, re_m = 37240.0, 1.8677, 1.27455
    e0_exact = float(morse_exact_energy(0, de_m, alpha_m, hcl.spec.c_mu))
    n_list = [501, 1001, 2001, 4001]
    err_o4, err_o2, h_list = [], [], []
    for n_pts in n_list:
        rg = np.linspace(0.45, 5.2, n_pts, dtype=np.float64)
        vm = morse_potential(rg, de_m, alpha_m, re_m)
        h_list.append(float(rg[1] - rg[0]))
        err_o4.append(abs(solve_radial_schrodinger(rg, vm, 0, 0, hcl.spec.c_mu, e_hint=e0_exact, c_coeff=1.0 / 12.0).energy - e0_exact))
        err_o2.append(abs(solve_radial_schrodinger(rg, vm, 0, 0, hcl.spec.c_mu, e_hint=e0_exact, c_coeff=0.0).energy - e0_exact))

    slope_o4 = float(np.polyfit(np.log(h_list), np.log(err_o4), 1)[0])
    slope_o2 = float(np.polyfit(np.log(h_list), np.log(err_o2), 1)[0])

    temps_check = np.array([100.0, 300.0, 1000.0, 3000.0])
    q_calc = partition_function(hcl.energy_cm, hcl.g_tot, temps_check)
    q_ref = np.array([hcl.pf_q_ref[int(t) - 1] for t in temps_check])
    pf_max_rel_err = float(np.max(np.abs(q_calc - q_ref) / q_ref))

    nu_300, s_300, j_lo_300, dj_300 = compute_stick_spectrum(hcl, 300.0, v_up=1, v_lo=0)
    nu_1500, s_1500, j_lo_1500, dj_1500 = compute_stick_spectrum(hcl, 1500.0, v_up=1, v_lo=0)
    ftir_300 = synthesize_ftir_interferogram(nu_300, s_300)

    j_peak_300 = int(j_lo_300[dj_300 == 1][np.argmax(s_300[dj_300 == 1])])
    j_peak_1500 = int(j_lo_1500[dj_1500 == 1][np.argmax(s_1500[dj_1500 == 1])])

    summary = {
        "hcl": {
            "n_states": int(len(hcl.state_id)),
            "n_transitions": int(len(hcl.trans_up_id)) if hcl.trans_up_id is not None else 0,
            "omega_e_cm": dunham_hcl.omega_e,
            "omega_e_xe_cm": dunham_hcl.omega_e_xe,
            "b_e_cm": dunham_hcl.b_e,
            "re_a": dunham_hcl.re_a,
            "morse_baseline_rms_cm": dunham_hcl.morse_rms_cm,
            "rkr_initial_rms_cm": cg_res.rms_initial_cm,
            "cg_refined_rms_cm": cg_res.rms_refined_cm,
            "virial_rel_err_v0": sample_states[0].virial_relative_error,
            "b0_hellmann_feynman_cm": sample_states[0].b_rot,
            "b0_observed_cm": float(hcl.b_v[0]),
            "pf_max_rel_err": pf_max_rel_err,
            "r_branch_peak_j_300k": j_peak_300,
            "r_branch_peak_j_1500k": j_peak_1500,
            "j_max_analytic_300k": thermal_j_max(300.0, float(hcl.b_v[0])),
            "j_max_analytic_1500k": thermal_j_max(1500.0, float(hcl.b_v[0])),
            "ftir_revival_opd_cm": ftir_300.revival_opd_cm,
            "ftir_expected_revival_cm": float(1.0 / (2.0 * hcl.b_v[0])),
            "cg_history": cg_res.history,
        },
        "co": {
            "n_states": int(len(co.state_id)),
            "omega_e_cm": dunham_co.omega_e,
            "omega_e_xe_cm": dunham_co.omega_e_xe,
            "b_e_cm": dunham_co.b_e,
            "re_a": dunham_co.re_a,
            "r_min_v0_a": float(rkr_co.r_min[0]),
            "r_max_v0_a": float(rkr_co.r_max[0]),
        },
        "convergence": {
            "n_grid": n_list, "h_a": h_list,
            "err_numerov_o4_cm": err_o4, "err_fd_o2_cm": err_o2,
            "order_numerov_o4": slope_o4, "order_fd_o2": slope_o2,
        },
    }

    (out_dir / "exomol_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_summary(
        out_dir / "exomol_summary.png", r_grid, v_refined, rkr_hcl, dunham_hcl,
        hcl.v_levels, hcl.e_v0, cg_res, sample_states, h_list, err_o4, err_o2,
        slope_o4, slope_o2, nu_300, s_300, nu_1500, s_1500, ftir_300,
    )
    return summary


def _plot_summary(
    fig_path: Path, r_grid: np.ndarray, v_refined: np.ndarray, rkr_hcl: RKRCurve, dunham_hcl: DunhamExpansion,
    v_levels: np.ndarray, e_v0: np.ndarray, cg_res: CGInversionResult, sample_states: dict[int, EigenState],
    h_list: list[float], err_o4: list[float], err_o2: list[float], slope_o4: float, slope_o2: float,
    nu_300: np.ndarray, s_300: np.ndarray, nu_1500: np.ndarray, s_1500: np.ndarray, ftir_300: FTIRResult,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), dpi=160)

    ax = axes[0, 0]
    alpha_bs = np.sqrt(dunham_hcl.morse_wexe_cm / rkr_hcl.c_mu)
    v_morse_bs = morse_potential(r_grid, dunham_hcl.morse_de_cm, alpha_bs, rkr_hcl.re_a)
    ax.plot(r_grid, v_morse_bs / 1e3, "--", color="#888888", lw=1.4, label="2-param Morse baseline")
    ax.plot(r_grid, v_refined / 1e3, color="#1f4e79", lw=2.0, label="CG-Refined EMO $V(r)$")
    ax.scatter(
        np.concatenate([rkr_hcl.r_min, rkr_hcl.r_max]), np.concatenate([rkr_hcl.g_v, rkr_hcl.g_v]) / 1e3,
        s=22, color="#d95f02", zorder=5, label="RKR turning points $(r_{\\min}, r_{\\max})$",
    )
    for v_idx, st in sample_states.items():
        label = f"$|u_{{v={v_idx}}}|^2$" if v_idx in (0, 17) else None
        ax.plot(r_grid, (st.energy / 1e3) + 1.1 * (st.u**2), lw=1.0, alpha=0.85, label=label)
    ax.set_xlim(0.8, 3.3)
    ax.set_ylim(0.0, 40.0)
    ax.set_xlabel("Internuclear distance $r$ ($\\mathrm{\\AA}$)")
    ax.set_ylabel("Potential energy $V(r)$ ($10^3\\,\\mathrm{cm^{-1}}$)")
    ax.set_title("A. $\\mathrm{^{1}H\\text{-}^{35}Cl}$ RKR & CG-EMO Potential Inversion")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    v_fl = v_levels.astype(float)
    e_bs = dunham_hcl.morse_we_cm * v_fl - dunham_hcl.morse_wexe_cm * (v_fl**2 + v_fl)
    ax.plot(v_levels, e_bs - e_v0, "s--", color="#888888", ms=4, label=f"2-param Morse (RMS = {dunham_hcl.morse_rms_cm:.2f} $\\mathrm{{cm^{{-1}}}}$)")
    ax.plot(v_levels, cg_res.e_initial_cm - e_v0, "o-", color="#d95f02", ms=4.5, label=f"1st-order RKR (RMS = {cg_res.rms_initial_cm:.2f} $\\mathrm{{cm^{{-1}}}}$)")
    ax.plot(v_levels, cg_res.e_refined_cm - e_v0, "^-", color="#1b9e77", ms=5, label=f"CG-Refined EMO (RMS = {cg_res.rms_refined_cm:.2f} $\\mathrm{{cm^{{-1}}}}$)")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_yscale("symlog", linthresh=2.0)
    ax.set_xlabel("Vibrational quantum number $v$")
    ax.set_ylabel("Eigenvalue residual $E_{\\mathrm{model}} - E_{\\mathrm{ExoMol}}$ ($\\mathrm{cm^{-1}}$)")
    ax.set_title("B. Inverse-Forward Schrodinger Spectral Residuals")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.loglog(h_list, err_o4, "o-", color="#1f4e79", lw=1.8, label=f"Numerov stencil (order = {slope_o4:.2f})")
    ax.loglog(h_list, err_o2, "s--", color="#c0392b", lw=1.6, label=f"2nd-order FD control (order = {slope_o2:.2f})")
    ax.set_xlabel("Grid spacing $h$ ($\\mathrm{\\AA}$)")
    ax.set_ylabel("Absolute eigenvalue error $|E(h) - E_{\\mathrm{exact}}|$ ($\\mathrm{cm^{-1}}$)")
    ax.set_title("C. Numerov $O(h^4)$ Convergence vs $O(h^2)$ Control")
    ax.legend(fontsize=8.5)
    ax.grid(True, which="both", alpha=0.25)

    ax = axes[1, 1]
    ax.vlines(nu_300, 0.0, s_300 / np.max(s_300), color="#1f4e79", lw=1.3, alpha=0.85, label="$T = 300\\,\\mathrm{K}$ ($J_{\\mathrm{peak}}=2$)")
    ax.vlines(nu_1500, 0.0, s_1500 / np.max(s_300), color="#d95f02", lw=1.3, alpha=0.75, label="$T = 1500\\,\\mathrm{K}$ ($J_{\\mathrm{peak}}=6$)")
    ax.set_xlim(2550.0, 3120.0)
    ax.set_ylim(0.0, 1.12)
    ax.set_xlabel("Transition wavenumber $\\tilde{\\nu}$ ($\\mathrm{cm^{-1}}$)")
    ax.set_ylabel("Normalized line intensity $S_{fi}(T) / S_{\\max}(300\\,\\mathrm{K})$")
    ax.set_title(f"D. Boltzmann P/R-Branch Spectrum (FTIR revival $\\delta={ftir_300.revival_opd_cm:.4f}\\,\\mathrm{{cm}}$)")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def main() -> None:
    summary = run_pipeline()
    hcl, conv = summary["hcl"], summary["convergence"]
    print(
        f"[ExoMol 1H-35Cl] r_e={hcl['re_a']:.4f} A | "
        f"Morse RMS={hcl['morse_baseline_rms_cm']:.2f} cm^-1 -> "
        f"RKR RMS={hcl['rkr_initial_rms_cm']:.2f} cm^-1 -> "
        f"CG-EMO RMS={hcl['cg_refined_rms_cm']:.2f} cm^-1 | "
        f"Numerov order={conv['order_numerov_o4']:.2f}"
    )


if __name__ == "__main__":
    main()
