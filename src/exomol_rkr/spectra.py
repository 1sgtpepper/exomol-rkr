"""Boltzmann partition function, radiative line intensities, and FTIR interferometry."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .data import C2_CM_K, C_LIGHT_CM_S, MoleculeDataset


@dataclass(frozen=True)
class FTIRResult:
    """Synthetic Michelson FTIR interferogram and inverse Fourier spectral recovery."""

    nu_grid_cm: np.ndarray
    spectrum_true: np.ndarray
    opd_cm: np.ndarray
    interferogram: np.ndarray
    spectrum_apodized: np.ndarray
    spectrum_unapodized: np.ndarray
    zpd_intensity: float
    integrated_intensity: float
    revival_opd_cm: float


def partition_function(
    energies_cm: np.ndarray, degeneracies: np.ndarray, temperatures_k: float | np.ndarray
) -> float | np.ndarray:
    """Compute Boltzmann partition function Q(T) = sum_i g_i exp(-c_2 E_i / T)."""
    t_arr = np.asarray(temperatures_k, dtype=np.float64)
    e_col = np.asarray(energies_cm, dtype=np.float64)[:, None]
    g_col = np.asarray(degeneracies, dtype=np.float64)[:, None]
    q_vals = np.sum(g_col * np.exp(-C2_CM_K * e_col / np.atleast_1d(t_arr)[None, :]), axis=0)
    return float(q_vals[0]) if np.ndim(temperatures_k) == 0 else q_vals


def thermal_j_max(temperature_k: float, b_rot_cm: float) -> float:
    """Analytical maximum of thermal rotational population J_max = sqrt(T / (2 c_2 B)) - 1/2."""
    return float(np.sqrt(temperature_k / (2.0 * C2_CM_K * b_rot_cm)) - 0.5)


def compute_stick_spectrum(
    dataset: MoleculeDataset, temperature_k: float, v_up: int | None = 1, v_lo: int | None = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute HITRAN-convention absorption line intensities S_fi(T) in [cm/molecule]."""
    if dataset.trans_up_id is None or dataset.trans_lo_id is None or dataset.trans_a_s is None:
        raise ValueError(f"Dataset {dataset.spec.slug} does not include radiative transitions")

    up_idx, lo_idx = dataset.trans_up_id - 1, dataset.trans_lo_id - 1
    mask = np.ones(len(up_idx), dtype=bool)
    if v_up is not None:
        mask &= dataset.v_vib[up_idx] == v_up
    if v_lo is not None:
        mask &= dataset.v_vib[lo_idx] == v_lo

    up_idx, lo_idx, a_fi = up_idx[mask], lo_idx[mask], dataset.trans_a_s[mask]
    e_up, e_lo = dataset.energy_cm[up_idx], dataset.energy_cm[lo_idx]
    g_up = dataset.g_tot[up_idx].astype(np.float64)
    j_lo = dataset.j_rot[lo_idx]
    delta_j = dataset.j_rot[up_idx] - j_lo

    nu_fi = e_up - e_lo
    q_t = float(partition_function(dataset.energy_cm, dataset.g_tot, temperature_k))
    boltz = np.exp(-C2_CM_K * e_lo / temperature_k)
    stim = -np.expm1(-C2_CM_K * nu_fi / temperature_k)
    s_fi = (g_up * a_fi / (8.0 * np.pi * C_LIGHT_CM_S * (nu_fi**2) * q_t)) * boltz * stim

    order = np.argsort(nu_fi)
    return nu_fi[order], s_fi[order], j_lo[order], delta_j[order]


def synthesize_ftir_interferogram(
    nu_lines: np.ndarray,
    s_lines: np.ndarray,
    nu_min: float = 2550.0,
    nu_max: float = 3150.0,
    n_grid: int = 4096,
    gamma_cm: float = 0.25,
) -> FTIRResult:
    """Synthesize Lorentzian infrared spectrum, Michelson FTIR interferogram, and inverse FFT recovery."""
    nu_grid = np.linspace(nu_min, nu_max, n_grid, endpoint=False, dtype=np.float64)
    d_nu = float(nu_grid[1] - nu_grid[0])

    diff = nu_grid[:, None] - np.asarray(nu_lines, dtype=np.float64)[None, :]
    spectrum_true = ((gamma_cm / np.pi) / (diff**2 + gamma_cm**2)) @ np.asarray(s_lines, dtype=np.float64)

    fft_coeffs = np.fft.rfft(spectrum_true) * d_nu
    interferogram = np.real(fft_coeffs)
    opd_cm = np.arange(len(fft_coeffs), dtype=np.float64) / (n_grid * d_nu)

    spectrum_unapodized = np.fft.irfft(fft_coeffs / d_nu, n=n_grid)
    hann_win = 0.5 * (1.0 + np.cos(np.pi * np.arange(len(fft_coeffs)) / (len(fft_coeffs) - 1)))
    spectrum_apodized = np.fft.irfft((fft_coeffs * hann_win) / d_nu, n=n_grid)

    rev_mask = (opd_cm >= 0.035) & (opd_cm <= 0.060)
    rev_idx = np.where(rev_mask)[0][int(np.argmax(np.abs(interferogram[rev_mask])))]
    return FTIRResult(
        nu_grid, spectrum_true, opd_cm, interferogram, spectrum_apodized,
        spectrum_unapodized, float(interferogram[0]), float(np.sum(spectrum_true) * d_nu), float(opd_cm[rev_idx]),
    )
