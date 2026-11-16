"""ExoMol dataset loader, physical constants, and rovibrational state extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import urllib.request
import numpy as np

# CODATA 2018 spectroscopic constants: h / (8 * pi^2 * c * u * 1e-20) in [cm^-1 * A^2 * Da]
HBAR2_OVER_2U_CM_A2: float = 16.8576291916
C2_CM_K: float = 1.43877736
C_LIGHT_CM_S: float = 2.99792458e10


@dataclass(frozen=True)
class MoleculeSpec:
    """Metadata and isotopic constants for an ExoMol diatomic molecule."""

    slug: str
    formula: str
    dataset: str
    base_url: str
    mass_1_da: float
    mass_2_da: float
    de_cm: float
    has_transitions: bool = True

    @property
    def prefix(self) -> str:
        return f"{self.slug}__{self.dataset}"

    @property
    def reduced_mass_da(self) -> float:
        return (self.mass_1_da * self.mass_2_da) / (self.mass_1_da + self.mass_2_da)

    @property
    def c_mu(self) -> float:
        return HBAR2_OVER_2U_CM_A2 / self.reduced_mass_da


MOLECULES: dict[str, MoleculeSpec] = {
    "1H-35Cl": MoleculeSpec(
        "1H-35Cl", "1H-35Cl", "HITRAN-HCl",
        "https://exomol.com/db/HCl/1H-35Cl/HITRAN-HCl",
        1.00782503223, 34.968852682, 37240.0, True,
    ),
    "12C-16O": MoleculeSpec(
        "12C-16O", "12C-16O", "Li2015",
        "https://exomol.com/db/CO/12C-16O/Li2015",
        12.000000000, 15.99491461957, 90670.0, False,
    ),
}


@dataclass(frozen=True)
class MoleculeDataset:
    """Parsed ExoMol rovibrational states, transitions, and partition function."""

    spec: MoleculeSpec
    state_id: np.ndarray
    energy_cm: np.ndarray
    g_tot: np.ndarray
    j_rot: np.ndarray
    v_vib: np.ndarray
    v_levels: np.ndarray
    e_v0: np.ndarray
    b_v: np.ndarray
    pf_temp_k: np.ndarray
    pf_q_ref: np.ndarray
    trans_up_id: np.ndarray | None = None
    trans_lo_id: np.ndarray | None = None
    trans_a_s: np.ndarray | None = None
    trans_nu_cm: np.ndarray | None = None


def _ensure_file(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            path.write_bytes(resp.read())
    return path


def load_molecule(
    slug: str = "1H-35Cl", data_dir: Path | str | None = None, max_v: int | None = None
) -> MoleculeDataset:
    """Load an ExoMol diatomic dataset and extract vibrational/rotational ladders."""
    if slug not in MOLECULES:
        raise ValueError(f"Unsupported molecule {slug!r}; choose from {tuple(MOLECULES)}")
    spec = MOLECULES[slug]
    root = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parents[2] / "data"

    states_path = _ensure_file(f"{spec.base_url}/{spec.prefix}.states.bz2", root / f"{spec.prefix}.states.bz2")
    pf_path = _ensure_file(f"{spec.base_url}/{spec.prefix}.pf", root / f"{spec.prefix}.pf")

    s_arr = np.loadtxt(states_path, usecols=(0, 1, 2, 3, 4), dtype=np.float64)
    state_id = s_arr[:, 0].astype(np.int64)
    energy_cm = s_arr[:, 1]
    g_tot = s_arr[:, 2].astype(np.int64)
    j_rot = s_arr[:, 3].astype(np.int64)
    v_vib = s_arr[:, 4].astype(np.int64)

    by_v: dict[int, dict[int, float]] = {}
    for v_val, j_val, e_val in zip(v_vib, j_rot, energy_cm):
        v_int, j_int = int(v_val), int(j_val)
        if (max_v is None or v_int <= max_v) and j_int <= 6:
            by_v.setdefault(v_int, {})[j_int] = float(e_val)

    v_levels = np.array(sorted(v for v, j_map in by_v.items() if 0 in j_map and len(j_map) >= 4), dtype=np.int64)
    e_v0 = np.array([by_v[int(v)][0] for v in v_levels], dtype=np.float64)
    b_v = np.zeros(len(v_levels), dtype=np.float64)
    for idx, v in enumerate(v_levels):
        j_list = sorted(by_v[int(v)].keys())
        jj = np.array([j * (j + 1) for j in j_list], dtype=np.float64)
        ee = np.array([by_v[int(v)][j] for j in j_list], dtype=np.float64)
        b_v[idx] = np.polyfit(jj, ee, 2)[1]

    pf_arr = np.loadtxt(pf_path, dtype=np.float64)
    trans_up = trans_lo = trans_a = trans_nu = None
    if spec.has_transitions:
        trans_path = _ensure_file(f"{spec.base_url}/{spec.prefix}.trans.bz2", root / f"{spec.prefix}.trans.bz2")
        t_arr = np.loadtxt(trans_path, dtype=np.float64)
        trans_up = t_arr[:, 0].astype(np.int64)
        trans_lo = t_arr[:, 1].astype(np.int64)
        trans_a = t_arr[:, 2]
        trans_nu = t_arr[:, 3]

    return MoleculeDataset(
        spec, state_id, energy_cm, g_tot, j_rot, v_vib, v_levels, e_v0, b_v,
        pf_arr[:, 0], pf_arr[:, 1], trans_up, trans_lo, trans_a, trans_nu,
    )
