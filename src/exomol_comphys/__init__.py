"""ExoMol diatomic quantum spectroscopy and inverse potential solver (PHYS 3178)."""

from .data import MOLECULES, MoleculeDataset, MoleculeSpec, load_molecule
from .invert import EMOPotential, brent_minimize, fit_emo_from_rkr, refine_emo_conjugate_gradient
from .numerov import (
    EigenState,
    diff_central_o4,
    morse_exact_energy,
    morse_potential,
    numerov_shoot,
    simpson_integrate,
    solve_radial_schrodinger,
)
from .rkr import DunhamExpansion, RKRCurve, fit_dunham, gauss_legendre, rkr_turning_points
from .spectra import (
    FTIRResult,
    compute_stick_spectrum,
    partition_function,
    synthesize_ftir_interferogram,
    thermal_j_max,
)

__all__ = [
    "MOLECULES", "DunhamExpansion", "EMOPotential", "EigenState", "FTIRResult",
    "MoleculeDataset", "MoleculeSpec", "RKRCurve", "brent_minimize", "compute_stick_spectrum",
    "diff_central_o4", "fit_dunham", "fit_emo_from_rkr", "gauss_legendre", "load_molecule",
    "morse_exact_energy", "morse_potential", "numerov_shoot", "partition_function",
    "refine_emo_conjugate_gradient", "rkr_turning_points", "simpson_integrate",
    "solve_radial_schrodinger", "synthesize_ftir_interferogram", "thermal_j_max",
]
