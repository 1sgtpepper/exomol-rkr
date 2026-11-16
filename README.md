# exomol-rkr

RKR turning-point inversion, Numerov–Cooley radial Schrödinger solver, and FTIR spectral synthesis for closed-shell diatomic molecules (`1H-35Cl` and `12C-16O`) from the [ExoMol database](https://exomol.com).

![ExoMol Summary](outputs/exomol_summary.png)

## Quickstart

```bash
pip install -e ".[dev]"
pytest
exomol-rkr
```

Running `exomol-rkr` writes `outputs/exomol_results.json` and `outputs/exomol_summary.png`.

## Architecture

For $^1\Sigma^+$ diatomic states ($S = \Lambda = 0$), nuclear motion follows the 1D radial Schrödinger equation:

$$-C_\mu \frac{d^2 u_{v,J}(r)}{dr^2} + \left[V(r) + \frac{C_\mu J(J+1)}{r^2}\right] u_{v,J}(r) = E_{v,J} u_{v,J}(r), \qquad C_\mu = \frac{\hbar^2}{2\mu}$$

| Module | Description |
| :--- | :--- |
| `src/exomol_rkr/data.py` | Parses `.states.bz2`, `.trans.bz2`, and `.pf` tables; extracts $E(v,0)$ and $B_v$. |
| `src/exomol_rkr/numerov.py` | Bidirectional $O(h^4)$ Numerov shooting, Sturm node bisection, and Cooley predictor–corrector. |
| `src/exomol_rkr/rkr.py` | Dunham expansion $Y_{kl}$, Gauss–Legendre quadrature, and singular RKR turning-point inversion. |
| `src/exomol_rkr/invert.py` | Extended Morse Oscillator $V_{\text{EMO}}(r)$ fit refined via Fletcher–Reeves conjugate gradient and Brent line search using the Hellmann–Feynman Jacobian. |
| `src/exomol_rkr/spectra.py` | Boltzmann partition sums $Q(T)$, HITRAN line intensities $S_{fi}(T)$, and cosine FFT interferograms. |
| `src/exomol_rkr/cli.py` | End-to-end verification pipeline and summary plot generation. |

## Verification Summary

- **Analytical Morse & $O(h^4)$ scaling**: Numerov eigenvalues match closed-form Morse energies to $< 5 \times 10^{-9}\text{ cm}^{-1}$ under Richardson extrapolation with grid-halving ratio $16.00$ ($O(h^2)$ control ratio $4.00$).
- **Quantum invariants**: Eigenstates satisfy Sturm node parity ($n_{\text{nodes}} = v$), orthonormality ($< 10^{-7}$), rotational Hellmann–Feynman identity ($< 10^{-7}$), and the Virial theorem $2\langle \hat{T} \rangle = \langle r V'_{\text{eff}}(r) \rangle$ ($1.3 \times 10^{-8}$).
- **RKR & EMO potential inversion**: Trigonometric substitution $x' = x\sin^2\theta$ recovers exact Morse turning-point half-widths to $< 10^{-13}\text{ \AA}$. On empirical `1H-35Cl` ($v = 0\dots 17$), forward Numerov RMS error drops from $106.31\text{ cm}^{-1}$ (2-parameter Morse) to $3.05\text{ cm}^{-1}$ (RKR) and $0.91\text{ cm}^{-1}$ (CG-refined EMO).
- **Thermodynamics & FTIR**: Partition sums match ExoMol `.pf` tables to $< 10^{-6}$ (`1H-35Cl`) and $< 10^{-5}$ (`12C-16O`), and the fundamental $v: 0 \to 1$ band reproduces the thermal $J_{\max}$ shift ($J=2 \to 6$ from $300\text{ K}$ to $1500\text{ K}$) and rotational revival at $\delta = (2B_0)^{-1} \approx 0.048\text{ cm}$.
