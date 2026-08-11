# Enumerating Maass Newforms by Symbolic Regression

Companion code for the paper **"Enumerating Maass Newforms by Symbolic
Regression"** by Yang-Hui He, Ritik Jain, Kyu-Hwan Lee, Yau Liu, and Thomas
Oliver.

Symbolic regression on the spectral parameters of Maass newforms on the modular
curves $X_0(N)$ recovers the counting law

$$\Lambda_{\mathrm{new}}(x;N) \sim \frac{\varphi(N)\,x}{12},
\qquad\Longleftrightarrow\qquad
\lambda_k^{\mathrm{new}}(N) \sim \frac{12k}{\varphi(N)},$$

which is then proved via Atkin–Lehner theory and Möbius inversion. Compared with
Weyl's law for the full spectrum, $\lambda_k(N)\sim 4\pi k / \big(\tfrac{\pi}{3}
N\prod_{p\mid N}(1+\tfrac1p)\big)$, the newform law replaces $\prod_{p\mid N}(1+
1/p)$ with $\prod_{p\mid N}(1-1/p)=\varphi(N)/N$.

## Repository structure

```
emnsr/                     shared library
  config.py                global constants (NMAX, K, SEED)
  arithmetic.py            phi, mu, sigma, squarefree/prime tests
  data.py                  download + parse the LMFDB Maass newform data
  geometry.py              Vol(X_0), Vol(X_1), genus, geodesic lengths
  spectral.py              spectral-parameter arrays (X_0(N), Bolza, Klein)
  features.py              geodesic length spectra + MLP feature vectors

symbolic_regression/       symbolic regression for Maass newforms
  example.py               worked example: recover the Prime Number Theorem
  features.py, metrics.py, plots.py, pysr_tools.py
  ex31_geometric.py        geometric features -> r_k(N)  (volume-only law)
  ex32_arithmetic.py       arithmetic features -> lambda_k(N)  (12k / phi law)

nonlinear_regression/      Selberg-trace-formula-constrained regression
  stf.py                   shared trace-formula machinery
  poisson_s1.py            circle S^1 (Poisson summation)
  bolza.py                 Bolza surface (verification + recovery)
  klein.py                 Klein quartic

mlp/                       multilayer perceptron for spectral parameters
  model.py                 model, training, evaluation, saliency, LOOCV
  run.py                   train on X_0(N); transfer to Bolza / Klein

save_datasets.py           cache feature/spectral arrays to datasets.npz
```

## Installation

```bash
pip install -e .          # installs the packages + dependencies
# or: pip install -r requirements.txt
```

The symbolic-regression scripts use [PySR](https://github.com/MilesCranmer/PySR),
which provisions its Julia backend automatically on first run. The datasets are
downloaded on first use: the Maass newform spectral parameters from the LMFDB
(mirrored on [Zenodo record 15490636](https://zenodo.org/records/15490636)) and
the Bolza eigenvalues from arXiv:1110.2150.

## Running the experiments

Each experiment is a module, run from the repository root:

```bash
# Symbolic regression
python -m symbolic_regression.example          # Prime Number Theorem demo
python -m symbolic_regression.ex32_arithmetic  # discovers 12k / phi(N)
python -m symbolic_regression.ex31_geometric   # volume-only law from geometry

# Selberg-trace-formula regression
python -m nonlinear_regression.poisson_s1
python -m nonlinear_regression.bolza
python -m nonlinear_regression.klein

# MLP
python -m mlp.run
```

## Data

- **Maass newforms** — LMFDB, mirrored on Zenodo record `15490636`
  (`MaassForms.txt`; colon-separated, `level = field[1]`, `r = field[2]`; the
  Laplace eigenvalue is `lambda = r^2 + 1/4`).
- **Bolza surface** — eigenvalues from Stohmaier & Uski (arXiv:1110.2150);
  primitive geodesic lengths and multiplicities from Aurich & Steiner (1988,
  *Physica D* 32:451).
- **Klein quartic** — spectral parameters and primitive geodesic lengths from
  the periodic-orbit literature.

## License

MIT — see [LICENSE](LICENSE).
