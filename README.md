# Enumerating Maass Newforms by Symbolic Regression

Companion code for the paper **"Enumerating Maass Newforms by Symbolic
Regression"** by Yang-Hui He, Ritik Jain, Kyu-Hwan Lee, Yau Liu, and Thomas
Oliver.

## Repository structure

```
emnsr/                     shared library
  config.py                global constants (NMAX, K, SEED)
  nt_features.py           phi, mu, sigma, squarefree/prime tests
  data.py                  download + parse the LMFDB Maass newform data
  geometric_features.py    Vol(X_0), Vol(X_1), genus, geodesic lengths
  spectral_features.py     spectral-parameter arrays (X_0(N), Bolza, Klein)
  feature_vectors.py       geodesic length spectra + MLP feature vectors

symbolic_regression/       symbolic regression for Maass newforms
  pnt_example.py           worked example: recover the Prime Number Theorem
  features.py, utils.py    dataset builders + metrics/plots/PySR toolkit
  example_1.py             geometric features -> r_k(N)  (volume-only law)
  example_2.py             arithmetic features -> lambda_k(N)  (12k / phi law)

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
python -m symbolic_regression.pnt_example      # Prime Number Theorem demo
python -m symbolic_regression.example_2        # discovers 12k / phi(N)
python -m symbolic_regression.example_1        # volume-only law from geometry

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

## Acknowledgements

The experiments in this repo were run using data from the [LMFDB](https://www.lmfdb.org/). It was created with the assistance of GPT-5.5 and Claude Opus 4.8.
