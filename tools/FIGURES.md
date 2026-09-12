# Provenance des figures du rapport

Ce fichier répond à une question simple : **pour chaque figure du rapport, quel
code la produit ?** Pour régénérer une figure, lance la source indiquée.

Les carnets s'exécutent depuis `notebooks/` ; les scripts depuis la racine du
projet (`python tools/xxx.py`).

| Figure (images/) | Produite par | Comment régénérer |
|---|---|---|
| `00_gbm_trajectories.png` | carnet `00_fondations.ipynb` | Run All |
| `00_gbm_lognormal.png` | carnet `00_fondations.ipynb` | Run All |
| `01_payoffs.png` | carnet `01_black_scholes.ipynb` | Run All |
| `01_greeks_curves.png` | carnet `01_black_scholes.ipynb` | Run All |
| `01_mc_convergence.png` | carnet `01_black_scholes.ipynb` | Run All |
| `02_tree.png` | carnet `02_binomial.ipynb` | Run All |
| `02_convergence.png` | carnet `02_binomial.ipynb` | Run All |
| `03_payoff_distribution.png` | carnet `03_monte_carlo.ipynb` | Run All |
| `03_barrier_price.png` | carnet `03_monte_carlo.ipynb` | Run All |
| `04_smile.png` | carnet `04_volatility_surface_MODIFIE.ipynb` (cellule smile) | Run All, en séance US |
| `04_term_structure.png` | carnet `04_volatility_surface_MODIFIE.ipynb` | Run All, en séance US |
| `04_vol_surface.png` | carnet `04_volatility_surface_MODIFIE.ipynb` (cellule surface) | Run All, en séance US |
| `05_convergence_methodes.png` | `tools/figures_diagnostic.py` → `figure_convergence()` | `python tools/figures_diagnostic.py` |
| `07_mc_reduction_variance.png` | `tools/figures_mc.py` (version 1 panneau, plus utilisée dans le rapport) | `python tools/figures_mc.py` |
| `07_mecanisme_n.png` | `tools/figures_mc.py` | `python tools/figures_mc.py` |
| `leave_one_out.png` | `tools/figures_diagnostic.py` → `figure_leave_one_out()` | `python tools/figures_diagnostic.py` |
| `smiles_bruts_aapl.png` | `tools/figures_diagnostic.py` → `figure_smiles_bruts()` | `python tools/figures_diagnostic.py` |

## Figures dont le générateur n'est pas (encore) dans le dépôt

Les trois figures suivantes ont été produites lors de sessions antérieures et
leur script de génération n'a pas été retrouvé dans le code actuel. Le PNG est
présent et correct, mais il n'est pas régénérable en l'état :

- `05_vega_amplification.png` (amplification du vega près de l'échéance) ;
- `06_arbre_americain.png` / `.svg` (schéma de l'arbre américain) ;
- `13_error_time.png` (erreur en fonction du temps de calcul).

Si tu veux les rendre reproductibles, je peux reconstruire un script dédié pour
chacune ; dis-le simplement.

## Chiffres (pas figures) du rapport

- Tableaux du laboratoire AAPL (§11) et tableaux régénérés (biais américain,
  sensibilité barrière) : `python tools/labo_aapl.py` → écrit aussi
  `tools/labo_aapl.json`.
- Cotation d'exotiques sur marché : `python tools/exotique_sur_marche.py`.
