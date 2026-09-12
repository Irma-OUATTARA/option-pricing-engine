# Reproduire les figures et valeurs des sections 9, 10 et 11

Toutes les commandes se lancent depuis la racine du projet
(`option-pricing-engine/`). Elles s'appuient sur la chaîne de marché
`data/chaine_options.csv`.

---

## Étape 0 — Quelle chaîne de marché ?

**La chaîne est FIGÉE.** Le fichier canonique est
`data/chaine_AAPL_2026-07-27.csv` (spot 336,19, 186 options). Tous les scripts et
le carnet pointent dessus. Rien ne se re-télécharge : les valeurs et les figures
sont donc **identiques à chaque exécution**, sans dépendance réseau.

Pour capturer une \emph{nouvelle} chaîne un jour : ouvrir le carnet
`04_volatility_surface_MODIFIE.ipynb`, mettre `CAPTURE = True` en tête, et faire
*Run All* **en séance US (15h30–22h Paris)**. Cela écrit un fichier daté
`data/chaine_AAPL_AAAA-MM-JJ.csv` \emph{sans écraser la base figée}. Il faut
ensuite pointer les scripts sur ce nouveau fichier et relancer les étapes 1 à 4.

---

## Étape 1 — Les figures du §9 (smile, structure par terme, surface)

**Hors-ligne, depuis la base figée** (plus besoin de yfinance) :

```bash
python tools/figures_surface_main.py
```

Produit :

| Figure | Contenu |
|---|---|
| `images/04_smile.png` | smiles en ln(K/F), une courbe par maturité |
| `images/04_term_structure.png` | vol ATM en fonction de la maturité |
| `images/04_vol_surface.png` | surface 3D en ln(K/F), axe Z borné [15 %, 55 %] |

---

## Étape 2 — Les VALEURS des §9, §10 et §11

Une seule commande produit la quasi-totalité des chiffres de marché :

```bash
python tools/labo_aapl.py
```

Elle affiche six blocs et écrit `tools/labo_aapl.json`. Correspondance :

| Où dans le rapport | Ce que ça alimente |
|---|---|
| §9 | tableau du biais américain (S = 336,19) — bloc [6] |
| §10 | sensibilité de la barrière `tab:pontsurface` — bloc [4] |
| §11 | structure ATM, européen, américain, asiatique, barrière, leave-one-out — tous les tableaux |

Graine fixée (12345) → tu dois retomber **exactement** sur les nombres du PDF.

---

## Étape 3 — Le diagnostic d'arbitrage du §10

```bash
python tools/diagnostic_arbitrage.py
```

Produit le tableau `tab:diagarbi` (§10) : test calendaire (variance totale
croissante) et test papillon (min de g(k) par maturité).

---

## Étape 4 — La figure leave-one-out du §11

```bash
python tools/figures_diagnostic.py
```

Régénère trois figures :

| Figure | Utilisée où |
|---|---|
| `images/leave_one_out.png` | §11 (validation de la surface) |
| `images/05_convergence_methodes.png` | §7 (réduction de variance, 2 panneaux) |
| `images/smiles_bruts_aapl.png` | note diagnostic « surface plate » (hors rapport) |

---

## Étape 4bis — Les trois éléments ajoutés au chapitre surface

```bash
python tools/figures_surface_extra.py            # 608 par defaut
python tools/figures_surface_extra.py 615        # pour donner le vrai brut du jour
```

Régénère `images/04_pipeline.png` (schéma), `images/04_filtrage.png` (barres
avant/après) et **imprime le tableau du smile** (`tab:smilecar`) à recopier dans
le rapport. Le nombre d'options brutes vient du dict `stats` du carnet 04 ;
passe-le en argument pour la valeur exacte du jour.

## Étape 5 — Recompiler le rapport

- Sur Overleaf : téléverser le zip, ou
- En local : `pdflatex rapport.tex` (trois fois, pour la table des matières).

---

## Récapitulatif express

```bash
# depuis option-pricing-engine/
python tools/labo_aapl.py            # valeurs §9, §10, §11  (+ labo_aapl.json)
python tools/diagnostic_arbitrage.py # tableau d'arbitrage §10
python tools/figures_diagnostic.py   # figure leave-one-out §11 (+ figures §7)
# figures §9 (smile/structure/surface) : carnet 04, Run All en séance US
```

Voir aussi `tools/FIGURES.md` pour la provenance de **toutes** les figures du
rapport (pas seulement §9–11).
