"""Produit TOUS les chiffres numeriques du rapport final, en un seul passage.

Sortie : un fichier JSON (chiffres_rapport.json) consomme par la redaction.

Ce que le script mesure, et rien d'autre :
  1. references par lots (batch means)        -> valeurs de reference + incertitude
  2. tableau "budget fixe"  (M = 400 000)     -> prix, ecart, SE, temps
  3. correlations et beta optimaux            -> rho, beta, 1-rho^2, gain
  4. pentes de convergence                    -> 4 points (M, SE) + regression
  5. M mesure pour IC95 <= 0,01               -> M, prix, IC obtenu, temps

Aucune extrapolation nulle part : le M rapporte est le nombre de trajectoires
effectivement simulees.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from src.black_scholes import BlackScholes
from src.monte_carlo import (mc_european, mc_asian, mc_barrier,
                             geometric_asian_closed_form,
                             payoff_european, payoff_asian, payoff_barrier,
                             control_terminal_spot, control_geometric_asian,
                             control_vanilla)
from src.simulation import simulate_gbm_paths, simulate_gbm_terminal

S, K, r, SG, T, Q = 100.0, 100.0, 0.05, 0.20, 1.0, 0.0
B, NFIX = 120.0, 252
TOL, Z95 = 0.01, 1.96
G_MES, G_REF = 12345, 999

OUT = {}
BS = BlackScholes(S, K, r, SG, T).price()
GEO = geometric_asian_closed_form(S, K, r, SG, T, NFIX)
OUT["exact"] = {"bs_call": BS, "bs_put": BlackScholes(S, K, r, SG, T).price("put"),
                "asian_geo": GEO}
print(f"BS call = {BS:.6f}   asiat.geo = {GEO:.6f}", flush=True)


# =====================================================================
# 1. REFERENCES PAR LOTS
# =====================================================================
def reference_par_lots(pricer, n_lots=10, M_lot=100_000, graine=G_REF):
    prix = np.array([pricer(M_lot, np.random.default_rng(graine + k))
                     for k in range(n_lots)])
    return dict(valeur=float(prix.mean()),
                ci95=float(Z95 * prix.std(ddof=1) / np.sqrt(n_lots)),
                ecart_type_lots=float(prix.std(ddof=1)),
                se=float(prix.std(ddof=1) / np.sqrt(n_lots)),
                lots=[float(x) for x in prix],
                n_lots=n_lots, M_lot=M_lot, M_total=n_lots * M_lot)


print("\n--- 1. references par lots ---", flush=True)
refs = {}
refs["asian_arith"] = reference_par_lots(
    lambda M, g: mc_asian(S, K, r, SG, T, NFIX, M, rng=g, control_variate=True).price)
refs["barrier_out"] = reference_par_lots(
    lambda M, g: mc_barrier(S, K, B, r, SG, T, NFIX, M, rng=g).price)
refs["barrier_in"] = reference_par_lots(
    lambda M, g: mc_barrier(S, K, B, r, SG, T, NFIX, M, rng=g,
                            barrier_type="up-and-in", control_variate=True).price)
for k, v in refs.items():
    print(f"{k:<14} {v['valeur']:.5f} +/- {v['ci95']:.5f}", flush=True)
OUT["references"] = refs


# =====================================================================
# 2. TABLEAU A BUDGET FIXE
# =====================================================================
VARIANTES = [("Monte-Carlo seul", dict()),
             ("+ antithetique", dict(antithetic=True)),
             ("+ variable de controle", dict(control_variate=True)),
             ("+ les deux", dict(antithetic=True, control_variate=True))]


def budget_fixe(pricer, M, variantes=VARIANTES, ref=None, repets=5):
    out = []
    for lib, kw in variantes:
        ts = []
        for _ in range(repets):
            t0 = time.perf_counter()
            res = pricer(M, np.random.default_rng(G_MES), **kw)
            ts.append(time.perf_counter() - t0)
        out.append(dict(methode=lib, prix=res.price, se=res.std_error,
                        ci95=res.ci95, temps_ms=1000 * float(np.median(ts)),
                        ecart=None if ref is None else res.price - ref,
                        M=M))
    base = out[0]
    for d in out:
        d["gain_se"] = base["se"] / d["se"]
        d["efficacite"] = ((base["se"] ** 2 * base["temps_ms"]) /
                           (d["se"] ** 2 * d["temps_ms"]))
    return out


print("\n--- 2. budget fixe M = 400 000 (europeenne) ---", flush=True)
OUT["budget_euro"] = budget_fixe(
    lambda M, g, **kw: mc_european(S, K, r, SG, T, M, rng=g, **kw),
    400_000, ref=BS)
for d in OUT["budget_euro"]:
    print(f"{d['methode']:<24} prix={d['prix']:8.4f} SE={d['se']:.5f} "
          f"t={d['temps_ms']:6.1f}ms gain={d['gain_se']:.2f} "
          f"eff=x{d['efficacite']:.1f}", flush=True)


# =====================================================================
# 3. CORRELATIONS ET BETA
# =====================================================================
def rho_beta(payoff, control, n_steps, M=100_000, n_lots=10, graine=G_REF,
             antithetic=False):
    """rho, beta et gain, mesures sur n_lots echantillons independants.

    Si antithetic=True, les quantites sont calculees sur les MOYENNES DE
    PAIRES : ce sont elles les unites i.i.d., et c'est sur elles que le moteur
    corrige regresse.
    """
    ctx = dict(S0=S, r=r, sigma=SG, T=T, q=Q, n_steps=n_steps)
    EX = float(control.mean(ctx))
    R, Bt, W0, W1 = [], [], [], []
    for k in range(n_lots):
        g = np.random.default_rng(graine + k)
        if n_steps == 1:
            ST = simulate_gbm_terminal(S, r - Q, SG, T, M, rng=g,
                                       antithetic=antithetic)
            paths = ST[:, None]
        else:
            paths = simulate_gbm_paths(S, r - Q, SG, T, n_steps, M, rng=g,
                                       antithetic=antithetic,
                                       include_spot=False)
        disc = np.exp(-r * T)
        Y = disc * payoff(paths)
        X = np.asarray(control.values(paths, ctx), dtype=float)
        if antithetic:
            m = len(Y) // 2
            Y = 0.5 * (Y[:m] + Y[m:2 * m])
            X = 0.5 * (X[:m] + X[m:2 * m])
        c = np.cov(Y, X, ddof=1)
        beta = c[0, 1] / c[1, 1]
        res = Y - beta * (X - EX)
        R.append(c[0, 1] / np.sqrt(c[0, 0] * c[1, 1]))
        Bt.append(beta)
        W0.append(Y.std(ddof=1))
        W1.append(res.std(ddof=1))
    R, Bt, W0, W1 = map(np.array, (R, Bt, W0, W1))
    return dict(rho=float(R.mean()), rho_sd=float(R.std(ddof=1)),
                beta=float(Bt.mean()), beta_sd=float(Bt.std(ddof=1)),
                omega_brut=float(W0.mean()), omega_res=float(W1.mean()),
                un_moins_rho2=float(1 - R.mean() ** 2),
                gain_var_theorique=float(1 / (1 - R.mean() ** 2)),
                gain_var_mesure=float((W0.mean() / W1.mean()) ** 2),
                gain_se_mesure=float(W0.mean() / W1.mean()),
                EX=EX, M=M, n_lots=n_lots)


print("\n--- 3. correlations et beta ---", flush=True)
corr = {}
corr["euro_ST"] = rho_beta(payoff_european(K), control_terminal_spot(), 1,
                           M=200_000)
corr["euro_ST_paires"] = rho_beta(payoff_european(K), control_terminal_spot(), 1,
                                  M=200_000, antithetic=True)
corr["asian_geo"] = rho_beta(payoff_asian(K), control_geometric_asian(K), NFIX)
corr["barrier_in"] = rho_beta(payoff_barrier(K, B, barrier_type="up-and-in"),
                              control_vanilla(K), NFIX)
corr["barrier_out"] = rho_beta(payoff_barrier(K, B, barrier_type="up-and-out"),
                               control_vanilla(K), NFIX)
for k, v in corr.items():
    print(f"{k:<18} rho={v['rho']:+.5f} beta={v['beta']:7.4f} "
          f"1-rho2={v['un_moins_rho2']:.6f} gain_var=x{v['gain_var_mesure']:.1f}",
          flush=True)
OUT["correlations"] = corr


# =====================================================================
# 3bis. REDUCTION DE VARIANCE DECOMPOSEE  (omega / n_iid / SE / gains)
# =====================================================================
def reduction_variance(pricer, M, variantes=VARIANTES):
    """Decompose le gain : SE = omega / sqrt(n_iid).

    omega    : ecart-type empirique de l'echantillon I.I.D. effectivement
               utilise -- payoffs bruts, ou moyennes de paires en antithetique,
               ou payoffs corriges en presence d'un controle.
    n_iid    : nombre de ces unites : M sans antithetique, M/2 avec.
    C'est cette decomposition qui montre OU agit chaque technique.
    """
    out, base = [], None
    for lib, kw in variantes:
        res = pricer(M, np.random.default_rng(G_MES), **kw)
        omega = float(np.std(res.sample, ddof=1))
        if base is None:
            base = res.std_error
        g = base / res.std_error
        out.append(dict(methode=lib, prix=res.price, omega=omega,
                        n_iid=int(res.sample.size), se=res.std_error,
                        M=M, gain_se=g, gain_var=g * g))
    return out


print("\n--- 3bis. reduction de variance decomposee ---", flush=True)
red = {}
red["euro"] = reduction_variance(
    lambda M, g, **kw: mc_european(S, K, r, SG, T, M, rng=g, **kw), 400_000)
red["asian"] = reduction_variance(
    lambda M, g, **kw: mc_asian(S, K, r, SG, T, NFIX, M, rng=g, **kw), 200_000)
red["out"] = reduction_variance(
    lambda M, g, **kw: mc_barrier(S, K, B, r, SG, T, NFIX, M, rng=g, **kw), 200_000)
red["in"] = reduction_variance(
    lambda M, g, **kw: mc_barrier(S, K, B, r, SG, T, NFIX, M, rng=g,
                                  barrier_type="up-and-in", **kw), 200_000)
for nom, rows in red.items():
    for d in rows:
        print(f"{nom:<6} {d['methode']:<24} omega={d['omega']:8.4f} "
              f"n_iid={d['n_iid']:>7,} SE={d['se']:.6f} "
              f"gainSE=x{d['gain_se']:.2f} gainVAR=x{d['gain_var']:.1f}", flush=True)
OUT["reduction_variance"] = red


# =====================================================================
# 4. PENTES DE CONVERGENCE
# =====================================================================
def pente(pricer, tailles, graine=G_MES, **kw):
    ses = [pricer(M, np.random.default_rng(graine), **kw).std_error
           for M in tailles]
    lm, ls = np.log(tailles), np.log(ses)
    a = -np.polyfit(lm, ls, 1)[0]
    deux_pts = -(ls[-1] - ls[0]) / (lm[-1] - lm[0])
    return dict(tailles=list(map(int, tailles)),
                ses=[float(x) for x in ses],
                ln_M=[float(x) for x in lm], ln_SE=[float(x) for x in ls],
                pente=float(a), ecart=float(a - 0.5),
                pente_deux_points=float(deux_pts))


print("\n--- 4. pentes ---", flush=True)
T_EU = [50_000, 100_000, 200_000, 400_000]
T_PD = [25_000, 50_000, 100_000, 200_000]
pentes = {}
for lib, kw in VARIANTES:
    pentes[f"euro|{lib}"] = pente(
        lambda M, g, **k: mc_european(S, K, r, SG, T, M, rng=g, **k), T_EU, **kw)
    pentes[f"asian|{lib}"] = pente(
        lambda M, g, **k: mc_asian(S, K, r, SG, T, NFIX, M, rng=g, **k), T_PD, **kw)
for lib, kw in VARIANTES:
    pentes[f"out|{lib}"] = pente(
        lambda M, g, **k: mc_barrier(S, K, B, r, SG, T, NFIX, M, rng=g, **k),
        T_PD, **kw)
    pentes[f"in|{lib}"] = pente(
        lambda M, g, **k: mc_barrier(S, K, B, r, SG, T, NFIX, M, rng=g,
                                     barrier_type="up-and-in", **k), T_PD, **kw)
for k, v in pentes.items():
    print(f"{k:<32} pente={v['pente']:.3f} (ecart {v['ecart']:+.3f})", flush=True)
OUT["pentes"] = pentes
json.dump(OUT, open("tools/chiffres_rapport.json", "w"), indent=1)
print("\n[sauvegarde intermediaire ecrite]", flush=True)


# =====================================================================
# 5. M MESURE POUR IC95 <= 0,01
# =====================================================================
def M_pour_atteindre(pricer, tol=TOL, M_max=25_000_000, graine=G_MES, **kw):
    n = somme = somme_carres = 0.0
    n = 0
    M_total, k = 0, 0
    t0 = time.perf_counter()
    while M_total < M_max:
        M_lot = int(max(500, min(200_000, M_total // 8)))
        M_lot -= M_lot % 2
        res = pricer(M_lot, np.random.default_rng(graine + k), **kw)
        x = res.sample
        n += x.size
        somme += float(x.sum())
        somme_carres += float(np.square(x).sum())
        M_total += res.n_paths
        k += 1
        if n > 1:
            var = (somme_carres - somme * somme / n) / (n - 1)
            ci = Z95 * np.sqrt(max(var, 0.0) / n)
            if ci <= tol:
                return dict(M=M_total, prix=somme / n, ci95=float(ci),
                            temps_s=time.perf_counter() - t0, n_paquets=k)
    var = (somme_carres - somme * somme / n) / (n - 1)
    return dict(M=M_total, prix=somme / n,
                ci95=float(Z95 * np.sqrt(var / n)),
                temps_s=time.perf_counter() - t0, n_paquets=k, sature=True)


print("\n--- 5. M mesure (IC95 <= 0,01) ---", flush=True)
mesures = {}
jobs = [("euro", lambda M, g, **k: mc_european(S, K, r, SG, T, M, rng=g, **k), 1),
        ("asian", lambda M, g, **k: mc_asian(S, K, r, SG, T, NFIX, M, rng=g, **k), NFIX),
        ("out", lambda M, g, **k: mc_barrier(S, K, B, r, SG, T, NFIX, M, rng=g, **k), NFIX),
        ("in", lambda M, g, **k: mc_barrier(S, K, B, r, SG, T, NFIX, M, rng=g,
                                            barrier_type="up-and-in", **k), NFIX)]
for nom, pr, nst in jobs:
    for lib, kw in VARIANTES:
        d = M_pour_atteindre(pr, **kw)
        d["n_steps"] = nst
        d["tirages"] = d["M"] * nst
        mesures[f"{nom}|{lib}"] = d
        print(f"{nom:<6} {lib:<24} M={d['M']:>11,} prix={d['prix']:8.4f} "
              f"IC={d['ci95']:.5f} t={d['temps_s']:6.1f}s", flush=True)
    OUT["mesures"] = mesures
    json.dump(OUT, open("tools/chiffres_rapport.json", "w"), indent=1)

json.dump(OUT, open("tools/chiffres_rapport.json", "w"), indent=1)
print("\n=== termine : tools/chiffres_rapport.json ===")
