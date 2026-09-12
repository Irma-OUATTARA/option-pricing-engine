"""LABORATOIRE AAPL : un sous-jacent reel, une surface, quatre contrats.

Un seul jeu de donnees de marche (la chaine AAPL nettoyee par le carnet 04)
alimente quatre contrats HYPOTHETIQUES distincts. On ne renomme jamais une
option cotee : on partage seulement les intrants (spot, r, q, surface de vol).

Sorties, dans l'ordre :
  0. structure par terme + interpolation en variance totale
  1. Contrat europeen  : BSM / CRR / MC (coherence interne des 3 moteurs)
  2. Contrat americain : CRR, prime d'exercice anticipe
  3. Contrat asiatique : MC brut / antithetique / controle / combine
  4. Contrat barriere  : MC + sensibilite au point de la surface
  5. Validation leave-one-out de la surface (test hors-echantillon)
  6. Tableau du biais americain regenere sur la vraie chaine

Usage : python tools/labo_aapl.py [chemin_csv]
Le CSV attendu est celui exporte par le carnet 04 (colonnes type,K,T,mid,iv,spot).
Toutes les valeurs numeriques sont ecrites aussi dans tools/labo_aapl.json,
pour etre reprises telles quelles dans le rapport.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from src.black_scholes import BlackScholes
from src.binomial import BinomialTree
from src.monte_carlo import mc_asian, mc_barrier, mc_european

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DEFAUT = os.path.join(RACINE, "data", "chaine_AAPL_2026-07-27.csv")
JSON_OUT = os.path.join(RACINE, "tools", "labo_aapl.json")

R_DISC = 0.04      # taux d'actualisation (T-bill US 1 an ~ 4 %)
Q_DIV = 0.004      # rendement de dividende AAPL (~ 0,4 %/an)
GRAINE = 12345     # graine fixe -> chiffres reproductibles pour le rapport


# ===========================================================================
#  0. SURFACE : structure par terme ATM + interpolation en variance totale
# ===========================================================================
def structure_atm(chain, spot):
    """Vol a la monnaie par maturite (strike le plus proche du spot)."""
    pts = []
    for T, g in chain.groupby("T"):
        i = (g["K"] - spot).abs().idxmin()
        pts.append((float(T), float(g.loc[i, "iv"]), float(g.loc[i, "K"])))
    return sorted(pts)


def vol_interpolee(pts, T_cible):
    """Interpolation en VARIANCE TOTALE sigma^2*T (seule sans arbitrage
    calendaire : la variance totale doit croitre avec T)."""
    Ts = np.array([p[0] for p in pts])
    var_tot = np.array([p[1] ** 2 * p[0] for p in pts])
    v = np.interp(T_cible, Ts, var_tot)
    return float(np.sqrt(v / T_cible))


def vol_au_strike(chain, T_cible, K_cible):
    """Vol lue sur le smile de la maturite cotee la plus proche, au strike voulu."""
    Ts = np.array(sorted(chain["T"].unique()))
    T_proche = Ts[np.argmin(np.abs(Ts - T_cible))]
    g = chain[chain["T"] == T_proche].sort_values("K")
    return float(np.interp(K_cible, g["K"], g["iv"])), float(T_proche)


# ===========================================================================
def main(chemin):
    chain = pd.read_csv(chemin)
    spot = float(chain["spot"].iloc[0])
    r, q = R_DISC, Q_DIV
    out = {"spot": spot, "r": r, "q": q, "n_options": int(len(chain)),
           "n_maturites": int(chain["T"].nunique())}

    print("=" * 74)
    print(f"LABORATOIRE AAPL   spot={spot:.2f}   r={r:.1%}   q={q:.2%}   "
          f"{len(chain)} options / {chain['T'].nunique()} maturites")
    print("=" * 74)

    pts = structure_atm(chain, spot)
    print("\n[0] STRUCTURE PAR TERME A LA MONNAIE")
    print(f"    {'maturite':>10}{'K ATM':>9}{'vol':>9}{'var.tot.':>11}")
    struct = []
    for T, iv, Kk in pts:
        print(f"    {T*365:>7.0f} j{Kk:>9.1f}{iv:>8.1%}{iv**2*T:>11.5f}")
        struct.append({"T_j": round(T*365), "K_atm": Kk, "iv": iv,
                       "var_tot": iv**2*T})
    out["structure_atm"] = struct

    # -----------------------------------------------------------------------
    # 1. CONTRAT EUROPEEN  (K=ATM, T=91j) : BSM / CRR / MC
    # -----------------------------------------------------------------------
    K1 = round(spot / 5) * 5.0
    T1 = 0.25
    sig1 = vol_interpolee(pts, T1)
    bsm = BlackScholes(spot, K1, r, sig1, T1, q).price("call")
    crr = BinomialTree(spot, K1, r, sig1, T1, n_steps=2000, q=q).price("call", "european")
    mc = mc_european(spot, K1, r, sig1, T1, n_paths=400_000, option_type="call",
                     q=q, rng=np.random.default_rng(GRAINE),
                     antithetic=True, control_variate=True)
    print("\n[1] CONTRAT EUROPEEN  call "
          f"K={K1:.0f}  T={T1*365:.0f}j  sigma(surface)={sig1:.2%}")
    print(f"    {'Black-Scholes (analytique)':<34}{bsm:>10.4f}")
    print(f"    {'CRR 2000 pas':<34}{crr:>10.4f}   ecart {abs(crr-bsm):.1e}")
    print(f"    {'Monte-Carlo (anti+controle)':<34}{mc.price:>10.4f}"
          f"   +/-{mc.ci95:.4f}   ecart {abs(mc.price-bsm):+.4f}")
    dansIC = abs(mc.price - bsm) <= mc.ci95
    print(f"    BSM dans l'IC95 du MC : {'oui' if dansIC else 'NON'}")
    out["europeen"] = {"K": K1, "T": T1, "sigma": sig1, "bsm": bsm, "crr": crr,
                       "ecart_crr": abs(crr-bsm), "mc": mc.price,
                       "mc_se": mc.std_error, "mc_ci95": mc.ci95,
                       "ecart_mc": abs(mc.price-bsm), "dans_ic": bool(dansIC)}

    # -----------------------------------------------------------------------
    # 2. CONTRAT AMERICAIN  (put ITM) : CRR, prime d'exercice
    # -----------------------------------------------------------------------
    K2 = round(spot * 1.05 / 5) * 5.0     # put ITM (strike > spot)
    T2 = 0.50
    sig2, _ = vol_au_strike(chain, T2, K2)
    euro = BinomialTree(spot, K2, r, sig2, T2, n_steps=2000, q=q).price("put", "european")
    amer = BinomialTree(spot, K2, r, sig2, T2, n_steps=2000, q=q).price("put", "american")
    prime = amer - euro
    print("\n[2] CONTRAT AMERICAIN  put ITM "
          f"K={K2:.0f}  T={T2*365:.0f}j  sigma(strike)={sig2:.2%}")
    print(f"    {'Put europeen (CRR)':<34}{euro:>10.4f}")
    print(f"    {'Put americain (CRR)':<34}{amer:>10.4f}")
    print(f"    {'Prime d exercice anticipe':<34}{prime:>10.4f}   ({prime/euro:.2%})")
    print("    Black-Scholes NON applicable (pas d'exercice anticipe) ;")
    print("    Monte-Carlo classique NON applicable (necessiterait Longstaff-Schwartz).")
    out["americain"] = {"K": K2, "T": T2, "sigma": sig2, "euro": euro,
                        "amer": amer, "prime": prime, "prime_pct": prime/euro}

    # -----------------------------------------------------------------------
    # 3. CONTRAT ASIATIQUE  : MC brut / anti / controle / combine
    # -----------------------------------------------------------------------
    K3 = K1
    T3 = 0.25
    sig3 = sig1
    nfix = 63
    print("\n[3] CONTRAT ASIATIQUE  call arithmetique "
          f"K={K3:.0f}  T={T3*365:.0f}j  {nfix} fixings  sigma={sig3:.2%}")
    print(f"    {'variante':<30}{'prix':>10}{'SE':>10}{'gain vs brut':>14}")
    asi = {}
    se_brut = None
    combos = [("MC brut", dict(antithetic=False, control_variate=False)),
              ("+ antithetiques", dict(antithetic=True, control_variate=False)),
              ("+ controle (geom.)", dict(antithetic=False, control_variate=True)),
              ("+ anti & controle", dict(antithetic=True, control_variate=True))]
    for lib, kw in combos:
        res = mc_asian(spot, K3, r, sig3, T3, n_steps=nfix, n_paths=200_000,
                       q=q, rng=np.random.default_rng(GRAINE), **kw)
        if se_brut is None:
            se_brut = res.std_error
        gain = (se_brut / res.std_error) ** 2
        print(f"    {lib:<30}{res.price:>10.4f}{res.std_error:>10.5f}"
              f"{gain:>12.1f}x")
        asi[lib] = {"prix": res.price, "se": res.std_error, "gain": gain}
    out["asiatique"] = {"K": K3, "T": T3, "sigma": sig3, "nfix": nfix,
                        "variantes": asi}

    # -----------------------------------------------------------------------
    # 4. CONTRAT BARRIERE  up-and-out + sensibilite au point de surface
    # -----------------------------------------------------------------------
    K4 = K1
    B4 = round(spot * 1.15 / 5) * 5.0
    T4 = 0.25
    sig_atm = sig1
    iv_K, _ = vol_au_strike(chain, T4, K4)
    iv_B, _ = vol_au_strike(chain, T4, B4)
    van4 = BlackScholes(spot, K4, r, sig_atm, T4, q).price("call")
    print("\n[4] CONTRAT BARRIERE  up-and-out call "
          f"K={K4:.0f}  B={B4:.0f}  T={T4*365:.0f}j")
    print(f"    {'Call vanille (reference, vol ATM)':<38}{van4:>10.4f}")
    print(f"    {'sensibilite au point de surface':<38}")
    print(f"    {'  vol retenue':<28}{'valeur':>9}{'prix':>11}{'ecart/ATM':>11}")
    barr = {}
    ref = None
    for lib, s in [("ATM interpolee", sig_atm),
                   (f"lue au strike K={K4:.0f}", iv_K),
                   (f"lue a la barriere B={B4:.0f}", iv_B)]:
        px = mc_barrier(spot, K4, B4, r, s, T4, n_steps=63, n_paths=300_000,
                        q=q, rng=np.random.default_rng(GRAINE)).price
        if ref is None:
            ref = px
        print(f"    {lib:<28}{s:>9.2%}{px:>11.4f}{px/ref-1:>10.1%}")
        barr[lib] = {"vol": s, "prix": px, "ecart": px/ref-1}
    out["barriere"] = {"K": K4, "B": B4, "T": T4, "vanille": van4,
                       "variantes": barr}

    # -----------------------------------------------------------------------
    # 5. VALIDATION LEAVE-ONE-OUT (test hors-echantillon de la surface)
    # -----------------------------------------------------------------------
    print("\n[5] VALIDATION LEAVE-ONE-OUT (reconstruction hors-echantillon)")
    print("    on retire un point interieur, on interpole le smile sans lui,")
    print("    on compare la vol predite a la vol observee.")
    erreurs = []
    detail = []
    for T, g in chain.groupby("T"):
        g = g.sort_values("K").reset_index(drop=True)
        if len(g) < 5:
            continue
        for i in range(1, len(g) - 1):          # points interieurs seulement
            K_test = g.loc[i, "K"]
            iv_obs = g.loc[i, "iv"]
            reste = g.drop(index=i)
            iv_pred = float(np.interp(K_test, reste["K"], reste["iv"]))
            err = abs(iv_pred - iv_obs)
            erreurs.append(err)
            detail.append((round(T*365), K_test, iv_obs, iv_pred, err))
    erreurs = np.array(erreurs)
    print(f"    points testes            : {len(erreurs)}")
    print(f"    erreur absolue moyenne   : {erreurs.mean()*100:.2f} pts de vol")
    print(f"    erreur mediane           : {np.median(erreurs)*100:.2f} pts")
    print(f"    90e centile              : {np.percentile(erreurs,90)*100:.2f} pts")
    print(f"    erreur maximale          : {erreurs.max()*100:.2f} pts")
    # 3 exemples representatifs (median, un bon, le pire)
    detail.sort(key=lambda d: d[4])
    ex = [detail[len(detail)//2], detail[1], detail[-1]]
    print("    exemples (maturite, strike, vol obs, vol predite, erreur) :")
    for T_j, Kk, o, p, e in ex:
        print(f"      {T_j:>4}j  K={Kk:>7.1f}  obs={o:.1%}  pred={p:.1%}  "
              f"err={e*100:.2f} pts")
    out["leave_one_out"] = {
        "n": int(len(erreurs)),
        "mae_pts": float(erreurs.mean()*100),
        "median_pts": float(np.median(erreurs)*100),
        "p90_pts": float(np.percentile(erreurs, 90)*100),
        "max_pts": float(erreurs.max()*100),
        "exemples": [{"T_j": t, "K": k, "obs": o, "pred": p, "err_pts": e*100}
                     for t, k, o, p, e in ex]}

    # -----------------------------------------------------------------------
    # 6. BIAIS AMERICAIN regenere sur la vraie chaine (spot reel)
    # -----------------------------------------------------------------------
    print("\n[6] BIAIS D'EXERCICE AMERICAIN (regenere, spot reel)")
    T6 = 0.25
    sig6, _ = vol_au_strike(chain, T6, spot)      # vol ATM ~ maturite 3 mois
    print(f"    S={spot:.2f}  sigma(ATM)={sig6:.2%}  q={q:.2%}  T={T6*365:.0f}j")
    print(f"    {'contrat':<22}{'europeen':>11}{'americain':>11}{'prime':>9}")
    biais = []
    cas = [("call OTM K=+8%", "call", round(spot*1.08/5)*5.0),
           ("put OTM  K=-8%", "put", round(spot*0.92/5)*5.0),
           ("put ATM  K=S",   "put", round(spot/5)*5.0),
           ("put ITM  K=+18%", "put", round(spot*1.18/5)*5.0)]
    for lib, typ, Kk in cas:
        e = BinomialTree(spot, Kk, r, sig6, T6, n_steps=2000, q=q).price(typ, "european")
        a = BinomialTree(spot, Kk, r, sig6, T6, n_steps=2000, q=q).price(typ, "american")
        pr = (a - e) / e if e > 1e-9 else 0.0
        print(f"    {lib:<22}{e:>11.4f}{a:>11.4f}{pr:>8.2%}")
        biais.append({"contrat": lib, "K": Kk, "euro": e, "amer": a, "prime_pct": pr})
    out["biais_americain"] = {"sigma_atm": sig6, "T": T6, "cas": biais}

    with open(JSON_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nToutes les valeurs -> {os.path.relpath(JSON_OUT, RACINE)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else CSV_DEFAUT)
