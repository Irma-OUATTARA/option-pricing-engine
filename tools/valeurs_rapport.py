"""Produit TOUS les chiffres cites dans le rapport, a convention fixee.

Pourquoi ce script existe
-------------------------
Les valeurs d'exotiques du rapport venaient de sources differentes (carnets a
n=100 ou n=50, etude de convergence a n=252), ce qui a produit deux valeurs
differentes pour le meme produit. Un prix d'exotique n'a aucun sens sans son
nombre de fixings n : plus n est petit, plus une barriere out survit et plus
elle est chere ; plus n est petit, moins la moyenne d'une asiatique est lissee
et plus elle est chere.

Ce script est donc la SOURCE UNIQUE des chiffres du rapport. Une commande :

    python tools/valeurs_rapport.py

Convention retenue et a citer dans le rapport :
    S0 = K = 100, r = 5 %, sigma = 20 %, T = 1 an, q = 0
    n = 252 fixings pour tous les produits path-dependent
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from src.black_scholes import BlackScholes
from src.binomial import BinomialTree
from src.monte_carlo import (mc_european, mc_asian, mc_barrier, mc_price,
                             payoff_digital, geometric_asian_closed_form)

S, K, r, sig, T, q = 100.0, 100.0, 0.05, 0.20, 1.0, 0.0
N_FIX = 252                 # fixings des produits path-dependent
GRAINE_REF = 999            # references
GRAINE_MES = 12345          # mesures ponctuelles


def reference_par_lots(pricer, n_lots=10, M_lot=100_000, graine=GRAINE_REF):
    """Moyenne de lots independants : precision d'un gros M, memoire d'un petit."""
    prix = np.array([pricer(M_lot, np.random.default_rng(graine + k))
                     for k in range(n_lots)])
    return prix.mean(), 1.96 * prix.std(ddof=1) / np.sqrt(n_lots)


def titre(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


print(f"Point de reference : S0 = K = {S:.0f}, r = {r:.0%}, sigma = {sig:.0%}, "
      f"T = {T:.0f} an, q = {q:.0f}")
print(f"Fixings des produits path-dependent : n = {N_FIX}")

# ---------------------------------------------------------------- vanilles
titre("VANILLES  (sections 5, 6, tableau 14)")
bs = BlackScholes(S, K, r, sig, T, q)
call_bs, put_bs = bs.price("call"), bs.price("put")
arbre_eur = BinomialTree(S, K, r, sig, T, 2000, q).price("call", "european")
arbre_am_put = BinomialTree(S, K, r, sig, T, 2000, q).price("put", "american")
arbre_am_call = BinomialTree(S, K, r, sig, T, 2000, q).price("call", "american")
mc_eur = mc_european(S, K, r, sig, T, 400_000,
                     rng=np.random.default_rng(GRAINE_MES))

print(f"call europeen  Black-Scholes           {call_bs:10.4f}")
print(f"put  europeen  Black-Scholes           {put_bs:10.4f}")
print(f"parite C - P                           {call_bs-put_bs:10.10f}")
print(f"           S0 - K e^(-rT)              {S-K*np.exp(-r*T):10.10f}")
print(f"call europeen  arbre CRR N=2000        {arbre_eur:10.4f}"
      f"   ecart a BS {arbre_eur-call_bs:+.4f}")
print(f"call europeen  Monte-Carlo M=400 000   {mc_eur.price:10.4f}"
      f"   IC95 +/-{mc_eur.ci95:.4f}")
print(f"put  americain arbre CRR N=2000        {arbre_am_put:10.4f}"
      f"   prime d'exercice {arbre_am_put-put_bs:+.4f}")
print(f"call americain arbre CRR N=2000        {arbre_am_call:10.4f}"
      f"   (= europeen, jamais exerce)")

# --------------------------------------------------------------- asiatiques
titre(f"ASIATIQUES  (sections 4.7, 8.1, tableaux 6, 8, 14)   n = {N_FIX}")
geo_cf = geometric_asian_closed_form(S, K, r, sig, T, N_FIX, "call", q)
geo_mc = mc_asian(S, K, r, sig, T, N_FIX, 200_000,
                  rng=np.random.default_rng(GRAINE_MES), average="geometric")
ari, ci_ari = reference_par_lots(
    lambda M, rng: mc_asian(S, K, r, sig, T, N_FIX, M, rng=rng,
                            control_variate=True).price)

print(f"asiatique geometrique  formule fermee  {geo_cf:10.4f}   <- exacte")
print(f"asiatique geometrique  Monte-Carlo     {geo_mc.price:10.4f}"
      f"   IC95 +/-{geo_mc.ci95:.4f}"
      f"   ecart {geo_mc.price-geo_cf:+.4f}")
print(f"asiatique arithmetique REFERENCE       {ari:10.4f}   IC95 +/-{ci_ari:.4f}")
print(f"decote face au call europeen           {(ari/call_bs-1)*100:9.1f} %")

print(f"\nEffet du nombre d'observations (section 4.7) :")
for n in (12, 52, 250, 252):
    p, c = reference_par_lots(
        lambda M, rng, n=n: mc_asian(S, K, r, sig, T, n, M, rng=rng,
                                     control_variate=True).price,
        n_lots=5, M_lot=50_000)
    print(f"   n = {n:>3} fixings                       {p:10.4f}   IC95 +/-{c:.4f}")

# ---------------------------------------------------------------- barrieres
titre(f"BARRIERES B = 120  (sections 8.2, tableaux 10, 14)   n = {N_FIX}")
B = 120.0
out, ci_out = reference_par_lots(
    lambda M, rng: mc_barrier(S, K, B, r, sig, T, N_FIX, M, rng=rng).price)
inn, ci_inn = reference_par_lots(
    lambda M, rng: mc_barrier(S, K, B, r, sig, T, N_FIX, M, rng=rng,
                              barrier_type="up-and-in",
                              control_variate=True).price)

print(f"up-and-out  REFERENCE                  {out:10.4f}   IC95 +/-{ci_out:.4f}")
print(f"up-and-in   REFERENCE                  {inn:10.4f}   IC95 +/-{ci_inn:.4f}")
print(f"somme in + out                         {out+inn:10.4f}")
print(f"call vanille Black-Scholes             {call_bs:10.4f}")
print(f"ecart de parite                        {out+inn-call_bs:+10.4f}"
      f"   (marge cumulee +/-{ci_out+ci_inn:.4f})")

print("\nNOTE : la reference de la up-and-out est la moins precise du projet.")
print("Aucune variable de controle ne fonctionne sur ce produit (rho = -0,01).")
print("C'est une limite structurelle, a assumer dans le rapport.")

# ----------------------------------------------------------------- digitale
titre("DIGITALE  (section 11.4, niveau 3)")
dig_mc = mc_price(payoff_digital(K, "call"), S, r, sig, T, n_steps=1,
                  n_paths=400_000, q=q, rng=np.random.default_rng(GRAINE_MES))
from scipy.stats import norm
dig_cf = np.exp(-r * T) * norm.cdf(bs.d2)     # formule fermee exacte
print(f"digitale cash-or-nothing  Monte-Carlo  {dig_mc.price:10.4f}"
      f"   IC95 +/-{dig_mc.ci95:.4f}")
print(f"digitale  formule e^(-rT) N(d2)        {dig_cf:10.4f}   <- exacte")
print(f"ecart                                  {dig_mc.price-dig_cf:+10.4f}")

print("\n" + "=" * 74)
print("Copier ces valeurs dans le rapport en citant n = %d." % N_FIX)
print("=" * 74)
