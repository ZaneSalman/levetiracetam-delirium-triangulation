# Stage 3 — Causal machine learning: heterogeneity of the LEV -> delirium effect (precision-medicine layer)
# Runs in the same Colab runtime after stage 2 (needs df, X, T). AGGREGATE OUTPUT ONLY.
# Method: cross-fitted doubly robust (DR) learner (Kennedy 2023) with gradient-boosted nuisance models,
# GATES calibration by CATE quintile-group (Chernozhukov et al.), permutation importance of effect modifiers.

import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
warnings.filterwarnings("ignore")
rng = np.random.RandomState(2026)


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def dr_learner(Xm, Tm, Ym, n_splits=5, seed=2026):
    Xa = Xm.values.astype(float)
    n = len(Ym)
    mu0, mu1, e = np.zeros(n), np.zeros(n), np.zeros(n)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    strat = Tm * 2 + Ym
    for tr, te in skf.split(Xa, strat):
        ps = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                            l2_regularization=1.0, random_state=seed)
        ps.fit(Xa[tr], Tm[tr]); e[te] = ps.predict_proba(Xa[te])[:, 1]
        for arm, store in [(0, mu0), (1, mu1)]:
            idx = tr[Tm[tr] == arm]
            m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                               l2_regularization=1.0, random_state=seed)
            m.fit(Xa[idx], Ym[idx]); store[te] = m.predict_proba(Xa[te])[:, 1]
    e = np.clip(e, 0.05, 0.95)
    phi = mu1 - mu0 + Tm * (Ym - mu1) / e - (1 - Tm) * (Ym - mu0) / (1 - e)
    # second stage: cross-fitted CATE regression on the pseudo-outcome
    cate = np.zeros(n)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed + 1)
    for tr, te in kf.split(Xa):
        r = HistGradientBoostingRegressor(max_depth=2, learning_rate=0.03, max_iter=200,
                                          min_samples_leaf=40, l2_regularization=2.0, random_state=seed)
        r.fit(Xa[tr], phi[tr]); cate[te] = r.predict(Xa[te])
    final = HistGradientBoostingRegressor(max_depth=2, learning_rate=0.03, max_iter=200,
                                          min_samples_leaf=40, l2_regularization=2.0, random_state=seed)
    final.fit(Xa, phi)
    return phi, cate, e, final


def report(label, mask_):
    Xm = X[mask_].reset_index(drop=True)
    Tm = T[mask_].astype(int)
    Ym = df.delirium.values[mask_].astype(int)
    section(f"{label}  (n={len(Ym)}, LEV={int(Tm.sum())}, events={int(Ym.sum())})")
    phi, cate, e, final = dr_learner(Xm, Tm, Ym)

    ate, se = phi.mean(), phi.std(ddof=1) / np.sqrt(len(phi))
    print(f"DR-learner ATE (risk difference): {100*ate:+.1f} pp  (95% CI {100*(ate-1.96*se):+.1f} to {100*(ate+1.96*se):+.1f})")
    print(f"Overlap: propensity 5th-95th pct {np.percentile(e,5):.2f}-{np.percentile(e,95):.2f}; "
          f"share at clip bounds {(np.isclose(e,0.05)|np.isclose(e,0.95)).mean()*100:.1f}%")
    q = np.percentile(cate, [5, 25, 50, 75, 95]) * 100
    print("Predicted CATE distribution (pp): 5%={:+.1f} 25%={:+.1f} 50%={:+.1f} 75%={:+.1f} 95%={:+.1f}".format(*q))
    print(f"Share with predicted harm > 0: {(cate > 0).mean()*100:.1f}%")

    # GATES: average DR pseudo-outcome within CATE quintile groups (honest: CATE was cross-fitted)
    grp = pd.qcut(cate, 5, labels=["G1 (lowest)", "G2", "G3", "G4", "G5 (highest)"], duplicates="drop")
    g = pd.DataFrame({"grp": grp, "phi": phi}).groupby("grp", observed=True)["phi"].agg(["count", "mean", "std"])
    g["se"] = g["std"] / np.sqrt(g["count"])
    g["GATE pp (95% CI)"] = [f"{100*m:+.1f} ({100*(m-1.96*s):+.1f} to {100*(m+1.96*s):+.1f})" for m, s in zip(g["mean"], g["se"])]
    print("\nGATES (sorted by predicted CATE):")
    print(g[["count", "GATE pp (95% CI)"]].to_string())
    top, bot = phi[grp == g.index[-1]], phi[grp == g.index[0]]
    diff = top.mean() - bot.mean(); sed = np.sqrt(top.var(ddof=1)/len(top) + bot.var(ddof=1)/len(bot))
    print(f"Top minus bottom group: {100*diff:+.1f} pp (95% CI {100*(diff-1.96*sed):+.1f} to {100*(diff+1.96*sed):+.1f})")

    # Effect modifiers: permutation importance of the CATE model
    pi = permutation_importance(final, Xm.values.astype(float), final.predict(Xm.values.astype(float)),
                                n_repeats=10, random_state=2026, scoring="r2")
    imp = pd.Series(pi.importances_mean, index=Xm.columns).sort_values(ascending=False).head(8)
    print("\nTop effect modifiers (permutation importance of CATE model):")
    print(imp.round(3).to_string())

    # Pre-specified subgroup GATEs (DR pseudo-outcome means)
    print("\nPre-specified subgroups (DR risk difference, pp):")
    subs = {
        "Age >= 65": Xm["age"].values >= 65, "Age < 65": Xm["age"].values < 65,
        "Charlson >= 5": Xm["charlson"].values >= 5, "Charlson < 5": Xm["charlson"].values < 5,
        "Ventilated <=48h": Xm["vent_48h"].values == 1, "Not ventilated": Xm["vent_48h"].values == 0,
        "Admission GCS <= 12": Xm["gcs_min"].values <= 12, "Admission GCS > 12": Xm["gcs_min"].values > 12,
    }
    for k in ["injury_TBI", "injury_SAH"]:
        if k in Xm:
            subs[k.replace("injury_", "") + " only"] = Xm[k].values == 1
    rows = []
    for k, mk in subs.items():
        if mk.sum() < 50:
            continue
        p = phi[mk]; m_, s_ = p.mean(), p.std(ddof=1) / np.sqrt(len(p))
        rows.append([k, int(mk.sum()), f"{100*m_:+.1f} ({100*(m_-1.96*s_):+.1f} to {100*(m_+1.96*s_):+.1f})"])
    print(pd.DataFrame(rows, columns=["subgroup", "n", "RD pp (95% CI)"]).to_string(index=False))


assessed = df.assessed.values == 1
report("A. Delirium among patients assessed (hours 48-168)", assessed)
report("B. INCIDENT delirium (assessed, no delirium in first 48 h)", assessed & (df.delirium_0_48.values == 0))
print("\nDone. All output above is aggregate.")
