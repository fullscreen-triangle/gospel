"""Response independence on the WCS417 drought-rescue GWAS data.

The three admissible constructions of 'the response of accession x to WCS417
under drought' all use the same cell means:
  gain     = W_D - M_D
  increase = W_D / M_D - 1
  rescue   = (W_D - M_D) / (M_N - M_D)
Question: does the ordering / top set of accessions depend on the construction?
Then: how much of each ordering survives resampling the plants we have raw data for.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

rng = np.random.default_rng(417)
d = pd.read_csv("gwas/easyGWAS_input_file.csv")
cells = pd.read_csv("gwas/recovered_cell_means.csv")
raw = pd.read_csv("gwas/raw_plants.csv")
acc = pd.read_csv("gwas/Accessions.csv")
res = {}

# ---- 1. agreement between constructions (shoot and root) -----------------
for o in ["Shoot", "Root"]:
    W, MD, MN = cells[f"{o}_W_D"], cells[f"{o}_M_D"], cells[f"{o}_M_N"]
    R = pd.DataFrame({
        "gain": W - MD,
        "increase": W / MD - 1,
        "rescue": (W - MD) / (MN - MD),
        "W_D (size)": W,
        "tolerance loss (mock only)": 1 - MD / MN,
    })
    rho = R.corr(method="spearman")
    k = int(round(0.2 * len(R)))
    tops = {c: set(R[c].nlargest(k).index) for c in ["gain", "increase", "rescue"]}
    jac = {f"{a}~{b}": len(tops[a] & tops[b]) / len(tops[a] | tops[b])
           for a, b in [("gain", "increase"), ("gain", "rescue"), ("increase", "rescue")]}
    in_all = len(tops["gain"] & tops["increase"] & tops["rescue"])
    print(f"\n==== {o}: Spearman between constructions ====")
    print(rho.round(2).to_string())
    print(f"top-20% (k={k}) Jaccard: " + ", ".join(f"{p} {v:.2f}" for p, v in jac.items()))
    print(f"accessions in top-20% under all three: {in_all}/{k}")
    res[o] = {"spearman": rho.round(3).to_dict(), "top20_jaccard": jac, "top20_in_all": [in_all, k]}

# ---- 2. rescuer labels ---------------------------------------------------
for c in ["Shoot_Rescuer", "Root_Rescuer"]:
    print(f"{c}: {d[c].value_counts().to_dict()}")
res["rescuer_counts"] = {c: d[c].value_counts().to_dict() for c in ["Shoot_Rescuer", "Root_Rescuer"]}
# is the label a threshold on some trait?
for c, o in [("Shoot_Rescuer", "Shoot"), ("Root_Rescuer", "Root")]:
    W, MD = cells[f"{o}_W_D"], cells[f"{o}_M_D"]
    g = W - MD
    print(f"  {c}: gain range rescuer {g[d[c]==1].min():.2f}..{g[d[c]==1].max():.2f}; "
          f"non-rescuer {g[d[c]==0].min():.2f}..{g[d[c]==0].max():.2f}; negative gain n={(g<0).sum()}")

# ---- 3. noise: resample only the WCS417-drought plants we have -----------
raw["Shoot FW"] = pd.to_numeric(raw["Shoot FW"], errors="coerce")
by = {i: g["Shoot FW"].dropna().to_numpy() for i, g in raw.groupby("ID")}
cv = np.median([v.std(ddof=1) / v.mean() for v in by.values() if len(v) > 2])
print(f"\nwithin-accession CV of shoot FW (WCS417, drought): median {cv:.2f}")
c = cells.set_index("genome_id")
ids = [i for i in c.index if i in by and len(by[i]) >= 5]
MD, MN = c.loc[ids, "Shoot_M_D"].to_numpy(), c.loc[ids, "Shoot_M_N"].to_numpy()

def constructions(W, MD, MN):
    return {"gain": W - MD, "increase": W / MD - 1, "rescue": (W - MD) / (MN - MD)}

base = constructions(np.array([by[i].mean() for i in ids]), MD, MN)
B = 500
k = int(round(0.2 * len(ids)))
rel = {n: [] for n in base}
stay = {n: np.zeros(len(ids)) for n in base}
top_base = {n: set(np.argsort(-v)[:k]) for n, v in base.items()}
# (a) W-only resampling: lower bound on noise (mock cells treated as exact)
# (b) all-cell resampling: mock cells perturbed with the same CV and n=7
modes = {"W only (lower bound)": False, "all cells, CV-matched": True}
res["noise"] = {"cv_shoot_WD": float(cv), "k": k, "n": len(ids)}
for label, all_cells in modes.items():
    rel = {n: [] for n in base}
    stay = {n: np.zeros(len(ids)) for n in base}
    for _ in range(B):
        W = np.array([rng.choice(by[i], len(by[i]), replace=True).mean() for i in ids])
        if all_cells:
            se = cv / np.sqrt(7)
            md = MD * (1 + rng.normal(0, se, len(ids)))
            mn = MN * (1 + rng.normal(0, se, len(ids)))
        else:
            md, mn = MD, MN
        bs = constructions(W, md, mn)
        for n in base:
            ok = np.isfinite(bs[n]) & np.isfinite(base[n]); rel[n].append(spearmanr(bs[n][ok], base[n][ok])[0])
            top = set(np.argsort(-bs[n])[:k])
            for j in top_base[n]:
                stay[n][j] += j in top
    print(f"\n-- resampling: {label} ({B} reps) --")
    res["noise"][label] = {}
    for n in base:
        frac = np.mean([stay[n][j] / B for j in top_base[n]])
        print(f"  {n:9s} rank reliability rho={np.median(rel[n]):.2f} "
              f"[{np.percentile(rel[n], 5):.2f}, {np.percentile(rel[n], 95):.2f}]; "
              f"top-20% members retained {frac:.0%} of the time")
        res["noise"][label][n] = {"rho_median": float(np.median(rel[n])),
                                  "rho_p5": float(np.percentile(rel[n], 5)),
                                  "top20_retained": float(frac)}

# ---- 4. candidate genes by trait family ----------------------------------
o = pd.read_csv("gwas/easyGWAS_output.csv")
def family(m):
    m = m.lower()
    if "rescue" in m: return "rescue (W vs mock, normalised by drought loss)"
    if "increase" in m: return "relative increase under drought"
    if "gain" in m and "drought" in m: return "absolute gain under drought"
    if "loss" in m: return "drought loss, MOCK ONLY (no bacterium)"
    if "fw" in m: return "WCS417-drought biomass (size)"
    return "other"
o["family"] = o["Metric"].map(family)
fam_per_gene = o.groupby("Gene")["family"].nunique()
print("\ncandidate rows by family:")
print(o.groupby("family")["Gene"].nunique().to_string())
print(f"genes: {o.Gene.nunique()}, in >1 metric: {(o.groupby('Gene')['Metric'].nunique() > 1).sum()}, "
      f"in >1 trait FAMILY: {(fam_per_gene > 1).sum()}")
res["candidates"] = {"genes": int(o.Gene.nunique()),
                     "multi_metric": int((o.groupby('Gene')['Metric'].nunique() > 1).sum()),
                     "multi_family": int((fam_per_gene > 1).sum()),
                     "by_family": o.groupby("family")["Gene"].nunique().to_dict()}

# ---- 5. the 98-plant line ------------------------------------------------
big = raw.groupby(["ID", "Seed Line"]).size().sort_values(ascending=False).head(4)
print("\nlargest lines:", big.to_dict())
json.dump(res, open("gwas/respind_real.json", "w"), indent=2, default=str)
