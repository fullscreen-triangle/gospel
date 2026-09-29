"""Col-0 (6909, H0263) appears 98 times in contiguous 7-plant blocks: use it as a
batch control. Compare between-block spread of Col-0 with between-accession spread."""
import numpy as np, pandas as pd
raw = pd.read_csv("gwas/raw_plants.csv").reset_index().rename(columns={"index": "row"})
raw["Shoot FW"] = pd.to_numeric(raw["Shoot FW"], errors="coerce")
raw["Root FW"] = pd.to_numeric(raw["Root FW"], errors="coerce")
col = raw[raw["ID"] == 6909].copy()
col["block"] = col["row"].diff().fillna(1).ne(1).cumsum()
print("Col-0 blocks:", col.groupby("block").size().to_dict())
print("block row starts:", col.groupby("block")["row"].min().tolist())
for o in ["Shoot FW", "Root FW"]:
    bm = col.groupby("block")[o].mean()
    acc = raw[raw["ID"] != 6909].groupby("ID")[o].mean()
    within = col.groupby("block")[o].std(ddof=1).median()
    print(f"\n{o}: Col-0 block means {bm.round(1).tolist()}")
    print(f"  Col-0 between-block SD {bm.std(ddof=1):.2f} (CV {bm.std(ddof=1)/bm.mean():.2f}); "
          f"within-block SD {within:.2f}")
    print(f"  between-accession SD of means {acc.std(ddof=1):.2f} (CV {acc.std(ddof=1)/acc.mean():.2f})")
    print(f"  ratio batch-SD / accession-SD = {bm.std(ddof=1)/acc.std(ddof=1):.2f}  "
          f"(=> ~{(bm.var(ddof=1)/acc.var(ddof=1)):.0%} of between-accession variance matchable by batch alone)")
    print(f"  Col-0 block range {bm.min():.1f}-{bm.max():.1f} spans accession percentiles "
          f"{(acc < bm.min()).mean():.0%}-{(acc < bm.max()).mean():.0%}")
