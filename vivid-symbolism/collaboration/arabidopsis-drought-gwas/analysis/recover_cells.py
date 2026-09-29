"""Recover the four cell means per accession and organ, and audit the sheet.

Verified identities (all 247 accessions, residual ~1e-10):
  gainD  = W - M_D
  loss   = 1 - M_D / M_N          ("AVG percent Loss ... under drought")
  rescue = (W - M_D) / (M_N - M_D)
gainNS is consistent with either W_N - M_N or W - W_N; both are reported.
"""
import numpy as np
import pandas as pd

d = pd.read_csv("gwas/easyGWAS_input_file.csv")
t1 = pd.read_csv("gwas/Table_1.csv")
t2 = pd.read_csv("gwas/Table_2.csv")
raw = pd.read_csv("gwas/raw_plants.csv")
raw["Total FW"] = pd.to_numeric(raw["Total FW"], errors="coerce")

org = {"Shoot": "SFW", "Root": "RFW", "Total": "TFW"}
out = pd.DataFrame({"genome_id": d["IID"]})
for o, a in org.items():
    W = d[f"AVG_{o}_FW"]
    MD = W - d[f"AVG_Gain_in_{a}_Drought"]
    MN = MD / (1 - d[f"AVG_percent_Loss_in_{a}_under_drought"])
    inc = d[f"AVG_percent_increase_in_{a}" + ("_" if o == "Shoot" else "")]
    out[f"{o}_W_D"], out[f"{o}_M_D"], out[f"{o}_M_N"] = W, MD, MN
    out[f"{o}_W_N_if_WN-MN"] = MN + d[f"AVG_Gain_in_{a}_Non_Stress"]
    out[f"{o}_W_N_if_W-WN"] = W - d[f"AVG_Gain_in_{a}_Non_Stress"]
    resid = (W / MD - 1) - inc
    bad = resid.abs() > 1e-6
    print(f"{o}: increase = W/M_D - 1 holds for {(~bad).sum()}/{len(d)}; "
          f"mismatches at IDs {d.loc[bad, 'IID'].tolist()[:12]}")
    print(f"   M_D median {MD.median():.2f}, M_N median {MN.median():.2f}, "
          f"median fractional loss {(1 - MD / MN).median():.2f}")
    print(f"   M_N <= M_D (loss<=0) in {(MN <= MD).sum()} accessions; "
          f"M_N<=0 in {(MN <= 0).sum()}")
    wn1, wn2 = out[f"{o}_W_N_if_WN-MN"], out[f"{o}_W_N_if_W-WN"]
    print(f"   W_N < M_N: {(wn1 < MN).mean():.0%} (if W_N-M_N) / {(wn2 < MN).mean():.0%} (if W-W_N)")

# raw per-plant WCS417-drought means vs AVG_Shoot_FW
g = raw.groupby("ID")[["Shoot FW", "Root FW", "Total FW"]].mean()
m = d.set_index("IID")[["AVG_Shoot_FW", "AVG_Root_FW", "AVG_Total_FW"]].join(g, how="inner")
print("\nraw-mean vs AVG_Shoot_FW max |diff|:", (m["Shoot FW"] - m["AVG_Shoot_FW"]).abs().max().round(4),
      " n =", len(m))
print("raw plants per accession:", raw.groupby("ID").size().value_counts().to_dict())

# which condition is Table 1?
j = t1.merge(out, left_on="Genome ID", right_on="genome_id", how="inner")
for col, lab in [("Average Mock SFW (mg)", "Mock"), ("Average WCS417 SFW (mg)", "WCS417")]:
    for c in ["Shoot_M_D", "Shoot_M_N", "Shoot_W_D", "Shoot_W_N_if_WN-MN", "Shoot_W_N_if_W-WN"]:
        r = np.corrcoef(j[col], j[c])[0, 1]
        print(f"Table1 {lab:6s} vs {c:20s} r={r:+.2f}  median ratio {np.median(j[col] / j[c]):.2f}")

# Table 2 'tolerance' vs recovered loss
j2 = t2.merge(d, left_on="Genome ID", right_on="IID")
print("\nTable 2 shoot 'drought tolerance' == 'percent loss' column:",
      np.allclose(j2[" Average Shoot Drought Tolerance"], j2["AVG_percent_Loss_in_SFW_under_drought"]))
out.to_csv("gwas/recovered_cell_means.csv", index=False)
