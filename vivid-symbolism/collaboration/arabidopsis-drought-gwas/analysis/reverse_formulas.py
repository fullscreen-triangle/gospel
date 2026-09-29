"""Recover the formula behind each derived trait.

W  = mean WCS417 drought (known: AVG_*_FW, matches raw plants).
Unknown cell means: M_D (mock drought), M_N (mock non-stress), W_N (WCS417 non-stress).
Enumerate candidate definitions, solve for the unknowns, and keep only the
combinations that reproduce every derived column for every accession.
"""
import itertools
import numpy as np
import pandas as pd

d = pd.read_csv("gwas/easyGWAS_input_file.csv")
org = {"S": "SFW", "R": "RFW", "T": "TFW"}
organ_word = {"S": "Shoot", "R": "Root", "T": "Total"}

for o in "SRT":
    W = d[f"AVG_{organ_word[o]}_FW"].to_numpy(float)
    inc = d[f"AVG_percent_increase_in_{org[o]}" + ("_" if o == "S" else "")].to_numpy(float)
    gD = d[f"AVG_Gain_in_{org[o]}_Drought"].to_numpy(float)
    gN = d[f"AVG_Gain_in_{org[o]}_Non_Stress"].to_numpy(float)
    loss = d[f"AVG_percent_Loss_in_{org[o]}_under_drought"].to_numpy(float)
    resc = d[f"AVG_percent_Rescue_{organ_word[o]}"].to_numpy(float)

    print(f"\n==== {organ_word[o]} ====")
    # 1) increase and gain under drought
    for name, MD in {"inc=W/MD-1": W / (1 + inc), "inc=(W-MD)/W": W * (1 - inc)}.items():
        err = np.nanmax(np.abs((W - MD) - gD))
        print(f"  {name:15s} -> gainD = W-MD residual {err:.2e}")
    MD = W - gD

    # 2) candidate definitions for gainNS and loss, each giving an expression
    gN_defs = {
        "W_N-M_N": lambda MN, WN: WN - MN,
        "W-M_N": lambda MN, WN: W - MN,
        "W-W_N": lambda MN, WN: W - WN,
        "M_D-M_N": lambda MN, WN: MD - MN,
    }
    loss_defs = {
        "M_D/M_N": lambda MN, WN: MD / MN,
        "1-M_D/M_N": lambda MN, WN: 1 - MD / MN,
        "W/W_N": lambda MN, WN: W / WN,
        "W_N/M_N-1": lambda MN, WN: WN / MN - 1,
    }
    # solve numerically per accession for (MN, WN) on a grid-free basis:
    # choose defs where one equation involves a single unknown.
    def solve(gdef, ldef):
        # try closed forms
        if gdef == "W-M_N":
            MN = W - gN
            if ldef in ("M_D/M_N", "1-M_D/M_N"):
                return MN, None
            WN = {"W/W_N": W / loss, "W_N/M_N-1": (loss + 1) * MN}[ldef]
            return MN, WN
        if gdef == "W-W_N":
            WN = W - gN
            if ldef == "M_D/M_N":
                return MD / loss, WN
            if ldef == "1-M_D/M_N":
                return MD / (1 - loss), WN
            if ldef == "W_N/M_N-1":
                return WN / (loss + 1), WN
            return None, WN
        if gdef == "M_D-M_N":
            MN = MD - gN
            return MN, None
        if gdef == "W_N-M_N":
            if ldef == "M_D/M_N":
                MN = MD / loss
            elif ldef == "1-M_D/M_N":
                MN = MD / (1 - loss)
            elif ldef == "W_N/M_N-1":
                MN = gN / loss
            else:
                MN = None
            if MN is None:
                return None, None
            return MN, MN + gN
        return None, None

    for g, l in itertools.product(gN_defs, loss_defs):
        MN, WN = solve(g, l)
        if MN is None:
            continue
        # consistency check of the other equation when both unknowns determined
        chk = []
        if WN is not None:
            chk.append(np.nanmax(np.abs(gN_defs[g](MN, WN) - gN)))
            chk.append(np.nanmax(np.abs(loss_defs[l](MN, WN) - loss)))
        else:
            chk.append(np.nanmax(np.abs(loss_defs[l](MN, np.nan) - loss)) if "W" not in l else np.nan)
        pos = np.nanmean(MN > 0) if MN is not None else np.nan
        # rescue candidates
        cands = {
            "(W-M_D)/(M_N-M_D)": (W - MD) / (MN - MD),
            "(W-M_D)/M_N": (W - MD) / MN,
            "(W-M_D)/(W_N-M_D)": None if WN is None else (W - MD) / (WN - MD),
            "(W-M_D)/(W_N-W)": None if WN is None else (W - MD) / (WN - W),
            "(W-M_D)/W_N": None if WN is None else (W - MD) / WN,
            "W/W_N - M_D/M_N": None if WN is None else W / WN - MD / MN,
        }
        best = []
        for k, v in cands.items():
            if v is None:
                continue
            r = np.nanmedian(np.abs(v - resc))
            best.append((r, k))
        best.sort()
        print(f"  gainNS={g:8s} loss={l:10s} check={['%.1e' % c for c in chk]} "
              f"MN>0={pos:.2f}  best rescue: {best[0][1]} (med|err|={best[0][0]:.3g})")
