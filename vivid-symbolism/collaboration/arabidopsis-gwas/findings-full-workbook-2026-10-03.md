# The full per-plant workbook (NS+DS.xlsx): what it changes

Data: `vivid-symbolism/public/gwas/NS+DS.xlsx`. Analysis: `validation/experiments_full.py`
(F1–F7, seed 417, ~2 min). Results: `results/F*.json`. Figure: `figures/panel_7_full_workbook.png`.
Per-accession phenotypes for any new association scan: `results/phenotypes_full.csv`.

## What the workbook contains

8,820 plants: 247 accessions × 4 cells (mock/WCS417 × drought/non-stress), shoot and
root fresh weight.

- **Drought:** replicate 1 (BR1) covers all 247 accessions, 7 plants per cell.
- **Replicates 2 and 3 (BR2, BR3):** 50 accessions, of which 49 are in both. Each was a
  fresh, independent run with 7 plants per cell.
- **Non-stress:** grown once.
- **Col-0 control:** grown in 7-plant blocks in every cell. Drought has 14 blocks in BR1,
  3 in BR2 and 4 in BR3; non-stress has 14.
- **Missing weights:** 324 shoot weights are missing, and 16 accession-cells are missing
  entirely.

## F1 — Consistency with the first workbook

- **The earlier GWAS used run 1 only.** The inoculated-drought means equal the first
  workbook's raw sheet exactly (244 accessions, max difference 0).
- **The earlier recovery of the mock means is confirmed by measurement.**
  - M_D matches for 241 of 243 accessions.
  - M_N matches for 235 of 239.
  - Every mismatch is one of the four lines grown in more than one block.
- **"Gain under non-stress" is W_N − M_N** (240 of 241 accessions match). This was the
  one definition the first workbook could not resolve.
- **The 50 replicated accessions were not random.**
  - 43 of them carry the "non-rescuer" label.
  - Their median run-1 rescue sits at the 26th percentile.
  - The lab regrew the apparent weak rescuers to confirm them.

## F2 — Measured noise (replaces the earlier modelling assumption)

| Shoot, median plant CV | mock | WCS417 |
|---|---|---|
| drought | **0.18** | 0.12 |
| non-stress | 0.15 | 0.18 |

Root follows the same pattern: 0.26 for mock-drought against 0.16 for inoculated-drought.

- **The mock-drought cell is the noisiest.** The earlier assumption of equal relative
  spread in every cell understated it.
- **Batch within a run, from the Col-0 blocks,** is also largest in mock-drought: CV
  0.177 against 0.081 for inoculated-drought in run 1.
- **Batch is shared under non-stress but not under drought.** Under non-stress, the Col-0
  mock and inoculated block means move together (r = 0.66 shoot, 0.88 root), so that
  batch cancels in the WCS417 effect. Under drought in run 1 they are uncorrelated
  (r = 0.01). The drought gain therefore carries two independent batch offsets.

## F3 — Does weak rescue replicate? No.

- **Col-0, the reference line, is not stable** (shoot gain, mg, 95% CI):
  - run 1: 6.2 [5.1, 7.3]
  - run 2: **1.4 [−1.2, 4.3]**
  - run 3: 6.0 [3.8, 8.1]

  In run 2 the reference genotype itself would have been called a non-rescuer. Root
  behaves the same way: 3.0, 0.3, 3.1.
- **No accession stays a non-rescuer.**
  - Five accessions had a gain compatible with zero in run 1 (Sei-0, Per-1, Böt, IP-Ala-0,
    Cas-0; Choto-1 is borderline).
  - Four different ones were compatible with zero when runs 2–3 are pooled (Yo-0,
    IP-Gua-1, IP-Orb-10, Goced-1; Wl-0 and Ini-0 are borderline).
  - None is in both lists, and no retest estimate is negative.
  - Per-1, the previous best loss-of-rescue candidate, was clearly rescued in both
    retests: 5.3 and 3.2 mg, both intervals above zero.
- **Regression to the mean.** The retested accessions moved 26% (shoot) and 32% (root)
  of the way back towards the all-accession mean (paired Wilcoxon p = 0.05 shoot, 0.002
  root). That is what selecting on a noisy single run predicts.

**Conclusion:** the screen contains no confirmed loss-of-rescue accession. Low rescue in
one run is mostly that run.

## F4 — Replicate reproducibility (no modelling: run 2 vs run 3, 48 accessions)

| Quantity | Shoot ρ [95% CI] | Root ρ [95% CI] |
|---|---|---|
| mock drought loss (no bacterium) | **0.73** [0.56, 0.84] | 0.68 [0.48, 0.81] |
| log M_D | 0.72 [0.55, 0.84] | 0.71 [0.53, 0.83] |
| log W | 0.52 [0.27, 0.71] | 0.58 [0.34, 0.74] |
| % increase | 0.45 [0.18, 0.65] | 0.38 [0.10, 0.61] |
| gain | 0.19 [−0.11, 0.46] | 0.11 [−0.19, 0.39] |
| rescue | 0.11 [−0.19, 0.39] | 0.14 [−0.16, 0.42] |
| canonical (λ̂) | 0.08 [−0.22, 0.36] | 0.15 [−0.15, 0.42] |

This table is the central result.

- **What replicates is drought sensitivity, not bacterial rescue.** An accession's
  drought sensitivity is a stable trait. Its benefit from WCS417, beyond that, is not
  distinguishable from zero reproducibility between independent runs.
- **% increase replicates better only because it carries drought sensitivity.** Its
  accession ranking correlates 0.80 with mock drought loss (E3). So its better
  reproducibility is evidence that it measures drought sensitivity, not rescue.
- **Run-to-run noise is larger than modelled.** It is about CV 0.18 for log W and 0.22
  for log M_D, beyond plant noise: more than twice the within-run batch CV of 0.081 the
  earlier analysis used. Part of each run's offset is shared by W and M_D of the same
  accession (r = 0.40), so part of it cancels in the gain.
- **Caveat:** these 48 accessions were selected as weak rescuers, though the spread of
  their run-1 gains is 91% of the full panel's. Two runs on 48 accessions give wide
  intervals; the intervals above are the honest statement.

## F5 — λ re-estimated with all four cells measured

- **Shoot:** λ̂ ≈ −0.15 [−0.56, −0.01]. **Root:** λ̂ ≈ 0.05 [−0.28, 0.14]. The values
  shift by about 0.01 between Monte Carlo reruns.
- The earlier estimates were −0.26 and −0.04.
- % increase (λ = 1) is still excluded in both organs. Plain loss-normalised rescue
  (λ = 0) lies inside the root interval and at the upper edge of the shoot interval, so
  for practical purposes the canonical phenotype is rescue.
- The bias correction is larger than before, because the measured mock-drought noise is
  larger than assumed.

## F6 — The benefit is drought-specific

- **Without drought, WCS417 mildly inhibits shoot growth.** The mean log(W_N/M_N) is
  −0.13, about −12%. 30% of accessions are significantly inhibited and only 1%
  significantly promoted. Root is unaffected on average.
- **Under drought, inoculation increases biomass in 95% of accessions (shoot) and 99%
  (root).**
- **The two responses are unrelated across accessions** (ρ = −0.08 shoot, −0.01 root).
  Drought rescue is not general growth promotion carried over into drought. This
  differs from the plate assays of Wintermans et al. (2016), where nearly all accessions
  were promoted without stress.
- **The inoculation × water interaction is not a clean phenotype either.**
  log(W/M_D) − log(W_N/M_N) is positive in 92–94% of accessions. But it correlates
  0.79–0.82 with drought loss, because it sits at the λ = 1 end of the family.

## F7 — Certification under measured noise

| Shoot, canonical rescue | median certifiable margin | tiers |
|---|---|---|
| measured plant noise only | 2.0 SD | 8 |
| + measured within-run batch | 3.3 SD | 3 |
| + measured run-to-run noise | **4.6 SD** | **2** |

Root: 1.9, 2.9 and 3.7 SD, with 9, 4 and 3 tiers. The earlier modelled bracket was
1.7–2.7 SD with 4–8 tiers. The real noise is worse than the pessimistic model.

## What this means

1. **In this screen, the reproducible accession signal is drought sensitivity.** The
   rescue signal from a single run is mostly that run's conditions. The 28 candidate
   genes are hits on single-run phenotypes, and should be expected to replicate poorly
   unless they are drought-sensitivity loci.
2. **A rescue GWAS needs a phenotype averaged over independent runs**, with mock and
   inoculated plants sharing trays as they did under non-stress. Run 2 vs run 3 suggests
   a single-run reliability for rescue of about 0.1–0.2. By the Spearman–Brown formula,
   reaching a reliability of 0.6 would need 6 runs at 0.2, and 13–14 at 0.1. Controlling
   drought severity, and co-locating mock with inoculated plants, are the levers that
   would cut this.
3. **The drought-sensitivity GWAS is the solid result in this dataset.** That is a
   legitimate paper in itself, and `phenotypes_full.csv` includes the mock drought loss
   for it.
4. **The biological headline survives and sharpens.** WCS417 helps under drought and not
   otherwise, and the help is reliably positive: zero confirmed non-rescuers, consistent
   with the redundancy hypothesis. What does not survive is the idea that accessions
   differ reproducibly in how much they are helped.
