# Is there a genotype that breaks the rescue ceiling, and why?

Answer to Abdul's voice note of 2026-10-05. The analysis covers:

- the three-run workbook (NS+DS.xlsx), 247 accessions;
- the 1001 Genomes v3.1 imputed SNP matrix (10.7 M SNPs, 1,135 accessions) and IBS kinship;
- the Cod-0 pseudogenome;
- the Araport11 annotation.

Scripts are in `validation/`: `superrescuers.py` (S1), `cod0_genetics.py` (S2), `cod0_effects.py` (S3), `gwas_lmm.py` (S4) and `blind_cod0.py` (S5). Results are in `results/S1–S5`, and the summary figure is `figures/panel_8_superrescuer.png`.

**Short answer.**

- **Yes: Cod-0** (Spain, 41.25 N, −1.32 E). It was in the rescreen and exceeds 40% shoot rescue in all three independent runs. It beats Col-0 in every run.
- **The "why" is not in common variation.** It is not a stack of common alleles, and its close relatives don't share the trait. Rare, Cod-0-specific variation is the best explanation.
- **A GWAS cannot fine-map that.** It needs a cross. The genome does give a short list of Cod-0-specific candidates that can be tested with existing Col-0 mutants now.

---

## 1. Cod-0 breaks the ceiling, reproducibly

Selection was made on run 1 (top 25%). Runs 2 and 3 were then checked independently.

| Shoot | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| Rescue (W−M_D)/(M_N−M_D) | 0.40 | 1.12 | 0.60 |
| Cod-0 − Col-0 rescue (95% CI) | 0.26 [0.09, 0.45] | 1.08 [0.60, 2.05] | 0.44 [0.23, 0.70] |
| WCS417 / mock under drought | 1.61× | 1.77× | 1.83× |
| Col-0, WCS417 / mock | 1.39× | 1.06× | 1.30× |
| Absolute gain W − M_D | 3.0 mg | 5.7 mg | 4.3 mg |

- Cod-0 is in the upper tail of rescue in both independent runs: p = 0.0004 (Bonferroni over the discovered set: 0.005; permutation: 0.005).
- Root rescue gives the same picture: 0.30 / 0.74 / 0.60 (Bonferroni 0.015).
- No other accession replicates this cleanly.

**Three things to say plainly:**

1. **Cod-0 is the steadiest responder, not the biggest.** By WCS417/mock fold change it ranks 13th–17th of 49 rescreened accessions in each run. By its worst run it ranks 8th. BI-4 has bigger folds (1.9× and 2.5×) but collapses to 1.2× in run 2.
2. **Part of its high rescue is that drought costs it less.** Mock drought loss is 0.41–0.60 against the panel mean of 0.78. That shrinks the rescue denominator. Its absolute gain is modest because it is tiny: the smallest plant in the panel by M_N.
3. **Without drought, WCS417 promotes Cod-0.** In run 1, the only run with a non-stress control, WCS417 almost doubles Cod-0's shoot (log ratio +0.64, 99.6th percentile). The panel median is inhibited (−0.12). If the 30–40% ceiling reflects a cost the plant pays for hosting WCS417, **Cod-0 looks like an accession that doesn't pay it.** This is one run, 7 plants per cell, and needs confirming.

## 2. Why Cod-0? Not common alleles

- **Heritability.** Kinship-based REML h² is 0.18–0.20 for shoot rescue (p = 0.012) and 0.31 for root rescue (p = 0.001).
- **Relatives.** Cod-0's ten closest relatives in the panel average 0.26 shoot rescue, the same as the panel. The trait does not travel with its lineage.
- **Blind allele test.** The GWAS was re-run with Cod-0 held out. At the top 10 shoot-rescue peaks, Cod-0 carries 7 rescue-raising alleles against 4.7 expected (p = 0.08). At 25 or more peaks it is average.
- **The 10/10 was circular.** With Cod-0 inside the GWAS it scored 10/10, but only because its own phenotype helped choose those alleles.
- **Conclusion:** Cod-0's behaviour is most likely carried by **rare or private variants**, which a 247-accession GWAS cannot see.

## 3. The GWAS done properly (S4)

The scan:

- 237 accessions with complete phenotypes;
- 1.84 M SNPs with MAF ≥ 5%, from the full 1001 Genomes set;
- an EMMAX mixed model with IBS kinship;
- rank-normal phenotypes.

λ_GC is 0.98–1.02 for every trait, so the model is well calibrated.

- **No SNP reaches genome-wide significance** for any construction (threshold −log10 p = 7.57). The best is 6.8.
- **Of the 25 easyGWAS candidate genes, two recur** at p < 1e-4 within 1 kb:
  - MORN4 / AT1G77660: drought loss, −log10 p 6.6;
  - AT1G58007: promotion and interaction, −log10 p 5.6.

  Neither is a rescue gene. None of the 7 root-rescue genes and neither Total-Rescue gene reappears. Two caveats: his phenotypes were different constructions, and easyGWAS ran on a different SNP set without our kinship.
- **Shoot rescue has no transferable signal.** Holding out all 48 rescreened accessions, the run-1 peaks predict nothing about their rescue (sign agreement 24/47).
- **Root rescue has a real, diffuse signal.** Of the top 50 root-rescue peaks, 36 have the same direction in the held-out rescreen (p = 0.001). A score built from them predicts held-out root rescue (ρ ≈ 0.3, p = 0.02–0.06). The architecture is polygenic: no single locus is strong enough to justify a mutant on its own.

## 4. What is unique about Cod-0's genome (S2, S3)

- Cod-0 carries 19,726 alleles found in at most 1% of the 1,135 accessions; 4,549 are private.
- Its alleles were read from its own pseudogenome, located by flanking sequence because the pseudogenome contains indels.
- Among the 3,026 coding and splice sites that could be resolved:

| Effect | Count |
|---|---|
| Missense | 1,787 (1,073 radical) |
| Synonymous | 1,179 |
| Stop-gained | 40 |
| Splice-site | 10 |
| Start-lost | 5 |
| Stop-lost | 5 |

- A curated set of 53 WCS417/ISR, iron, immunity, ABA and auxin genes is **not enriched** (2 hit vs 2.96 expected, p = 0.80). There is no pathway-level smoking gun.

**Correction to an earlier draft.** FRO2, BTS and PYL2 were listed as missense changes. They are **synonymous** in Cod-0. The earlier calls were an artefact of reading the pseudogenome at reference coordinates.

### Candidate genes, ranked for testing with existing Col-0 mutants

| Gene | Cod-0 change | Rarity | Why it could matter |
|---|---|---|---|
| **PAD4** (AT3G52430) | F73S, radical | private (1/1,135) | EDS1–PAD4 immune hub. Lower immune cost of colonisation would fit "no non-stress penalty" and a higher ceiling. Test *pad4-1*. |
| **FKBP15-2** (AT5G48580) | splice-site + I124M | private | Sits about 25 kb from the Total-Rescue easyGWAS hits AT5G48640/48650, inside a dense rare haplotype: 24 rare alleles per 70 kb, against a median of 5. Add it next to the mutants already growing there. |
| **CYP707A2** (AT2G29090) | L442F, S280I, K253N, all radical | ~10 accessions | ABA 8′-hydroxylase. Would explain Cod-0's drought tolerance, and tests whether ABA sets the ceiling. |
| **ARF7** (AT5G20730) | Q577K, radical | 5 accessions | Auxin-dependent lateral-root response to WCS417. |
| RLP9, TIR-NBS-LRR AT5G41550 | stop-gained | 1–2 accessions | Immune receptors lost in Cod-0. Speculative. |

These are hypotheses ranked by biology and rarity. None is statistically implicated: with one accession, every private variant is equally associated with the trait.

## 5. How to get to "we found the gene"

1. **Run Cod-0 through the existing setups**, alongside Col-0 and BI-4, the biggest absolute responder:
   - NS and DS, mock and WCS417;
   - the SynCom-germination + WCS417-invasion design;
   - the rescuer-only and non-rescuer SynComs.

   First question: does Cod-0 also exceed 40% when WCS417 invades a community? Add WCS417 CFU in rhizosphere and shoot. Is Cod-0 hosting more bacteria, or responding differently to the same load?
2. **Test the candidate mutants** in Col-0: *pad4*, *cyp707a2*, *arf7*, *fkbp15-2*. Use the same four-cell design, and report rescue together with the non-stress effect.
3. **Map it with a Cod-0 × Col-0 cross.**
   - Rescue cannot be scored on a single F2 plant, because it needs mock and WCS417 for the same genotype.
   - So score **F2:3 families**: about 150 families, each split mock/WCS417 under drought.
   - Then use bulk-segregant sequencing (QTL-seq) of the extreme families.
   - This is the route that ends in a gene. The GWAS cannot do it, because the effect lives in variation Cod-0 nearly alone carries.
