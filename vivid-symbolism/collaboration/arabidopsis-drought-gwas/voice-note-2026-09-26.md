# Voice note — Arabidopsis × Pseudomonas drought-rescue GWAS

Source: `vivid-symbolism/public/audio/WhatsApp Ptt 2026-09-26 at 01.21.51.ogg`
(7 min 24 s). Transcribed locally with faster-whisper `medium`, English, then
lightly corrected (e.g. "snips" → SNPs, "easy GWAS" → easyGWAS). Filler and
greetings trimmed; timestamps kept so any line can be checked against the audio.

## Transcript

**[00:26]** I thought I'd give you background so that when you get the file you
already know what's in it.

**[00:39]** Similar to my previous work, I'm working with a *Pseudomonas* strain —
a model beneficial bacterium, published a lot, many people work on it. I'm trying
to understand how this strain can rescue plants from drought stress, i.e. make
them grow better under drought.

**[01:07]** I'm doing a time-resolved RNA-seq. That data is pretty much ready for
making a figure. Let's see how the GWAS goes; maybe then we go into the RNA-seq.

**[01:29]** The GWAS is not on the strain, not on the bacterium, it's on the
plant. There are natural populations of *Arabidopsis*, each with a different
genomic composition. You screen these plants for what you're looking for — in
our case drought rescue by this *Pseudomonas* strain — and tie the phenotypes to
the genomic sequences to identify SNPs that correlate. For example, some
populations fail to rescue or show increased rescue, and through pipelines you
see which SNPs are associated.

**[02:42]** There are ~1,000 *Arabidopsis* accessions, but we screened ~200 or
250 (my students did it, so I don't remember the exact number).

**[03:05]** The problem: we did not find any accessions that **lose** rescue.
That's unfortunate, because that's what you want — loss of rescue — so you can
tie that trait to the genome.

**[03:22]** What the data does have: several traits, all based on **biomass** —
shoot or root biomass, **with or without the bacterium**, grown under
**non-stress or drought**. From this, using formulas, you get percentage rescue,
percentage increase in biomass, absolute gain in biomass.

**[03:53]** You can also look at drought tolerance: from the mock plants (no
bacteria), non-stress vs drought gives a **drought tolerance index**, also in the
sheet.

**[04:13]** We did get some SNPs associated with the *degree* of rescue — some
rescue a bit less, some a bit more.

**[04:26]** Caveats: we only ran this **once**, so statistically it's a bit
faulty. Only 200–250 accessions, and they were **selected based on a paper on
annual precipitation**, so there's some fault in the design from the get-go.

**[04:47]** Nevertheless, we have SNPs, and I ordered **mutants for the genes
upstream and downstream of those SNPs**. We're growing them now and will screen
them once we have seeds. If that works out, it goes in the paper.

**[05:15]** We used the **easyGWAS** pipeline to find SNPs, and also did it
manually.

**[05:27]** There may also be a **primary root length** trait — not sure if it
was measured.

**[05:46]** Sheet layout: one column with the **accession ID**, then trait
columns — biomass, increase in biomass, loss in biomass, rescue capacity.
Rescue = how much of what is lost to drought is recovered by the bacterium (as a
percentage). Plus the drought index. The sheet should show the formulas.

**[06:30]** I can also send the **table of SNPs** we found, so you can check
against it if you run anything.

**[06:48]** Send voice notes back and forth — I'm pushing out data before I land
the job in ~3 months, so I'm busy. Hopefully I'll send the data this weekend.

## Follow-up notes (20:13, 20:14, 20:18) — answers to our questions

### 20:13 (4 min)

**[00:00]** You're right that the formulas are different expressions of the same
thing — the biomass. In the candidate-gene table you'll see the category each
gene came from: did it show up using rescue, or absolute gain? **Different
formulas give different results, but some genes agree** — you find them in
multiple traits. (I'll use "trait" to mean one formula: rescue is one trait, gain
in biomass another.)

**[00:52]** Combinations: with/without bacteria × drought/non-stress × shoot/root
= **8**. Replicates: **5–7 plants** per accession, per treatment, per condition —
that's the raw data.

**[01:32]** Candidate table: gene name, AGI (AT…G…) ID, and the metric (trait)
the SNP was found in. We'll also send **Manhattan plots**.

**[01:56]** Tweaks: I think he restricted to our ~250 accessions. We looked at the
p-value several ways — **lowered the threshold a bit** below the default
(~−log10 p = 7), because some SNPs were borderline, to see which genes those
were. That's how those genes showed up. Then SNP position: upstream,
downstream, promoter, CDS — all in the table.

**[02:58]** Accession table: ID, genotype name, country of origin; not sure about
latitude/longitude. easyGWAS settings: probably defaults.

### 20:14 (30 s)

Extra layer: the mock plants give a drought tolerance index, so we can ask
whether accessions from **dry regions are more drought-tolerant**.

### 20:18 (3 min)

**[00:00]** I'll send a paper: a GWAS on **this exact strain** in *Arabidopsis*,
but not under drought, looking at the strain's **volatiles**. Ours is on the
roots, so there's a **contact effect we can decouple from non-contact**; they did
only non-stress. Media and plant age differ, but there should be **shared
accessions under non-stress**, so we might separate volatile from non-volatile
effects. Another group did the strain under drought, also volatiles, with
RNA-seq at **two early time points**; mine has **four**.

**[01:40]** Also worth finding: GWAS of *Arabidopsis* under drought without
bacteria.

**[01:56]** Biggest bias a reviewer will raise: **how the accession subset was
selected** — maybe that's why no loss of rescue. My other hypothesis: **the
strain rescues by multiple routes**. From the RNA-seq I have **~50 candidate
genes**; I've screened many and **nothing loses rescue** — knock one out and
another takes over. **Redundancy** on both the bacterial and plant side.

## Notes of 2026-09-27

### 17:22 (4 min) — the RNA-seq

**[00:00]** Two approaches. **First:** at the early time point, take genes
up-regulated by drought in mock plants (the drought response) and ask which are
**reversed when the bacterium is present** — up under drought but down with
WCS417, and vice versa. GO enrichment of those genes → **iron-related
responses**.

**[01:09]** **Second:** genes consistently regulated across **time points 2, 6,
9, 12** — up in mock across time, down in mock across time, up with bacteria
across time, down with bacteria across time. Intersect these, to get genes
consistently regulated by drought across time and regulated in the **opposite
direction by the bacterium** across time. Result: a heat map of **~50 genes,
mostly root**; only **two in the shoot**, because far more genes are regulated
in the root.

**[02:38]** Also a **STRING network** of the iron-related genes. Working
hypothesis: **the bacterium provides iron to the plant**. **Iron-uptake mutants
show even higher rescue**, so the rescue is independent of the plant's normal
iron-uptake machinery. **Iron content is higher with the bacterium.** Mechanism
unknown; perhaps acidification or bacterial iron chelators.

### 17:25 (1 min) — accession selection

**[00:00]** Likes the idea (the climate layer). Will send the paper that led to
those accessions. **He didn't select them himself: they're the genotypes already
in the lab**, bulked by someone else, so they could be screened immediately
instead of ordering and bulking the full 1,000. Precipitation levels could come
from a database via the collection sites, or from that paper.

## Dataset as described

| Item | Detail |
|---|---|
| Organism screened | *Arabidopsis thaliana*, ~200–250 natural accessions (of ~1,000) |
| Accession selection | Chosen from a study of annual precipitation — not random |
| Treatment | *Pseudomonas* strain (beneficial model strain) vs mock |
| Condition | Non-stress vs drought |
| Design | 2 treatments × 2 conditions × 2 organs = 8 cells per accession; **one experimental run** |
| Replicates | 5–7 plants per accession × treatment × condition (raw per-plant data exists) |
| Raw traits | Shoot biomass, root biomass; possibly primary root length |
| Derived traits | % rescue, % biomass increase, absolute biomass gain, drought tolerance index (mock only) |
| Key negative | No accession loses rescue; variation is only in degree |
| GWAS | easyGWAS, likely default settings, threshold lowered below −log10 p ≈ 7 |
| Candidate table | Gene name, AGI ID, source trait, SNP position class; candidates differ by trait with partial overlap |
| Accession table | ID, genotype name, country of origin (coordinates uncertain) |
| Follow-up | Mutants near candidate SNPs being grown |
| Also available | Time-resolved RNA-seq, 4 time points, ~50 candidate genes, single knockouts do not lose rescue |
| Related literature | Volatile-based GWAS on the same strain (non-stress); a drought + volatiles study with 2-time-point RNA-seq |
