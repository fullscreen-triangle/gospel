# Reply 4: script for voice notes (the super-rescuer, and a route to a gene)

Three notes, each about 2½ minutes read aloud. Numbers are rounded for speaking.
The exact values are in `arabidopsis-gwas/findings-superrescuer-2026-10-07.md`, and the figure to send with it is `arabidopsis-gwas/figures/panel_8_superrescuer.png`.

---

## Note 1: yes, there is one, and it was in the rescreen

Thanks. That's really kind about the paper, and it helps a lot to know about the ceiling. I took your question literally: is there an accession that pushes past thirty to forty percent, compared with Col-0, and holds up when it's grown again?

There is one, and it's **Cod-0**, from Spain. It was in the rescreen, so we have three independent runs for it. Its shoot rescue is forty percent in run one, over a hundred percent in run two, and sixty percent in run three. It's above Col-0 in every run, and the confidence interval excludes zero each time. Under drought, WCS417 increases its shoot by about one-point-six to one-point-eight-fold in all three runs. Col-0 ranges from one-point-one to one-point-four.

I checked it properly, because picking the best of two hundred and forty-seven lines will always find something by chance. I selected on run one only, then asked whether runs two and three agree, since they played no part in the choice. They do: the chance of that by luck is about one in two hundred after correcting for everything I looked at. No other accession replicates that cleanly.

I want to be straight about three things. First, it isn't the biggest responder. By fold change it sits around the top third. What sets it apart is that it never drops off. Second, it's a tiny plant, the smallest in the panel, and it's quite drought tolerant. Part of its high rescue is simply that drought takes less from it, so the absolute gain is only a few milligrams. Third, and I think this is the interesting part: in run one, without drought, WCS417 nearly doubles Cod-0. Most accessions are slightly inhibited, like you described. So if the ceiling is a cost the plant pays for hosting the bacterium, Cod-0 may be a line that doesn't pay it. That's one run with seven plants, so it needs repeating.

---

## Note 2: why Cod-0? Not common variants, which explains the GWAS

You said you should have used the 1001 Genomes, so I did. I took the full SNP matrix and ran a mixed-model GWAS with kinship correction on about one-point-eight million SNPs. The model is well calibrated, with genomic inflation right at one.

The honest result is that nothing reaches genome-wide significance, for any version of rescue. Of the easyGWAS candidates, only two come back, MORN4 and AT1G58007, and both track drought loss or plant size rather than rescue. Shoot rescue shows no signal that transfers. When I hold out all the rescreened accessions and predict them from the rest, it's chance. Root rescue is different. It has a real but diffuse signal: about seventy percent of its top peaks point the same way in the rescreen, and that holds up. But it's spread thinly across many loci, so no single one justifies a mutant.

Then I asked whether Cod-0 is simply carrying lots of the common rescue-raising alleles. At first it looked that way: ten out of ten at the top peaks. But Cod-0's own phenotype was helping to pick those peaks. With Cod-0 held out, it drops to seven out of ten, which is close to chance. Its closest relatives also rescue at the panel average.

So the trait doesn't travel with its lineage, and it isn't common variation. It most likely sits in rare variants that Cod-0 almost alone carries. That's exactly the kind of thing a two-hundred-and-fifty-accession GWAS cannot see. It's not that your GWAS failed. The answer just isn't in that kind of data.

---

## Note 3: what's unique about it, and how we get to a gene

I went through Cod-0's own genome sequence for what's rare or unique. There are about twenty thousand rare variants, and around three thousand change a protein or a splice site. A few stand out biologically. I'd treat them as hypotheses, not hits, because with one accession every unique variant is equally associated with the trait.

The first is **PAD4**. Cod-0 has a change at position seventy-three that no other accession in the 1001 Genomes carries. PAD4 is part of the immune hub with EDS1, so a weaker immune cost of hosting WCS417 would fit "no penalty without drought" and "a higher ceiling". *pad4-1* is in Col-0, so you could test it straight away.

The second is **FKBP15-2**. Cod-0 has a unique splice-site mutation there, about twenty-five kilobases from your own Total Rescue hits, the cyclin and the G3BP-like gene, in a stretch where Cod-0 is unusually divergent. Since you're already growing those mutants, I'd add this one.

The third is **CYP707A2**, the ABA-breakdown enzyme, with three rare protein changes. That would explain why Cod-0 tolerates drought, and it would tell you whether ABA sets the ceiling. The fourth is **ARF7**, which matters for the lateral-root response to WCS417.

I also need to correct something I told you earlier. Iron-uptake genes like FRO2 and BTS looked mutated in Cod-0, but that was a coordinate error. They're unchanged at the protein level.

So my suggestion has three steps:

1. Put Cod-0 through your SynCom designs, next to Col-0, with CFU counts. Does it still break forty percent when WCS417 invades a community? And is it carrying more bacteria, or responding differently to the same load?
2. Test those four mutants in Col-0.
3. If Cod-0 holds up, cross it to Col-0. Rescue can't be scored on a single F2 plant, so score F3 families split into mock and WCS417, then sequence the extreme pools.

That's the route that ends in "we found the gene", and Cod-0 is the reason it's worth doing. I can send the full write-up and the figure whenever suits you.
