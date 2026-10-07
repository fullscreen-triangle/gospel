# Reply 3 — script for voice notes (after the full per-plant workbook)

Three notes, each about 2½ minutes read aloud. The numbers are rounded for speaking.
The exact values are in `arabidopsis-gwas/findings-full-workbook-2026-10-03.md`.

---

## Note 1 — what the new file confirmed

Thanks for the full sheet, it made a big difference. Before, I only had individual
plants for the inoculated drought condition, and I had to work the other three out
from your averages. Now everything is measured, so I could check the earlier analysis
against real data.

The good news first: it holds up. The means I reconstructed for the mock plants match
your measured values for almost every accession. The only exceptions are the few lines
grown more than once. The file also answered something I couldn't tell from the
formulas: "gain under non-stress" is inoculated minus mock under non-stress.

It also showed me something I hadn't realised. The GWAS used only the first replicate.
And the fifty accessions in replicates two and three weren't random. Forty-three of
them were labelled non-rescuers, so I assume you regrew them to confirm they really
don't respond. Is that right?

If so, that's a very useful experiment, because it tests directly whether low rescue
holds up in a new run. There's one subtlety in reading it. Those fifty were picked
because they looked weak in run one, so comparing run one with the retests is biased:
anything picked for being low will tend to look better next time. Runs two and three
are a fair comparison, though, because neither was used to choose them. So that's the
comparison I trusted most.

---

## Note 2 — what the replicates showed

This part is important, and I want to be careful with it, because it affects the
candidate genes.

First, the low rescuers didn't stay low. In run one, about five accessions had a gain
you couldn't distinguish from zero. In the retest, a different four did, and none was
in both lists. Per-1, which looked like the best loss-of-rescue line, was clearly
rescued in both retests. On average, the regrown accessions moved about a quarter to a
third of the way back towards the population mean. That's exactly what you'd expect if
run one was partly luck.

Second, and most striking, Col-0. Its shoot gain was about six milligrams in run one,
one and a half in run two, and six again in run three. In run two, the reference line
itself would have counted as a non-rescuer.

Third, I compared run two with run three directly for each phenotype. Drought
sensitivity, meaning how much the mock plant loses to drought, replicates well, with a
rank correlation around zero point seven. But the rescue itself, whichever way it's
calculated (gain, percent rescue, or the version the data prefer), comes out at about
zero point one to zero point two. That's statistically indistinguishable from zero.
Percent increase does a bit better, around zero point four. But that's because, as I
showed before, it largely ranks drought sensitivity.

So the honest reading is this. In one run, the stable difference between accessions
is how drought-sensitive they are. How much extra they get from WCS417 changes from run
to run. The candidate genes were all mapped on one run's rescue values, so I'd expect
them to replicate poorly, unless they turn out to be drought-sensitivity genes.

---

## Note 3 — the biology, and what I'd suggest

The biology actually comes out of this looking strong.

Without drought, WCS417 doesn't promote growth in your system. It slightly reduces
shoot weight, by about twelve percent on average, and roots are unchanged. Under
drought, it increases biomass in ninety-five percent of accessions. And how an
accession responds in one condition tells you nothing about the other. So this is a
genuinely drought-specific benefit, not general growth promotion. That's different from
the plate assays in the volatile paper, where almost everything was promoted without
stress. That difference is worth a sentence in the paper.

And not having a single confirmed non-rescuer fits your redundancy idea well: the strain
helps essentially every genotype under drought.

On what to do with it, three suggestions.

One: the drought sensitivity data are reproducible, so a GWAS on mock drought loss
would stand on solid ground right now. I've prepared that phenotype per accession.

Two: for a rescue GWAS, the phenotype would need to be averaged over several
independent runs. From these numbers, roughly six or more. Two things would cut that
down a lot. One is controlling drought severity between runs. The other is growing each
accession's mock and inoculated plants on the same tray. Under non-stress, your Col-0
mock and inoculated plants moved together from tray to tray, so that noise cancelled.
Under drought they didn't.

Three: before the mutant results go in, it would be worth checking whether the mutants
were grown alongside their wild type in the same run. If not, the run effect alone
could create or hide a difference.

A few quick questions, whenever you have time. Were the drought mock and inoculated
plants on separate trays? Was the non-stress experiment a single run? And how was
drought severity controlled between runs?

I've updated the write-up and made a short slide deck that goes through all of this.
I'll send it over.
