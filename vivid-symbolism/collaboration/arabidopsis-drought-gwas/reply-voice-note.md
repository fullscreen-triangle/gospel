# Reply — script for voice notes

Two notes, each about 2½ minutes read at a normal pace. Short sentences so they
can be read aloud without stumbling.

---

## Note 1 — the data, and a first analysis

Thanks for laying all of that out, it's really clear now. Let me say back what
I've understood, so you can correct me.

For each accession you have eight cells: with or without the strain, under
drought or non-stress, in shoot and in root. Each cell has five to seven
individual plants. And every trait in the sheet — rescue, percentage gain,
absolute gain, the drought tolerance index — is a different formula applied to
those same biomass values.

What you said about the candidate genes is actually the most interesting part
for me. The same measurements, expressed through different formulas, give
partly different gene lists, with some genes shared. That's exactly the
question my method is built around. When you have several reasonable ways of
defining the same response, does the conclusion depend on which one you pick?
In my framework that property is called response independence. I've tested it
on metabolic networks, but never on real phenotype data like yours.

So here's what I'd suggest as a first step. It's concrete, and it should be
useful for your paper directly. I'd take the raw per-plant biomass and ask
three things.

First, how much does the ranking of accessions change between trait
definitions?

Second, which candidate genes come up regardless of how rescue is defined, and
which ones only appear under one formula?

Third, can we define rescue once, directly from the raw data — as the
interaction between the strain and drought in a single model — rather than as a
ratio? Ratios of noisy biomass values tend to inflate variance.

The practical payoff is prioritisation. The genes that hold up across every
trait definition are your strongest candidates for the mutant screen. That's
also a reasonable answer to a reviewer about the lowered significance
threshold. If a borderline SNP keeps showing up under every definition of the
trait, that's independent support for it. If it only appears under one, it's
more likely to be noise.

---

## Note 2 — redundancy, climate, and what I need

Your point about redundancy really caught my attention. You have around fifty
candidates from the RNA-seq, and knocking any one of them out doesn't abolish
rescue. That's exactly what you'd expect if rescue is a property of the
network, not of any single gene. My method can say something concrete about
that case.

If we build a network from your time-resolved RNA-seq, we can compute the
smallest set of genes that would have to be removed together to disconnect the
strain's effect from the growth outcome. In graph terms that's a minimum cut.
It would tell us whether single knockouts should ever be enough, and it would
give a ranked shortlist of double or triple mutant combinations to test. So
instead of screening combinations blindly, you'd have a prediction that the
next experiment either confirms or rules out. I'd see that as a second phase,
after the GWAS part.

On the climate layer, I agree it's worth adding. There's also a strategic angle
to it. The accessions were chosen on annual precipitation, so climate and
population structure are tangled together. If we include the climate of origin
explicitly as a covariate, alongside kinship, the selection bias becomes
something we've measured and accounted for. That's a stronger position with a
reviewer than defending the selection. If the accession IDs match the 1001
Genomes IDs, I can pull the coordinates and climate data myself, so no need to
dig for them.

The volatile study is a good idea too. If there are shared accessions under
non-stress, that's a natural way to separate the contact effect from the
volatile effect. Please do send the paper.

So what I need, whenever you have a moment: the raw per-plant biomass sheet,
the candidate gene table, the Manhattan plots, and the accession table. The
RNA-seq can wait until we're into the second phase.

I know the next three months are intense for you, so I'll keep this low-effort
on your side. I'll do the analysis and come back with results rather than
questions. Good luck in the lab.
