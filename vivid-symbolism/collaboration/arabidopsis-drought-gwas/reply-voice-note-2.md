# Reply 2 — script for voice notes (after receiving the GWAS workbook)

Three notes, each about 2½ minutes read aloud. Numbers are rounded for
speaking. The exact values are in `findings-2026-09-29.md` if he asks.

---

## Note 1 — what's in the sheet

Thanks for the data and for the notes on the RNA-seq, I've gone through
everything. Let me start with the spreadsheet itself, because there are a few
things worth knowing before we go any further.

The raw per-plant data in the file is only for one of the four conditions:
WCS417 under drought. Seven plants per accession, and Col-0 grown fourteen
times across the screen. The mock plants and the non-stress plants only appear
as averages inside the derived traits. So I reconstructed the formulas from the
numbers, and they fit every accession exactly.

Rescue is the bacterial gain under drought divided by how much that accession
loses to drought in mock. Percent increase is the treated plants over the mock
drought plants, minus one. And percent loss under drought is calculated only
from mock plants, non-stress versus drought.

Three things follow from that.

First, the columns in Table 2 labelled "drought tolerance" are actually the
loss values. A higher number there means less tolerant, not more. Worth fixing
before it goes into a figure.

Second, because the loss trait doesn't involve the bacterium at all, the genes
that came from it — MORN4, TPS21, and the antisense RNA on chromosome two —
are drought-response genes, not rescue genes. They might still be interesting,
but they belong to a different story.

Third, "non-rescuer" in the sheet is a threshold on the gain. Every single
accession gains biomass with the strain. So your data really does say rescue
varies in strength and is never lost, which fits your redundancy idea.

---

## Note 2 — the main result

Now the part I was most curious about: do the different formulas agree on which
accessions rescue best?

Mostly, they don't. Among the top twenty percent of accessions, gain, percent
increase and rescue share only about a third to less than half of the same
lines in shoot. In root it's lower still. Only sixteen accessions are in the top
fifth under all three definitions in shoot, and seven in root.

To check this isn't just noise, I resampled the individual plants five hundred
times. The rankings barely move — correlations above point nine. So noise
isn't driving the differences. The choice of formula is.

The reason is that each formula carries a different confounder. Percent
increase divides by the mock drought biomass, so drought-sensitive accessions
get inflated values. Its ranking correlates at point eight with the mock-only
drought loss. Twelve of your twenty-eight candidate genes come from
percent-increase traits, so some of those may be drought-sensitivity genes
rather than rescue genes. Absolute gain in milligrams tracks plant size. Rescue
normalised by each accession's own drought loss is the least confounded of the
three.

That also explains the overlap in your candidate list. The genes that appear
under several metrics are always average versus median of the same trait, or
shoot versus total. No gene appears under two genuinely different definitions.

So my suggestion is to fix one rescue phenotype before any further GWAS. The
cleanest version is a single model on the per-plant data from all four
conditions, with the bacterium-by-drought interaction as the phenotype, and
batch and kinship included. Col-0 is ideal for the batch part: its tray-to-tray
spread is about half the spread between accessions, which is too large to
ignore.

---

## Note 3 — iron, climate, and what I need

The iron story is really nice. A reversal of the drought response that points
to iron, higher iron content with the bacterium, and uptake mutants that don't
lose rescue — that's a coherent mechanism.

One thing I'd check, because it's the same issue as the GWAS traits: which
metric did you use when you said the iron-uptake mutants show higher rescue? If
it's percent increase, a mutant that grows worse in mock under drought will
look like it's rescued more, even with the same absolute gain. If the absolute
gain and the loss-normalised rescue are also higher in the mutants, the result
is solid, and then it's a strong result.

And this is where the network side of my method can come in later. Your fifty
reversal genes plus the iron STRING network are exactly the kind of network I
can analyse. The question would be: what is the smallest set of genes you'd
have to remove together to break rescue? That gives a shortlist of double or
triple mutants to test, instead of screening them blindly.

On the accessions, that makes sense, and it's actually a better answer to a
reviewer than a deliberate selection. The sheet already has latitude and
longitude for every accession, so I can pull precipitation and temperature from
a public climate database myself. You don't need to dig up that paper for me.

What would help most, whenever you have time: the per-plant data for the other
three conditions, which tray or batch each accession was grown in, the easyGWAS
settings if you find them, and whether "gain in non-stress" means treated minus
mock under non-stress, or drought-treated minus non-stress-treated. The
formulas fit both, so I can't tell from the numbers.

I'll put all of this into a short write-up with figures so you can look at it
properly when you have a moment.
