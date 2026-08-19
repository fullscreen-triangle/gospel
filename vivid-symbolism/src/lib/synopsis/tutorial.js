// The tutorial script library for the /ide page.
//
// Two groups. `tutorial` is twelve lessons that build the four corpus
// programs from their pieces; `refusals` is six programs that must be
// rejected, because the language is defined as much by what it refuses.
//
// Every script here was run through the Rust binary before being
// committed, and lessons 6, 9 and 12 are corpus programs reproduced
// verbatim. That matters: the IDE is a third consumer of the parser,
// and a tutorial that disagreed with the compiler would be worse than
// no tutorial at all.
//
// `stage` on a refusal records WHERE the refusal comes from:
//   "A" -- the grammar refuses it. The program cannot be written, so
//          there is no tree and the views say so.
//   "B" -- the checker refuses it. The program parses, so the tree IS
//          drawn: the rule being taught is usually a relation between
//          two nodes a reader can point at.
//
// Both stages produce real diagnostics now. The field is no longer a
// note about what is unimplemented; check_ide.mts asserts that each
// refusal actually fires at the stage claimed here, so a mislabelled
// lesson is a test failure rather than a quiet inaccuracy.

export const TUTORIAL = [
  {
    id: "01-first-program",
    title: "The smallest program",
    blurb:
      "Every program names its inputs, does its work inside a frame of reference, and says where its report goes. These three parts are required; there is no shorter legal program than this.",
    tryThis:
      "Delete the last line. The program is refused with a ParseError — a program that does not say where its results go is not a program.",
    src: `open a = "x.fa"

under nucleotide {
    record a
}

report to "r.report"`,
  },
  {
    id: "02-projection",
    title: "Projection gives a value a type",
    blurb:
      "`open` names a file, but a file is not yet something you can compute with. `project ... by` turns it into a typed value inside the current frame. The projector chosen — here `channels(dna)` — determines that type.",
    tryThis:
      "Change `channels(dna)` to `spectral(coeffs = 8)`. It still parses: both are projectors. The difference between them is a matter for the type checker.",
    src: `open motif = "motif.fa"

under nucleotide {
    let q = project motif by channels(dna)
    record q
}

report to "r.report"`,
  },
  {
    id: "03-comparison",
    title: "Comparison, and the residue",
    blurb:
      "`compare ... against ... by` relates two projected values. Note that `bind` introduces TWO names, not one: the result, and the residue — the part of the comparison that the method did not explain. Naming the residue is not decoration. A later stage requires that it be consumed before the frame closes, so a program cannot quietly discard what its method failed to account for.",
    tryThis:
      "Remove `res_r` from the `record` line. That is the `unconsumed-residue` refusal, in the refusals folder below.",
    src: `open q = "q.fa"
open t = "t.fa"

under nucleotide {
    let a = project q by channels(dna)
    let b = project t by channels(dna)
    bind r, res_r = compare a against b by xcorr(normalised)
    record r, res_r
}

report to "r.report"`,
  },
  {
    id: "04-detection",
    title: "Detection, and parameters without defaults",
    blurb:
      "`detect` finds structure in a comparison. Every parameter it needs must be written down: there are no defaults anywhere in the language. A default is a decision made by the implementer that silently becomes part of someone's result, so the language declines to make any.",
    tryThis:
      "Delete the `min_score` line. That is a ParameterError — the third refusal below.",
    src: `open q = "q.fa"
open t = "t.fa"

under nucleotide {
    let a = project q by channels(dna)
    let b = project t by channels(dna)
    bind r, res_r = compare a against b by xcorr(normalised)
    let hits = detect peaks in r {
        z            = 4.0 ;
        min_distance = 30  ;
        min_score    = 0.35
    }
    record hits, res_r
}

report to "r.report"`,
  },
  {
    id: "05-claims",
    title: "Stating what the program asserts",
    blurb:
      "A `claim` attaches a sentence in English to the value that is supposed to support it. This is what makes a synopsis program self-describing: the report carries the claim and the evidence side by side, so a reader can see what was asserted without reconstructing it from the code.",
    tryThis:
      "Change the claim text. Nothing else changes — the claim is a statement about the value, not an instruction to the machine.",
    src: `open q = "q.fa"
open t = "t.fa"

under nucleotide {
    let a = project q by channels(dna)
    let b = project t by channels(dna)
    bind r, res_r = compare a against b by xcorr(normalised)
    let hits = detect peaks in r {
        z            = 4.0 ;
        min_distance = 30  ;
        min_score    = 0.35
    }
    claim "motif occurs at least once above z=4" = hits
    record hits, res_r
}

report to "r.report"`,
  },
  {
    id: "06-motif-scan",
    title: "Complete: motif scan",
    blurb:
      "This is a corpus program, reproduced exactly. You have now written it line by line: open, project, compare with a residue, detect with explicit parameters, claim, record, report. Nothing here is new.",
    tryThis:
      "Compare this against lesson 5. The only additions are the second input and the comment.",
    corpus: true,
    src: `# Locate a motif in a target by coherent multichannel matched filtering.
open motif  = "motif.fa"
open target = "chr7_region.fa"

under nucleotide {
    let q = project motif  by channels(dna)
    let t = project target by channels(dna)
    bind r, res_r = compare q against t by xcorr(normalised)
    let hits = detect peaks in r {
        z            = 4.0 ;
        min_distance = 30  ;
        min_score    = 0.35
    }
    claim "motif occurs at least once above z=4" = hits
    record hits, res_r
}

report to "motif_scan.report"`,
  },
  {
    id: "07-spectral",
    title: "A different projector, and dimension",
    blurb:
      "`spectral(coeffs = 8)` embeds a sequence as eight coefficients. The coefficient count becomes part of the value's type, so comparing an 8-coefficient embedding against a 16-coefficient one is a type error rather than a silent truncation.",
    tryThis:
      "This is the basis of the `dimension_mismatch` refusal in the corpus: two projections with different `coeffs` cannot be compared.",
    src: `open query = "query.fa"

under residue_space {
    let e = project query by spectral(coeffs = 8)
    record e
}

report to "r.report"`,
  },
  {
    id: "08-reranking",
    title: "Chaining: a cheap pass, then an exact one",
    blurb:
      "`detect top` takes the best k candidates from a comparison, and those candidates can themselves be compared again by a more expensive method. This is the ordinary shape of a homology search — a fast approximate filter followed by an exact rerank — expressed so that both passes and both residues are on the record.",
    tryThis:
      "Note that the second `compare` produces its own residue, `res_f`. Each comparison accounts for itself separately.",
    src: `open query = "query.fa"
open db    = "db.fa"

under residue_space {
    let e  = project query by spectral(coeffs = 8)
    let DB = project db    by spectral(coeffs = 8)
    bind s, res_s = compare e against DB by shader(cosine)
    let cands = detect top in s {
        k     = 200 ;
        depth = 12
    }
    record cands, res_s
}

report to "r.report"`,
  },
  {
    id: "09-homology",
    title: "Complete: homology search",
    blurb:
      "The second corpus program, verbatim. It adds the exact rerank to lesson 8 and records both residues — the approximate pass and the exact one each answer for what they left unexplained.",
    tryThis:
      "Try removing `res_s` from the record line and predict the error class before you run it.",
    corpus: true,
    src: `open query = "query.fa"
open db    = "swissprot_subset.fa"

under residue_space {
    let e  = project query by spectral(coeffs = 8)
    let DB = project db    by spectral(coeffs = 8)
    bind s, res_s = compare e against DB by shader(cosine)
    let cands = detect top in s {
        k      = 200 ;
        depth  = 12
    }
    bind final, res_f = compare e against cands by smith_waterman(
        match = 2 ; mismatch = -1 ; gap = -2
    )
    claim "top 20 by exact rerank" = final
    record final, res_s, res_f
}

report to "homology.report"`,
  },
  {
    id: "10-units-and-response",
    title: "Units, and declared perturbations",
    blurb:
      "A `method` declaration names a perturbation once, with its magnitude, at the top of the file. `unit ... anchors 3` picks out a local neighbourhood, and `response ... by` records how that neighbourhood answers the named perturbation. The method must be declared before it can be used — an anonymous perturbation is refused.",
    tryThis:
      "Delete the `method` line. That is the `anonymous_response` refusal: you cannot perturb something by an unnamed rule.",
    src: `open wt = "wildtype.solved"

method pi_edge_raise = perturb_incident(magnitude = 0.10)

under cell_receiver {
    let Nwt = project wt by contact(medium = 0.05)
    let cA  = unit Nwt anchors 3
    let rA  = response cA by pi_edge_raise
    record cA, rA
}

report to "r.report"`,
  },
  {
    id: "11-alignment",
    title: "Alignment: sufficient, not exact",
    blurb:
      "This is the centre of the language. `align` asks whether two things correspond well enough to carry a conclusion across — not whether they are identical. It requires a `response` clause and TWO correspondences, because agreeing on structure alone, or on behaviour alone, is not sufficient evidence. `relax ... until quiescent` then settles the alignment, and its `eta` must be strictly positive so that the settling is guaranteed to terminate.",
    tryThis:
      "Set `eta = 0` in the relax block. That is a TerminationError: a step size of zero would never converge, so the language refuses the program rather than letting it hang.",
    src: `open wt  = "wildtype.solved"
open var = "variant.solved"

method pi_edge_raise = perturb_incident(magnitude = 0.10)

under cell_receiver {
    let Nwt  = project wt  by contact(medium = 0.05)
    let Nvar = project var by contact(medium = 0.05)
    let cA = unit Nwt  anchors 3
    let cB = unit Nvar anchors 3
    let rA = response cA by pi_edge_raise
    let rB = response cB by pi_edge_raise
    let phi_c = corr from cA to cB by species_name
    let phi_r = corr from rA to rB by species_name
    let v = align central(cA, cB) response(rA, rB)
            under phi_c, phi_r { theta = 0.01 }
    relax v until quiescent { eta = 0.005 ; theta = 0.01 }
    claim "variant is functionally equivalent to wild type" = v
    record v
}

report to "r.report"`,
  },
  {
    id: "12-transfer",
    title: "Complete: annotation transfer",
    blurb:
      "The last corpus program. `for u in items(N) where ...` iterates over a bounded collection with a guard — note that it is a `for` over items that already exist, not a `while`. The language has no `while` production at all, which is why every program is guaranteed to terminate. `nearest_unit` finds each unit's correspondent in the second network.",
    tryThis:
      "There is no way to write an unbounded loop in this language. Try it in the editor and see what the parser says.",
    corpus: true,
    src: `open ref     = "reference.solved"
open subject = "subject.solved"

method pi_uniform = perturb_incident(magnitude = 0.10)

under tissue_receiver {
    let Nr = project ref     by contact(medium = 0.05)
    let Ns = project subject by contact(medium = 0.05)
    for u in items(Nr) where separation(u) > 0.10 {
        let cA = unit u  anchors 3
        let cB = nearest_unit Ns to cA
        let rA = response cA by pi_uniform
        let rB = response cB by pi_uniform
        let phi_c = corr from cA to cB by anchor_match
        let phi_r = corr from rA to rB by anchor_match
        let v = align central(cA, cB) response(rA, rB)
                under phi_c, phi_r { theta = 0.02 }
        claim "annotation of u transfers to its correspondent" = v
        record v
    }
}

report to "transfer.report"`,
  },
];

export const REFUSALS = [
  {
    id: "no-report",
    title: "A program must name its report",
    expect: "ParseError",
    stage: "A",
    blurb:
      "The frame closes and the program simply stops. Where were the results supposed to go? The grammar requires `report to` as the final form, so this is refused before any analysis happens.",
    src: `open q = "q.fa"
under N {
    let a = project q by channels(dna)
    record a
}`,
  },
  {
    id: "sequence-indexing",
    title: "There is no way to index a sequence",
    expect: "SynopsisError",
    reported: "ParseError",
    stage: "A",
    blurb:
      "`q[3]` does not parse, because the grammar has no indexing production. This is deliberate and it is load-bearing: a language that can read position 3 can be used to hand-tune an exact answer, and the guarantee that results come from stated methods rather than from poking at coordinates only holds if reaching into a sequence is impossible to write. The reference records this as the base SynopsisError; our parsers catch it earlier and report the more specific ParseError, which the hierarchy permits.",
    src: `open q = "q.fa"
under N {
    let x = q[3]
    record x
}
report to "r"`,
  },
  {
    id: "undefined-variable",
    title: "Every name must be bound",
    expect: "ScopeError",
    stage: "B",
    blurb:
      "`nonexistent` was never opened or bound. The grammar is satisfied — it is a well-formed name in a well-formed position — so this is caught by the checker rather than the parser.",
    src: `open q = "q.fa"
under N {
    let a = project nonexistent by channels(dna)
    record a
}
report to "r"`,
  },
  {
    id: "cross-frame-reference",
    title: "A value from another frame is a different type",
    expect: "ScopeError",
    stage: "B",
    blurb:
      "`a` was projected inside `nucleotide_frame`, and the second block tries to use it inside `protein_frame`. The frame is part of the type, so this is not merely bad style — the two values are of genuinely different types, and there is no coercion between them. This is the single device that makes frame confusion unwriteable rather than merely discouraged.",
    src: `open wildtype = "wt.fa"
open variant = "var.fa"
under nucleotide_frame {
    let a = project wildtype by channels(dna)
    let b = project variant by channels(dna)
    bind r, res = compare a against b by xcorr(normalised)
    record res
}
under protein_frame {
    let p = project wildtype by channels(protein)
    bind bad, res2 = compare a against p by xcorr(normalised)
    record res2
}
report to "r"`,
  },
  {
    id: "unconsumed-residue",
    title: "Residue must be consumed",
    expect: "ResidueError",
    stage: "B",
    blurb:
      "The comparison bound `res`, and the frame closes without it ever being recorded or used. The residue is what the method could not explain; allowing a program to drop it would allow a program to report a clean result while quietly discarding the evidence that it was not clean.",
    src: `open q = "q.fa"
open t = "t.fa"
under N {
    let a = project q by channels(dna)
    let b = project t by channels(dna)
    bind r, res = compare a against b by xcorr(normalised)
    record r
}
report to "r"`,
  },
  {
    id: "peaks-missing-parameter",
    title: "No defaults, anywhere",
    expect: "ParameterError",
    stage: "B",
    blurb:
      "`detect peaks` requires z, min_distance and min_score. Two are given. Rather than supplying the third from a table of sensible values, the language refuses the program — because a default silently becomes part of a published result, and nobody reading the report would know it had been chosen by the implementer rather than the author.",
    src: `open q = "q.fa"
open t = "t.fa"
under N {
    let a = project q by channels(dna)
    let b = project t by channels(dna)
    bind r, res = compare a against b by xcorr(normalised)
    let h = detect peaks in r { z = 4.0 ; min_distance = 30 }
    record h, res
}
report to "r"`,
  },
];

export const ALL_FILES = [
  ...TUTORIAL.map((f) => ({ ...f, group: "tutorial", name: `${f.id}.syp` })),
  ...REFUSALS.map((f) => ({ ...f, group: "refusals", name: `${f.id}.syp` })),
];

export const DEFAULT_FILE = "01-first-program.syp";
