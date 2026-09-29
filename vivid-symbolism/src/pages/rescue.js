import Head from "next/head";
import Link from "next/link";
import { useEffect, useState } from "react";

import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";
import {
  BatchBlocks, CellMeansChart, GainCaterpillar, ReliabilityBars,
} from "@/components/rescue/DataCharts";
import {
  CandidateChart, LambdaExplorer, RatioArtefact, TopKExplorer,
} from "@/components/rescue/ConstructionCharts";
import {
  CalibrationCurves, DesignHeatmap, GeoMap, TierChart, VerdictExplorer,
} from "@/components/rescue/VerdictCharts";

const BASE = "/rescue";
const FILES = {
  accessions: "accessions", index: "index", e1: "E1_audit", e2: "E2_heterogeneity",
  e3: "E3_dependence", e4: "E4_canonical", e5: "E5_four_column", e6: "E6_calibration",
  e7: "E7_batch", e8: "E8_candidates", e9: "E9_ratio", e10: "E10_classes",
  e11: "E11_geography", e12: "E12_design",
};

const TOC = [
  ["overview", "Overview"],
  ["question", "1 · The question before the question"],
  ["screen", "2 · The screen as received"],
  ["audit", "3 · E1 — Auditing the workbook"],
  ["constructions", "4 · What a phenotype is"],
  ["heterogeneity", "5 · E2 — One strain, many effects"],
  ["batch", "6 · E7 — The batch control already in the design"],
  ["dependence", "7 · E3 — Do the readings agree?"],
  ["canonical", "8 · E4 — Letting the data choose"],
  ["fourcolumn", "9 · E5, E6 — Four columns and three verdicts"],
  ["tiers", "10 · E10 — Resolution depth"],
  ["candidates", "11 · E8 — Triage of the candidate list"],
  ["mutants", "12 · E9 — Checking a mutant claim"],
  ["design", "13 · E12 — The next experiment"],
  ["geography", "14 · E11 — Collection sites"],
  ["protocol", "15 · Protocol for the reanalysis"],
  ["limits", "16 · Limitations"],
  ["files", "17 · Files and reproduction"],
];

// ------------------------------------------------------------------ prose kit
function Section({ id, title, kicker, children }) {
  return (
    <section id={id} className="mt-20 scroll-mt-28">
      {kicker ? (
        <div className="text-xs font-bold uppercase tracking-[0.18em] text-primary dark:text-primaryDark">{kicker}</div>
      ) : null}
      <h2 className="mt-1 text-3xl font-bold leading-tight md:text-2xl">{title}</h2>
      <div className="mt-5 space-y-4 text-[15px] font-medium leading-relaxed text-dark/90 dark:text-light/90">
        {children}
      </div>
    </section>
  );
}

function Eq({ children }) {
  return (
    <div className="my-4 overflow-x-auto rounded-lg border border-dark/20 bg-dark/[0.03] px-4 py-3 text-center
      font-mono text-sm dark:border-light/20 dark:bg-light/[0.04]">
      {children}
    </div>
  );
}

function Callout({ tone = "note", title, children }) {
  const border = tone === "warn" ? "border-[#e34948]" : "border-primary dark:border-primaryDark";
  return (
    <div className={`my-6 rounded-r-lg border-l-4 ${border} bg-dark/[0.03] px-5 py-4 dark:bg-light/[0.04]`}>
      {title ? <div className="mb-1 text-sm font-bold uppercase tracking-wide">{title}</div> : null}
      <div className="space-y-2 text-sm leading-relaxed">{children}</div>
    </div>
  );
}

function Spec({ id, purpose, inputs, procedure, output, acceptance, result }) {
  const rows = [
    ["Purpose", purpose], ["Inputs", inputs], ["Procedure", procedure],
    ["Output", output], ["Acceptance", acceptance], ["Result", result],
  ];
  return (
    <div className="my-6 overflow-hidden rounded-xl border-2 border-dark dark:border-light">
      <div className="bg-dark px-4 py-2 font-mono text-sm font-bold text-light dark:bg-light dark:text-dark">
        Experiment {id}
      </div>
      <dl className="divide-y divide-dark/15 dark:divide-light/15">
        {rows.map(([k, v]) => (
          <div key={k} className="grid grid-cols-[8.5rem_1fr] gap-3 px-4 py-2.5 text-sm md:grid-cols-1 md:gap-1">
            <dt className="font-bold text-dark/70 dark:text-light/70">{k}</dt>
            <dd className="leading-relaxed">{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function Tile({ label, value, sub }) {
  return (
    <div className="rounded-xl border-2 border-dark p-4 dark:border-light">
      <div className="text-xs font-semibold uppercase tracking-wide text-primary dark:text-primaryDark">{label}</div>
      <div className="mt-1 text-2xl font-bold">{value}</div>
      {sub ? <div className="mt-1 text-xs font-medium text-dark/70 dark:text-light/70">{sub}</div> : null}
    </div>
  );
}

function Code({ children }) {
  return <code className="rounded bg-dark/[0.06] px-1 py-0.5 font-mono text-[13px] dark:bg-light/[0.08]">{children}</code>;
}

function JsonLink({ name }) {
  return (
    <Link href={`${BASE}/results/${name}.json`} target="_blank"
      className="font-mono text-sm font-semibold text-primary underline-offset-4 hover:underline dark:text-primaryDark">
      {name}.json
    </Link>
  );
}

// ------------------------------------------------------------------ page
export default function Rescue() {
  const [D, setD] = useState({});
  const [err, setErr] = useState(null);

  useEffect(() => {
    let alive = true;
    Promise.all(Object.entries(FILES).map(([k, f]) =>
      fetch(`${BASE}/results/${f}.json`).then((r) => {
        if (!r.ok) throw new Error(`${f}.json: ${r.status}`);
        return r.json();
      }).then((j) => [k, j])))
      .then((entries) => { if (alive) setD(Object.fromEntries(entries)); })
      .catch((e) => { if (alive) setErr(String(e)); });
    return () => { alive = false; };
  }, []);

  const acc = D.accessions ? D.accessions.accessions : null;
  const lamS = D.e4 ? D.e4.organs.Shoot : null;
  const lamR = D.e4 ? D.e4.organs.Root : null;

  return (
    <>
      <Head>
        <title>Drought rescue &middot; a reanalysis specification</title>
        <meta name="description"
          content="Specification and results of a reanalysis of a WCS417 drought-rescue natural-variation screen in Arabidopsis: construction dependence, batch, and the limits of certification." />
      </Head>
      <TransitionEffect />
      <main className="mb-16 flex w-full flex-col items-center justify-center dark:text-light">
        <Layout className="pt-16">
          <AnimatedText
            text="Rescue Is a Relation, Not a Trait"
            className="mb-6 !text-6xl !leading-tight lg:!text-5xl sm:!text-4xl xs:!text-3xl"
          />
          <p className="max-w-4xl text-lg font-medium leading-relaxed md:text-base">
            A reanalysis of a natural-variation screen for drought rescue of <i>Arabidopsis thaliana</i> by the
            beneficial rhizobacterium <i>Pseudomonas simiae</i> WCS417. The screen covers 247 accessions, two
            inoculation treatments, two water regimes and two organs. This page is the experiment&apos;s
            specification. For each of twelve analyses it states the purpose, the inputs, the procedure, the
            output file, what would count as success, and what happened. Every chart reads the same JSON files
            the manuscript was built from, and several recompute their statistics live in your browser.
          </p>

          <Callout tone="warn" title="Unpublished data">
            <p>
              The phenotypes, candidate genes and design described here come from an unpublished screen in the
              collaborating laboratory. They are shown for the collaboration. Do not cite, reuse or redistribute
              them without the laboratory&apos;s agreement.
            </p>
          </Callout>

          <div className="mt-6 flex flex-wrap gap-3 text-sm font-semibold">
            <Link href={`${BASE}/arabdopsis-drought-gwas-rescue.pdf`} target="_blank"
              className="rounded-lg border-2 border-dark px-4 py-2 hover:border-primary dark:border-light dark:hover:border-primaryDark">
              Manuscript (PDF)
            </Link>
            <Link href={`${BASE}/references.bib`} target="_blank"
              className="rounded-lg border-2 border-dark px-4 py-2 hover:border-primary dark:border-light dark:hover:border-primaryDark">
              Bibliography (.bib)
            </Link>
            <Link href={`${BASE}/results/index.json`} target="_blank"
              className="rounded-lg border-2 border-dark px-4 py-2 hover:border-primary dark:border-light dark:hover:border-primaryDark">
              Results index (JSON)
            </Link>
          </div>

          {err ? <p className="mt-6 font-mono text-sm text-[#e34948]">Could not load results: {err}</p> : null}

          <nav className="mt-10 rounded-xl border-2 border-dark p-5 dark:border-light" aria-label="contents">
            <div className="mb-2 text-sm font-bold uppercase tracking-wide">Contents</div>
            <ol className="grid grid-cols-2 gap-x-8 gap-y-1 text-sm md:grid-cols-1">
              {TOC.map(([id, t]) => (
                <li key={id}><a href={`#${id}`} className="hover:text-primary hover:underline dark:hover:text-primaryDark">{t}</a></li>
              ))}
            </ol>
          </nav>

          {/* ============================================================ overview */}
          <Section id="overview" kicker="In one screen" title="Overview">
            <p>
              The screen asked which host genes let WCS417 protect <i>Arabidopsis</i> from drought. To map genes, one
              needs one number per accession. The screen, however, produced a small factorial per accession:
              inoculated or mock, watered or droughted, shoot and root. The laboratory derived several numbers from
              that factorial: biomass under inoculation, absolute gain, percentage increase, percentage rescue,
              percentage loss and a binary rescuer label. It mapped each one separately. The candidate loci differed
              from trait to trait, and no accession lost rescue altogether.
            </p>
            <p>
              This reanalysis asks the question that has to be answered before any of those scans can be
              interpreted: <b>which of those numbers is the rescue of an accession, and how much does the choice
              matter?</b> The answer has four parts, each backed by an experiment below.
            </p>
            <ol className="ml-6 list-decimal space-y-2">
              <li>The workbook fixes three of the four cell means of every accession exactly, and the readings in
                use are exact formulas in them. One of those readings contains no inoculated plants at all.</li>
              <li>The readings disagree about which accessions rescue best. In root the disagreement exceeds every
                noise model, and under plant-level noise two readings can certify the same pair of accessions in
                opposite order.</li>
              <li>The choice of reading is the choice of an exponent λ, and the data can estimate λ. They reject
                percentage increase, the reading behind the largest group of candidate genes, and select a
                loss-normalised rescue. Within the range the data allow, the choice no longer matters more than
                noise.</li>
              <li>Comparing accessions through four columns (two mock baselines and two responses) with a
                three-valued verdict shows what the screen can and cannot certify. It can certify that accessions
                differ. It almost never can certify that they rescue equally. It resolves four to five tiers of
                rescue, and its largest source of noise is batch, which its own Col-0 control can measure.</li>
            </ol>
            <div className="mt-6 grid grid-cols-4 gap-4 lg:grid-cols-2 sm:grid-cols-1">
              <Tile label="Analysis set" value={D.e1 ? `${D.e1.n_main}` : "…"} sub="accessions with all three cell means" />
              <Tile label="λ̂ shoot" value={lamS ? lamS.lambda_hat_corrected.toFixed(2) : "…"}
                sub={lamS ? `95% [${lamS.lambda_ci95_corrected[0].toFixed(2)}, ${lamS.lambda_ci95_corrected[1].toFixed(2)}]; λ = 1 excluded` : ""} />
              <Tile label="λ̂ root" value={lamR ? lamR.lambda_hat_corrected.toFixed(2) : "…"}
                sub={lamR ? `95% [${lamR.lambda_ci95_corrected[0].toFixed(2)}, ${lamR.lambda_ci95_corrected[1].toFixed(2)}]` : ""} />
              <Tile label="Tiers resolved" value={D.e10 ? `${D.e10.Shoot.M3_plant_plus_batch.max_depth.canonical} / ${D.e10.Root.M3_plant_plus_batch.max_depth.canonical}` : "…"} sub="shoot / root, batch-inclusive noise" />
            </div>
            <p>
              The page is long, about half an hour of reading, because it is meant to be used as a specification.
              Someone should be able to rerun, extend or challenge every step from what is written here and in the
              files it links.
            </p>
          </Section>

          {/* ============================================================ question */}
          <Section id="question" kicker="Motivation" title="1 · The question before the question">
            <p>
              A genome-wide association study correlates a number per accession with genotypes. The number has to
              exist before the correlation does, and the study inherits every property of that number. If the
              number partly measures plant size, the scan partly maps plant size. If it partly measures drought
              sensitivity, the scan partly maps drought sensitivity. None of this shows up in a Manhattan plot. A
              peak looks the same whatever the phenotype is made of.
            </p>
            <p>
              In a screen for a beneficial microbe under stress, the thing of interest is an <b>interaction</b>: how
              much better an inoculated plant does under drought than it would have done without the bacterium,
              compared with how much drought hurt it in the first place. That quantity is not stored anywhere in a
              plant. It is a relation among four groups of plants: inoculated and mock, each watered and droughted.
              Any single number is one <i>reading</i> of that relation, and there are several reasonable readings.
            </p>
            <p>
              Two situations in this screen made the question pressing. First, different readings produced different
              candidate genes, with apparent agreement only between near-identical readings. That is either noise,
              which would argue for pooling the readings, or systematic difference, which would mean they measure
              different things. Second, no accession lost rescue, so the classic contrast between responders and
              non-responders did not exist. Both situations are questions about the phenotype, not about the
              genotypes, and both can be investigated from the phenotype data alone.
            </p>
            <Callout title="What this reanalysis does not do">
              <p>
                It does not rerun the association scan. The workbook contains no genotypes, and none were retrieved.
                Statements about candidate genes concern <i>which reading</i> a candidate came from, not whether it
                is real.
              </p>
              <p>
                It does not measure noise in the three cells that have no per-plant data. It models that noise and
                reports every noise-sensitive result under several models.
              </p>
              <p>
                It does not claim that any accession fails to be rescued. It identifies accessions whose rescue is
                statistically compatible with zero, which is a reason to test them again.
              </p>
            </Callout>
          </Section>

          {/* ============================================================ screen */}
          <Section id="screen" kicker="Materials" title="2 · The screen as received">
            <p>
              The material was a single workbook, a slide deck and a voice-recorded description of the design by the
              experimenter. The workbook has six sheets:
            </p>
            <ul className="ml-6 list-disc space-y-1">
              <li>per-plant biomass rows;</li>
              <li>the input table given to easyGWAS: one row per accession, 38 derived traits and two binary
                labels;</li>
              <li>the easyGWAS output: 40 candidate rows over 28 genes;</li>
              <li>accession metadata: identifier, name, country and coordinates;</li>
              <li>two summary tables prepared for a manuscript.</li>
            </ul>
            <p>Accessions are keyed by their 1001 Genomes identifiers.</p>
            <p>
              The per-plant sheet holds 1,841 plants, and every one of them is in a single cell of the design:
              <b> WCS417-inoculated, drought</b>. There are 243 accessions with 7 plants, three accessions (7394, 9807,
              9874) with 14 plants in two blocks, and Col-0 (6909) with 98 plants in 14 blocks, one of which is empty.
              Mock plants and non-stress plants appear only as per-accession averages and medians inside the derived
              traits. Primary root length was recorded for 1,042 plants.
            </p>
            <p>
              The four repeatedly grown lines are excluded because their derived traits mix plant subsets (E1).
              Eight further accessions lack mock values. That leaves an analysis set of <b>235 accessions</b>.
            </p>
            <p>
              The accessions come from 30 countries, led by Spain (66), Germany (44), Russia (18) and France (17).
              They were not chosen for this experiment. They are the collection that was already bulked and
              available in the laboratory, originally assembled for a study of annual precipitation. That matters
              for how to read geographic patterns (§14), and it is a known reviewer concern.
            </p>
            {acc ? <CellMeansChart accessions={acc} /> : null}
            <p>
              The chart shows the three cell means recovered for every accession, sorted by the unstressed mock
              biomass. Three features stand out:
            </p>
            <ul className="ml-6 list-disc space-y-1">
              <li>drought is severe: the median mock plant loses 78% of its shoot biomass and 84% of its root
                biomass;</li>
              <li>inoculation recovers some of that loss in every accession;</li>
              <li>the amount recovered varies a great deal.</li>
            </ul>
            <p>
              Hovering an accession shows its three cell means and its drought loss. Those four numbers are all that
              any reading of rescue can use.
            </p>
          </Section>

          {/* ============================================================ audit */}
          <Section id="audit" kicker="Experiment E1" title="3 · Auditing the workbook">
            <p>
              Before comparing readings, one has to know what each reading is. The workbook stores results, not
              formulas. So we reverse-engineered the formulas by testing candidate definitions against all
              accessions at once, and kept only those that fit every accession to rounding error.
            </p>
            <Spec
              id="E1 — audit"
              purpose="Recover the unobserved cell means and the exact definition of every derived trait; find labelling problems."
              inputs="Workbook sheets: per-plant biomass, easyGWAS input, Table 1, Table 2."
              procedure={<>Write W for the mean of the recorded inoculated–drought plants. Solve the absolute-gain and loss traits for the mock means. Test the remaining traits as closed-form functions of the recovered means. Compare Table 1 and Table 2 with the recovered quantities.</>}
              output={<JsonLink name="E1_audit" />}
              acceptance="An identity is accepted if it holds to < 10⁻⁶ for every accession in the analysis set."
              result={<>All accepted. Rescue = (W − M_D)/(M_N − M_D) holds for all 235 accessions (median residual 4 × 10⁻¹⁰). % increase = W/M_D − 1 holds for 239 of 247 accessions; every failure is a repeatedly grown line.</>}
            />
            <p>The recovery rests on two identities in the workbook:</p>
            <Eq>M_D = W − gain_D ,&nbsp;&nbsp;&nbsp;&nbsp;M_N = M_D / (1 − loss)</Eq>
            <p>
              W is simply the mean of the recorded plants, matching the workbook to 4 × 10⁻⁹. So both mock means
              follow for every accession. Two further traits, not used in the recovery, then serve as independent
              checks, and both pass. The one trait that involves the inoculated non-stress plants, &ldquo;gain under
              non-stress&rdquo;, fits two different definitions equally well. That cell, W_N, therefore cannot be
              recovered, and nothing below uses it.
            </p>
            <p>The audit found three things the laboratory should know.</p>
            <ol className="ml-6 list-decimal space-y-2">
              <li><b>The loss trait contains no inoculated plants.</b> &ldquo;Percent loss under drought&rdquo; is
                1 − M_D/M_N, computed from mock plants only. It is a drought-sensitivity trait. Candidate genes found
                with it are drought genes, not rescue genes.</li>
              <li><b>Table 2 is mislabelled.</b> Its &ldquo;drought tolerance&rdquo; columns equal that loss for 239 of
                240 accessions. A higher value means <i>less</i> tolerant.</li>
              <li><b>The rescuer label is a threshold, not a loss of rescue.</b> Every accession gains biomass from
                inoculation. Labelled non-rescuers gain 0.73–7.75 mg shoot; labelled rescuers gain 6.1–25.6 mg. The
                label is a cut near 6–8 mg in a continuous distribution.</li>
            </ol>
            <p>
              Table 1 is the drought condition. Its mock and WCS417 columns match M_D and W with correlations of 0.96
              and 0.97, so the ΔSFW panels in the slide deck show the gain under drought.
            </p>
          </Section>

          {/* ============================================================ constructions */}
          <Section id="constructions" kicker="Theory" title="4 · What a phenotype is">
            <p>
              To talk precisely about &ldquo;readings&rdquo;, we define them. A <b>response construction</b> is any
              function f(W, M_D, M_N) with two properties:
            </p>
            <ul className="ml-6 list-disc space-y-1">
              <li><i>null calibration</i>: f is zero exactly when inoculation does nothing (W = M_D);</li>
              <li><i>monotonicity</i>: f increases with the inoculated biomass W.</li>
            </ul>
            <p>The three constructions in use all qualify:</p>
            <Eq>
              gain = W − M_D &nbsp;&nbsp;·&nbsp;&nbsp; % increase = W/M_D − 1 &nbsp;&nbsp;·&nbsp;&nbsp; rescue = (W − M_D)/(M_N − M_D)
            </Eq>
            <p>
              A construction is <b>scale invariant</b> if multiplying all of an accession&apos;s cells by the same
              factor leaves it unchanged. This encodes a biological requirement. Accessions differ in vigour, a
              vigorous accession is bigger in every cell, and vigour is not rescue. Gain fails the requirement: a
              plant twice as large gains twice as many milligrams for the same biology. Percentage increase and
              rescue both satisfy it.
            </p>
            <p>The two scale-invariant constructions are members of one family:</p>
            <Eq>f_λ = (W − M_D) / ( M_D^λ · (M_N − M_D)^(1−λ) )</Eq>
            <p>
              Here λ = 1 gives percentage increase and λ = 0 gives rescue. The denominator is the baseline the gain is
              measured against: the size of the stressed mock plant, the size of the drought loss, or a geometric
              mixture of the two. Every member of the family is scale invariant.
            </p>
            <p>
              The key observation is that <b>choosing λ is choosing a mechanism</b>. Suppose the gain in accession a
              is generated as θ_a · M_D^λ* · (M_N − M_D)^(1−λ*), with a rescue capacity θ_a that has nothing to do
              with the baselines. Then f at λ* recovers θ_a exactly. Any other λ equals θ_a multiplied by a power of
              M_D/(M_N − M_D), a monotone function of drought sensitivity. Two special cases make this concrete:
            </p>
            <ul className="ml-6 list-disc space-y-1">
              <li>percentage increase is the right reading only if WCS417 <i>multiplies the stressed biomass</i> by
                an accession-specific factor;</li>
              <li>rescue is right only if WCS417 <i>restores an accession-specific share of what drought
                removed</i>.</li>
            </ul>
            <p>
              The wrong choice loads the phenotype onto drought sensitivity, and the size of the error sets how
              strongly. Nothing in the word &ldquo;rescue&rdquo; decides between the mechanisms, but the data can:
              taking logarithms turns the mechanism into a linear model whose slope on log M_D estimates λ (§8).
            </p>
            {acc && D.e4 ? <LambdaExplorer accessions={acc} e4={D.e4} /> : null}
            <p>
              The explorer recomputes f_λ for every accession as you move the slider. The left panel shows how
              strongly f_λ is correlated with mock-only drought loss (solid) and with vigour (dashed); the shaded
              band is the range of λ the data allow. The right panel shows the ranking itself. Two positions are
              worth comparing:
            </p>
            <ul className="ml-6 list-disc space-y-1">
              <li>at λ = 1 the accessions line up along the diagonal of drought sensitivity, because % increase
                largely ranks how badly drought hurt the mock plant;</li>
              <li>near λ = 0 the cloud loses that structure.</li>
            </ul>
          </Section>

          {/* ============================================================ heterogeneity */}
          <Section id="heterogeneity" kicker="Experiment E2" title="5 · One strain, many effects">
            <p>
              If WCS417 had one effect under drought, every accession&apos;s response would differ from every other
              only by noise. The first test is therefore whether the responses are more variable than their standard
              errors allow. A large excess means the effect of the strain is not a single value. It is indexed by
              accession: a property of the pair (strain, accession), not of the strain alone. That indexed set is the
              object a mapping study maps.
            </p>
            <Spec
              id="E2 — heterogeneity and detectability"
              purpose="Test whether the inoculation effect is heterogeneous; measure how reliable each construction is; ask whether a loss of rescue would have been visible."
              inputs="Recovered cell means; per-plant inoculated–drought weights; noise models M2–M4 (below)."
              procedure={<>Draw 1,000 bootstrap/perturbation replicates per accession and take the standard deviation of each construction as its standard error. Compute Cochran&apos;s Q, I² and the reliability τ²/(τ² + se²), where τ² is the between-accession variance net of noise. For the gain, compute 95% intervals and flag those that reach zero.</>}
              output={<JsonLink name="E2_heterogeneity" />}
              acceptance="Heterogeneity is declared if p < 0.001 under every noise model."
              result={<>Heterogeneity holds everywhere (shoot Q = 396–877 on 234 df under batch-inclusive noise). Reliability ranges from 0.35 (shoot rescue, + batch) to 0.95 (root % increase, plant noise). 18 of 235 shoot gain intervals include zero under batch-inclusive noise; 6 do under plant noise alone.</>}
            />
            <Callout title="The noise models">
              <p>
                Per-plant spread was measured only in the inoculated–drought cell. There, the median
                within-accession coefficient of variation is 0.121 (shoot) and 0.156 (root). We assume the same
                relative spread in the other three cells. Batch is estimated from the Col-0 blocks (§6) as a
                coefficient of variation of 0.081 (shoot) and 0.093 (root) per cell.
              </p>
              <ul className="ml-5 list-disc space-y-1">
                <li><b>M1</b> resamples the observed inoculated plants and holds the mock means fixed.</li>
                <li><b>M2</b> also perturbs the mock means with plant-level noise.</li>
                <li><b>M3</b>, the reference model, also adds an independent batch factor to every cell.</li>
                <li><b>M4</b> is M3 with mock noise multiplied by 1.5.</li>
              </ul>
              <p>
                M3 is pessimistic: if mock and inoculated plants shared trays, part of the batch effect would cancel
                in the gain. So M2 and M3 bracket the truth.
              </p>
            </Callout>
            {D.e2 ? <ReliabilityBars e2={D.e2} /> : null}
            <p>
              The chart shows reliability for each construction under each noise model. Two points matter. First,
              the constructions differ sharply in reliability. Rescue is a ratio of two differences, so it
              accumulates noise from every cell. Under batch-inclusive noise its true between-accession spread is
              smaller than its per-accession standard error. Second, batch noise costs rescue far more than it
              costs the other two. That is not an argument against rescue, as §8 shows. It is an argument for a
              better experiment (§13).
            </p>
            {D.e2 && acc ? <GainCaterpillar e2={D.e2} accessions={acc} /> : null}
            <p>
              <b>Could a loss of rescue have been seen?</b> Mostly yes. The median shoot gain is 4.5 standard errors
              from zero, and 92% of intervals exclude zero (96% in root). The smallest gain detectable with 80% power
              is 5.8 mg shoot, against a median gain of about 9 mg. A complete loss of rescue in most genetic
              backgrounds would have stood out. The red intervals are the exception: accessions whose gain is
              compatible with zero. <b>Per-1</b> has the smallest gain in both organs (0.73 mg shoot, 0.27 mg root)
              and is compatible with zero under every noise model. It is the first line to retest as a candidate
              loss-of-rescue accession.
            </p>
          </Section>

          {/* ============================================================ batch */}
          <Section id="batch" kicker="Experiment E7" title="6 · The batch control already in the design">
            <p>
              Col-0 was sown about every 119 rows through the screen, 14 times in blocks of 7. The design therefore
              already contains a control that measures how much the growing conditions drifted from block to block.
              Because the genotype is the same in every block, any difference between Col-0 blocks beyond plant
              noise is batch.
            </p>
            <Spec
              id="E7 — batch"
              purpose="Estimate the batch component of noise from the internal control and compare it with the spread between accessions."
              inputs="Per-plant weights of Col-0 (6909), 13 usable blocks; the three lines grown twice."
              procedure="One-way ANOVA of Col-0 plants on block; intraclass correlation; method-of-moments batch coefficient of variation (between-block variance of means minus plant variance divided by block size)."
              output={<JsonLink name="E7_batch" />}
              acceptance="Batch is declared material if the ANOVA rejects at p < 0.01 and the between-block SD is at least a quarter of the between-accession SD."
              result="Material in both organs. Shoot: F = 3.76, p = 1.7 × 10⁻⁴, ICC = 0.28; block SD is 0.43 of the between-accession SD. Root: F = 3.12, p = 1.2 × 10⁻³, ICC = 0.23; ratio 0.53."
            />
            {D.e7 && acc ? <BatchBlocks e7={D.e7} accessions={acc} /> : null}
            <p>
              The same genotype, grown in different blocks of the same experiment, moves from the 54th to the 92nd
              percentile of all accessions in shoot, and from the 71st to the 99th in root. The three lines grown
              twice agree: their two block means differ by 19–42%. Line 9807 had 27.7 mg in one block and 19.5 mg in
              the other.
            </p>
            <p>
              The screen&apos;s analysis has no block term, so every accession&apos;s value carries the offset of the
              block it happened to be in. The fix is already paid for: with block membership, the Col-0 means give
              the block offsets for the inoculated–drought cell directly.
            </p>
          </Section>

          {/* ============================================================ dependence */}
          <Section id="dependence" kicker="Experiment E3" title="7 · Do the readings agree?">
            <p>
              Two readings can disagree because they measure different things, or because each is noisy. Only the
              first is a property of the readings. We call a verdict <b>construction independent</b> if the
              disagreement between two readings is no larger than the disagreement between one reading and a noisy
              replicate of itself.
            </p>
            <Spec
              id="E3 — construction dependence"
              purpose="Measure how much the three readings in use disagree, and compare that with resampling noise."
              inputs="Gain, % increase and rescue for 235 accessions per organ; noise models M1–M4."
              procedure="Spearman correlations among the constructions and three nuisance quantities (size under inoculation, vigour, mock drought loss). Top-k sets at k = 10, 20, 30% and their Jaccard overlap and turnover. Noise ceiling: the 95th percentile of the turnover between a construction and 300 resampled versions of itself."
              output={<JsonLink name="E3_dependence" />}
              acceptance="Construction independence is declared if the between-construction turnover is at or below the noise ceiling."
              result="Rejected in root under every noise model (turnover 0.57–0.70 against ceilings 0.36–0.40). In shoot rejected under plant noise, and exceeding the batch-inclusive ceiling by only 0.00–0.09."
            />
            <p>
              The correlations explain the disagreement. Each reading carries a different contaminant, as the
              mechanism argument of §4 predicts:
            </p>
            <ul className="ml-6 list-disc space-y-2">
              <li><b>% increase</b> ranks accessions almost exactly as mock-only drought loss does (ρ = 0.80 in both
                organs). A drought-sensitive accession has a small M_D, and dividing by a small number gives a large
                ratio. This is the spurious correlation of ratios described by Pearson in 1897, reappearing as a
                phenotype.</li>
              <li><b>Gain</b> in milligrams tracks plant size (ρ = 0.60 with the inoculated biomass).</li>
              <li><b>Rescue</b> is the least loaded on either, but, as E2 showed, the noisiest.</li>
            </ul>
            {acc && D.e3 ? <TopKExplorer accessions={acc} e3={D.e3} /> : null}
            <p>
              At the top 20%, only 15 of 47 shoot accessions and 7 of 47 root accessions are in the top set under all
              three readings. At the top 10%, no root accession is. Move the slider: the overlap improves as k grows,
              because large top sets overlap by construction, but it never becomes close. The strip plot shows where
              the disagreement comes from. Many accessions that % increase places at the top sit in the middle of
              the rescue ranking.
            </p>
            <Callout title="Stronger than disagreement: contradiction">
              <p>
                Disagreeing about which accessions are &ldquo;top&rdquo; is one thing. Asserting incompatible facts
                is another. Call a pair <i>certified</i> in order a &gt; b by a construction if the difference exceeds
                the margin by more than 1.645 standard errors. Under plant noise alone, % increase and the canonical
                reading of §8 certify <b>433 shoot pairs and 700 root pairs in opposite order</b>. Under
                batch-inclusive noise, 9 and 62 such contradictions survive. Each one is two admissible readings of
                the same data stating, each at the 5% level, that a beats b and that b beats a.
              </p>
            </Callout>
          </Section>

          {/* ============================================================ canonical */}
          <Section id="canonical" kicker="Experiment E4" title="8 · Letting the data choose">
            <p>
              If readings disagree, the choice among them must be made by something other than habit. §4 showed that
              the choice is an exponent λ with a mechanistic meaning. So we estimate it.
            </p>
            <Spec
              id="E4 — canonical construction"
              purpose="Estimate λ from the data, correct the estimator's bias, and test whether restricting to the data-admissible range restores construction independence."
              inputs="Recovered cell means; noise model M3."
              procedure={<>Fit log(W − M_D) = b₀ + b₁ log M_D + b₂ log(M_N − M_D). Under the mechanism, b₁ = λ and b₁ + b₂ = 1. Bootstrap jointly over accessions and noise (1,000 replicates). Because M_D appears on both sides with error, the slope is biased; simulate 200 synthetic screens per true λ from −0.75 to 1 with the observed baselines and noise, record the mean estimate, and invert that curve.</>}
              output={<JsonLink name="E4_canonical" />}
              acceptance="A construction is data-admissible if its λ lies in the bias-corrected 95% interval. Construction independence is restored if the turnover between the ends of that interval is at or below the noise ceiling."
              result={lamS && lamR ? <>λ̂ = {lamS.lambda_hat_corrected.toFixed(2)} [{lamS.lambda_ci95_corrected[0].toFixed(2)}, {lamS.lambda_ci95_corrected[1].toFixed(2)}] shoot; {lamR.lambda_hat_corrected.toFixed(2)} [{lamR.lambda_ci95_corrected[0].toFixed(2)}, {lamR.lambda_ci95_corrected[1].toFixed(2)}] root. λ = 1 excluded in both. Turnover within the interval {lamS.within_admissible.flip_rate_top20.toFixed(2)} (shoot) and {lamR.within_admissible.flip_rate_top20.toFixed(2)} (root), below the ceilings of 0.43 and 0.40.</> : "…"}
            />
            <p>
              The uncorrected slopes are −0.34 (shoot) and −0.09 (root). Simulation at the observed noise level shows
              that the estimator underestimates λ by about 0.10 (shoot) and 0.04 (root) near λ = 0, and by more at
              larger λ, so we invert the calibration curve. After correction:
            </p>
            <ul className="ml-6 list-disc space-y-1">
              <li><b>Percentage increase is excluded</b> with a wide margin in both organs.</li>
              <li><b>Rescue</b> (λ = 0) lies inside the root interval and just beyond the upper end of the shoot
                interval.</li>
            </ul>
            <p>
              We adopt f at λ̂, per organ, as the <b>canonical</b> reading. For practical purposes it is rescue: the
              two never certify opposite orders, and their resolution tiers agree at ρ = 0.88–0.99.
            </p>
            <p>
              Biologically, this says that WCS417 behaves as though it restores a share of what drought removed,
              rather than multiplying whatever biomass the stressed plant has. That is a small result about the
              interaction in its own right. It also has a methodological consequence: the reading that produced
              twelve of the 28 candidate genes is the one the data reject.
            </p>
            <Callout title="A caveat the data insist on">
              <p>
                The mechanism also predicts b₁ + b₂ = 1: the gain should scale in exact proportion to plant size.
                Allowing for the same attenuation, simulation expects 0.81 ± 0.17 (shoot) and 0.86 ± 0.12 (root).
                Shoot is compatible (0.61, 1.2 SD below). Root is not (0.48, 3.2 SD below): larger root systems gain
                proportionally less.
              </p>
              <p>
                No scale-invariant reading therefore fully removes vigour in root. At λ ≈ 0 the root phenotype still
                correlates at −0.34 with M_N. In root, vigour belongs in the association model as a covariate, not in
                the phenotype.
              </p>
            </Callout>
            <p>
              The last line of the specification is the important one. Across the range of λ the data allow, the
              top-20% turnover is 0.32 (shoot) and 0.30 (root), below the noise ceilings. Between % increase and
              rescue it is 0.51 and 0.70, above them. <b>Once the data choose the reading, the remaining freedom no
              longer changes the verdict more than noise does.</b> Construction dependence was a symptom of an
              unexamined choice, not an irreducible feature of the screen.
            </p>
          </Section>

          {/* ============================================================ four column */}
          <Section id="fourcolumn" kicker="Experiments E5 and E6" title="9 · Four columns and three verdicts">
            <p>
              Ranking accessions by one number discards what they share. Two accessions can have the same rescue for
              different reasons, or different rescue from indistinguishable starting points. So we compare every
              pair of accessions through <b>four columns</b>:
            </p>
            <ul className="ml-6 list-disc space-y-1">
              <li>two <i>baseline</i> columns: each accession&apos;s mock phenotype, (log M_D, log M_N), which
                describes the plant without the bacterium;</li>
              <li>two <i>response</i> columns: each accession&apos;s canonical rescue.</li>
            </ul>
            <p>Each pair of columns receives one of three verdicts.</p>
            <Eq>correspond &nbsp;if&nbsp; |d| + 1.645·s &lt; δ &nbsp;&nbsp;·&nbsp;&nbsp; diverge &nbsp;if&nbsp; |d| − 1.645·s &gt; δ &nbsp;&nbsp;·&nbsp;&nbsp; decline &nbsp;otherwise</Eq>
            <p>
              Here d is the observed difference, s its standard error and δ the margin that counts as
              &ldquo;the same&rdquo;. <i>Correspond</i> is the two one-sided tests rule for equivalence;
              <i> diverge</i> is its mirror image. <i>Decline</i> is not a failure. It is the correct report when
              the data do not settle the question. A method that must return a score for every pair cannot say
              that, and returns a confident-looking number where the honest answer is &ldquo;cannot tell&rdquo;.
            </p>
            <p>Two cells of the resulting 3 × 3 table carry the diagnostic content:</p>
            <ul className="ml-6 list-disc space-y-2">
              <li><b>False friends</b> (baseline correspond, response diverge): accessions indistinguishable without
                the bacterium but different with it. These are the most informative contrasts for mapping, because
                their difference in rescue cannot be blamed on growth or drought sensitivity.</li>
              <li><b>Convergent pairs</b> (baseline diverge, response correspond): different plants, same benefit.</li>
            </ul>
            <Spec
              id="E6 — calibration"
              purpose="Verify that the three-valued verdict has the error rates it claims, including at this screen's noise level."
              inputs="Pair standard error relative to margin, s/δ ∈ {0.1, 0.25, 0.5, 1, 2.0}; the last is the empirical level of this screen."
              procedure="Simulate 40,000 differences per true difference Δ/δ ∈ [0, 3] and s/δ level; record P(correspond), P(diverge), P(decline)."
              output={<JsonLink name="E6_calibration" />}
              acceptance="P(correspond) ≤ α when |Δ| ≥ δ; P(diverge) ≤ α + Φ(−2δ/s − 1.645) when |Δ| ≤ δ (the analytic bound)."
              result="Worst false-correspond 0.051 (Monte Carlo error 0.001); worst false-diverge 0.054, equal to the analytic bound at s/δ = 2. Accepted."
            />
            {D.e6 ? <CalibrationCurves e6={D.e6} /> : null}
            <p>
              The calibration also shows the problem. At this screen&apos;s ratio of standard error to margin (2.0 at a
              margin of half a between-accession SD), <i>correspond</i> is essentially unreachable, whatever the
              truth.
            </p>
            <Spec
              id="E5 — four-column verdicts"
              purpose="Apply the four-column comparison to all 27,495 pairs; find the smallest margin at which each pair could be certified equivalent; count decisive contradictions between constructions."
              inputs="Baselines (log M_D, log M_N) and responses (gain, % increase, rescue, canonical) with standard errors under M2 and M3."
              procedure="Three-valued verdicts at baseline margins of 15/30/50% and response margins of 0.25–2 SD; per-pair minimum certifiable margin |d| + 1.645·s; certified orders and opposite orders between constructions."
              output={<JsonLink name="E5_four_column" />}
              acceptance="Descriptive; the design-level summary is the distribution of the minimum certifiable margin."
              result="Median pair certifiable only at 2.75 SD (shoot) and 2.41 SD (root) under M3; 1.80 and 1.69 SD under plant noise alone. Under plant noise, 818 shoot false friends and 20 convergent pairs."
            />
            {acc && D.index && D.e5 ? <VerdictExplorer accessions={acc} index={D.index} e5={D.e5} /> : null}
            <p>
              The explorer recomputes all 27,495 pair verdicts as you move the margins. The left panel is the most
              useful single summary of a screen like this: the <b>minimum certifiable margin</b>, the smallest margin
              at which each pair could be declared equivalent. With batch-inclusive noise the median pair needs a
              margin of about 2.5 between-accession SDs, wider than the spread of the whole population. Even with
              plant noise alone (dashed) only 14–16% of pairs certify below 1 SD. <b>The screen can say that two
              accessions differ. It cannot say that two accessions rescue equally.</b>
            </p>
            <p>
              The right panel is the four-column table. Its outlined cells are false friends and convergent pairs.
              Widen the baseline margin and false friends appear, because more pairs certify as baseline-equivalent.
              Narrow the response margin and they turn into declines. Under plant noise the most extreme false
              friends in shoot include IP-Mur-0 with Lu3-30 (canonical rescue 0.05 against 0.31) and Neo-6 with Uk-1.
              These are good pairs for a follow-up with more replication.
            </p>
          </Section>

          {/* ============================================================ tiers */}
          <Section id="tiers" kicker="Experiment E10" title="10 · Resolution depth">
            <p>
              A ranking of 235 accessions suggests 235 distinguishable levels. The data support far fewer. We define
              the <b>resolution depth</b> as the length of the longest chain of accessions in which each is certified
              greater than the next. The <b>tier</b> of an accession is the length of the longest such chain ending at
              it from below.
            </p>
            <Spec
              id="E10 — resolution depth"
              purpose="Count how many levels of rescue the screen can tell apart, and compare tier assignments across constructions."
              inputs="Each construction's values and standard errors under M2 and M3."
              procedure="Sort by value; dynamic programming over the certified order v_a − v_b > 1.645·√(s_a² + s_b²)."
              output={<JsonLink name="E10_classes" />}
              acceptance="Descriptive."
              result={D.e10 ? `Canonical: ${D.e10.Shoot.M3_plant_plus_batch.max_depth.canonical} tiers (shoot) and ${D.e10.Root.M3_plant_plus_batch.max_depth.canonical} (root) with batch; ${D.e10.Shoot.M2_plant_all_cells.max_depth.canonical} and ${D.e10.Root.M2_plant_all_cells.max_depth.canonical} with plant noise only.` : "…"}
            />
            {acc && D.e10 ? <TierChart accessions={acc} e10={D.e10} /> : null}
            <p>
              Read the screen as sorting accessions into a handful of classes, not into 235 ranks. Tier assignments
              of the readings in use agree with the canonical tiers only moderately (ρ = 0.44–0.82), least for
              % increase. A follow-up should sample across tiers, not across ranks.
            </p>
          </Section>

          {/* ============================================================ candidates */}
          <Section id="candidates" kicker="Experiment E8" title="11 · Triage of the candidate list">
            <p>
              The easyGWAS output lists 40 rows covering 28 genes, each tagged with the trait (&ldquo;metric&rdquo;)
              it was found with. E1 tells us what each trait is. So the candidate list can be sorted by what each
              locus is a locus <i>for</i>.
            </p>
            <Spec
              id="E8 — candidates"
              purpose="Assign every candidate to the trait family it came from, and measure each family's loading on the mock-only drought loss."
              inputs="easyGWAS output table; trait columns from the easyGWAS input."
              procedure="Classify metrics into five families (rescue, % increase, gain, size under WCS417, mock-only loss); Spearman correlation of each metric with the mock-only loss; count genes recurring across metrics and across families."
              output={<JsonLink name="E8_candidates" />}
              acceptance="Descriptive."
              result="28 genes; 7 recur across metrics, 0 across families. By family: rescue 9, % increase 12, gain 1, size 3, mock-only 3."
            />
            {D.e8 ? <CandidateChart e8={D.e8} /> : null}
            <p>The seven genes that recur across metrics recur only across near-duplicates of one trait: average and median, or shoot and total (which contains shoot). The triage is then:</p>
            <ul className="ml-6 list-disc space-y-2">
              <li><b>Mock-only (3 genes):</b> MORN4 (AT1G77660), the antisense transcript AT2G07213 and TPS21
                (AT5G23960). They were found with a trait computed without inoculated plants, so they are candidates
                for drought sensitivity.</li>
              <li><b>% increase (12 genes):</b> found with traits whose rankings correlate at 0.78–0.81 with drought
                loss, so they are at risk of being drought-sensitivity loci.</li>
              <li><b>Rescue (9 genes):</b> the most interpretable candidates for the mutant screen.</li>
              <li><b>Size and gain (4 genes):</b> these carry the size loading.</li>
            </ul>
            <p>None of this says whether a locus is real. It says which phenotype it would be real for.</p>
          </Section>

          {/* ============================================================ mutants */}
          <Section id="mutants" kicker="Experiment E9" title="12 · Checking a mutant claim">
            <p>
              The experimenter&apos;s transcriptome points to iron: drought-induced genes that the bacterium reverses
              are enriched for iron responses, and inoculated plants contain more iron. Mutants in plant iron-uptake
              genes reportedly show <i>higher</i> rescue. That fits a model in which the bacterium supplies iron
              independently of the plant&apos;s own uptake. Before building on it, one check is needed, and it is the
              same issue as the GWAS traits.
            </p>
            <Spec
              id="E9 — baseline-only mutant"
              purpose="Show what each reading reports for a mutant that only grows worse under drought and receives exactly the same absolute benefit."
              inputs="Recovered shoot cell means of the 235 accessions as wild-type backgrounds."
              procedure="Reduce M_D by a fraction ε ∈ [0, 0.5], keep the absolute gain and M_N fixed, and report each construction's mutant/wild-type ratio (median and interquartile range over backgrounds)."
              output={<JsonLink name="E9_ratio" />}
              acceptance="Descriptive."
              result="At ε = 0.5: % increase 2.00×, gain 1.00×, rescue 0.88×, canonical 0.71×."
            />
            {D.e9 ? <RatioArtefact e9={D.e9} /> : null}
            <p>
              A mutant that is simply sicker under drought, with <i>no</i> change in how much the bacterium helps it,
              looks up to twice as well rescued by % increase. The other readings report no change or a decrease.
              The claim of enhanced rescue in the iron mutants is established if the absolute gain and the
              loss-normalised rescue both rise. It is an artefact of a weaker baseline if only % increase does.
              This check needs no new experiment, only the numbers already collected.
            </p>
          </Section>

          {/* ============================================================ design */}
          <Section id="design" kicker="Experiment E12" title="13 · The next experiment">
            <p>
              The single most useful thing the reanalysis can offer is a design for the next screen. E2 and E7
              identify batch, not plant number, as the dominant noise. E12 turns that into numbers.
            </p>
            <Spec
              id="E12 — replication design"
              purpose="Predict the reliability and certifiability of canonical rescue under alternative designs."
              inputs="Plant and batch coefficients of variation; recovered cell means; the canonical λ."
              procedure="For n ∈ {7, 14, 21, 28} plants per cell and r ∈ {1, 2, 3, 4, 6} independent runs (each with its own batch effect), set the relative standard error of a cell mean to √(c_p²/(n·r) + c_b²/r). Simulate standard errors, and compute reliability and the probabilities of certifying equality at 0.5τ and 1τ and a difference at 1τ and 2τ."
              output={<JsonLink name="E12_design" />}
              acceptance="Descriptive; the design question is how reliability and certifiability grow with n versus r."
              result="Shoot reliability 0.34 now; 0.36 with twice the plants; 0.50 with two runs, 0.61 with three. Root: 0.57 → 0.80 with three runs. Equality at 1τ becomes certifiable (P = 0.49) only for root at 28 plants × 6 runs."
            />
            {D.e12 ? <DesignHeatmap e12={D.e12} /> : null}
            <p>Three recommendations follow directly.</p>
            <ol className="ml-6 list-decimal space-y-2">
              <li><b>Repeat in independent runs rather than adding plants.</b> A second run is worth more than
                quadrupling the plants in one run.</li>
              <li><b>Grow each accession&apos;s mock and inoculated plants together</b>, on the same tray and in the
                same block, so that batch cancels in the gain.</li>
              <li><b>Keep a common control in every block and record block membership</b>, then enter block as a
                term in the model.</li>
            </ol>
          </Section>

          {/* ============================================================ geography */}
          <Section id="geography" kicker="Experiment E11" title="14 · Collection sites">
            <Spec
              id="E11 — geography"
              purpose="Describe how rescue, drought loss and vigour vary with collection site."
              inputs="Latitude and longitude of the 235 accessions."
              procedure="Spearman correlations of five quantities with latitude and longitude, per organ (20 tests); Bonferroni threshold 0.0025."
              output={<JsonLink name="E11_geography" />}
              acceptance="Associations are reported as descriptive; only those passing the Bonferroni threshold are highlighted."
              result="Two survive: canonical shoot rescue falls with longitude (ρ = −0.25, p = 1.2 × 10⁻⁴), and root vigour rises with longitude (ρ = 0.20, p = 0.0024). Mock drought loss shows no geographic pattern in shoot."
            />
            {acc && D.e11 ? <GeoMap accessions={acc} e11={D.e11} /> : null}
            <p>
              Because the accessions were not sampled to represent the species, these patterns are descriptive. The
              proper next step is a climate covariate taken from the collection coordinates, such as WorldClim
              precipitation. The coordinates are already in the workbook, so this needs no further information from
              the laboratory. It turns the non-random composition of the panel from a reviewer&apos;s objection into a
              measured variable.
            </p>
          </Section>

          {/* ============================================================ protocol */}
          <Section id="protocol" kicker="Specification" title="15 · Protocol for the reanalysis">
            <p>
              What follows is the procedure the data now support, written as steps. Steps 1–3 need no new data. Steps
              4–6 need the per-plant mock and non-stress weights, which exist in the laboratory but not in the
              workbook, and the genotypes.
            </p>
            <ol className="ml-6 list-decimal space-y-3">
              <li><b>Fix the phenotype before mapping.</b> Use the canonical reading per organ: f at λ̂, which for
                practical purposes is loss-normalised rescue (W − M_D)/(M_N − M_D). Report λ̂ and its interval.</li>
              <li><b>Retire the mock-only trait from the rescue analysis</b> and correct the Table 2 label.</li>
              <li><b>Triage the existing candidates</b> by trait family (§11). Prioritise the rescue-derived ones for
                mutants, and treat % increase and mock-only candidates as drought-sensitivity candidates.</li>
              <li><b>Fit one model on per-plant data from all four cells</b>, with the inoculation × drought
                interaction as the phenotype and block as a term, estimated from the repeated Col-0.</li>
              <li><b>Map with kinship</b> (a mixed model), and in root add log M_N as a covariate.</li>
              <li><b>Report certifiability with any ranking</b>: the resolution depth, the minimum certifiable margin,
                and the fraction of declined pairs.</li>
              <li><b>Retest Per-1</b> and the accessions whose gain is compatible with zero, as loss-of-rescue
                candidates.</li>
              <li><b>For the iron mutants</b>, report absolute gain and loss-normalised rescue next to % increase.</li>
              <li><b>For the next screen</b>, use independent runs, co-located mock and inoculated plants, and a
                common control per block.</li>
            </ol>
            <Callout title="Beyond this reanalysis">
              <p>
                The absence of loss-of-rescue lines, together with 50 transcriptome candidates none of which
                abolishes rescue alone, suggests redundancy: rescue runs through several routes. If so, the
                informative next experiment is combinatorial. In a network built from the time-resolved
                transcriptome, find the smallest set of genes whose joint removal disconnects the bacterial signal
                from the growth outcome, and test that set as a higher-order mutant. That analysis needs the
                transcriptome and has not been done here.
              </p>
            </Callout>
          </Section>

          {/* ============================================================ limits */}
          <Section id="limits" kicker="Honesty" title="16 · Limitations">
            <ol className="ml-6 list-decimal space-y-2">
              <li><b>Three cells have modelled, not measured, noise.</b> M4 (mock noise × 1.5) changes no
                qualitative conclusion, but per-plant data would replace the assumption with a measurement.</li>
              <li><b>Batch is estimated from one cell and one genotype.</b> Batch effects are assumed independent
                across cells; shared trays would make the batch-inclusive model pessimistic.</li>
              <li><b>No genotypes.</b> No association was rerun, corrected for structure, or tested.</li>
              <li><b>One run.</b> A second run could sharpen every conclusion about construction dependence, and in
                shoot could reverse it.</li>
              <li><b>The λ family is a model.</b> The root rejection of b₁ + b₂ = 1 shows that it is incomplete there.
                The bias correction assumes the noise model.</li>
              <li><b>Margins are choices.</b> The four-column table depends on δ. The minimum certifiable margin does
                not.</li>
              <li><b>W_N cannot be recovered</b>, so no reading that uses the inoculated non-stress plants can be
                evaluated.</li>
            </ol>
          </Section>

          {/* ============================================================ files */}
          <Section id="files" kicker="Reproduction" title="17 · Files and reproduction">
            <p>
              The validation suite lives in <Code>collaboration/arabidopsis-gwas/validation/</Code>. Running
              <Code> python experiments.py</Code> regenerates every result from the workbook in about ten seconds,
              deterministically from seed 417. <Code>python make_figures.py</Code> rebuilds the six manuscript panels
              from the JSON, and <Code>python export_site.py</Code> copies the results to this page. Every chart here
              reads these files:
            </p>
            <ul className="grid grid-cols-3 gap-2 lg:grid-cols-2 sm:grid-cols-1">
              {Object.values(FILES).map((f) => <li key={f}><JsonLink name={f} /></li>)}
            </ul>
            {D.index ? (
              <p className="text-sm text-dark/70 dark:text-light/70">
                Last run: {D.index.n_main} accessions, bootstrap B = {D.index.bootstrap_B}, reference noise model{" "}
                {D.index.reference_model}, total {D.index.total_seconds} s.
              </p>
            ) : null}
          </Section>
        </Layout>
      </main>
    </>
  );
}
