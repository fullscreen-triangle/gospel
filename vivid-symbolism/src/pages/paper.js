import Head from "next/head";
import Link from "next/link";

import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";

const PAPER_TITLE = "Shader-Based Genomic Homology Search via Spectral Coordinate Embeddings";

const paperLinks = [
  {
    label: "PDF (paper)",
    description: "The full manuscript with method, experiments and discussion.",
    href: "/paper/shader-based-genomic-homology-search.pdf",
    external: false,
  },
  {
    label: "Source repository",
    description: "All code, experiment scripts, raw JSON results and figure generators.",
    href: "https://github.com/fullscreen-triangle/gospel",
    external: true,
  },
  {
    label: "Bibliography (.bib)",
    description: "References cited by the paper, in BibTeX format.",
    href: "/paper/references.bib",
    external: false,
  },
];

const experiments = [
  {
    id: "exp_01",
    title: "Embedding separation vs. substitution rate",
    summary:
      "ROC-AUC of within-family vs. between-family cosine similarities over eight substitution rates. AUC = 1.000 for mu <= 0.10; >= 0.97 through mu = 0.20.",
    file: "/paper/results/exp_01_embedding_separation.json",
  },
  {
    id: "exp_02",
    title: "Ranking correlation with Smith-Waterman and k-mer Jaccard",
    summary:
      "Spearman rho and recall@K against both exact local alignment and exhaustive k-mer Jaccard baselines. Recall@5 is 0.93 to 1.00 across all tested conditions.",
    file: "/paper/results/exp_02_ranking_correlation.json",
  },
  {
    id: "exp_03",
    title: "Scaling: 10^2 to 10^5 sequences",
    summary:
      "Per-query wall-clock time for the shader kernel vs. Jaccard and extrapolated Smith-Waterman. 1.1 ms at N = 10^5. Speedup 13,319x over Jaccard, 8x10^6 over SW.",
    file: "/paper/results/exp_03_scaling.json",
  },
  {
    id: "exp_04",
    title: "Hierarchical prefix addressing",
    summary:
      "Recall-vs-filter Pareto for 3D random-projection addressing with Hamming-1 expansion. Honest negative finding: the 3D projection is too low-rank to serve as a strong prefilter at demo scale.",
    file: "/paper/results/exp_04_prefix_addressing.json",
  },
  {
    id: "exp_05",
    title: "End-to-end pipeline with Smith-Waterman re-rank",
    summary:
      "Full retrieval pipeline on a 1000-sequence protein benchmark. Shader first stage + top-20 SW rerank recovers 95% of true family members.",
    file: "/paper/results/exp_05_end_to_end.json",
  },
];

const figures = [
  { src: "/paper/figures/fig_06_method.png", caption: "Fig. 1. Spectral coordinate embedding, from channelised signal to clustered coordinate space." },
  { src: "/paper/figures/fig_01_separation.png", caption: "Fig. 2. Homolog vs. non-homolog separation across the twilight zone." },
  { src: "/paper/figures/fig_02_recall.png", caption: "Fig. 3. Top-K retrieval against Smith-Waterman and k-mer Jaccard." },
  { src: "/paper/figures/fig_03_scaling.png", caption: "Fig. 4. Per-query cost and speedup vs. database size." },
  { src: "/paper/figures/fig_04_prefix.png", caption: "Fig. 5. Hierarchical prefix addressing: recall vs. filter trade-off." },
  { src: "/paper/figures/fig_05_pareto.png", caption: "Fig. 6. Speed vs. recall Pareto of the end-to-end pipeline." },
];

export default function Paper() {
  return (
    <>
      <Head>
        <title>Paper &middot; Gospel Homology Search</title>
        <meta
          name="description"
          content="Paper, validation results and figures for the shader-based genomic homology search framework."
        />
      </Head>

      <TransitionEffect />
      <main className="mb-16 flex w-full flex-col items-center justify-center dark:text-light">
        <Layout className="pt-16">
          <AnimatedText
            text="Paper & Validation"
            className="mb-8 !text-7xl !leading-tight lg:!text-6xl sm:!text-5xl xs:!text-4xl"
          />

          <p className="mb-8 max-w-3xl text-base font-medium dark:text-light md:text-sm">
            {PAPER_TITLE}
          </p>

          <section className="grid w-full grid-cols-3 gap-6 lg:grid-cols-1">
            {paperLinks.map((p) => (
              <Link
                key={p.label}
                href={p.href}
                target={p.external ? "_blank" : "_self"}
                className="flex flex-col rounded-xl border-2 border-dark p-5 transition hover:border-primary
                  dark:border-light dark:hover:border-primaryDark"
              >
                <span className="text-lg font-bold">{p.label}</span>
                <span className="mt-2 text-sm font-medium dark:text-light/90">
                  {p.description}
                </span>
              </Link>
            ))}
          </section>

          <h2 className="mt-16 text-3xl font-bold">Validation experiments</h2>
          <p className="mt-2 max-w-3xl text-base font-medium dark:text-light md:text-sm">
            Each experiment is backed by a JSON file generated by a single
            Python script, released with the paper. All numbers shown on this
            site are read directly from these files.
          </p>

          <ul className="mt-6 space-y-4">
            {experiments.map((e) => (
              <li
                key={e.id}
                className="rounded-xl border-2 border-dark p-5 dark:border-light"
              >
                <div className="flex flex-wrap items-baseline justify-between gap-4">
                  <h3 className="text-xl font-bold">{e.title}</h3>
                  <Link
                    href={e.file}
                    target="_blank"
                    className="font-mono text-sm font-semibold text-primary underline-offset-4
                      hover:underline dark:text-primaryDark"
                  >
                    {e.id}.json
                  </Link>
                </div>
                <p className="mt-2 text-sm font-medium dark:text-light/90">
                  {e.summary}
                </p>
              </li>
            ))}
          </ul>

          <h2 className="mt-16 text-3xl font-bold">Figures</h2>
          <p className="mt-2 max-w-3xl text-base font-medium dark:text-light md:text-sm">
            Figures are rebuilt from the JSON files by{" "}
            <code className="font-mono text-sm">experiments/make_figures.py</code> in the source repository.
          </p>

          <div className="mt-6 grid grid-cols-2 gap-6 md:grid-cols-1">
            {figures.map((f) => (
              <figure
                key={f.src}
                className="rounded-xl border-2 border-dark p-4 dark:border-light"
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={f.src}
                  alt={f.caption}
                  className="h-auto w-full rounded"
                />
                <figcaption className="mt-3 text-sm font-medium dark:text-light/90">
                  {f.caption}
                </figcaption>
              </figure>
            ))}
          </div>
        </Layout>
      </main>
    </>
  );
}
