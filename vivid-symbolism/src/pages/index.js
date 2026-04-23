import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";
import Head from "next/head";
import Link from "next/link";

const cards = [
  {
    title: "Spectral coordinate embedding",
    body: "A sequence is mapped to a fixed-dimensional vector by taking the lowest-frequency magnitudes of the discrete Fourier transform of its channelised representation. Substitutions perturb the signal locally but preserve the low-frequency envelope; that envelope is the embedding.",
  },
  {
    title: "Fragment-shader similarity kernel",
    body: "First-pass retrieval reduces to one dot product per database entry. The operation is the native per-pixel compute of the GPU rendering pipeline, so the same code runs on desktop, laptop, phone, or inside a browser tab, with no driver install.",
  },
  {
    title: "Two-stage pipeline",
    body: "The shader kernel ranks the whole database in milliseconds; a conventional local aligner re-ranks the top-K only. The filter stage is 10^3 to 10^4 times faster than exhaustive k-mer Jaccard; the re-rank preserves alignment score ordering.",
  },
  {
    title: "Reproducible and open",
    body: "Every claim on this site is backed by a released JSON file in the paper directory and a Python script that generated it. The JavaScript embedding implemented here is a direct port of the reference Python and produces matching coordinates to numerical precision.",
  },
];

function Metric({ label, value, sub }) {
  return (
    <div className="flex flex-col rounded-xl border-2 border-dark p-6 dark:border-light">
      <span className="text-sm font-semibold uppercase tracking-wide text-primary dark:text-primaryDark">
        {label}
      </span>
      <span className="mt-2 font-mono text-3xl font-bold dark:text-light xs:text-2xl">
        {value}
      </span>
      <span className="mt-1 text-sm font-medium dark:text-light/80">{sub}</span>
    </div>
  );
}

export default function Home() {
  return (
    <>
      <Head>
        <title>Gospel &middot; Shader-Based Genomic Homology Search</title>
        <meta
          name="description"
          content="Shader-based genomic homology search via a spectral coordinate embedding. Paste a DNA or protein sequence; the search runs entirely in your browser."
        />
      </Head>

      <TransitionEffect />
      <article className="flex min-h-screen items-center text-dark dark:text-light sm:items-start">
        <Layout className="!pt-0 md:!pt-16 sm:!pt-16">
          <div className="flex w-full flex-col items-start">
            <AnimatedText
              text="Shader-based genomic homology search."
              className="!text-left !text-7xl !leading-tight xl:!text-6xl lg:!text-center lg:!text-5xl md:!text-4xl sm:!text-3xl"
            />

            <p className="mt-6 max-w-3xl text-base font-medium md:text-sm sm:!text-xs">
              A sequence becomes a short coordinate vector. The rendering
              pipeline of any modern GPU evaluates cosine similarity against
              every entry in a database in a single pass. Local alignment is
              kept only as a re-ranker on the top-K candidates. The whole
              pipeline runs client-side.
            </p>

            <div className="mt-6 flex flex-wrap items-center gap-4">
              <Link
                href="/search"
                className="flex items-center rounded-lg border-2 border-solid bg-dark p-2.5 px-6 text-lg font-semibold
                  capitalize text-light hover:border-dark hover:bg-transparent hover:text-dark
                  dark:bg-light dark:text-dark dark:hover:border-light dark:hover:bg-dark dark:hover:text-light
                  md:p-2 md:px-4 md:text-base"
              >
                Run a search
              </Link>
              <Link
                href="/method"
                className="text-lg font-medium capitalize text-dark underline dark:text-light md:text-base"
              >
                How it works
              </Link>
              <Link
                href="/paper"
                className="text-lg font-medium capitalize text-dark underline dark:text-light md:text-base"
              >
                The paper
              </Link>
            </div>

            <div className="mt-14 grid w-full grid-cols-3 gap-6 lg:grid-cols-2 sm:grid-cols-1">
              <Metric
                label="per-query time"
                value="1.1 ms"
                sub="on a 10^5-sequence database, single-thread CPU reference"
              />
              <Metric
                label="speedup vs Jaccard"
                value="13,319×"
                sub="at N = 10^5, exhaustive k-mer Jaccard baseline"
              />
              <Metric
                label="retained accuracy"
                value="AUC ≥ 0.97"
                sub="through 20% substitution rate (twilight zone)"
              />
            </div>

            <div className="mt-14 grid w-full grid-cols-2 gap-6 md:grid-cols-1">
              {cards.map((c) => (
                <div
                  key={c.title}
                  className="rounded-xl border-2 border-dark p-6 dark:border-light"
                >
                  <h2 className="text-2xl font-bold dark:text-light">
                    {c.title}
                  </h2>
                  <p className="mt-3 text-base font-medium dark:text-light/90 md:text-sm">
                    {c.body}
                  </p>
                </div>
              ))}
            </div>

            <p className="mt-12 max-w-3xl text-sm font-medium text-dark/80 dark:text-light/80">
              The framework is positioned as a first-stage filter, not a
              replacement for local alignment. Its value is that it runs
              anywhere the rendering pipeline exists, including a web browser,
              with no server round-trip. Numbers are reproducible from the
              scripts in the source repository.
            </p>
          </div>
        </Layout>
      </article>
    </>
  );
}
