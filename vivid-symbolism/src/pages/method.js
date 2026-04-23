import Head from "next/head";

import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";

function CodeBlock({ children, title }) {
  return (
    <div className="my-4 overflow-hidden rounded-xl border-2 border-dark dark:border-light">
      {title && (
        <div className="border-b-2 border-dark bg-dark px-4 py-2 font-mono text-sm font-semibold text-light
          dark:border-light dark:bg-light dark:text-dark">
          {title}
        </div>
      )}
      <pre className="overflow-x-auto bg-light p-4 font-mono text-xs leading-relaxed dark:bg-dark dark:text-light">
        <code>{children}</code>
      </pre>
    </div>
  );
}

function Section({ title, children }) {
  return (
    <section className="mt-16 w-full max-w-4xl first:mt-0">
      <h2 className="text-3xl font-bold dark:text-light md:text-2xl">{title}</h2>
      <div className="mt-4 space-y-4 text-base font-medium leading-relaxed dark:text-light md:text-sm">
        {children}
      </div>
    </section>
  );
}

export default function Method() {
  return (
    <>
      <Head>
        <title>Method &middot; Gospel Homology Search</title>
        <meta
          name="description"
          content="How the spectral coordinate embedding and fragment-shader similarity kernel turn homology search into a single rendering pass."
        />
      </Head>

      <TransitionEffect />
      <main className="mb-16 flex w-full flex-col items-center justify-center dark:text-light">
        <Layout className="pt-16">
          <AnimatedText
            text="Method"
            className="mb-8 !text-7xl !leading-tight lg:!text-6xl sm:!text-5xl xs:!text-4xl"
          />

          <Section title="1. Channelise the sequence">
            <p>
              A sequence of length L becomes a real-valued signal matrix of
              shape c by L. For DNA we use four one-hot channels,
              A/C/G/T. For protein we use three physicochemical indices:
              Kyte-Doolittle hydropathy, normalised van der Waals
              side-chain volume, and net charge at neutral pH. Each channel
              is mean-centred so the DC Fourier bin is empty and carries no
              information.
            </p>
            <CodeBlock title="channelise(seq, kind)">
{`for (let i = 0; i < L; i += 1) {
  const aa = seq[i];
  channels[0][i] = HYDROPATHY[aa];
  channels[1][i] = VOLUME[aa];
  channels[2][i] = CHARGE[aa];
}
// per-channel mean subtraction`}
            </CodeBlock>
          </Section>

          <Section title="2. Spectral embedding">
            <p>
              The signal is decomposed by the real discrete Fourier
              transform. The lowest K non-DC magnitude bins per channel are
              retained, length-normalised by L, and then L2-normalised
              across channels so that cosine similarity of two embeddings
              is their dot product. The final vector has dimension c times
              K, which is 48 for DNA at K=12 and 36 for protein at K=12 -
              small enough to fit in a handful of shader uniforms.
            </p>
            <p>
              Substitutions perturb the signal locally while preserving the
              low-frequency envelope. Homologous sequences therefore share
              the same low-frequency spectrum, while unrelated sequences
              have unrelated spectra. The truncation to K bins is an
              explicit band-limit that keeps the substitution-robust
              features and discards high-frequency noise.
            </p>
          </Section>

          <Section title="3. Fragment-shader similarity kernel">
            <p>
              First-pass retrieval over a database of N sequences with
              precomputed embeddings reduces to one matrix-vector product
              of size N by d. Pack the database embeddings into a
              floating-point texture, upload the query as a uniform, render
              a full-screen quad: each fragment computes one cosine
              similarity. The kernel is entirely branch-free, performs a
              single texture read and a single dot product per fragment,
              and saturates on the fragment-stage texture bandwidth of the
              device.
            </p>
            <CodeBlock title="GLSL ES fragment shader">
{`precision highp float;
uniform sampler2D uDB;     // N_x x N_y x 4 floats
uniform vec4      uQuery;  // query embedding
in      vec2      vUV;
out     float     fragOut;
void main() {
  vec4 v = texture(uDB, vUV);
  fragOut = dot(v, uQuery);
}`}
            </CodeBlock>
            <p>
              The web tool on this site runs the CPU reference of this
              kernel - the same arithmetic expressed as a vectorised
              Float32Array dot product. Switching to the actual GLSL path
              is an engineering substitution, not a new method.
            </p>
          </Section>

          <Section title="4. Two-stage pipeline">
            <p>
              The spectral embedding is a coarse similarity predicate, not
              an alignment-score replacement. Top-K by cosine similarity is
              accurate set membership but loose rank ordering. The
              production pipeline is therefore a filter followed by a
              re-ranker:
            </p>
            <ul className="ml-6 list-decimal space-y-2 dark:text-light">
              <li>
                <strong>Shader pass.</strong> Compute cosine similarity of
                query to every database entry. Keep top K1 = 20 candidates.
              </li>
              <li>
                <strong>Re-rank.</strong> Run Smith-Waterman (or any local
                aligner) on the 20 candidates; sort by alignment score.
              </li>
            </ul>
            <p>
              The filter stage is the one that must scale to large N; the
              re-rank is bounded by K1, a constant. In our measurements the
              shader stage runs at 1.1 ms per query against a
              100,000-sequence database on a single CPU thread, against a
              14.8-second exhaustive k-mer Jaccard baseline at the same
              scale - a thirteen-thousand-fold speedup.
            </p>
          </Section>

          <Section title="5. Complexity">
            <p>
              Let N be the database size, d the embedding dimension, m and
              n the two sequence lengths, and K1 the top-K budget for the
              re-ranker.
            </p>
            <ul className="ml-6 list-disc space-y-1 dark:text-light">
              <li>Shader stage: O(d N) floating-point operations per query.</li>
              <li>Re-rank stage: O(K1 m n) alignment operations per query, independent of N.</li>
              <li>Exhaustive Smith-Waterman: O(m n N) per query.</li>
              <li>Exhaustive k-mer Jaccard: O(L N) per query with large set-hashing constants.</li>
            </ul>
            <p>
              The shader kernel is bandwidth-bound. Consumer GPUs deliver on
              the order of 500 GB/s of texture bandwidth, so a database of
              10^8 sequences with d = 48 corresponds to 20 GB of streamed
              data and a theoretical lower bound of 40 ms per query.
            </p>
          </Section>

          <Section title="6. When the embedding loses">
            <p>
              We do not hide the limits. The embedding is translation-
              sensitive: long insertions or deletions shift the spectrum and
              degrade retrieval. Spearman rank correlation between embedding
              and Smith-Waterman score is moderate (0.05 to 0.38 in our
              synthetic benchmarks); what survives is top-K set membership,
              not exact order. Random 3D projections are too low-rank to
              serve as a useful prefix prefilter at demo scale; the full
              shader scan is cheaper. And we have evaluated only synthetic
              substitution-only divergence - not indels, duplications or
              domain shuffling. Those are future work; the paper lists them
              explicitly.
            </p>
          </Section>
        </Layout>
      </main>
    </>
  );
}
