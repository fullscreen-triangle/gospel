import dynamic from "next/dynamic";
import Head from "next/head";
import Link from "next/link";

import TransitionEffect from "@/components/TransitionEffect";

// Three.js needs the browser; load the model viewer client-side only.
const ProteinModel = dynamic(() => import("@/components/ProteinModel"), {
  ssr: false,
  loading: () => null,
});

export default function Home() {
  return (
    <>
      <Head>
        <title>Gospel &middot; Shader-Based Genomic Homology Search</title>
        <meta
          name="description"
          content="Shader-based genomic homology search via spectral coordinate embeddings."
        />
      </Head>

      <TransitionEffect />
      <main className="relative flex min-h-[calc(100vh-7rem)] w-full items-center justify-center overflow-hidden text-dark dark:text-light">
        <div className="pointer-events-auto absolute inset-0">
          <ProteinModel className="h-full w-full" />
        </div>

        <div className="pointer-events-none relative z-10 flex flex-col items-center px-8 text-center">
          <h1 className="text-gray-700 font-mont text-5xl font-bold tracking-tight md:text-3xl sm:text-2xl">
            Gospel
          </h1>
          <p className="mt-4 max-w-md text-base font-medium text-dark/80 dark:text-light/80 md:text-sm">
            Shader-based genomic homology search.
          </p>
          {/* Two entry points, because the site does two separable
              things: it searches, and it hosts a language for stating
              how a search was performed. The second is not a subsection
              of the first and is not discoverable from it. */}
          <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
            <Link
              href="/search"
              className="pointer-events-auto rounded-lg border-2 border-dark bg-light px-6 py-2 text-base font-semibold
                text-dark transition hover:bg-dark hover:text-light
                dark:border-light dark:bg-dark dark:text-light dark:hover:bg-light dark:hover:text-dark"
            >
              Open the search
            </Link>
            <Link
              href="/ide"
              className="pointer-events-auto rounded-lg border-2 border-dark/40 px-6 py-2 text-base font-semibold
                text-dark transition hover:border-dark hover:bg-dark hover:text-light
                dark:border-light/40 dark:text-light dark:hover:border-light dark:hover:bg-light dark:hover:text-dark"
            >
              Write a method
            </Link>
          </div>

          {/* "parser", not "compiler": the type checker and evaluator
              are not implemented, and the editor says so on the programs
              that need them. The landing page must not promise more than
              the tool behind it delivers. */}
          <p className="mt-4 max-w-sm text-[13px] leading-relaxed text-dark/55 dark:text-light/55">
            <span className="font-semibold">synopsis</span> is a small
            language for genomic comparisons in which the method is
            recoverable from the script. The editor runs its parser in
            your browser, against a worked tutorial.
          </p>
        </div>
      </main>
    </>
  );
}
