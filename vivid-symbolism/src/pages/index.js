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
          <h1 className="font-mont text-5xl font-bold tracking-tight md:text-3xl sm:text-2xl">
            Gospel
          </h1>
          <p className="mt-4 max-w-md text-base font-medium text-dark/80 dark:text-light/80 md:text-sm">
            Shader-based genomic homology search.
          </p>
          <Link
            href="/search"
            className="pointer-events-auto mt-8 rounded-lg border-2 border-dark bg-light px-6 py-2 text-base font-semibold
              text-dark transition hover:bg-dark hover:text-light
              dark:border-light dark:bg-dark dark:text-light dark:hover:bg-light dark:hover:text-dark"
          >
            Open the search
          </Link>
        </div>
      </main>
    </>
  );
}
