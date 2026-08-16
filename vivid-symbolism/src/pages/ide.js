import Head from "next/head";
import { useEffect, useMemo, useState } from "react";

import TransitionEffect from "@/components/TransitionEffect";
import Explorer from "@/components/ide/Explorer";
import Editor from "@/components/ide/Editor";
import OutputPanel from "@/components/ide/OutputPanel";
import { analyse, tokensOf } from "@/lib/synopsis/analyse";
import { ALL_FILES, TUTORIAL, REFUSALS } from "@/lib/synopsis/tutorial";

/**
 * The synopsis IDE.
 *
 * Three columns: the lesson files, the source, and what the compiler
 * makes of it. The compiler here is the real one -- the same front-end
 * the command-line tool runs, compiled to the browser -- so nothing on
 * this page is a simulation of a diagnostic. Everything runs locally;
 * no source leaves the tab.
 *
 * This page is deliberately NOT a general editor. It is a reading
 * surface for eighteen scripts, twelve of which build up the corpus
 * programs and six of which must be refused, and the edit affordance
 * exists so a reader can break a program on purpose and see what the
 * language says about it.
 */

const TUTORIAL_FILES = ALL_FILES.filter((f) => f.group === "tutorial");
const REFUSAL_FILES = ALL_FILES.filter((f) => f.group === "refusals");

function Chrome({ children }) {
  return (
    <div className="flex items-center gap-2 border-b border-black/40 bg-[#3c3c3c] px-3 py-1.5">
      <span className="flex gap-1.5" aria-hidden="true">
        <span className="h-3 w-3 rounded-full bg-[#ff5f57]" />
        <span className="h-3 w-3 rounded-full bg-[#febc2e]" />
        <span className="h-3 w-3 rounded-full bg-[#28c840]" />
      </span>
      {children}
    </div>
  );
}

export default function IDE() {
  const [activeId, setActiveId] = useState(TUTORIAL_FILES[0].id);
  const [sources, setSources] = useState(() =>
    Object.fromEntries(ALL_FILES.map((f) => [f.id, f.src]))
  );
  const [caret, setCaret] = useState({ line: 1, col: 1 });
  const [jumpTo, setJumpTo] = useState(null);

  const file = ALL_FILES.find((f) => f.id === activeId);
  const src = sources[activeId];

  // Debounced, because analysing on every keystroke makes the diagnostic
  // flicker through every intermediate half-typed state while someone is
  // in the middle of a line -- which reads as noise, not as feedback.
  const [settled, setSettled] = useState(src);
  useEffect(() => {
    const t = setTimeout(() => setSettled(src), 300);
    return () => clearTimeout(t);
  }, [src]);

  const result = useMemo(() => analyse(settled), [settled]);
  const tokens = useMemo(() => tokensOf(settled), [settled]);

  const dirty = src !== file.src;
  const errorLine = result.ok ? null : result.line;

  const setSrc = (next) => setSources((s) => ({ ...s, [activeId]: next }));
  const revert = () => setSrc(file.src);

  return (
    <>
      <Head>
        <title>synopsis — IDE | Gospel</title>
        <meta
          name="description"
          content="An in-browser editor for synopsis, running the same compiler front-end as the command-line tool. Twelve lessons and six refusals."
        />
      </Head>
      <TransitionEffect />

      <main className="flex w-full flex-col items-center bg-light px-4 pb-16 pt-24 dark:bg-dark sm:px-8 lg:px-12">
        <div className="w-full max-w-7xl">
          <header className="mb-6">
            <h1 className="text-4xl font-bold text-dark dark:text-light sm:text-5xl">
              The synopsis editor
            </h1>
            <p className="mt-3 max-w-3xl text-base font-medium leading-relaxed text-dark/80 dark:text-light/80">
              Eighteen scripts, in order. The first twelve build the
              corpus programs one construct at a time; the last six are
              programs the language must refuse, because a language is
              defined as much by what it rejects as by what it accepts.
              The compiler running below is the same front-end the
              command-line tool uses, compiled to WebAssembly-free
              JavaScript and run entirely in this tab — nothing you type
              is sent anywhere.
            </p>
          </header>

          {/* the window */}
          <div className="overflow-hidden rounded-lg border border-black/50 shadow-2xl">
            <Chrome>
              <span className="ml-2 font-mono text-[12px] text-[#cccccc]/80">
                {file.name}
                {dirty ? " •" : ""}
              </span>
              <span className="ml-auto text-[11px] text-[#cccccc]/40">
                synopsis — stage A (parser)
              </span>
            </Chrome>

            <div className="grid h-[34rem] grid-cols-1 md:grid-cols-[15rem_minmax(0,1fr)] lg:grid-cols-[15rem_minmax(0,1fr)_22rem]">
              <div className="hidden min-h-0 border-r border-black/40 md:block">
                <Explorer
                  tutorial={TUTORIAL_FILES}
                  refusals={REFUSAL_FILES}
                  activeId={activeId}
                  onSelect={(id) => {
                    setActiveId(id);
                    setJumpTo(null);
                  }}
                />
              </div>

              <div className="min-h-0 border-r border-black/40">
                <Editor
                  value={src}
                  onChange={setSrc}
                  errorLine={errorLine}
                  onCaret={setCaret}
                  jumpTo={jumpTo}
                />
              </div>

              <div className="hidden min-h-0 lg:block">
                <OutputPanel
                  result={result}
                  file={file}
                  tokens={tokens}
                  onJump={(line) => setJumpTo({ line, at: Date.now() })}
                />
              </div>
            </div>

            {/* status bar */}
            <div className="flex items-center gap-4 border-t border-black/40 bg-[#007acc] px-3 py-1 font-mono text-[11px] text-white">
              <span>{result.ok ? "no problems" : result.error}</span>
              <span className="ml-auto">
                Ln {caret.line}, Col {caret.col}
              </span>
              <span>synopsis</span>
              {dirty && (
                <button
                  onClick={revert}
                  className="rounded bg-white/20 px-1.5 hover:bg-white/30"
                >
                  revert
                </button>
              )}
            </div>
          </div>

          {/* the lesson, below the window rather than inside it, so the
              editor keeps the full height it needs */}
          <section className="mt-6 grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
            <div>
              <h2 className="text-xs font-bold uppercase tracking-widest text-primary dark:text-primaryDark">
                {file.group === "refusals" ? "Refusal" : "Lesson"}
                {file.corpus ? " — corpus program" : ""}
              </h2>
              <h3 className="mt-1 text-2xl font-bold text-dark dark:text-light">
                {file.title}
              </h3>
              <p className="mt-3 text-[15px] leading-relaxed text-dark/80 dark:text-light/80">
                {file.blurb}
              </p>
            </div>

            {file.tryThis && (
              <aside className="self-start rounded-lg border-2 border-dark p-4 dark:border-light">
                <div className="text-xs font-bold uppercase tracking-widest text-primary dark:text-primaryDark">
                  Try this
                </div>
                <p className="mt-2 text-[14px] leading-relaxed text-dark/80 dark:text-light/80">
                  {file.tryThis}
                </p>
              </aside>
            )}
          </section>

          <p className="mt-8 max-w-3xl text-[13px] leading-relaxed text-dark/55 dark:text-light/55">
            The parser is complete; the type checker is not. Four of the
            six refusals below are caught by the checker rather than the
            grammar, and for those the output panel says so plainly
            instead of printing an error it did not derive. The two that
            are refused here — a missing report, and indexing a sequence
            — are refused by the shape of the grammar itself, which is
            what makes them impossible to write rather than merely
            discouraged.
          </p>

          <p className="mt-3 max-w-3xl text-[13px] leading-relaxed text-dark/55 dark:text-light/55">
            On a narrow screen the output panel is hidden and the file
            list collapses; the editor needs three columns to be worth
            reading, so this page is best on a wide display.
          </p>
        </div>
      </main>
    </>
  );
}
