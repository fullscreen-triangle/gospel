import { useState } from "react";
import dynamic from "next/dynamic";

/**
 * The output pane: Problems, six diagrams, and the token stream.
 *
 * The diagrams split into two kinds. Tree, Frames and Residue answer
 * what SHAPE the program has. Dataflow, Params and Arity answer
 * questions with numeric answers -- how far each value stands from
 * the files it came from, what the complete set of stated parameters
 * is, and whether an alignment carries all four of its columns.
 *
 * The tree is worth showing precisely because it is the artefact the
 * two compilers are held to. The trees this page draws are compared,
 * node for node and field for field, against trees dumped from the
 * reference implementation; that comparison is what makes "the web tool
 * and the CLI implement the same language" a checkable claim rather
 * than an assurance.
 *
 * The panel will not invent a diagnostic. Where a refusal belongs to a
 * checking stage that is not implemented yet, it says so in those words
 * instead of printing an error the parser did not raise. The same rule
 * governs the diagrams: every one of them is computed from the tree the
 * parser just built, and from nothing else. The evaluator does not
 * exist, so this page has no scores to plot -- and a chart of numbers
 * this program did not produce would be indistinguishable, to a reader,
 * from one it did. What the front-end DOES measure is real and it moves
 * when the source moves: derivation depth, the parameter surface, the
 * arity of an alignment. Those are the axes here.
 */

// d3 measures and mutates DOM, so it must not run during the server
// render. Loading the views client-side avoids a hydration mismatch
// between markup produced without layout and markup produced with it.
const Views = dynamic(() => import("./Views"), {
  ssr: false,
  loading: () => (
    <p className="p-3 text-[12px] text-[#cccccc]/40">drawing…</p>
  ),
});

const TABS = [
  "Problems",
  "Tree",
  "Dataflow",
  "Frames",
  "Residue",
  "Params",
  "Arity",
  "Tokens",
];

/** The tabs Views renders. Kept beside TABS so the two cannot drift. */
const DIAGRAMS = ["Tree", "Dataflow", "Frames", "Residue", "Params", "Arity"];

function Tab({ id, active, onClick, badge }) {
  return (
    <button
      onClick={() => onClick(id)}
      className={`relative border-b-2 px-2 py-1.5 text-[10px] font-semibold uppercase tracking-wide transition-colors ${
        active
          ? "border-[#007acc] text-white"
          : "border-transparent text-[#cccccc]/50 hover:text-[#cccccc]"
      }`}
    >
      {id}
      {badge ? (
        <span className="ml-1.5 rounded-full bg-[#f48771] px-1.5 text-[9px] font-bold text-[#1e1e1e]">
          {badge}
        </span>
      ) : null}
    </button>
  );
}

function Row({ label, value }) {
  return (
    <div className="flex gap-2 text-[12px]">
      <span className="w-20 shrink-0 text-[#cccccc]/45">{label}</span>
      <span className="font-mono text-[#d4d4d4]">{value}</span>
    </div>
  );
}

function Problems({ result, file, onJump }) {
  // A stage-B refusal that the checker did NOT make. This used to be
  // the common case, and the panel said so: the checker did not exist,
  // and announcing the expected class would have been claiming a
  // diagnostic we had not derived. The checker exists now, so reaching
  // here means the two disagree -- the corpus says this program must be
  // refused and our checker accepted it. That is a defect in the
  // checker, and it is reported as one rather than as a pending stage.
  const missed = file?.group === "refusals" && result.ok;

  if (missed) {
    return (
      <div className="space-y-3">
        <div className="flex items-start gap-2 rounded border border-[#cca700]/40 bg-[#cca700]/10 p-3">
          <span className="mt-0.5 text-[#cca700]">▲</span>
          <div className="space-y-1.5">
            <p className="text-[13px] font-semibold text-[#cca700]">
              Accepted — but the corpus says it must be refused
            </p>
            <p className="text-[12px] leading-relaxed text-[#cccccc]/70">
              The conformance corpus records this program as refused with{" "}
              <code className="rounded bg-black/40 px-1 font-mono text-[#f48771]">
                {file.expect}
              </code>
              , and this build accepted it. That is a defect in the
              checker, not a property of the program. The page reports
              what the compiler actually did.
            </p>
          </div>
        </div>
        <Row label="expected" value={file.expect} />
        <Row label="got" value="accepted" />
      </div>
    );
  }

  if (result.ok) {
    const rep = result.report;
    return (
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-[13px] font-semibold text-[#4ec9b0]">
          <span>✓</span>
          <span>Parsed and typechecked — no problems</span>
        </div>
        <div className="space-y-1">
          <Row label="declarations" value={result.counts.decls} />
          <Row label="frames" value={result.counts.frames} />
          <Row label="statements" value={result.counts.stmts} />
          <Row label="report to" value={result.counts.report ?? "—"} />
        </div>
        {rep ? (
          // What the checker learned, as opposed to what the parser
          // counted. Parameters are here because the language has no
          // defaults: every threshold a program relies on was written
          // down, and this is where a reader sees the whole set.
          <div className="space-y-1 border-t border-[#cccccc]/10 pt-2">
            {/* claims and responses are lists; parameters and residues
                are keyed maps, mirroring the reference's dicts. */}
            <Row label="claims" value={rep.claims.length || "—"} />
            <Row
              label="residues"
              value={Object.keys(rep.residues).length || "—"}
            />
            <Row
              label="parameters"
              value={Object.keys(rep.parameters).length || "—"}
            />
            <Row label="responses" value={rep.responses.length || "—"} />
          </div>
        ) : null}
      </div>
    );
  }

  const expected = file?.group === "refusals";
  return (
    <div className="space-y-3">
      <button
        onClick={() => result.line && onJump(result.line)}
        className="block w-full rounded border border-[#f48771]/40 bg-[#f48771]/10 p-3 text-left transition-colors hover:bg-[#f48771]/20"
      >
        <div className="flex items-center gap-2">
          <span className="text-[#f48771]">✕</span>
          <span className="font-mono text-[13px] font-semibold text-[#f48771]">
            {result.error}
          </span>
          {result.line ? (
            <span className="ml-auto font-mono text-[11px] text-[#cccccc]/50">
              line {result.line}
            </span>
          ) : null}
        </div>
        <p className="mt-1.5 text-[12px] leading-relaxed text-[#cccccc]/80">
          {result.message}
        </p>
      </button>

      {expected && (
        <p className="text-[11px] leading-relaxed text-[#cccccc]/50">
          This refusal is the point of the lesson —{" "}
          {result.stage === "A" ? (
            <>
              the program is rejected by the grammar itself, before any
              analysis runs. It cannot be written, so there is nothing
              for the checker to decide.
            </>
          ) : (
            <>
              the program parses, so the tree beside this pane is
              complete. It is rejected by the type checker: it can be
              written, but it cannot be meant.
            </>
          )}
          {file.reported && file.reported !== file.expect ? (
            <>
              {" "}The reference records this as{" "}
              <code className="font-mono text-[#cccccc]/70">{file.expect}</code>;{" "}
              reporting the more specific{" "}
              <code className="font-mono text-[#cccccc]/70">{file.reported}</code>{" "}
              is permitted, since a compiler may narrow a refusal but never
              widen one.
            </>
          ) : null}
        </p>
      )}
    </div>
  );
}

function Tokens({ tokens }) {
  if (!tokens) {
    return (
      <p className="text-[12px] text-[#cccccc]/50">
        The source does not tokenise; see Problems.
      </p>
    );
  }
  const tone = {
    kw: "text-[#569cd6]",
    ident: "text-[#9cdcfe]",
    num: "text-[#b5cea8]",
    string: "text-[#ce9178]",
    punct: "text-[#d4d4d4]",
    range: "text-[#d4d4d4]",
    eof: "text-[#cccccc]/40",
  };
  return (
    <table className="w-full border-collapse font-mono text-[11.5px]">
      <thead className="sticky top-0 bg-[#1e1e1e] text-[10px] uppercase tracking-wide text-[#cccccc]/40">
        <tr>
          <th className="py-1 pr-3 text-right font-semibold">#</th>
          <th className="py-1 pr-3 text-left font-semibold">kind</th>
          <th className="py-1 pr-3 text-left font-semibold">text</th>
          <th className="py-1 text-right font-semibold">line</th>
        </tr>
      </thead>
      <tbody>
        {tokens.map((t, i) => (
          <tr key={i} className="hover:bg-white/5">
            <td className="py-[1px] pr-3 text-right text-[#cccccc]/30">{i}</td>
            <td className={`py-[1px] pr-3 ${tone[t.kind] || ""}`}>{t.kind}</td>
            <td className="py-[1px] pr-3 text-[#d4d4d4]">
              {t.text === "" ? <span className="text-[#cccccc]/30">∅</span> : t.text}
            </td>
            <td className="py-[1px] text-right text-[#cccccc]/40">{t.line}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/**
 * Nothing has been compiled yet.
 *
 * This replaces the whole panel rather than emptying its body. A tab
 * strip above an empty pane invites clicking through seven views that
 * each say nothing, which reads as a broken page instead of one that
 * is waiting. `stale` distinguishes the two ways of arriving here --
 * never having run, and having run and then edited -- because the
 * second means a result existed and was deliberately discarded, and
 * saying so is the difference between a page that lost your output
 * and one that refused to show you a report about code you have
 * since changed.
 */
function NotRun({ stale, onRun }) {
  return (
    <section className="flex h-full flex-col items-center justify-center gap-3 bg-[#1e1e1e] p-6 text-center">
      <p className="text-[13px] font-semibold text-[#cccccc]/80">
        {stale ? "Edited since the last run" : "Not compiled yet"}
      </p>
      <p className="max-w-[16rem] text-[12px] leading-relaxed text-[#cccccc]/45">
        {stale
          ? "The source has changed, so the previous report describes code that is no longer on screen. Run it again to see what the compiler makes of this version."
          : "Diagnostics, the syntax tree and the six diagrams are all derived from a parse. Run the program to produce one."}
      </p>
      <button
        onClick={onRun}
        className="rounded bg-[#28c840]/90 px-3 py-1 font-mono text-[11px] text-[#1e1e1e] transition-colors hover:bg-[#28c840]"
      >
        run
      </button>
      <p className="font-mono text-[10px] text-[#cccccc]/30">Ctrl+Enter</p>
    </section>
  );
}

export default function OutputPanel({ result, file, tokens, stale, onRun, onJump }) {
  const [tab, setTab] = useState("Problems");

  // Before the first run there is no result to describe, and after an
  // edit the one we have describes different source. Either way the
  // honest thing to draw is the absence, not a stale report.
  if (!result) return <NotRun stale={stale} onRun={onRun} />;

  const problemCount = result.ok ? 0 : 1;

  return (
    <section className="flex h-full flex-col bg-[#1e1e1e]">
      <div className="flex shrink-0 flex-wrap items-center gap-0.5 border-b border-black/40 bg-[#252526] px-2">
        {TABS.map((t) => (
          <Tab
            key={t}
            id={t}
            active={tab === t}
            onClick={setTab}
            badge={t === "Problems" && problemCount ? problemCount : null}
          />
        ))}
      </div>

      {/* The diagrams manage their own padding and scrolling, because a
          wide tree needs to scroll under the tab strip rather than
          inside a padded box that clips it. */}
      <div
        className={`min-h-0 flex-1 overflow-auto ${
          DIAGRAMS.includes(tab) ? "" : "p-3"
        }`}
      >
        {tab === "Problems" && (
          <Problems result={result} file={file} onJump={onJump} />
        )}
        {DIAGRAMS.includes(tab) && (
          <Views view={tab} result={result} file={file} onJump={onJump} />
        )}
        {tab === "Tokens" && <Tokens tokens={tokens} />}
      </div>
    </section>
  );
}
