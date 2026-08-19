import Head from "next/head";
import { useCallback, useEffect, useRef, useState } from "react";

import TransitionEffect from "@/components/TransitionEffect";
import Explorer from "@/components/ide/Explorer";
import Editor from "@/components/ide/Editor";
import OutputPanel from "@/components/ide/OutputPanel";
import { analyse, tokensOf } from "@/lib/synopsis/analyse";
import { ALL_FILES } from "@/lib/synopsis/tutorial";

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
 *
 * Compilation is EXPLICIT. An earlier version analysed on a debounced
 * timer, so a lesson arrived already compiled and its diagnostic was
 * simply present, as though it were a property of the file rather than
 * the outcome of running something. That taught the wrong thing about
 * a compiler. Here the source sits un-run until someone asks, and the
 * result is pinned to the exact text it was produced from -- so what
 * the panel shows is always a report about code that was actually
 * submitted, never about code that has since been edited.
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

function RunIcon() {
  return (
    <svg viewBox="0 0 16 16" className="h-3 w-3" fill="currentColor" aria-hidden="true">
      <path d="M4 2.5v11l9-5.5-9-5.5z" />
    </svg>
  );
}

/** A window split into panes, with the named pane filled in. */
function PanelIcon({ side }) {
  return (
    <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" aria-hidden="true">
      <rect x="1.5" y="2.5" width="13" height="11" rx="1"
            fill="none" stroke="currentColor" strokeWidth="1.2" />
      {side === "left" ? (
        <rect x="1.5" y="2.5" width="4.5" height="11" fill="currentColor" />
      ) : (
        <rect x="10" y="2.5" width="4.5" height="11" fill="currentColor" />
      )}
    </svg>
  );
}

function ExpandIcon({ full }) {
  return (
    <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" fill="currentColor" aria-hidden="true">
      {full ? (
        <path d="M2 6h4V2H4.5v2.5H2V6zm8-4v4h4V4.5h-2.5V2H10zM2 10v1.5h2.5V14H6v-4H2zm8 0v4h1.5v-2.5H14V10h-4z" />
      ) : (
        <path d="M2 2v4h1.5V3.5H6V2H2zm8 0v1.5h2.5V6H14V2h-4zM3.5 10H2v4h4v-1.5H3.5V10zm9 0v2.5H10V14h4v-4h-1.5z" />
      )}
    </svg>
  );
}

/** Column widths in px. The editor takes whatever is left. */
const DEFAULT_W = { left: 240, right: 352 };
const MIN_W = 140; // narrower than this and the file names stop being readable
const MIN_EDITOR = 280;

/**
 * A draggable divider between two columns.
 *
 * The drag does NOT go through React state. A pointermove fires at
 * screen refresh rate, and putting the width in state would re-render
 * the editor and re-run every d3 view on each of those frames, which
 * turns a smooth drag into a stutter. Instead the handle writes the
 * new track size straight onto the grid element's inline style and
 * reports the final width once, on release. React owns the committed
 * width; the drag owns the pixels in between.
 *
 * Pointer events rather than mouse events, so a trackpad, a touch
 * screen and a stylus all work, and so `setPointerCapture` keeps the
 * drag alive when the cursor outruns the 6px handle -- which it will,
 * because a fast drag moves further per frame than the handle is wide.
 */
function Splitter({ onDrag, onCommit, side, label }) {
  const down = (e) => {
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    const x0 = e.clientX;
    const w0 = onDrag(null); // read the current width without changing it

    const move = (ev) => {
      // Dragging the right-hand divider rightwards SHRINKS its column,
      // so that side's delta is inverted.
      const d = side === "left" ? ev.clientX - x0 : x0 - ev.clientX;
      onDrag(w0 + d);
    };
    const up = (ev) => {
      const d = side === "left" ? ev.clientX - x0 : x0 - ev.clientX;
      onCommit(w0 + d);
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  return (
    <div
      onPointerDown={down}
      onDoubleClick={() => onCommit(side === "left" ? DEFAULT_W.left : DEFAULT_W.right)}
      role="separator"
      aria-orientation="vertical"
      aria-label={label}
      title={`${label} — drag to resize, double-click to reset`}
      className="group relative z-10 w-1.5 shrink-0 cursor-col-resize bg-black/40 transition-colors hover:bg-[#007acc]"
    >
      {/* A 1.5px target is hard to hit, so the grab area is widened
          past the visible line without widening the line itself. */}
      <span className="absolute inset-y-0 -left-1 -right-1" />
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
  const [full, setFull] = useState(false);

  // Committed column widths, and which columns are folded away. A
  // collapsed column keeps its width so re-opening it restores the
  // size you chose rather than snapping back to the default.
  const [w, setW] = useState(DEFAULT_W);
  const [hidden, setHidden] = useState({ left: "auto", right: "auto" });
  const [auto, setAuto] = useState({ left: false, right: false });
  const gridRef = useRef(null);

  // What the compiler was last handed, and what it said. `src` is the
  // text that produced this -- keeping it is what lets the page tell
  // "you have not run this yet" apart from "you have run it and then
  // changed it", which are different things to say to a reader.
  const [run, setRun] = useState(null);
  const [running, setRunning] = useState(false);

  const file = ALL_FILES.find((f) => f.id === activeId);
  const src = sources[activeId];

  const compile = useCallback(() => {
    const text = sources[activeId];
    setRunning(true);
    // A frame's delay so the button's pressed state actually paints.
    // The parse itself is microseconds -- this is honest about being
    // for the eye, not a simulated workload.
    requestAnimationFrame(() => {
      setRun({
        id: activeId,
        src: text,
        result: analyse(text),
        tokens: tokensOf(text),
        at: Date.now(),
      });
      setRunning(false);
    });
  }, [activeId, sources]);

  // Ctrl/Cmd+Enter, the shortcut every REPL and notebook already uses.
  useEffect(() => {
    const onKey = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        compile();
      }
      if (e.key === "Escape" && full) setFull(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [compile, full]);

  // Below 1024px the output panel does not have room to be worth
  // reading, and below 768px the file list goes with it. This used to
  // be `lg:hidden` / `md:hidden` on the tracks themselves; it has to be
  // state now, because a CSS-hidden column would keep its inline track
  // width and leave a 240px gap where its contents used to be.
  //
  // `auto` is a third value, distinct from true and false: it means
  // nobody has expressed a preference, so the viewport decides. Once
  // someone clicks a toggle their choice sticks, and widening the
  // window does not silently undo it.
  useEffect(() => {
    const apply = () =>
      setAuto({
        left: window.innerWidth <= 767,
        right: window.innerWidth <= 1023,
      });
    apply();
    window.addEventListener("resize", apply);
    return () => window.removeEventListener("resize", apply);
  }, []);

  // Fullscreen takes over the viewport, so the page behind it must not
  // scroll underneath.
  useEffect(() => {
    if (!full) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [full]);

  const dirty = src !== file.src;

  // The result belongs to this file AND to this exact text. A result
  // from before an edit is a report about code that is no longer on
  // screen, and showing it would be worse than showing nothing.
  const current = run && run.id === activeId && run.src === src ? run : null;
  const stale = run && run.id === activeId && run.src !== src;
  const errorLine = current && !current.result.ok ? current.result.line : null;

  const setSrc = (next) => setSources((s) => ({ ...s, [activeId]: next }));
  const revert = () => setSrc(file.src);

  const status = running
    ? "compiling…"
    : current
    ? current.result.ok
      ? "no problems"
      : current.result.error
    : stale
    ? "edited since last run"
    : "not run";

  // "auto" defers to the viewport; true/false is a stated preference.
  const off = {
    left: hidden.left === "auto" ? auto.left : hidden.left,
    right: hidden.right === "auto" ? auto.right : hidden.right,
  };

  /**
   * The grid template. A collapsed column becomes a zero-width track
   * rather than an unmounted subtree, so its scroll position and the
   * output panel's selected tab survive being folded away and back.
   */
  const tracks = (left, right) =>
    [
      off.left ? "0px" : `${left}px`,
      off.left ? "" : "6px",
      "minmax(0,1fr)",
      off.right ? "" : "6px",
      off.right ? "0px" : `${right}px`,
    ]
      .filter(Boolean)
      .join(" ");

  // Clamp so a drag can neither erase a column nor squeeze the editor
  // out of existence. Without the MIN_EDITOR term you can drag the two
  // dividers together and lose the thing being edited.
  const clamp = (px, side) => {
    const box = gridRef.current?.getBoundingClientRect().width ?? 1200;
    const other = side === "left" ? (off.right ? 0 : w.right) : (off.left ? 0 : w.left);
    return Math.max(MIN_W, Math.min(px, box - other - MIN_EDITOR - 12));
  };

  // Called on every pointermove. Passing null reads the current width
  // without writing, which is how the drag learns its own origin.
  const drag = (side) => (px) => {
    if (px === null) return w[side];
    const next = clamp(px, side);
    if (gridRef.current)
      gridRef.current.style.gridTemplateColumns =
        side === "left" ? tracks(next, w.right) : tracks(w.left, next);
    return next;
  };
  const commit = (side) => (px) =>
    setW((prev) => ({ ...prev, [side]: clamp(px, side) }));

  const window_ = (
    <div
      className={
        full
          ? "flex h-full flex-col overflow-hidden bg-[#1e1e1e]"
          : "overflow-hidden rounded-lg border border-black/50 shadow-2xl"
      }
    >
      <Chrome>
        <span className="ml-2 font-mono text-[12px] text-[#cccccc]/80">
          {file.name}
          {dirty ? " •" : ""}
        </span>

        <button
          onClick={compile}
          disabled={running}
          title="Compile (Ctrl+Enter)"
          className={`ml-4 flex items-center gap-1.5 rounded px-2 py-0.5 font-mono text-[11px] transition-colors ${
            current
              ? "bg-white/10 text-[#cccccc]/70 hover:bg-white/20 hover:text-white"
              : "bg-[#28c840]/90 text-[#1e1e1e] hover:bg-[#28c840]"
          } disabled:opacity-50`}
        >
          <RunIcon />
          {current ? "run again" : "run"}
        </button>

        {stale && (
          <span className="font-mono text-[10px] text-[#cca700]">
            edited — run again
          </span>
        )}

        <span className="ml-auto flex items-center gap-3">
          <span className="text-[11px] text-[#cccccc]/40">
            synopsis — parser and checker
          </span>
          <span className="flex items-center gap-0.5">
            {[
              ["left", "File list"],
              ["right", "Output panel"],
            ].map(([k, name]) => (
              <button
                key={k}
                onClick={() => setHidden((h) => ({ ...h, [k]: !off[k] }))}
                title={`${off[k] ? "Show" : "Hide"} ${name.toLowerCase()}`}
                aria-label={`${off[k] ? "Show" : "Hide"} ${name.toLowerCase()}`}
                aria-pressed={!off[k]}
                className={`rounded px-1 py-0.5 transition-colors hover:bg-white/10 ${
                  off[k] ? "text-[#cccccc]/30" : "text-[#cccccc]/75"
                }`}
              >
                <PanelIcon side={k} />
              </button>
            ))}
          </span>

          <button
            onClick={() => setFull((f) => !f)}
            title={full ? "Exit full screen (Esc)" : "Full screen"}
            aria-label={full ? "Exit full screen" : "Full screen"}
            className="flex items-center gap-1.5 rounded px-1.5 py-0.5 text-[#cccccc]/60 transition-colors hover:bg-white/10 hover:text-white"
          >
            <ExpandIcon full={full} />
            <span className="font-mono text-[10px]">
              {full ? "exit" : "full screen"}
            </span>
          </button>
        </span>
      </Chrome>

      <div
        ref={gridRef}
        style={{ gridTemplateColumns: tracks(w.left, w.right) }}
        className={`grid ${full ? "min-h-0 flex-1" : "h-[34rem]"}`}
      >
        <div
          className={`min-h-0 ${off.left ? "overflow-hidden" : ""}`}
          aria-hidden={off.left}
        >
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

        {!off.left && (
          <Splitter
            side="left"
            label="File list width"
            onDrag={drag("left")}
            onCommit={commit("left")}
          />
        )}

        <div className="min-h-0">
          <Editor
            value={src}
            onChange={setSrc}
            errorLine={errorLine}
            onCaret={setCaret}
            jumpTo={jumpTo}
          />
        </div>

        {!off.right && (
          <Splitter
            side="right"
            label="Output panel width"
            onDrag={drag("right")}
            onCommit={commit("right")}
          />
        )}

        <div
          className={`min-h-0 ${off.right ? "overflow-hidden" : ""}`}
          aria-hidden={off.right}
        >
          <OutputPanel
            result={current?.result ?? null}
            file={file}
            tokens={current?.tokens ?? null}
            stale={stale}
            onRun={compile}
            onJump={(line) => setJumpTo({ line, at: Date.now() })}
          />
        </div>
      </div>

      <div className="flex items-center gap-4 border-t border-black/40 bg-[#007acc] px-3 py-1 font-mono text-[11px] text-white">
        <span>{status}</span>
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
  );

  if (full) {
    return (
      <>
        <Head>
          <title>synopsis — IDE | Gospel</title>
        </Head>
        <div className="fixed inset-0 z-50 bg-[#1e1e1e]">{window_}</div>
      </>
    );
  }

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
              Press <span className="font-mono text-[15px]">run</span> to
              compile the one on screen. The compiler is the same
              front-end the command-line tool uses, running entirely in
              this tab — nothing you type is sent anywhere.
            </p>
          </header>

          {window_}

          {/* the lesson, below the window rather than inside it, so the
              editor keeps the full height it needs */}
          <section className="mt-6 grid gap-6 lg:grid-cols-1">
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
            Both stages run here. Two of the six refusals — a missing
            report, and indexing a sequence — are refused by the shape
            of the grammar itself, which is what makes them impossible
            to write rather than merely discouraged. The other four
            parse cleanly and are refused by the type checker, so for
            those the tree beside the editor is complete: the rule being
            broken is a relation between two nodes you can point at.
          </p>

          <p className="mt-3 max-w-3xl text-[13px] leading-relaxed text-dark/55 dark:text-light/55">
            Below 1024px the output panel is hidden and below 768px the
            file list collapses with it; the editor needs its three
            columns to be worth reading, so this page is best on a wide
            display — or in full screen.
          </p>
        </div>
      </main>
    </>
  );
}
