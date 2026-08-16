import { useEffect, useRef } from "react";
import { highlightLines } from "@/lib/synopsis/highlight";

/**
 * The editor pane.
 *
 * A transparent <textarea> layered over a highlighted <pre>. This is the
 * standard technique and it is chosen over a real editor component
 * deliberately: Monaco is several megabytes to render twenty-five lines,
 * and it would bring its own tokenizer that would drift from ours. Here
 * the colouring comes from the same scan the compiler uses, so the
 * editor cannot claim a word is a keyword when the parser disagrees.
 *
 * The two layers must share font metrics EXACTLY or the caret will
 * separate from the glyphs, which is why the font stack, size, line
 * height, padding and tab size are set identically on both.
 */

const CLS = {
  kw: "text-[#569cd6]",
  str: "text-[#ce9178]",
  num: "text-[#b5cea8]",
  comment: "text-[#6a9955] italic",
  def: "text-[#4ec9b0]",
  ident: "text-[#9cdcfe]",
  punct: "text-[#d4d4d4]",
  plain: "text-[#d4d4d4]",
};

// Shared metrics. Any change here must be made once and inherited by
// both layers; that is the entire reason it is a constant.
// Written out literally, NOT interpolated from LINE_H: Tailwind scans
// source statically and cannot see a class name assembled at runtime,
// so `leading-[${n}px]` would simply never be generated.
const TYPE = "font-mono text-[13px] leading-[20px] tracking-normal";
const PAD = "px-4 py-3";

/** The leading above, in px. Used for scroll arithmetic; keep in step. */
const LINE_H = 20;

export default function Editor({ value, onChange, errorLine, onCaret, jumpTo }) {
  const taRef = useRef(null);
  const preRef = useRef(null);

  // Keep the highlight scrolled with the textarea.
  const syncScroll = () => {
    if (preRef.current && taRef.current) {
      preRef.current.scrollTop = taRef.current.scrollTop;
      preRef.current.scrollLeft = taRef.current.scrollLeft;
    }
  };

  useEffect(syncScroll, [value]);

  // Clicking a problem puts the caret on the offending line. `jumpTo`
  // carries a timestamp rather than a bare line number so that clicking
  // the SAME problem twice still moves the caret back -- with a plain
  // number the prop would be unchanged and the effect would not re-run.
  useEffect(() => {
    if (!jumpTo || !taRef.current) return;
    const ta = taRef.current;
    const lines = ta.value.split("\n");
    const idx = Math.min(Math.max(jumpTo.line, 1), lines.length) - 1;
    const start = lines.slice(0, idx).reduce((n, l) => n + l.length + 1, 0);
    ta.focus();
    ta.setSelectionRange(start, start + lines[idx].length);
    // Bring the line into view without yanking the whole page around.
    const top = idx * LINE_H - ta.clientHeight / 2;
    ta.scrollTop = Math.max(0, top);
    syncScroll();
    if (onCaret) onCaret({ line: idx + 1, col: 1 });
  }, [jumpTo]); // eslint-disable-line react-hooks/exhaustive-deps

  const report = () => {
    const ta = taRef.current;
    if (!ta || !onCaret) return;
    const upto = ta.value.slice(0, ta.selectionStart);
    const lines = upto.split("\n");
    onCaret({ line: lines.length, col: lines[lines.length - 1].length + 1 });
  };

  // Tab inserts four spaces rather than moving focus. Without this the
  // indentation the language's blocks invite is impossible to type.
  const onKeyDown = (e) => {
    if (e.key !== "Tab") return;
    e.preventDefault();
    const ta = e.target;
    const { selectionStart: s, selectionEnd: t } = ta;
    const next = `${value.slice(0, s)}    ${value.slice(t)}`;
    onChange(next);
    requestAnimationFrame(() => {
      ta.selectionStart = ta.selectionEnd = s + 4;
    });
  };

  const lines = highlightLines(value);

  return (
    <div className="relative h-full overflow-hidden bg-[#1e1e1e]">
      <div className="flex h-full">
        {/* gutter */}
        <div
          className={`shrink-0 select-none overflow-hidden bg-[#1e1e1e] py-3 pl-3 pr-2 text-right ${TYPE} text-[#858585]`}
          aria-hidden="true"
        >
          {lines.map((_, i) => (
            <div
              key={i}
              className={
                errorLine === i + 1
                  ? "bg-[#f4877120] font-bold text-[#f48771]"
                  : undefined
              }
            >
              {i + 1}
            </div>
          ))}
        </div>

        <div className="relative flex-1 overflow-hidden">
          {/* highlighted layer */}
          <pre
            ref={preRef}
            aria-hidden="true"
            className={`pointer-events-none absolute inset-0 overflow-auto whitespace-pre ${TYPE} ${PAD}`}
          >
            {lines.map((spans, i) => (
              <div
                key={i}
                className={errorLine === i + 1 ? "bg-[#f4877115]" : undefined}
              >
                {spans.length === 0 ? (
                  "\n"
                ) : (
                  spans.map((s, j) => (
                    <span key={j} className={CLS[s.cls] || CLS.plain}>
                      {s.text}
                    </span>
                  ))
                )}
              </div>
            ))}
          </pre>

          {/* input layer */}
          <textarea
            ref={taRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onScroll={syncScroll}
            onKeyDown={onKeyDown}
            onKeyUp={report}
            onClick={report}
            spellCheck={false}
            autoComplete="off"
            autoCorrect="off"
            autoCapitalize="off"
            className={`absolute inset-0 h-full w-full resize-none overflow-auto whitespace-pre break-normal bg-transparent text-transparent caret-white outline-none ${TYPE} ${PAD}`}
            aria-label="synopsis source"
          />
        </div>
      </div>
    </div>
  );
}
