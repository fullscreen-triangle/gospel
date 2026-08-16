import { useState } from "react";

/**
 * The file tree.
 *
 * Two groups, because the split is pedagogical rather than
 * organisational: `tutorial/` is what the language admits, `refusals/`
 * is what it rejects, and a reader who only ever sees the first half
 * has learned half the language.
 */

function Chevron({ open }) {
  return (
    <svg
      viewBox="0 0 16 16"
      className={`h-3 w-3 shrink-0 transition-transform ${open ? "rotate-90" : ""}`}
      fill="currentColor"
      aria-hidden="true"
    >
      <path d="M6 4l4 4-4 4V4z" />
    </svg>
  );
}

function FileIcon({ tone }) {
  return (
    <svg viewBox="0 0 16 16" className={`h-3.5 w-3.5 shrink-0 ${tone}`} fill="currentColor" aria-hidden="true">
      <path d="M9 1H4a1 1 0 0 0-1 1v12a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V5L9 1zm0 1.5L11.5 5H9V2.5z" />
    </svg>
  );
}

function Group({ label, count, files, activeId, onSelect, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="mb-1">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-1 px-2 py-1 text-left text-[11px] font-semibold uppercase tracking-wide text-[#cccccc]/70 hover:text-[#cccccc]"
      >
        <Chevron open={open} />
        <span>{label}</span>
        <span className="ml-auto tabular-nums text-[#cccccc]/40">{count}</span>
      </button>

      {open && (
        <ul>
          {files.map((f) => {
            const active = f.id === activeId;
            return (
              <li key={f.id}>
                <button
                  onClick={() => onSelect(f.id)}
                  title={f.title}
                  className={`group flex w-full items-center gap-2 border-l-2 py-[3px] pl-5 pr-2 text-left text-[13px] transition-colors ${
                    active
                      ? "border-[#007acc] bg-[#37373d] text-white"
                      : "border-transparent text-[#cccccc]/80 hover:bg-[#2a2d2e] hover:text-white"
                  }`}
                >
                  <FileIcon
                    tone={
                      f.group === "refusals"
                        ? "text-[#f48771]"
                        : f.corpus
                        ? "text-[#4ec9b0]"
                        : "text-[#519aba]"
                    }
                  />
                  <span className="truncate font-mono">{f.name}</span>
                  {f.corpus && (
                    <span
                      title="reproduced verbatim from the conformance corpus"
                      className="ml-auto shrink-0 rounded bg-[#4ec9b0]/15 px-1 text-[9px] font-semibold uppercase text-[#4ec9b0]"
                    >
                      corpus
                    </span>
                  )}
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

export default function Explorer({ tutorial, refusals, activeId, onSelect }) {
  return (
    <aside className="flex h-full flex-col overflow-y-auto bg-[#252526] py-1">
      <div className="px-3 py-2 text-[11px] font-bold uppercase tracking-widest text-[#cccccc]/60">
        Explorer
      </div>

      <Group
        label="tutorial"
        count={tutorial.length}
        files={tutorial}
        activeId={activeId}
        onSelect={onSelect}
      />
      <Group
        label="refusals"
        count={refusals.length}
        files={refusals}
        activeId={activeId}
        onSelect={onSelect}
      />

      <p className="mt-auto px-3 py-3 text-[11px] leading-relaxed text-[#cccccc]/40">
        Lessons build the corpus programs from their pieces. Refusals are
        programs the language must reject.
      </p>
    </aside>
  );
}
