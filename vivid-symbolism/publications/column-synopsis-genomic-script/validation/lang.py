"""
lang.py -- tokeniser, parser and typechecker for `synopsis` (.syp).

This exists to test the REFUSAL theorems of Sec 9 (Thms 9.4-9.7). Those
theorems assert that certain programs have no derivation and no typing.
A claim of the form "X is inexpressible" cannot be checked numerically;
it can only be checked by building the checker and confirming that X is
rejected -- and, just as importantly, that the four programs of Sec 10
are ACCEPTED. A checker that rejected everything would satisfy every
refusal theorem vacuously, so the positive corpus is load-bearing.

The typechecker implements:
  * frame indices phi (Thm 9.6, scope safety)
  * the (beta, varrho) accountability discipline (Cor 9.8)
  * four-ary `align` (Thm 9.5, arity)
  * positive `eta` (Assumption 4.6 discharged at compile time)
  * no default parameters (Thm 9.7, parameter completeness)
  * three distinct result types (Rule 6.11)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field


# =====================================================================
# Errors
# =====================================================================

class SynopsisError(Exception):
    """Base: a program the language refuses."""
    kind = "error"

    def __init__(self, msg: str, line: int = 0):
        super().__init__(msg)
        self.msg = msg
        self.line = line

    def __str__(self) -> str:
        return f"{self.kind}: {self.msg}" + (f" (line {self.line})" if self.line else "")


class ParseError(SynopsisError):
    kind = "parse error"


class TypeError_(SynopsisError):
    kind = "type error"


class ArityError(TypeError_):
    kind = "arity error"


class ScopeError(TypeError_):
    kind = "scope error"


class ResidueError(TypeError_):
    kind = "residue error"


class ParameterError(TypeError_):
    kind = "parameter error"


class TerminationError(TypeError_):
    kind = "termination error"


# =====================================================================
# Tokeniser
# =====================================================================

KEYWORDS = {
    "open", "let", "bind", "method", "require", "under", "project", "by",
    "compare", "against", "detect", "in", "align", "central", "response",
    "unit", "anchors", "perturb", "join", "and", "relax", "until",
    "quiescent", "sweep", "step", "for", "items", "where", "claim",
    "record", "drop", "report", "to", "corr", "from", "peaks", "top",
    "nearest_unit", "separation",
}

TOKEN_RE = re.compile(r"""
      (?P<ws>[ \t\r\n]+)
    | (?P<comment>\#[^\n]*)
    | (?P<string>"(?:[^"\\]|\\.)*")
    | (?P<range>\.\.)
    | (?P<num>-?\d+\.\d+|-?\d+)
    | (?P<ident>[A-Za-z_][A-Za-z_0-9]*)
    | (?P<punct>[{}()\[\],;=<>+*/])
""", re.VERBOSE)


@dataclass
class Token:
    kind: str      # 'kw' | 'ident' | 'num' | 'string' | 'punct' | 'range'
    text: str
    line: int
    value: float | str | None = None


def tokenise(src: str) -> list[Token]:
    toks: list[Token] = []
    pos, line = 0, 1
    n = len(src)
    while pos < n:
        m = TOKEN_RE.match(src, pos)
        if m is None:
            raise ParseError(f"unexpected character {src[pos]!r}", line)
        kind = m.lastgroup
        text = m.group()
        if kind == "ws":
            line += text.count("\n")
        elif kind == "comment":
            pass
        elif kind == "string":
            toks.append(Token("string", text, line, value=text[1:-1]))
        elif kind == "num":
            toks.append(Token("num", text, line, value=float(text)))
        elif kind == "ident":
            toks.append(Token("kw" if text in KEYWORDS else "ident", text, line))
        elif kind == "range":
            toks.append(Token("range", text, line))
        else:
            toks.append(Token("punct", text, line))
        pos = m.end()
    toks.append(Token("eof", "", line))
    return toks


# =====================================================================
# Types  (Sec 7.2)
# =====================================================================

@dataclass(frozen=True)
class Ty:
    """A synopsis type.

    `frame` is the frame index phi of Sec 7.4. Types with different
    frame indices are DIFFERENT TYPES -- that single device is what
    gives Thm 9.6.

    `dim` carries the coefficient count of coord_phi, so that comparing
    embeddings of different dimension is a type error (Sec 10.2).
    """
    name: str
    frame: str | None = None
    dim: int | None = None

    def __str__(self) -> str:
        s = self.name
        if self.frame:
            s += f"_{self.frame}"
        if self.dim is not None:
            s += f"<{self.dim}>"
        return s


SEQ = Ty("seq")
VERDICT = Ty("verdict")
REAL = Ty("real")
DEPTH = Ty("depth")

# The three result types of Rule 6.11 -- deliberately not unified.
RESULT_TYPES = {"profile", "ranked", "verdict"}


# =====================================================================
# AST
# =====================================================================

@dataclass
class Node:
    line: int = 0


@dataclass
class Open(Node):
    name: str = ""
    path: str = ""


@dataclass
class MethodDecl(Node):
    name: str = ""
    spec: str = ""
    params: dict = field(default_factory=dict)


@dataclass
class Project(Node):
    src: "Expr | None" = None
    projector: str = ""
    args: dict = field(default_factory=dict)


@dataclass
class Compare(Node):
    left: "Expr | None" = None
    right: "Expr | None" = None
    method: str = ""
    params: dict = field(default_factory=dict)


@dataclass
class Detect(Node):
    kind: str = ""           # peaks | top
    src: "Expr | None" = None
    params: dict = field(default_factory=dict)


@dataclass
class Align(Node):
    central: tuple = ()
    resp: tuple = ()
    corrs: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    has_response_clause: bool = True


@dataclass
class UnitExpr(Node):
    src: "Expr | None" = None
    anchors: int = 0


@dataclass
class ResponseExpr(Node):
    src: "Expr | None" = None
    method: str | None = None   # None means ANONYMOUS -> Rule 6.7 violation


@dataclass
class NearestUnit(Node):
    net: "Expr | None" = None
    to: "Expr | None" = None


@dataclass
class CorrExpr(Node):
    src: "Expr | None" = None
    dst: "Expr | None" = None
    by: str = ""


@dataclass
class Var(Node):
    name: str = ""


@dataclass
class Num(Node):
    value: float = 0.0


Expr = object


@dataclass
class Let(Node):
    name: str = ""
    ann: str | None = None
    expr: Expr | None = None


@dataclass
class Bind(Node):
    value: str = ""
    residue: str = ""
    expr: Expr | None = None


@dataclass
class Relax(Node):
    target: str = ""
    params: dict = field(default_factory=dict)


@dataclass
class Sweep(Node):
    var: str = ""
    lo: float = 0.0
    hi: float = 0.0
    step: float = 1.0
    values: list | None = None
    body: list = field(default_factory=list)


@dataclass
class For(Node):
    var: str = ""
    src: Expr | None = None
    guard: str | None = None
    body: list = field(default_factory=list)


@dataclass
class Claim(Node):
    text: str = ""
    expr: Expr | None = None


@dataclass
class Record(Node):
    names: list = field(default_factory=list)


@dataclass
class Drop(Node):
    name: str = ""


@dataclass
class FrameBlock(Node):
    name: str = ""
    body: list = field(default_factory=list)


@dataclass
class Program(Node):
    decls: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    report: str | None = None


# =====================================================================
# Parser
# =====================================================================

class Parser:
    def __init__(self, toks: list[Token]):
        self.toks = toks
        self.i = 0

    # -- helpers -----------------------------------------------------
    @property
    def cur(self) -> Token:
        return self.toks[self.i]

    def at(self, text: str) -> bool:
        return self.cur.text == text and self.cur.kind in ("kw", "punct", "range")

    def eat(self, text: str) -> Token:
        if not self.at(text):
            raise ParseError(f"expected {text!r}, found {self.cur.text!r}", self.cur.line)
        t = self.cur
        self.i += 1
        return t

    def ident(self) -> str:
        if self.cur.kind != "ident":
            raise ParseError(f"expected identifier, found {self.cur.text!r}", self.cur.line)
        t = self.cur
        self.i += 1
        return t.text

    def number(self) -> float:
        if self.cur.kind != "num":
            raise ParseError(f"expected number, found {self.cur.text!r}", self.cur.line)
        t = self.cur
        self.i += 1
        return float(t.value)

    def string(self) -> str:
        if self.cur.kind != "string":
            raise ParseError(f"expected string, found {self.cur.text!r}", self.cur.line)
        t = self.cur
        self.i += 1
        return str(t.value)

    # -- params ------------------------------------------------------
    def params(self) -> dict:
        """param { (";"|",") param }  -- a param is `ident = num`."""
        out: dict = {}
        while self.cur.kind == "ident":
            k = self.ident()
            self.eat("=")
            out[k] = self.number()
            if self.at(";") or self.at(","):
                self.i += 1
            else:
                break
        return out

    # -- program -----------------------------------------------------
    def program(self) -> Program:
        p = Program(line=1)
        while self.at("open") or self.at("method") or self.at("require"):
            p.decls.append(self.decl())
        while self.at("under"):
            p.frames.append(self.frame_block())
        if self.at("report"):
            ln = self.eat("report").line
            self.eat("to")
            p.report = self.string()
        else:
            raise ParseError(
                "every program must end with `report to <file>` (Rule 6.11: "
                "programs emit reports)", self.cur.line)
        if self.cur.kind != "eof":
            raise ParseError(f"trailing input {self.cur.text!r}", self.cur.line)
        return p

    def decl(self):
        if self.at("open"):
            ln = self.eat("open").line
            n = self.ident()
            self.eat("=")
            return Open(line=ln, name=n, path=self.string())
        if self.at("method"):
            ln = self.eat("method").line
            n = self.ident()
            self.eat("=")
            spec = self.ident()
            self.eat("(")
            ps = self.params()
            self.eat(")")
            return MethodDecl(line=ln, name=n, spec=spec, params=ps)
        raise ParseError(f"unexpected declaration {self.cur.text!r}", self.cur.line)

    def frame_block(self) -> FrameBlock:
        ln = self.eat("under").line
        name = self.ident()
        self.eat("{")
        body = []
        while not self.at("}"):
            if self.cur.kind == "eof":
                raise ParseError("unclosed `under` block", ln)
            body.append(self.stmt())
        self.eat("}")
        return FrameBlock(line=ln, name=name, body=body)

    def stmt(self):
        t = self.cur
        if self.at("let"):
            ln = self.eat("let").line
            n = self.ident()
            ann = None
            if self.at(":"):
                self.i += 1
                ann = self.ident()
            self.eat("=")
            return Let(line=ln, name=n, ann=ann, expr=self.expr())
        if self.at("bind"):
            ln = self.eat("bind").line
            v = self.ident()
            self.eat(",")
            r = self.ident()
            self.eat("=")
            return Bind(line=ln, value=v, residue=r, expr=self.expr())
        if self.at("relax"):
            ln = self.eat("relax").line
            tgt = self.ident()
            self.eat("until")
            self.eat("quiescent")
            self.eat("{")
            ps = self.params()
            self.eat("}")
            return Relax(line=ln, target=tgt, params=ps)
        if self.at("sweep"):
            return self.sweep_stmt()
        if self.at("for"):
            return self.for_stmt()
        if self.at("claim"):
            ln = self.eat("claim").line
            txt = self.string()
            self.eat("=")
            return Claim(line=ln, text=txt, expr=self.expr())
        if self.at("record"):
            ln = self.eat("record").line
            names = [self.ident()]
            while self.at(","):
                self.i += 1
                names.append(self.ident())
            return Record(line=ln, names=names)
        if self.at("drop"):
            ln = self.eat("drop").line
            return Drop(line=ln, name=self.ident())
        raise ParseError(f"unexpected statement {t.text!r}", t.line)

    def sweep_stmt(self) -> Sweep:
        ln = self.eat("sweep").line
        var = self.ident()
        self.eat("in")
        if self.at("["):
            self.i += 1
            vals = [self.number()]
            while self.at(","):
                self.i += 1
                vals.append(self.number())
            self.eat("]")
            node = Sweep(line=ln, var=var, values=vals)
        else:
            lo = self.number()
            self.eat("..")
            hi = self.number()
            self.eat("step")
            st = self.number()
            node = Sweep(line=ln, var=var, lo=lo, hi=hi, step=st)
        self.eat("{")
        while not self.at("}"):
            if self.cur.kind == "eof":
                raise ParseError("unclosed `sweep` block", ln)
            node.body.append(self.stmt())
        self.eat("}")
        return node

    def for_stmt(self) -> For:
        ln = self.eat("for").line
        var = self.ident()
        self.eat("in")
        self.eat("items")
        self.eat("(")
        src = self.expr()
        self.eat(")")
        guard = None
        if self.at("where"):
            self.i += 1
            # guard: separation(ident) > num
            self.eat("separation")
            self.eat("(")
            gv = self.ident()
            self.eat(")")
            op = self.cur.text
            self.i += 1
            thr = self.number()
            guard = f"separation({gv}) {op} {thr}"
        node = For(line=ln, var=var, src=src, guard=guard)
        self.eat("{")
        while not self.at("}"):
            if self.cur.kind == "eof":
                raise ParseError("unclosed `for` block", ln)
            node.body.append(self.stmt())
        self.eat("}")
        return node

    # -- expressions -------------------------------------------------
    def expr(self):
        t = self.cur
        ln = t.line

        if self.at("project"):
            self.i += 1
            src = self.expr()
            self.eat("by")
            proj = self.cur.text
            self.i += 1
            self.eat("(")
            args: dict = {}
            if proj == "channels":
                args["enc"] = self.ident()
            elif not self.at(")"):
                args = self.params()
            self.eat(")")
            return Project(line=ln, src=src, projector=proj, args=args)

        if self.at("compare"):
            self.i += 1
            left = self.expr()
            self.eat("against")
            right = self.expr()
            self.eat("by")
            meth = self.cur.text
            self.i += 1
            ps: dict = {}
            if self.at("("):
                self.i += 1
                if self.cur.kind == "ident" and self.toks[self.i + 1].text != "=":
                    ps["mode"] = self.ident()
                elif not self.at(")"):
                    ps = self.params()
                self.eat(")")
            return Compare(line=ln, left=left, right=right, method=meth, params=ps)

        if self.at("detect"):
            self.i += 1
            kind = self.cur.text
            self.i += 1
            self.eat("in")
            src = self.expr()
            self.eat("{")
            ps = self.params()
            self.eat("}")
            return Detect(line=ln, kind=kind, src=src, params=ps)

        if self.at("align"):
            self.i += 1
            self.eat("central")
            self.eat("(")
            ca = self.expr()
            self.eat(",")
            cb = self.expr()
            self.eat(")")
            has_resp = self.at("response")
            ra = rb = None
            if has_resp:
                self.i += 1
                self.eat("(")
                ra = self.expr()
                self.eat(",")
                rb = self.expr()
                self.eat(")")
            self.eat("under")
            corrs = [self.ident()]
            while self.at(","):
                self.i += 1
                corrs.append(self.ident())
            self.eat("{")
            ps = self.params()
            self.eat("}")
            return Align(line=ln, central=(ca, cb),
                         resp=(ra, rb) if has_resp else (),
                         corrs=corrs, params=ps,
                         has_response_clause=has_resp)

        if self.at("unit"):
            self.i += 1
            src = self.expr()
            self.eat("anchors")
            return UnitExpr(line=ln, src=src, anchors=int(self.number()))

        if self.at("response"):
            self.i += 1
            src = self.expr()
            meth = None
            if self.at("by"):
                self.i += 1
                meth = self.ident()
            return ResponseExpr(line=ln, src=src, method=meth)

        if self.at("nearest_unit"):
            self.i += 1
            net = self.expr()
            self.eat("to")
            return NearestUnit(line=ln, net=net, to=self.expr())

        if self.at("corr"):
            self.i += 1
            self.eat("from")
            a = self.expr()
            self.eat("to")
            b = self.expr()
            self.eat("by")
            return CorrExpr(line=ln, src=a, dst=b, by=self.ident())

        if t.kind == "ident":
            self.i += 1
            return Var(line=ln, name=t.text)

        if t.kind == "num":
            self.i += 1
            return Num(line=ln, value=float(t.value))

        raise ParseError(f"cannot parse expression at {t.text!r}", ln)


def parse(src: str) -> Program:
    return Parser(tokenise(src)).program()


# =====================================================================
# Typechecker
# =====================================================================

# Required parameter sets -- Thm 9.7. There are no defaults anywhere;
# omission is an error, never a silent fill-in.
REQUIRED_PARAMS = {
    "peaks": {"z", "min_distance", "min_score"},
    "top": {"k", "depth"},
    "align": {"theta"},
    "relax": {"eta", "theta"},
    "smith_waterman": {"match", "mismatch", "gap"},
}


@dataclass
class Report:
    """What a checked program emits (Sec 11)."""
    frames: list = field(default_factory=list)
    parameters: dict = field(default_factory=dict)
    residues: dict = field(default_factory=dict)
    abandoned: list = field(default_factory=list)
    claims: list = field(default_factory=list)
    bounds: dict = field(default_factory=dict)
    iterations: dict = field(default_factory=dict)
    responses: list = field(default_factory=list)


class Checker:
    def __init__(self) -> None:
        self.env: dict[str, Ty] = {}
        self.methods: dict[str, dict] = {}
        self.pending_residue: dict[str, int] = {}
        self.report = Report()
        self.frame: str | None = None

    # -- entry -------------------------------------------------------
    def check(self, p: Program) -> Report:
        for d in p.decls:
            if isinstance(d, Open):
                self.env[d.name] = SEQ
            elif isinstance(d, MethodDecl):
                self.methods[d.name] = d.params
                self.report.responses.append(
                    {"name": d.name, "spec": d.spec, "params": d.params})

        for fb in p.frames:
            self.frame = fb.name
            self.report.frames.append(fb.name)
            self.pending_residue = {}
            for s in fb.body:
                self.stmt(s)
            # Cor 9.8: residue must be consumed before the frame closes.
            if self.pending_residue:
                nm, ln = sorted(self.pending_residue.items())[0]
                raise ResidueError(
                    f"residue `{nm}` is bound but never recorded or dropped; "
                    f"every comparison value carries (beta, varrho) and the "
                    f"residue must be accounted for", ln)
            self.frame = None
        return self.report

    # -- statements --------------------------------------------------
    def stmt(self, s) -> None:
        if isinstance(s, Let):
            ty = self.expr(s.expr)
            if s.ann and s.ann != ty.name:
                raise TypeError_(
                    f"`{s.name}` annotated {s.ann} but has type {ty}", s.line)
            self.env[s.name] = ty

        elif isinstance(s, Bind):
            ty = self.expr(s.expr)
            self.env[s.value] = ty
            self.env[s.residue] = REAL
            self.pending_residue[s.residue] = s.line

        elif isinstance(s, Record):
            for n in s.names:
                if n not in self.env:
                    raise ScopeError(f"`{n}` is not in scope", s.line)
                self.pending_residue.pop(n, None)
                self.report.residues[n] = str(self.env[n])

        elif isinstance(s, Drop):
            if s.name not in self.env:
                raise ScopeError(f"`{s.name}` is not in scope", s.line)
            self.pending_residue.pop(s.name, None)
            self.report.abandoned.append(
                {"name": s.name, "line": s.line,
                 "note": f"residue abandoned at line {s.line}"})

        elif isinstance(s, Claim):
            ty = self.expr(s.expr)
            self.report.claims.append({"text": s.text, "type": str(ty)})

        elif isinstance(s, Relax):
            self.check_params("relax", s.params, s.line)
            eta = s.params["eta"]
            if eta <= 0.0:
                # The exact message the manuscript prints (Sec 7.6).
                raise TerminationError(
                    "`relax` requires a step floor bounded below (Assumption: "
                    f"effective update).\n  eta = {eta} is not positive; no "
                    "termination bound can be emitted.", s.line)
            if s.target not in self.env:
                raise ScopeError(f"`{s.target}` is not in scope", s.line)
            self.report.bounds[s.target] = {"eta": eta, "theta": s.params["theta"]}

        elif isinstance(s, Sweep):
            # Thm 9.1: the trip count is fixed on entry.
            if s.values is not None:
                count = len(s.values)
            else:
                if s.step <= 0.0:
                    raise TerminationError(
                        "`sweep` requires a positive step", s.line)
                count = int(math.floor((s.hi - s.lo) / s.step + 1e-9)) + 1
            self.report.iterations[s.var] = count
            for b in s.body:
                self.stmt(b)

        elif isinstance(s, For):
            self.expr(s.src)
            self.env[s.var] = Ty("unit", self.frame)
            self.report.iterations[s.var] = "finite: items(N)"
            if s.guard:
                self.report.parameters[f"guard:{s.var}"] = s.guard
            for b in s.body:
                self.stmt(b)

        else:
            raise TypeError_(f"unknown statement {type(s).__name__}", getattr(s, "line", 0))

    # -- parameters --------------------------------------------------
    def check_params(self, what: str, got: dict, line: int) -> None:
        need = REQUIRED_PARAMS.get(what, set())
        missing = need - set(got)
        if missing:
            raise ParameterError(
                f"`{what}` requires {sorted(need)}; missing "
                f"{sorted(missing)}. There are no defaults (Rule: no default "
                f"thresholds) -- state the value or the report cannot print it.",
                line)
        for k, v in got.items():
            self.report.parameters[f"{what}.{k}"] = v

    # -- expressions -------------------------------------------------
    def expr(self, e) -> Ty:
        if isinstance(e, Var):
            if e.name not in self.env:
                raise ScopeError(f"`{e.name}` is not in scope", e.line)
            ty = self.env[e.name]
            # Thm 9.6: a value made in another frame is a different type.
            if ty.frame is not None and ty.frame != self.frame:
                raise ScopeError(
                    f"`{e.name}` has type {ty} but the enclosing frame is "
                    f"`{self.frame}`; {ty.name}_{ty.frame} and "
                    f"{ty.name}_{self.frame} are distinct types", e.line)
            return ty

        if isinstance(e, Num):
            return REAL

        if isinstance(e, Project):
            src = self.expr(e.src)
            if src != SEQ and src.name not in ("seq", "net"):
                raise TypeError_(
                    f"`project` takes a seq, got {src}", e.line)
            p = e.projector
            if p == "channels":
                return Ty("frame", self.frame)
            if p == "spectral":
                if "coeffs" not in e.args:
                    raise ParameterError(
                        "spectral(...) requires `coeffs`", e.line)
                return Ty("coord", self.frame, dim=int(e.args["coeffs"]))
            if p == "contact":
                if "medium" not in e.args:
                    raise ParameterError(
                        "contact(...) requires `medium`", e.line)
                self.report.parameters["contact.medium"] = e.args["medium"]
                return Ty("net", self.frame)
            if p == "cardinal":
                return Ty("frame", self.frame)
            raise TypeError_(f"unknown projector `{p}`", e.line)

        if isinstance(e, Compare):
            lt = self.expr(e.left)
            rt = self.expr(e.right)
            m = e.method
            # Rule 6.11: one keyword, three result types. The index sets
            # genuinely differ, so a single return type would erase what
            # makes each family correct.
            if m == "xcorr":
                if lt.name != "frame" or rt.name != "frame":
                    raise TypeError_(
                        f"xcorr compares frame against frame, got {lt} and {rt}",
                        e.line)
                return Ty("profile", self.frame)
            if m == "shader":
                if lt.name != "coord" or rt.name != "coord":
                    raise TypeError_(
                        f"shader compares coord against coord, got {lt} and {rt}",
                        e.line)
                if lt.dim != rt.dim:
                    raise TypeError_(
                        f"embedding dimensions differ: {lt.dim} vs {rt.dim}; "
                        f"the dimension is part of coord_phi", e.line)
                return Ty("ranked", self.frame)
            if m == "smith_waterman":
                self.check_params("smith_waterman", e.params, e.line)
                return Ty("ranked", self.frame)
            if m == "jaccard":
                return Ty("ranked", self.frame)
            if m == "demand":
                return Ty("verdict", self.frame)
            raise TypeError_(f"unknown method `{m}`", e.line)

        if isinstance(e, Detect):
            st = self.expr(e.src)
            self.check_params(e.kind, e.params, e.line)
            if e.kind == "peaks":
                if st.name != "profile":
                    raise TypeError_(
                        f"`detect peaks` consumes profile (lag-indexed), got "
                        f"{st}", e.line)
                return Ty("peaks", self.frame)
            if e.kind == "top":
                if st.name != "ranked":
                    raise TypeError_(
                        f"`detect top` consumes ranked (entry-indexed), got "
                        f"{st}", e.line)
                return Ty("coord", self.frame, dim=st.dim)
            raise TypeError_(f"unknown detector `{e.kind}`", e.line)

        if isinstance(e, UnitExpr):
            st = self.expr(e.src)
            if st.name not in ("net", "unit"):
                raise TypeError_(f"`unit` takes a net, got {st}", e.line)
            return Ty("unit", self.frame)

        if isinstance(e, ResponseExpr):
            st = self.expr(e.src)
            if e.method is None:
                # Rule 6.7 -- responses may not be anonymous, because the
                # verdict's independence of this choice is OPEN (Sec 13).
                raise ParameterError(
                    "a response must name its perturbation map "
                    "(`response X by <method>`); anonymous responses are "
                    "rejected because the verdict's independence of this "
                    "choice is an open problem", e.line)
            if e.method not in self.methods:
                raise ScopeError(
                    f"response method `{e.method}` is not declared", e.line)
            return Ty("response", self.frame)

        if isinstance(e, NearestUnit):
            self.expr(e.net)
            self.expr(e.to)
            return Ty("unit", self.frame)

        if isinstance(e, CorrExpr):
            a = self.expr(e.src)
            b = self.expr(e.dst)
            if a.name != b.name:
                raise TypeError_(
                    f"`corr` relates like to like, got {a} and {b}", e.line)
            return Ty("corr", self.frame)

        if isinstance(e, Align):
            # Thm 9.5: align is FOUR-ARY. There is no two-argument form.
            if not e.has_response_clause:
                raise ArityError(
                    "`align` requires four columns: central(a,b) AND "
                    "response(a,b). There is no two-column form, so a verdict "
                    "cannot be reached from content alone (see the "
                    "false-friend separation).", e.line)
            for x in e.central:
                t = self.expr(x)
                if t.name != "unit":
                    raise TypeError_(
                        f"central columns take units, got {t}", e.line)
            for x in e.resp:
                t = self.expr(x)
                if t.name != "response":
                    raise TypeError_(
                        f"response columns take responses, got {t}", e.line)
            if len(e.corrs) != 2:
                raise ArityError(
                    f"`align` needs two correspondences (central and "
                    f"response), got {len(e.corrs)}", e.line)
            for c in e.corrs:
                if c not in self.env:
                    raise ScopeError(f"`{c}` is not in scope", e.line)
            self.check_params("align", e.params, e.line)
            return Ty("verdict", self.frame)

        raise TypeError_(f"cannot type {type(e).__name__}", getattr(e, "line", 0))


def check(src: str) -> Report:
    """Parse and typecheck. Raises SynopsisError on refusal."""
    return Checker().check(parse(src))


def accepts(src: str) -> tuple[bool, str]:
    try:
        check(src)
        return True, ""
    except SynopsisError as exc:
        return False, str(exc)
