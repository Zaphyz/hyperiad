"""Read Wolfram Language sources without a Wolfram kernel.

Reference implementations (e.g. the Hypergraphs paclet) are read, never run:
the mathematics is reimplemented here and checked against the reference's own
test cases. This module provides what that needs:

* :func:`split_statements` -- top-level statements with line ranges
  (strings, nested comments, brackets and line continuation handled);
* :func:`classify` -- what a statement defines (usage message, definition,
  upvalue, options, formatting, attributes, ...) and for which symbol;
* :class:`WLIndex` -- symbol -> usage + definition chunks, plus the test
  cases in the reference's test scripts, searchable by symbol.

It is a reader, not a full parser: it recovers structure reliably enough to
point at and extract code, which agents then translate.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

SYMBOL = r"[$A-Za-z][$A-Za-z0-9`]*"
_SYMBOL_RE = re.compile(SYMBOL)

# Text at the end of a line that leaves the expression incomplete, so it continues.
_CONTINUES = re.compile(r"(\^?:=|\^?=|<>|->|:>|&&|\|\||,|@@@|@@|@|//\.|//|/;|/\.|/:|/@|::|[+\-*/^|.~:<>!=?])$")
_POSTFIX_COMPLETE = ("++", "--", "!")  # a trailing & (Function) never matches _CONTINUES

_ASSIGN_RE = re.compile(r"(?<![=!<>:^])(\^:=|\^=|:=|=)(?![=!])")
_TAG_RE = re.compile(rf"^\s*({SYMBOL})\s*/:\s*")
_SUBVALUE_RE = re.compile(rf"^\s*\(\s*{SYMBOL}_+({SYMBOL})\s*\)\s*\[")  # (x_Head)[args] := ...
_MESSAGE_RE = re.compile(rf"^\s*({SYMBOL})::(\w+)\s*$")
# Wrapper[f[...]], Wrapper[x_f, ...] or Wrapper[x : f[...]]: the defined symbol is f.
_WRAPPED_RE = re.compile(
    rf"^\s*(Options|Attributes|SyntaxInformation|Default|Format|MakeBoxes|N|HoldPattern)\s*\[\s*"
    rf"(?:{SYMBOL}_+(?=[$A-Za-z])|{SYMBOL}\s*:\s*(?:HoldPattern\s*\[\s*)?)?({SYMBOL})"
)
_DIRECTIVES = {
    "BeginPackage",
    "Begin",
    "End",
    "EndPackage",
    "Needs",
    "Get",
    "Protect",
    "Unprotect",
    "ClearAll",
    "Clear",
}
_WRAPPER_KIND = {
    "Options": "options",
    "Attributes": "attributes",
    "SyntaxInformation": "syntax",
    "Default": "default",
    "Format": "format",
    "MakeBoxes": "format",
    "N": "numeric",
    "HoldPattern": "definition",
}


# -- scanning -----------------------------------------------------------


@dataclass(frozen=True)
class Token:
    kind: str  # string | comment | newline | open | close | semi | text
    text: str
    line: int  # line where the token starts


def tokenize(src: str) -> list[Token]:
    src = src.lstrip("﻿")
    if src.startswith("#!"):
        src = "(*" + src[2 : src.find("\n") if "\n" in src else len(src)] + "*)" + src[src.find("\n") :]
    tokens: list[Token] = []
    i, n, line = 0, len(src), 1
    text_start = None

    def flush_text(end: int) -> None:
        nonlocal text_start
        if text_start is not None and end > text_start:
            tokens.append(Token("text", src[text_start:end], line))
        text_start = None

    while i < n:
        c = src[i]
        two = src[i : i + 2]
        if c == '"':
            flush_text(i)
            j, start_line = i + 1, line
            while j < n and src[j] != '"':
                if src[j] == "\\":
                    j += 1
                if j < n and src[j] == "\n":
                    line += 1
                j += 1
            tokens.append(Token("string", src[i : j + 1], start_line))
            i = j + 1
        elif two == "(*":
            flush_text(i)
            j, depth, start_line = i + 2, 1, line
            while j < n and depth:
                if src[j : j + 2] == "(*":
                    depth, j = depth + 1, j + 2
                elif src[j : j + 2] == "*)":
                    depth, j = depth - 1, j + 2
                else:
                    line += src[j] == "\n"
                    j += 1
            tokens.append(Token("comment", src[i:j], start_line))
            i = j
        elif c == "\\" and src[i + 1 : i + 2] == "[":
            # named character such as \[Alpha]: part of the surrounding text, not a bracket
            j = src.find("]", i)
            j = n if j < 0 else j + 1
            if text_start is None:
                text_start = i
            i = j
        elif c == "\n":
            flush_text(i)
            tokens.append(Token("newline", c, line))
            line += 1
            i += 1
        elif two in ("<|", "|>"):
            flush_text(i)
            tokens.append(Token("open" if two == "<|" else "close", two, line))
            i += 2
        elif c in "[({":
            flush_text(i)
            tokens.append(Token("open", c, line))
            i += 1
        elif c in "])}":
            flush_text(i)
            tokens.append(Token("close", c, line))
            i += 1
        elif c == ";" and src[i + 1 : i + 2] != ";" and (i == 0 or src[i - 1] not in ";/"):
            # a lone ; ends a statement; ;; is Span and /; is Condition
            flush_text(i)
            tokens.append(Token("semi", c, line))
            i += 1
        else:
            if text_start is None:
                text_start = i
            i += 1
    flush_text(n)
    return tokens


@dataclass(frozen=True)
class Statement:
    text: str
    start_line: int
    end_line: int
    path: str = ""
    comment: str = ""  # the comment block immediately before the statement, if any

    def __str__(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


def split_statements(src: str, path: str = "") -> list[Statement]:
    """Split source into top-level statements.

    A statement ends at a top-level ``;`` or at a top-level newline after
    which the expression is complete (WL package-file semantics).
    """
    out: list[Statement] = []
    cur: list[Token] = []
    depth = 0
    last_comment = ""

    def significant(toks: list[Token]) -> str:
        return "".join(t.text for t in toks if t.kind not in ("comment", "newline")).strip()

    def flush() -> None:
        nonlocal cur, last_comment
        text = "".join(t.text for t in cur).strip()
        if significant(cur):
            end = max(t.line + t.text.count("\n") for t in cur)
            first = next(t for t in cur if t.kind != "newline")
            out.append(Statement(text, first.line, end, path, last_comment))
            last_comment = ""
        cur = []

    for tok in tokenize(src):
        if tok.kind == "comment" and depth == 0 and not significant(cur):
            last_comment = tok.text[2:-2].strip()
            cur = []
            continue
        if tok.kind == "newline" and depth == 0:
            sig = significant(cur)
            if not sig:
                cur = []
                continue
            if sig.endswith(_POSTFIX_COMPLETE) or not _CONTINUES.search(sig):
                flush()
                continue
        if tok.kind == "open":
            depth += 1
        elif tok.kind == "close":
            depth = max(depth - 1, 0)
        cur.append(tok)
        if tok.kind == "semi" and depth == 0:
            flush()
    flush()
    return out


def split_args(text: str) -> tuple[str, list[str]]:
    """``f[a, b, c]`` -> ``("f", ["a", "b", "c"])`` using top-level commas."""
    m = re.match(rf"\s*({SYMBOL})\s*\[", text)
    if not m:
        raise ValueError(f"not a call: {text[:40]!r}")
    args, depth, cur = [], 0, []
    for tok in tokenize(text[m.end() - 1 :]):
        if tok.kind == "open":
            depth += 1
            if depth == 1:
                continue
        elif tok.kind == "close":
            depth -= 1
            if depth == 0:
                break
        if depth == 1 and tok.kind == "text" and "," in tok.text:
            parts = tok.text.split(",")
            cur.append(parts[0])
            for p in parts[1:]:
                args.append("".join(cur).strip())
                cur = [p]
            continue
        cur.append(tok.text)
    if "".join(cur).strip() or args:
        args.append("".join(cur).strip())
    return m.group(1), args


def string_literals(text: str) -> str:
    """Concatenated, unescaped string literals of an expression (e.g. a usage message)."""
    parts = [t.text[1:-1] for t in tokenize(text) if t.kind == "string"]
    joined = "".join(parts)
    return re.sub(r"\\(.)", lambda m: {"n": "\n", "t": "\t"}.get(m.group(1), m.group(1)), joined)


# -- classification -----------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """A statement that defines something about a symbol."""

    kind: str  # message | definition | upvalue | subvalue | options | attributes | format | syntax | default | numeric
    symbol: str
    statement: Statement
    message: str = ""  # message name for kind == "message" (e.g. "usage")
    context: str = ""

    @property
    def text(self) -> str:
        return self.statement.text

    @property
    def location(self) -> str:
        return str(self.statement)


def _top_level_text(text: str) -> str:
    """The statement with bracketed parts and strings blanked out (same length)."""
    out, depth = [], 0
    for tok in tokenize(text):
        if tok.kind == "open":
            depth += 1
            out.append(" " * len(tok.text))
        elif tok.kind == "close":
            depth -= 1
            out.append(" " * len(tok.text))
        elif depth or tok.kind in ("string", "comment"):
            out.append(re.sub(r"[^\n]", " ", tok.text))
        else:
            out.append(tok.text)
    return "".join(out)


def classify(st: Statement, context: str = "") -> list[Chunk]:
    text = st.text.rstrip(";").strip()
    flat = _top_level_text(text)
    m = _ASSIGN_RE.search(flat)
    if m is None:
        call = re.match(rf"\s*({SYMBOL})\s*\[", text)
        if call and call.group(1) == "SetAttributes":
            _, args = split_args(text)
            names = _SYMBOL_RE.findall(args[0]) if args else []
            return [Chunk("attributes", s, st, context=context) for s in names]
        return []
    lhs = text[: m.start()]
    tag = _TAG_RE.match(lhs)
    if tag:
        return [Chunk("upvalue", tag.group(1), st, context=context)]
    sub = _SUBVALUE_RE.match(lhs)
    if sub:
        return [Chunk("subvalue", sub.group(1), st, context=context)]
    msg = _MESSAGE_RE.match(lhs)
    if msg:
        return [Chunk("message", msg.group(1), st, message=msg.group(2), context=context)]
    wrapped = _WRAPPED_RE.match(lhs)
    if wrapped:
        return [Chunk(_WRAPPER_KIND[wrapped.group(1)], wrapped.group(2), st, context=context)]
    head = re.match(rf"\s*({SYMBOL})", lhs)
    if head and head.group(1) not in _DIRECTIVES:
        return [Chunk("definition", head.group(1), st, context=context)]
    return []


# -- index --------------------------------------------------------------


@dataclass
class SymbolInfo:
    name: str
    chunks: list[Chunk] = field(default_factory=list)

    @property
    def usage(self) -> str | None:
        for c in self.chunks:
            if c.kind == "message" and c.message == "usage":
                return string_literals(c.text[c.text.index("=") + 1 :])
        return None

    @property
    def definitions(self) -> list[Chunk]:
        return [c for c in self.chunks if c.kind != "message"]

    @property
    def files(self) -> list[str]:
        return sorted({c.statement.path for c in self.chunks})

    def span(self, kinds: Iterable[str] = ("definition", "upvalue", "subvalue")) -> tuple[str, int, int] | None:
        """(path, first line, last line) covering this symbol's main definitions in one file."""
        defs = [c for c in self.chunks if c.kind in set(kinds)] or self.chunks
        if not defs:
            return None
        path = defs[0].statement.path
        same = [c for c in defs if c.statement.path == path]
        return path, min(c.statement.start_line for c in same), max(c.statement.end_line for c in same)


@dataclass(frozen=True)
class TestCase:
    path: str
    line: int
    form: str  # the test head, e.g. "check" or "VerificationTest"
    label: str
    got: str
    want: str | None

    def mentions(self, symbol: str) -> bool:
        pat = re.compile(rf"(?<![\w$`]){re.escape(symbol)}(?![\w$])")
        return bool(pat.search(self.got) or (self.want and pat.search(self.want)))


# Test call forms and the role of their positional arguments.
TEST_FORMS: dict[str, tuple[str, ...]] = {
    "VerificationTest": ("got", "want"),
    "check": ("label", "got", "want"),
}


class WLIndex:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.symbols: dict[str, SymbolInfo] = {}
        self.statements: dict[str, list[Statement]] = {}
        self.tests: list[TestCase] = []

    @classmethod
    def build(
        cls,
        root: Path | str,
        sources: Iterable[str] = ("src/**/*.wl",),
        tests: Iterable[str] = ("tests/**/*.wls", "tests/**/*.wlt"),
    ) -> WLIndex:
        index = cls(root)
        for pattern in sources:
            for path in sorted(index.root.glob(pattern)):
                index.add_source(path)
        for pattern in tests:
            for path in sorted(index.root.glob(pattern)):
                index.add_tests(path)
        return index

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def add_source(self, path: Path) -> None:
        rel = self._rel(path)
        stmts = split_statements(path.read_text(encoding="utf-8"), rel)
        self.statements[rel] = stmts
        context = ""
        for st in stmts:
            call = re.match(r'\s*(BeginPackage|Begin|EndPackage|End)\s*\[\s*(?:"([^"]+)")?', st.text)
            if call:
                head, arg = call.groups()
                if head == "BeginPackage":
                    context = arg or ""
                elif head == "Begin":
                    context += (arg or "").lstrip("`")
                elif head == "End":
                    context = context[: context.rstrip("`").rfind("`") + 1]
                else:
                    context = ""
                continue
            for chunk in classify(st, context):
                name = chunk.symbol.rsplit("`", 1)[-1]
                self.symbols.setdefault(name, SymbolInfo(name)).chunks.append(chunk)

    def add_tests(self, path: Path) -> None:
        rel = self._rel(path)
        for st in split_statements(path.read_text(encoding="utf-8"), rel):
            m = re.match(rf"\s*({SYMBOL})\s*\[", st.text)
            if not m or m.group(1) not in TEST_FORMS:
                continue
            if _ASSIGN_RE.search(_top_level_text(st.text)):  # the helper's own definition, check[...] := ...
                continue
            _, args = split_args(st.text)
            roles = dict(zip(TEST_FORMS[m.group(1)], args, strict=False))
            if "got" not in roles:
                continue
            label = string_literals(roles.get("label", "")) or f"{rel}:{st.start_line}"
            self.tests.append(TestCase(rel, st.start_line, m.group(1), label, roles["got"], roles.get("want")))

    # -- queries ---------------------------------------------------------

    def __contains__(self, name: object) -> bool:
        return name in self.symbols

    def symbol(self, name: str) -> SymbolInfo:
        return self.symbols[name.rsplit("`", 1)[-1]]

    def public_symbols(self) -> list[str]:
        """Symbols with a usage message: the reference's documented API."""
        return sorted(n for n, s in self.symbols.items() if s.usage is not None)

    def tests_for(self, name: str) -> list[TestCase]:
        return [t for t in self.tests if t.mentions(name)]

    def search(self, text: str) -> list[str]:
        needle = text.lower()
        return sorted(n for n, s in self.symbols.items() if needle in n.lower() or needle in (s.usage or "").lower())

    def lines(self, path: str, start: int, end: int) -> str:
        all_lines = (self.root / path).read_text(encoding="utf-8").lstrip("﻿").splitlines()
        return "\n".join(all_lines[start - 1 : end])

    def iter_chunks(self) -> Iterator[Chunk]:
        for info in self.symbols.values():
            yield from info.chunks
