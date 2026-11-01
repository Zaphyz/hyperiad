"""The text syntax of Hyperiad (provisional): tokenizer, parser and printer.

The syntax is close to Python, with a few Wolfram Language touches. It is
provisional and will be developed with the research. The printer is exact:
``parse(unparse(tree)) == tree`` for every tree the parser produces.

=====================  ===========================================  ===================
written                meaning                                      tree head
=====================  ===========================================  ===================
``f(x, y, k = v)``     call, with a keyword option                  ``f``, ``Option``
``[a, b]``             list                                         ``List``
``{k: v}``             association                                  ``Association``
``a[i, j]``            part, 1-based (``a["Order"]``: property)     ``Part``
``a -> b``             rule                                         ``Rule``
``x => body``          function (``(x, y) => body``)                ``Function``
``x = e``              assignment                                   ``Set``
``f(x, y) = body``     function definition                          ``Set``
``a + b``  ``a - b``   sum, difference                              ``Plus``, ``Subtract``
``a * b``  ``a / b``   product, quotient                            ``Times``, ``Divide``
``-a``                 negation                                     ``Minus``
``a ^ b``  ``a ** b``  power                                        ``Power``
``==  !=  <  <=  >  >=``  comparison                                ``Equal`` ...
``and  or  not``       logic                                        ``And``, ``Or``, ``Not``
``true false null``    constants
``# ...``              comment
=====================  ===========================================  ===================

Statements are separated by newlines or ``;``. Newlines inside brackets do not end
a statement. A cell's value is its last statement's; a trailing ``;`` suppresses it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from hyperiad.expr import Expr, Symbol

KEYWORDS = {"and", "or", "not", "true", "false", "null"}
_OPERATORS = ["->", "=>", "==", "!=", "<=", ">=", "**", "+", "-", "*", "/", "^", "<", ">", "=", ":", ",", ";"]
_TOKEN_RE = re.compile(
    r"""
    (?P<ws>[ \t\r]+|\\\n)
  | (?P<comment>\#[^\n]*)
  | (?P<newline>\n)
  | (?P<number>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<string>"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')
  | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<open>[(\[{])
  | (?P<close>[)\]}])
  | (?P<op>->|=>|==|!=|<=|>=|\*\*|[-+*/^<>=:,;])
    """,
    re.VERBOSE,
)

COMPARISONS = {"==": "Equal", "!=": "Unequal", "<": "Less", "<=": "LessEqual", ">": "Greater", ">=": "GreaterEqual"}
BINARY = {
    "->": ("Rule", 10, "right"),
    "or": ("Or", 20, "left"),
    "and": ("And", 30, "left"),
    **{op: (head, 50, "none") for op, head in COMPARISONS.items()},
    "+": ("Plus", 60, "left"),
    "-": ("Subtract", 60, "left"),
    "*": ("Times", 70, "left"),
    "/": ("Divide", 70, "left"),
    "^": ("Power", 90, "right"),
    "**": ("Power", 90, "right"),
}
PREC = {
    "Set": 1,
    "Function": 5,
    "Rule": 10,
    "Or": 20,
    "And": 30,
    "Not": 40,
    **dict.fromkeys(COMPARISONS.values(), 50),
    "Plus": 60,
    "Subtract": 60,
    "Times": 70,
    "Divide": 70,
    "Minus": 80,
    "Power": 90,
}
OP_TEXT = {
    "Rule": "->",
    "Or": "or",
    "And": "and",
    "Plus": "+",
    "Subtract": "-",
    "Times": "*",
    "Divide": "/",
    "Power": "^",
}
OP_TEXT.update({head: op for op, head in COMPARISONS.items()})
ASSOC = {head: assoc for head, _, assoc in BINARY.values()}
CALL_PREC = 100
ATOM_PREC = 110


class ParseError(SyntaxError):
    def __init__(self, message: str, line: int, col: int) -> None:
        super().__init__(f"{message} (line {line}, column {col})")
        self.msg_text, self.line, self.col = message, line, col


@dataclass(frozen=True)
class Token:
    kind: str  # number | string | name | open | close | op | newline | end
    text: str
    line: int
    col: int


def tokenize(src: str) -> list[Token]:
    tokens: list[Token] = []
    depth, line, line_start, pos = 0, 1, 0, 0
    while pos < len(src):
        m = _TOKEN_RE.match(src, pos)
        if not m:
            raise ParseError(f"unexpected character {src[pos]!r}", line, pos - line_start + 1)
        kind, text, col = m.lastgroup, m.group(), pos - line_start + 1
        pos = m.end()
        if kind in ("ws", "comment"):
            if text == "\\\n":
                line, line_start = line + 1, pos
            continue
        if kind == "newline":
            if depth == 0 and tokens and tokens[-1].kind != "newline":
                tokens.append(Token("newline", "\n", line, col))
            line, line_start = line + 1, pos
            continue
        if kind == "open":
            depth += 1
        elif kind == "close":
            depth = max(0, depth - 1)
        tokens.append(Token(kind, text, line, col))
    tokens.append(Token("end", "", line, pos - line_start + 1))
    return tokens


def _unquote(text: str) -> str:
    body = text[1:-1]
    escapes = {"n": "\n", "t": "\t", "\\": "\\", '"': '"', "'": "'", "r": "\r", "0": "\0"}
    return re.sub(r"\\(.)", lambda m: escapes.get(m.group(1), "\\" + m.group(1)), body)


class Parser:
    def __init__(self, src: str) -> None:
        self.tokens = tokenize(src)
        self.i = 0

    # -- token helpers ---------------------------------------------------

    @property
    def tok(self) -> Token:
        return self.tokens[self.i]

    def peek(self, k: int = 1) -> Token:
        return self.tokens[min(self.i + k, len(self.tokens) - 1)]

    def at(self, text: str) -> bool:
        return self.tok.kind in ("op", "open", "close", "name") and self.tok.text == text

    def advance(self) -> Token:
        t = self.tok
        self.i += 1
        return t

    def expect(self, text: str) -> Token:
        if not self.at(text):
            self.fail(f"expected {text!r} but found {self.tok.text or 'end of input'!r}")
        return self.advance()

    def separator(self, close: str) -> None:
        if not self.at(","):
            self.fail(f"expected ',' or {close!r} but found {self.tok.text or 'end of input'!r}")
        self.advance()

    def fail(self, message: str, tok: Token | None = None):
        t = tok or self.tok
        raise ParseError(message, t.line, t.col)

    def skip_separators(self) -> bool:
        seen_semi = False
        while self.tok.kind == "newline" or self.at(";"):
            seen_semi = self.at(";") or seen_semi
            self.advance()
        return seen_semi

    # -- program ---------------------------------------------------------

    def program(self) -> Any:
        stmts: list[Any] = []
        self.skip_separators()
        trailing_semi = False
        while self.tok.kind != "end":
            stmts.append(self.statement())
            trailing_semi = False
            if self.tok.kind != "end":
                if not (self.tok.kind == "newline" or self.at(";")):
                    self.fail(f"unexpected {self.tok.text!r}; statements are separated by newlines or ';'")
                # a ';' directly after the statement suppresses it, even if newlines follow
                trailing_semi = self.at(";")
                self.skip_separators()
        if trailing_semi:
            stmts.append(None)
        if len(stmts) == 1:
            return stmts[0]
        return Expr(Symbol("CompoundExpression"), tuple(stmts))

    def statement(self) -> Any:
        lhs = self.expr(0)
        if self.at("="):
            eq = self.advance()
            if not (isinstance(lhs, Symbol) or (isinstance(lhs, Expr) and isinstance(lhs.head, Symbol))):
                self.fail("only a name or a call f(x, ...) can be assigned to", eq)
            return Expr(Symbol("Set"), (lhs, self.statement()))
        return lhs

    # -- expressions (Pratt) -------------------------------------------------

    def expr(self, min_bp: int) -> Any:
        left = self.prefix()
        while True:
            t = self.tok
            if t.kind == "open" and t.text in "([" and CALL_PREC >= min_bp:
                left = self.postfix(left)
                continue
            key = t.text if t.kind in ("op", "name") else None
            if key not in BINARY:
                return left
            head, bp, assoc = BINARY[key]
            if bp < min_bp or (bp == min_bp and assoc != "right"):
                return left
            self.advance()
            right = self.expr(bp if assoc == "right" else bp + 1)
            left = Expr(Symbol(head), (left, right))
            if assoc == "none" and self.tok.text in COMPARISONS and self.tok.kind == "op":
                self.fail("comparisons do not chain; combine them with 'and'")

    def prefix(self) -> Any:
        t = self.tok
        if t.kind == "number":
            self.advance()
            return float(t.text) if any(c in t.text for c in ".eE") else int(t.text)
        if t.kind == "string":
            self.advance()
            return _unquote(t.text)
        if t.kind == "name":
            if t.text in ("true", "false"):
                self.advance()
                return t.text == "true"
            if t.text == "null":
                self.advance()
                return None
            if t.text == "not":
                self.advance()
                return Expr(Symbol("Not"), (self.expr(PREC["Not"]),))
            if t.text in KEYWORDS:
                self.fail(f"unexpected {t.text!r}")
            self.advance()
            if self.at("=>"):
                self.advance()
                return Expr(Symbol("Function"), (Expr(Symbol("List"), (Symbol(t.text),)), self.expr(PREC["Function"])))
            return Symbol(t.text)
        if t.kind == "op" and t.text in ("-", "+"):
            self.advance()
            operand = self.expr(PREC["Minus"])
            return Expr(Symbol("Minus"), (operand,)) if t.text == "-" else operand
        if t.kind == "open":
            return {"(": self.group, "[": self.list_, "{": self.association}[t.text]()
        self.fail(f"unexpected {t.text or 'end of input'!r}")

    def group(self) -> Any:
        start = self.expect("(")
        items = []
        while not self.at(")"):
            items.append(self.expr(0))
            if not self.at(")"):
                self.separator(")")
        self.expect(")")
        if self.at("=>"):
            self.advance()
            if not all(isinstance(x, Symbol) for x in items):
                self.fail("function parameters must be names", start)
            return Expr(Symbol("Function"), (Expr(Symbol("List"), tuple(items)), self.expr(PREC["Function"])))
        if len(items) != 1:
            self.fail("parentheses group exactly one expression (lists are written [a, b])", start)
        return items[0]

    def list_(self) -> Any:
        self.expect("[")
        items = []
        while not self.at("]"):
            items.append(self.expr(0))
            if not self.at("]"):
                self.separator("]")
        self.expect("]")
        return Expr(Symbol("List"), tuple(items))

    def association(self) -> Any:
        self.expect("{")
        pairs = []
        while not self.at("}"):
            key = self.expr(PREC["Rule"] + 1)
            self.expect(":")
            pairs.append(Expr(Symbol("Rule"), (key, self.expr(0))))
            if not self.at("}"):
                self.separator("}")
        self.expect("}")
        return Expr(Symbol("Association"), tuple(pairs))

    def postfix(self, left: Any) -> Any:
        if self.at("["):
            self.advance()
            idx = []
            while not self.at("]"):
                idx.append(self.expr(0))
                if not self.at("]"):
                    self.separator("]")
            self.expect("]")
            if not idx:
                self.fail("a part needs at least one index")
            return Expr(Symbol("Part"), (left, *idx))
        self.expect("(")
        args = []
        while not self.at(")"):
            if self.tok.kind == "name" and self.peek().text == "=" and self.peek().kind == "op":
                name = Symbol(self.advance().text)
                self.advance()
                args.append(Expr(Symbol("Option"), (name, self.expr(0))))
            else:
                args.append(self.expr(0))
            if not self.at(")"):
                self.separator(")")
        self.expect(")")
        return Expr(left, tuple(args))


def parse(src: str) -> Any:
    return Parser(src).program()


# -- printing --------------------------------------------------------------


def _prec(e: Any) -> int:
    if isinstance(e, Expr):
        name = e.head_name
        if name in OP_TEXT and len(e.args) == 2:
            return PREC[name]
        if name in ("Minus", "Not") and len(e.args) == 1:
            return PREC[name]
        if name == "Function" and len(e.args) == 2 and _params(e.args[0]) is not None:
            return PREC["Function"]
        if name == "Set" and len(e.args) == 2:
            return PREC["Set"]
        return CALL_PREC
    if isinstance(e, (int, float)) and not isinstance(e, bool) and e < 0:
        return PREC["Minus"]
    return ATOM_PREC


def _params(e: Any) -> tuple[Symbol, ...] | None:
    if isinstance(e, Expr) and e.head_name == "List" and all(isinstance(p, Symbol) for p in e.args):
        return e.args
    return None


def _wrap(e: Any, min_prec: int) -> str:
    text = unparse(e)
    return f"({text})" if _prec(e) < min_prec else text


def _atom(e: Any) -> str:
    if e is None:
        return "null"
    if isinstance(e, bool):
        return "true" if e else "false"
    if isinstance(e, int):
        return str(e)
    if isinstance(e, float):
        text = repr(e)
        return text if any(c in text for c in ".e") or text in ("inf", "nan") else text + ".0"
    if isinstance(e, str):
        return json.dumps(e, ensure_ascii=False)
    if isinstance(e, Symbol):
        return e.name
    raise TypeError(f"not an expression atom: {e!r}")


def unparse(e: Any) -> str:
    """Write a tree as text that parses back to the same tree."""
    if not isinstance(e, Expr):
        return _atom(e)
    name, args = e.head_name, e.args
    if name in OP_TEXT and len(args) == 2:
        p, assoc = PREC[name], ASSOC[name]
        left = _wrap(args[0], p + (assoc != "left"))
        right = _wrap(args[1], p + (assoc != "right"))
        return f"{left} {OP_TEXT[name]} {right}"
    if name == "Minus" and len(args) == 1:
        return "-" + _wrap(args[0], PREC["Minus"])
    if name == "Not" and len(args) == 1:
        return "not " + _wrap(args[0], PREC["Not"])
    if name == "Function" and len(args) == 2 and (ps := _params(args[0])) is not None:
        params = ps[0].name if len(ps) == 1 else "(" + ", ".join(p.name for p in ps) + ")"
        return f"{params} => {_wrap(args[1], PREC['Function'])}"
    if name == "Set" and len(args) == 2:
        return f"{unparse(args[0])} = {unparse(args[1])}"
    if name == "List":
        return "[" + ", ".join(unparse(a) for a in args) + "]"
    if name == "Association" and all(isinstance(a, Expr) and a.head_name == "Rule" and len(a.args) == 2 for a in args):
        return "{" + ", ".join(f"{_wrap(a.args[0], PREC['Rule'] + 1)}: {unparse(a.args[1])}" for a in args) + "}"
    if name == "Part" and len(args) >= 2:
        return _wrap(args[0], CALL_PREC) + "[" + ", ".join(unparse(a) for a in args[1:]) + "]"
    if name == "CompoundExpression" and args:
        stmts = list(args)
        suppressed = stmts[-1] is None and len(stmts) > 1
        if suppressed:
            stmts = stmts[:-1]
        return "\n".join(_statement(s) for s in stmts) + (";" if suppressed else "")
    head = e.head.name if isinstance(e.head, Symbol) else _wrap(e.head, CALL_PREC)
    return head + "(" + ", ".join(_argument(a) for a in args) + ")"


def _statement(e: Any) -> str:
    text = unparse(e)
    return f"({text})" if isinstance(e, Expr) and e.head_name == "CompoundExpression" else text


def _argument(a: Any) -> str:
    if isinstance(a, Expr) and a.head_name == "Option" and len(a.args) == 2 and isinstance(a.args[0], Symbol):
        return f"{a.args[0].name} = {unparse(a.args[1])}"
    return unparse(a)
