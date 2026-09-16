"""Deterministic generator for the DS-012 code-syntax torture corpus (Track B).

Builds 200 realistic code snippets across 6 languages (Python, JavaScript, C,
Rust, Go, Java) with roughly 33 samples per language.  Each snippet is 10-40
lines and contains realistic functions, structs, comments, and control flow —
all generated from seeded templates with variation, NOT copied from any
external source.

The corpus is a TRUE-positive coverage fixture for the Track B structural code
syntax filter (``is_code_syntax_context`` in AVG.core.metrics): every line of
every snippet is genuine source code, so the per-language detection rate is a
measure of how well the production regex recognizes each language's structural
markers.  Rates are RECORDED, not forced — if a language reads below 90% the
fixture keeps the templates as-is and the build prints a finding.

Mirrors the discipline of scripts/build_t2s_degenerate.py and
scripts/build_schema_corpus.py: a single fixed seed, an explicit
random.Random instance, structural self-checks at generation time, and a
byte-identical fixture across runs.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))  # parents[2] header; import AVG.* via /AVG

from AVG.core.metrics import is_code_syntax_context  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed seed provenance:
# SEED = 4242 is the DS-012 task identifier.  Using a single fixed seed and an
# explicit random.Random instance guarantees the corpus is byte-identical
# across runs and reviewers (double-run proof).
# ---------------------------------------------------------------------------
SEED = 4242
NUM_SAMPLES = 200
TARGET_PER_LANGUAGE = 33
LANGUAGE_TOLERANCE = 3
MIN_LINES = 10
MAX_LINES = 40
LABEL = "code"

# 6 languages * ~33 = 200; two languages carry the extra sample.  Order is
# fixed so the generation (and therefore the fixture) is deterministic.
LANGUAGES = ["python", "javascript", "c", "rust", "go", "java"]
COUNTS = {lang: TARGET_PER_LANGUAGE for lang in LANGUAGES}
COUNTS["python"] += 1
COUNTS["javascript"] += 1  # 34, 34, 33, 33, 33, 33 => 200

# ---------------------------------------------------------------------------
# Vocabularies.  These are original, generic identifiers and comment phrases;
# nothing is lifted from any external tutorial or codebase.
# ---------------------------------------------------------------------------

# "lambda" is deliberately excluded: it is a Python keyword and would produce
# non-parseable Python.  All other names are valid identifiers in every target
# language.
PY_IDENTS = [
    "alpha", "beta", "gamma", "delta", "sigma", "omega",
    "total", "count", "offset", "buffer", "record", "index", "limit",
    "threshold", "factor", "result", "accum", "partial", "items",
    "values", "data", "seq", "payload", "cache", "pending", "score",
]

JS_IDENTS = [
    "alpha", "beta", "gamma", "delta", "sigma", "lambda", "omega",
    "total", "count", "offset", "buffer", "record", "index", "limit",
    "threshold", "factor", "result", "accum", "items", "values", "data",
    "seq", "payload", "cache", "config", "entry", "element", "stack",
]

C_IDENTS = [
    "alpha", "beta", "gamma", "delta", "sigma", "lambda", "omega",
    "total", "count", "offset", "buffer", "record", "index", "limit",
    "threshold", "factor", "result", "accum", "items", "values", "data",
    "size", "temp", "ptr", "node", "sum", "acc", "value",
]

RUST_IDENTS = [
    "alpha", "beta", "gamma", "delta", "sigma", "lambda", "omega",
    "total", "count", "offset", "buffer", "record", "index", "limit",
    "threshold", "factor", "result", "accum", "items", "values", "data",
    "size", "value", "acc", "payload", "cache", "entry",
]

GO_IDENTS = [
    "alpha", "beta", "gamma", "delta", "sigma", "lambda", "omega",
    "total", "count", "offset", "buffer", "record", "index", "limit",
    "threshold", "factor", "result", "accum", "items", "values", "data",
    "size", "value", "acc", "payload", "cache", "entry", "sum",
]

JAVA_IDENTS = [
    "alpha", "beta", "gamma", "delta", "sigma", "lambda", "omega",
    "total", "count", "offset", "buffer", "record", "index", "limit",
    "threshold", "factor", "result", "accum", "items", "values", "data",
    "size", "value", "acc", "payload", "cache", "entry", "min", "max",
]

# PascalCase type / class names.
TYPE_NAMES = [
    "Buffer", "Record", "Cache", "Index", "Frame", "Packet", "Counter",
    "Tracker", "Storage", "Session", "Metrics", "Profile", "Window",
    "Queue", "Stack", "Vector", "Matrix", "Config", "Handler", "Service",
    "Adapter", "Factory", "Node", "Graph", "Item", "Entry", "Chunk",
    "Sample", "Snapshot", "Token",
]

# Generic, original comment phrases.
COMMENTS = [
    "initialize the buffer",
    "compute the running total",
    "clamp the value to the limit",
    "retry on transient failure",
    "collect matching entries",
    "check the invariant before use",
    "flush pending changes",
    "reset the accumulator",
    "validate the input range",
    "apply the transform in place",
    "skip empty records",
    "cache the computed result",
    "drop stale entries",
    "round to the nearest integer",
    "guard against overflow",
    "bail out early on invalid state",
    "sort the slice before returning",
    "merge the two halves",
    "track the high water mark",
    "normalize the input",
]


# ---------------------------------------------------------------------------
# Small deterministic helpers
# ---------------------------------------------------------------------------

def _num(rng: random.Random, lo: int = 1, hi: int = 100) -> int:
    return rng.randint(lo, hi)


def _comment(rng: random.Random) -> str:
    return rng.choice(COMMENTS)


def _ident_ns(rng: random.Random, pool: Sequence[str]) -> Iterator[str]:
    """Return an iterator over a shuffled identifier pool (unique per snippet)."""
    items = list(pool)
    rng.shuffle(items)
    return iter(items)


def _type_name(rng: random.Random) -> str:
    return rng.choice(TYPE_NAMES)


def _type_names(rng: random.Random, k: int) -> List[str]:
    """Draw ``k`` distinct type names (avoids trait/struct name collisions)."""
    names = list(TYPE_NAMES)
    rng.shuffle(names)
    return names[:k]


def _render(lines: List[str]) -> str:
    """Join template lines into a snippet with no trailing newline."""
    return "\n".join(lines)


def _non_empty_lines(text: str) -> int:
    return sum(1 for ln in text.splitlines() if ln.strip())


# ---------------------------------------------------------------------------
# Python templates
# ---------------------------------------------------------------------------

def _py_loop(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, PY_IDENTS)
    a1, a2, total, partial = next(ns), next(ns), next(ns), next(ns)
    n, m, limit = _num(rng, 2, 30), _num(rng, 2, 9), _num(rng, 50, 500)
    c1, c2, c3 = _comment(rng), _comment(rng), _comment(rng)
    return [
        f"def {a1}({a2}, {total}):",
        f'    """{c1}."""',
        f"    # {c2}",
        f"    {partial} = {a2}",
        f"    {total} = 0",
        f"    for i in range({n}):",
        f"        {partial} = {partial} + i",
        f"        if {partial} % {m} == 0:",
        f"            {total} += {partial}",
        f"        elif {total} > {limit}:",
        f"            break",
        f"        else:",
        f"            {total} -= {a2}",
        f"    # {c3}",
        f"    return {total}",
    ]


def _py_class(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, PY_IDENTS)
    attr1, attr2, method, param = next(ns), next(ns), next(ns), next(ns)
    cname = _type_name(rng)
    n = _num(rng, 2, 20)
    c1, c2 = _comment(rng), _comment(rng)
    return [
        f"class {cname}:",
        f'    """{c1}."""',
        "",
        f"    def __init__(self, {attr1}, {attr2}):",
        f"        self.{attr1} = {attr1}",
        f"        self.{attr2} = {attr2}",
        f"        self._cache = []",
        "",
        f"    def {method}(self, {param}):",
        f'        """{c2}."""',
        f"        for i in range({n}):",
        f"            if i % 2 == 0:",
        f"                self._cache.append(i)",
        f"        return sum(self._cache) + {param}",
    ]


def _py_main(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, PY_IDENTS)
    fname, items, result = next(ns), next(ns), next(ns)
    n, m = _num(rng, 5, 40), _num(rng, 2, 9)
    c1 = _comment(rng)
    return [
        f"def {fname}({items}):",
        f'    """{c1}."""',
        f"    {result} = []",
        f"    for i, value in enumerate({items}):",
        f"        if value % {m} == 0:",
        f"            {result}.append(value)",
        f"        elif value < 0:",
        f"            break",
        f"        else:",
        f"            {result}.append(-value)",
        f"    return {result}",
        "",
        "",
        'if __name__ == "__main__":',
        f"    data = list(range({n}))",
        f"    {fname}(data)",
    ]


def _py_try(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, PY_IDENTS)
    fname, arg, value = next(ns), next(ns), next(ns)
    key = next(ns)
    n = _num(rng, 2, 20)
    c1 = _comment(rng)
    return [
        f"def {fname}({arg}):",
        f'    """{c1}."""',
        f"    try:",
        f"        {value} = {arg}[{key!r}]",
        f"        return {value} * {n}",
        f"    except KeyError:",
        f"        return None",
        f"    except TypeError:",
        f"        return -1",
        f"    finally:",
        f"        pass",
    ]


def _py_comprehensions(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, PY_IDENTS)
    fname, items, squares, evens, total = next(ns), next(ns), next(ns), next(ns), next(ns)
    n = _num(rng, 2, 20)
    c1, c2 = _comment(rng), _comment(rng)
    return [
        f"def {fname}({items}):",
        f'    """{c1}."""',
        f"    {squares} = [x * x for x in {items} if x > 0]",
        f"    {evens} = [x for x in {squares} if x % 2 == 0]",
        f"    # {c2}",
        f"    {total} = 0",
        f"    for x in {evens}:",
        f"        if x > 0:",
        f"            {total} += x",
        f"    return {total}",
    ]


def _py_dict(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, PY_IDENTS)
    fname, seq, result = next(ns), next(ns), next(ns)
    m = _num(rng, 2, 9)
    c1, c2 = _comment(rng), _comment(rng)
    return [
        f"def {fname}({seq}):",
        f'    """{c1}."""',
        f"    # {c2}",
        f"    if not {seq}:",
        f"        return {{}}",
        f"    {result} = {{}}",
        f"    for idx, value in enumerate({seq}):",
        f"        if idx % {m} == 0:",
        f"            {result}[idx] = value",
        f"        else:",
        f"            {result}[-idx] = value * 2",
        f"    return {result}",
    ]


# ---------------------------------------------------------------------------
# JavaScript templates
# ---------------------------------------------------------------------------

def _js_function(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, JS_IDENTS)
    fname, a1, a2, s, count = next(ns), next(ns), next(ns), next(ns), next(ns)
    n = _num(rng, 2, 20)
    c1 = _comment(rng)
    return [
        f"// {c1}",
        f"function {fname}({a1}, {a2}) {{",
        f"    const {s} = {a1} + {a2};",
        f"    let {count} = 0;",
        f"    for (let i = 0; i < {n}; i++) {{",
        f"        if (i % 2 === 0) {{",
        f"            {count} += i;",
        f"        }}",
        f"    }}",
        f"    return {s} + {count};",
        f"}}",
    ]


def _js_arrow(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, JS_IDENTS)
    fname, gname, a1, a2, result, items, total = (
        next(ns), next(ns), next(ns), next(ns), next(ns), next(ns), next(ns),
    )
    n = _num(rng, 2, 20)
    c1 = _comment(rng)
    return [
        f"/* {c1} */",
        f"const {fname} = ({a1}, {a2}) => {{",
        f"    const {result} = {a1} * {a2};",
        f"    return {result} + {n};",
        f"}};",
        "",
        f"const {gname} = ({items}) => {{",
        f"    let {total} = 0;",
        f"    for (const item of {items}) {{",
        f"        {total} += item;",
        f"    }}",
        f"    return {total};",
        f"}};",
    ]


def _js_class(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, JS_IDENTS)
    attr, attr2, method, param, acc = next(ns), next(ns), next(ns), next(ns), next(ns)
    cname = _type_name(rng)
    n = _num(rng, 2, 20)
    c1 = _comment(rng)
    return [
        f"// {c1}",
        f"class {cname} {{",
        f"    constructor({attr}) {{",
        f"        this.{attr} = {attr};",
        f"        this.{attr2} = 0;",
        f"    }}",
        "",
        f"    {method}({param}) {{",
        f"        let {acc} = 0;",
        f"        for (let i = 0; i < {n}; i++) {{",
        f"            {acc} += i * this.{attr};",
        f"        }}",
        f"        return {acc} + {param};",
        f"    }}",
        f"}}",
    ]


def _js_var_style(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, JS_IDENTS)
    limit, fname, items, result = next(ns), next(ns), next(ns), next(ns)
    n = _num(rng, 5, 40)
    c1 = _comment(rng)
    return [
        f"// {c1}",
        f"const {limit} = {n};",
        "",
        f"function {fname}({items}) {{",
        f"    var {result} = [];",
        f"    for (var i = 0; i < {items}.length; i++) {{",
        f"        if ({items}[i] < {limit}) {{",
        f"            {result}.push({items}[i]);",
        f"        }}",
        f"    }}",
        f"    return {result};",
        f"}}",
    ]


def _js_callback(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, JS_IDENTS)
    fname, items, filtered, mapped = next(ns), next(ns), next(ns), next(ns)
    c1 = _comment(rng)
    return [
        f"// {c1}",
        f"function {fname}({items}) {{",
        f"    const {filtered} = {items}.filter(function (item) {{",
        f"        return item > 0;",
        f"    }});",
        f"    const {mapped} = {filtered}.map(function (item) {{",
        f"        return item * 2;",
        f"    }});",
        f"    return {mapped};",
        f"}}",
    ]


def _js_object(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, JS_IDENTS)
    config, fname = next(ns), next(ns)
    n, m = _num(rng, 2, 20), _num(rng, 20, 200)
    c1 = _comment(rng)
    return [
        f"// {c1}",
        f"const {config} = {{",
        f'    name: "sample",',
        f"    retries: {n},",
        f"    timeout: {m},",
        f"}};",
        "",
        f"function {fname}({config}) {{",
        f"    const {{ name, retries }} = {config};",
        f"    let attempt = 0;",
        f"    while (attempt < retries) {{",
        f"        attempt += 1;",
        f"    }}",
        f"    return name;",
        f"}}",
    ]


# ---------------------------------------------------------------------------
# C templates
# ---------------------------------------------------------------------------

def _c_struct_fn(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, C_IDENTS)
    f1, f2, f3, fname, p = next(ns), next(ns), next(ns), next(ns), next(ns)
    sname = _type_name(rng)
    n, limit = _num(rng, 4, 40), _num(rng, 50, 500)
    c1 = _comment(rng)
    return [
        "#include <stdio.h>",
        "",
        f"/* {c1} */",
        f"struct {sname} {{",
        f"    int {f1};",
        f"    float {f2};",
        f"    char {f3}[{n}];",
        f"}};",
        "",
        f"int {fname}(struct {sname} *{p}) {{",
        f"    int total = 0;",
        f"    for (int i = 0; i < {n}; i++) {{",
        f"        total += {p}->{f1};",
        f"        if (total > {limit}) {{",
        f"            break;",
        f"        }}",
        f"    }}",
        f"    return total;",
        f"}}",
    ]


def _c_typedef_main(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, C_IDENTS)
    f1, f2, v = next(ns), next(ns), next(ns)
    sname, tname = _type_names(rng, 2)
    d = round(rng.uniform(0.5, 99.5), 2)
    c1 = _comment(rng)
    return [
        "#include <stdio.h>",
        "",
        f"/* {c1} */",
        f"typedef struct {sname} {tname};",
        "",
        f"struct {sname} {{",
        f"    double {f1};",
        f"    int {f2};",
        f"}};",
        "",
        "int main(void) {",
        f"    {tname} {v} = {{0}};",
        f"    {v}.{f1} = {d};",
        f'    printf("%f\\n", {v}.{f1});',
        "    return 0;",
        "}",
    ]


def _c_array_fn(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, C_IDENTS)
    fname, arr, s, data, res, cnt = next(ns), next(ns), next(ns), next(ns), next(ns), next(ns)
    size = _num(rng, 4, 20)
    c1 = _comment(rng)
    return [
        "#include <stdio.h>",
        "",
        f"/* {c1} */",
        f"static int {fname}(const int *{arr}, int {cnt}) {{",
        f"    int {s} = 0;",
        f"    for (int i = 0; i < {cnt}; i++) {{",
        f"        if ({arr}[i] > 0) {{",
        f"            {s} += {arr}[i];",
        f"        }}",
        f"    }}",
        f"    return {s};",
        f"}}",
        "",
        "int main(void) {",
        f"    int {data}[{size}] = {{0}};",
        f"    int {res} = {fname}({data}, {size});",
        f'    printf("%d\\n", {res});',
        "    return 0;",
        "}",
    ]


def _c_plain_fn(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, C_IDENTS)
    fname, a, b, result, x = next(ns), next(ns), next(ns), next(ns), next(ns)
    n, m, limit = _num(rng, 2, 30), _num(rng, 2, 30), _num(rng, 50, 500)
    return [
        "#include <stdio.h>",
        "",
        f"int {fname}(int {a}, int {b}) {{",
        f"    int {result} = {a} + {b};",
        f"    if ({result} > {limit}) {{",
        f"        {result} = {result} - {b};",
        f"    }} else {{",
        f"        {result} = {result} * 2;",
        f"    }}",
        f"    return {result};",
        f"}}",
        "",
        "int main(void) {",
        f"    int {x} = {fname}({n}, {m});",
        f'    printf("%d\\n", {x});',
        "    return 0;",
        "}",
    ]


def _c_typedef_fn(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, C_IDENTS)
    f1, f2, fname, p, acc, cnt = next(ns), next(ns), next(ns), next(ns), next(ns), next(ns)
    sname, tname = _type_names(rng, 2)
    n = _num(rng, 2, 20)
    c1 = _comment(rng)
    return [
        "#include <stdio.h>",
        "",
        f"/* {c1} */",
        f"typedef struct {sname} {tname};",
        "",
        f"struct {sname} {{",
        f"    int {f1};",
        f"    int {f2};",
        f"}};",
        "",
        f"int {fname}({tname} *{p}, int {cnt}) {{",
        f"    int {acc} = 0;",
        f"    for (int i = 0; i < {cnt}; i++) {{",
        f"        {acc} += {p}->{f1} + {p}->{f2};",
        f"    }}",
        f"    return {acc};",
        f"}}",
    ]


def _c_header_comments(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, C_IDENTS)
    fname, a, r, x = next(ns), next(ns), next(ns), next(ns)
    n = _num(rng, 2, 30)
    c1, c2, c3 = _comment(rng), _comment(rng), _comment(rng)
    return [
        "/*",
        f" * {c1}",
        f" * {c2}",
        " */",
        "#include <stdio.h>",
        "",
        f"int {fname}(int {a}) {{",
        f"    /* {c3} */",
        f"    int {r} = {a} * {a};",
        f"    if ({r} < 0) {{",
        f"        {r} = -{r};",
        f"    }}",
        f"    return {r};",
        f"}}",
        "",
        "int main(void) {",
        f"    int {x} = {fname}({n});",
        f'    printf("%d\\n", {x});',
        "    return 0;",
        "}",
    ]


# ---------------------------------------------------------------------------
# Rust templates
# ---------------------------------------------------------------------------

def _rs_struct_impl(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, RUST_IDENTS)
    f1, f2, method, param, r = next(ns), next(ns), next(ns), next(ns), next(ns)
    sname = _type_name(rng)
    limit = _num(rng, 50, 500)
    c1 = _comment(rng)
    return [
        f"// {c1}",
        f"struct {sname} {{",
        f"    {f1}: i32,",
        f"    {f2}: f64,",
        f"}}",
        "",
        f"impl {sname} {{",
        f"    fn {method}(&self, {param}: i32) -> i32 {{",
        f"        let {r} = self.{f1} + {param};",
        f"        if {r} > {limit} {{",
        f"            {r} - self.{f2} as i32",
        f"        }} else {{",
        f"            {r} * 2",
        f"        }}",
        f"    }}",
        f"}}",
    ]


def _rs_loop(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, RUST_IDENTS)
    fname, a1, a2, acc = next(ns), next(ns), next(ns), next(ns)
    n = _num(rng, 2, 20)
    c1 = _comment(rng)
    return [
        f"// {c1}",
        f"fn {fname}({a1}: i32, {a2}: i32) -> i32 {{",
        f"    let mut {acc} = 0;",
        f"    for i in 0..{n} {{",
        f"        if i % 2 == 0 {{",
        f"            {acc} += {a1};",
        f"        }} else {{",
        f"            {acc} += {a2};",
        f"        }}",
        f"    }}",
        f"    {acc}",
        f"}}",
    ]


def _rs_enum_match(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, RUST_IDENTS)
    ename, fname = _type_name(rng), next(ns)
    v1, v2 = "First", "Second"
    c1 = _comment(rng)
    return [
        f"// {c1}",
        f"enum {ename} {{",
        f"    {v1},",
        f"    {v2}(i32),",
        f"}}",
        "",
        f"fn {fname}(value: {ename}) -> i32 {{",
        f"    match value {{",
        f"        {ename}::{v1} => 0,",
        f"        {ename}::{v2}(n) => n * 2,",
        f"    }}",
        f"}}",
    ]


def _rs_struct_fn(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, RUST_IDENTS)
    f1, f2, fname, s, r = next(ns), next(ns), next(ns), next(ns), next(ns)
    sname = _type_name(rng)
    c1, c2 = _comment(rng), _comment(rng)
    return [
        f"// {c1}",
        f"struct {sname} {{",
        f"    {f1}: u32,",
        f"    {f2}: String,",
        f"}}",
        "",
        f"fn {fname}({s}: {sname}) -> u32 {{",
        f"    // {c2}",
        f"    if {s}.{f1} == 0 {{",
        f"        return 0;",
        f"    }}",
        f"    let {r} = {s}.{f1} + 1;",
        f"    {r}",
        f"}}",
    ]


def _rs_main(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, RUST_IDENTS)
    a, b, c = next(ns), next(ns), next(ns)
    limit = _num(rng, 50, 500)
    c1 = _comment(rng)
    return [
        "fn main() {",
        f"    // {c1}",
        f"    let {a} = 5;",
        f"    let {b} = 10;",
        f"    let {c} = {a} + {b};",
        f"    if {c} > {limit} {{",
        f'        println!("big");',
        f"    }} else {{",
        f'        println!("small");',
        f"    }}",
        "}",
    ]


def _rs_trait(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, RUST_IDENTS)
    tname, sname = _type_names(rng, 2)
    method, f1, r = next(ns), next(ns), next(ns)
    c1 = _comment(rng)
    return [
        f"// {c1}",
        f"trait {tname} {{",
        f"    fn {method}(&self) -> i32;",
        f"}}",
        "",
        f"struct {sname} {{",
        f"    {f1}: i32,",
        f"}}",
        "",
        f"impl {tname} for {sname} {{",
        f"    fn {method}(&self) -> i32 {{",
        f"        let {r} = self.{f1} * 2;",
        f"        {r}",
        f"    }}",
        f"}}",
    ]


# ---------------------------------------------------------------------------
# Go templates
# ---------------------------------------------------------------------------

def _go_struct_method(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, GO_IDENTS)
    f1, f2, method = next(ns), next(ns), next(ns)
    sname = _type_name(rng)
    n = _num(rng, 2, 20)
    c1 = _comment(rng)
    return [
        f"// {c1}",
        "package main",
        "",
        'import "fmt"',
        "",
        f"type {sname} struct {{",
        f"    {f1} int",
        f"    {f2} float64",
        f"}}",
        "",
        f"func (s *{sname}) {method}() int {{",
        "    total := 0",
        f"    for i := 0; i < {n}; i++ {{",
        f"        total += s.{f1}",
        "    }",
        "    return total",
        "}",
        "",
        "func main() {",
        f"    s := &{sname}{{{f1}: {n}, {f2}: 1.5}}",
        f"    fmt.Println(s.{method}())",
        "}",
    ]


def _go_var_fn(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, GO_IDENTS)
    fname, a, b, result = next(ns), next(ns), next(ns), next(ns)
    limit = _num(rng, 50, 500)
    c1 = _comment(rng)
    return [
        f"// {c1}",
        "package main",
        "",
        f"func {fname}({a} int, {b} int) int {{",
        f"    var {result} int",
        f"    {result} = {a} + {b}",
        f"    if {result} > {limit} {{",
        f"        {result} -= {b}",
        "    }",
        f"    return {result}",
        "}",
    ]


def _go_plain_fn(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, GO_IDENTS)
    fname, a, b = next(ns), next(ns), next(ns)
    n = _num(rng, 2, 30)
    return [
        "package main",
        "",
        f"func {fname}({a} int, {b} int) int {{",
        f"    sum := {a} + {b}",
        f"    diff := {a} - {b}",
        f"    if sum > {n} {{",
        f"        sum -= diff",
        f"    }} else if sum == 0 {{",
        f"        return 0",
        f"    }} else {{",
        f"        sum += diff",
        f"    }}",
        "    return sum",
        "}",
    ]


def _go_range(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, GO_IDENTS)
    fname, items = next(ns), next(ns)
    n, m, k = _num(rng, 2, 30), _num(rng, 2, 30), _num(rng, 2, 30)
    c1 = _comment(rng)
    return [
        f"// {c1}",
        "package main",
        "",
        'import "fmt"',
        "",
        f"func {fname}({items} []int) int {{",
        "    total := 0",
        f"    for _, v := range {items} {{",
        "        if v > 0 {",
        "            total += v",
        "        }",
        "    }",
        "    return total",
        "}",
        "",
        "func main() {",
        f"    data := []int{{{n}, {m}, {k}}}",
        f"    fmt.Println({fname}(data))",
        "}",
    ]


def _go_const_switch(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, GO_IDENTS)
    limit, fname, x = next(ns), next(ns), next(ns)
    n = _num(rng, 2, 30)
    return [
        "package main",
        "",
        f"const {limit} = {n}",
        "",
        f"func {fname}({x} int) string {{",
        "    switch {",
        f"    case {x} < {limit}:",
        '        return "low"',
        f"    case {x} == {limit}:",
        '        return "exact"',
        "    default:",
        '        return "high"',
        "    }",
        "}",
    ]


def _go_map(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, GO_IDENTS)
    fname, items, keys = next(ns), next(ns), next(ns)
    c1, c2 = _comment(rng), _comment(rng)
    return [
        f"// {c1}",
        "package main",
        "",
        f"func {fname}({items} map[string]int) int {{",
        f"    // {c2}",
        f"    {keys} := make([]string, 0, len({items}))",
        f"    for k := range {items} {{",
        f"        {keys} = append({keys}, k)",
        "    }",
        "    total := 0",
        f"    for _, v := range {items} {{",
        "        total += v",
        "    }",
        "    return total",
        "}",
    ]


# ---------------------------------------------------------------------------
# Java templates
# ---------------------------------------------------------------------------

def _java_main(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, JAVA_IDENTS)
    x, y, z = next(ns), next(ns), next(ns)
    cname = _type_name(rng)
    n, m = _num(rng, 2, 30), _num(rng, 2, 30)
    c1, c2 = _comment(rng), _comment(rng)
    return [
        f"// {c1}",
        f"public class {cname} {{",
        "    public static void main(String[] args) {",
        f"        // {c2}",
        f"        int {x} = {n};",
        f"        int {y} = {m};",
        f"        int {z} = {x} + {y};",
        f"        System.out.println({z});",
        "    }",
        "}",
    ]


def _java_method(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, JAVA_IDENTS)
    method, a, b, result = next(ns), next(ns), next(ns), next(ns)
    cname = _type_name(rng)
    n = _num(rng, 2, 20)
    c1 = _comment(rng)
    return [
        f"/* {c1} */",
        f"public class {cname} {{",
        f"    public int {method}(int {a}, int {b}) {{",
        f"        int {result} = 0;",
        f"        for (int i = 0; i < {n}; i++) {{",
        f"            {result} += {a} * i + {b};",
        f"        }}",
        f"        return {result};",
        f"    }}",
        f"}}",
    ]


def _java_plain_class(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, JAVA_IDENTS)
    f1, f2, method, acc = next(ns), next(ns), next(ns), next(ns)
    cname = _type_name(rng)
    return [
        f"public class {cname} {{",
        f"    private final int {f1};",
        f"    private final String {f2};",
        "",
        f"    public {cname}(int {f1}, String {f2}) {{",
        f"        this.{f1} = {f1};",
        f"        this.{f2} = {f2};",
        "    }",
        "",
        f"    public int {method}() {{",
        f"        int {acc} = 0;",
        f"        for (int i = 0; i < this.{f1}; i++) {{",
        f"            {acc} += i;",
        "        }",
        f"        return {acc};",
        "    }",
        "}",
    ]


def _java_interface(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, JAVA_IDENTS)
    iname, cname = _type_names(rng, 2)
    method = next(ns)
    n = _num(rng, 2, 20)
    c1, c2 = _comment(rng), _comment(rng)
    return [
        f"// {c1}",
        f"public interface {iname} {{",
        f"    int {method}(int value);",
        "}",
        "",
        f"public class {cname} implements {iname} {{",
        f"    public int {method}(int value) {{",
        f"        // {c2}",
        f"        return value * {n};",
        "    }",
        "}",
    ]


def _java_static_util(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, JAVA_IDENTS)
    method, a, b, mn, mx, diff = next(ns), next(ns), next(ns), next(ns), next(ns), next(ns)
    cname = _type_name(rng)
    c1 = _comment(rng)
    return [
        f"public class {cname} {{",
        f"    // {c1}",
        f"    public static int {method}(int {a}, int {b}) {{",
        f"        int {mn} = {a} < {b} ? {a} : {b};",
        f"        int {mx} = {a} > {b} ? {a} : {b};",
        f"        int {diff} = {mx} - {mn};",
        f"        if ({diff} < 0) {{",
        f"            {diff} = -{diff};",
        "        }",
        f"        return {diff};",
        "    }",
        "}",
    ]


def _java_loop_array(rng: random.Random) -> List[str]:
    ns = _ident_ns(rng, JAVA_IDENTS)
    method, items, total = next(ns), next(ns), next(ns)
    cname = _type_name(rng)
    return [
        f"public class {cname} {{",
        f"    public int {method}(int[] {items}) {{",
        f"        int {total} = 0;",
        f"        for (int i = 0; i < {items}.length; i++) {{",
        f"            if ({items}[i] > 0) {{",
        f"                {total} += {items}[i];",
        "            }",
        "        }",
        f"        return {total};",
        "    }",
        "}",
    ]


# ---------------------------------------------------------------------------
# Template registry (fixed order => deterministic generation)
# ---------------------------------------------------------------------------

TEMPLATES: Dict[str, List[Callable[[random.Random], List[str]]]] = {
    "python": [
        _py_loop, _py_class, _py_main, _py_try, _py_comprehensions, _py_dict,
    ],
    "javascript": [
        _js_function, _js_arrow, _js_class, _js_var_style, _js_callback,
        _js_object,
    ],
    "c": [
        _c_struct_fn, _c_typedef_main, _c_array_fn, _c_plain_fn, _c_typedef_fn,
        _c_header_comments,
    ],
    "rust": [
        _rs_struct_impl, _rs_loop, _rs_enum_match, _rs_struct_fn, _rs_main,
        _rs_trait,
    ],
    "go": [
        _go_struct_method, _go_var_fn, _go_plain_fn, _go_range,
        _go_const_switch, _go_map,
    ],
    "java": [
        _java_main, _java_method, _java_plain_class, _java_interface,
        _java_static_util, _java_loop_array,
    ],
}


def _generate_corpus(rng: random.Random) -> List[dict]:
    """Generate the 200-sample corpus in language-major order."""
    samples: List[dict] = []
    sid = 0
    for lang in LANGUAGES:
        tpls = TEMPLATES[lang]
        count = COUNTS[lang]
        for i in range(count):
            builder = tpls[i % len(tpls)]
            lines = builder(rng)
            text = _render(lines)
            line_count = _non_empty_lines(text)
            samples.append(
                {
                    "id": sid,
                    "seed": SEED,
                    "language": lang,
                    "line_count": line_count,
                    "text": text,
                    "label": LABEL,
                }
            )
            sid += 1
    return samples


def _validate(samples: List[dict]) -> Dict[str, float]:
    """Structural self-checks + per-language Track B rate measurement.

    Returns the per-language TRUE-positive rates for ``is_code_syntax_context``.
    """
    assert len(samples) == NUM_SAMPLES, (
        f"expected {NUM_SAMPLES} samples, got {len(samples)}"
    )

    counts = Counter(s["language"] for s in samples)
    for lang in LANGUAGES:
        assert abs(counts[lang] - TARGET_PER_LANGUAGE) <= LANGUAGE_TOLERANCE, (
            f"language {lang} count {counts[lang]} outside "
            f"{TARGET_PER_LANGUAGE} +/- {LANGUAGE_TOLERANCE}"
        )

    ids = [s["id"] for s in samples]
    assert ids == list(range(NUM_SAMPLES)), "ids are not 0..199 in order"

    language_rates: Dict[str, float] = {}
    for lang in LANGUAGES:
        lang_samples = [s for s in samples if s["language"] == lang]
        hits = 0
        for s in lang_samples:
            assert s["seed"] == SEED, f"sample {s['id']} seed mismatch"
            assert s["label"] == LABEL, f"sample {s['id']} label mismatch"
            assert 10 <= s["line_count"] <= 40, (
                f"sample {s['id']} line_count {s['line_count']} outside 10..40"
            )
            assert _non_empty_lines(s["text"]) == s["line_count"], (
                f"sample {s['id']} recorded line_count {s['line_count']} != actual"
            )
            assert json.loads(json.dumps(s)), f"sample {s['id']} not JSON serializable"
            hits += 1 if is_code_syntax_context(s["text"]) else 0
        language_rates[lang] = hits / float(len(lang_samples))

    return language_rates


def build(output_path: str) -> List[dict]:
    """Build, validate, and write the corpus to ``output_path``."""
    rng = random.Random(SEED)
    samples = _generate_corpus(rng)
    rates = _validate(samples)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")

    digest = hashlib.sha256(out.read_bytes()).hexdigest()

    print(f"SEED={SEED}  samples={NUM_SAMPLES}  output={output_path}")
    print(f"fixture sha256: {digest}")
    print("per-language counts: " + ", ".join(
        f"{lang}={sum(1 for s in samples if s['language'] == lang)}"
        for lang in LANGUAGES
    ))
    print("per-language is_code_syntax_context TRUE rate:")
    for lang in LANGUAGES:
        print(f"  {lang:10s} {rates[lang] * 100.0:6.1f}%")
    overall = sum(1 for s in samples if is_code_syntax_context(s["text"])) / len(samples)
    print(f"  overall    {overall * 100.0:6.1f}%")

    # Findings: languages below 90% are reported, not tuned away.
    for lang in LANGUAGES:
        if rates[lang] < 0.90:
            first_false = next(
                (
                    s
                    for s in samples
                    if s["language"] == lang
                    and not is_code_syntax_context(s["text"])
                ),
                None,
            )
            print(f"FINDING: {lang} Track B rate {rates[lang] * 100.0:.1f}% < 90%")
            if first_false is not None:
                print(f"  example false-negative id={first_false['id']}:")
                print("  " + first_false["text"].replace("\n", "\n  "))

    return samples


def main() -> int:
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    else:
        output_path = str(ROOT / "tests" / "fixtures" / "code_torture.jsonl")
    build(output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
