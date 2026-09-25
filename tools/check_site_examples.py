#!/usr/bin/env python
"""check_site_examples.py — run every code block on the site, page by page, in order.

An example that does not run is worse than no example: it invites the reader to copy something
broken and suggests the documentation is not checked. This script extracts every
``<pre><code>`` from each language version of the site (``site/index.html`` in English,
``site/pt-br/index.html`` in Portuguese) and executes it, in page order and in one namespace per
page, because the examples form a narrative in which a block uses what an earlier one defined.

Blocks marked ``class="plain"`` are skipped on purpose: shell lines, syntax illustrations and
snippets that need the reader's own data. The mark is what declares them not executable.

    python tools/check_site_examples.py
"""
from __future__ import annotations

import html
import re
import sys
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Each language is a list of pages, in the same order, so a translation can be
# checked against its pair rather than against a different page of the site.
PAGES = {
    "en": [ROOT / "site" / "index.html", ROOT / "site" / "replications.html"],
    "pt-br": [ROOT / "site" / "pt-br" / "index.html",
              ROOT / "site" / "pt-br" / "replications.html"],
}
BLOCK = re.compile(r'<pre><code(?P<attr>[^>]*)>(?P<src>.*?)</code></pre>', re.S)


def blocks(page: Path):
    """The page's executable blocks, in order, with HTML escapes undone."""
    text = page.read_text(encoding="utf-8")
    for i, m in enumerate(BLOCK.finditer(text), start=1):
        if "plain" in m.group("attr"):
            continue
        line = text[:m.start()].count("\n") + 1
        yield i, line, html.unescape(m.group("src")).strip()


def run(page: Path, verbose: bool = True) -> int:
    """Execute one page's blocks in a fresh namespace; return the number that failed."""
    # A real MODULE registered in sys.modules, not a bare dict: `dataclasses` resolves
    # annotations through sys.modules[cls.__module__].__dict__, and an exec in an anonymous dict
    # makes that lookup return None. The example works in a script; the checker was wrong.
    name = f"__cgekit_site_examples_{page.parent.name}__"
    module = types.ModuleType(name)
    sys.modules[name] = module
    ns = module.__dict__
    failures = 0
    for i, line, src in blocks(page):
        label = f"block {i} (line {line})"
        try:
            exec(compile(src, f"<site {label}>", "exec"), ns)
        except Exception:
            failures += 1
            if verbose:
                print(f"  FAILED  {label}")
                print("    " + "\n    ".join(traceback.format_exc().strip().splitlines()[-4:]))
                print("    --- code ---")
                print("    " + "\n    ".join(src.splitlines()[:6]))
        else:
            if verbose:
                print(f"  ok      {label}  ({len(src.splitlines())} lines)")
    return failures


def main() -> int:
    failures = 0
    for lang, pages in PAGES.items():
        for page in pages:
            print(f"\n{lang}: {page.relative_to(ROOT)}")
            failures += run(page)
    print(f"\n{'every example ran' if failures == 0 else f'{failures} block(s) failed'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
