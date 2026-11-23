#!/usr/bin/env python3
"""End-to-end browser test: drives a real Chrome against a running instance.

The unit tests check the pieces and the integration tests check the API. This
checks the thing a user actually touches - that clicking the Keyword toggle
re-runs the search, that a shared link restores its filters, that the page does
not scroll sideways on a phone - none of which a request-level test can see.

It is deliberately not part of `make test`: it needs a browser and a running
server, so it is a deployment check rather than a unit check.

    python scripts/e2e_browser_test.py                        # the live demo
    python scripts/e2e_browser_test.py --base-url http://127.0.0.1:8000
    python scripts/e2e_browser_test.py --shots out/           # keep screenshots

Exit code is 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e2e_cdp import Browser, ChromeProcess  # noqa: E402

DEFAULT_BASE_URL = "https://icd10-hybrid-search.onrender.com"

TOP = "(() => {const e = document.querySelector('.result .code'); return e && e.dataset.code;})()"
CODES = "Array.from(document.querySelectorAll('.result .code')).map(e => e.dataset.code)"
COUNT = "document.querySelectorAll('.result').length"
EVIDENCE = (
    "(() => {const e = document.querySelector('.result .evidence');"
    " return e ? e.textContent.replace(/\\s+/g, ' ').trim() : '';})()"
)


class Report:
    def __init__(self) -> None:
        self.passed = 0
        self.failures: list[str] = []

    def check(self, label: str, condition: object, detail: str = "") -> None:
        mark = "PASS" if condition else "FAIL"
        if condition:
            self.passed += 1
        else:
            self.failures.append(label)
        print(f"  [{mark}] {label}" + (f"  -> {detail}" if detail else ""), flush=True)

    def section(self, title: str) -> None:
        print(f"\n--- {title} ---", flush=True)

    def summary(self) -> int:
        print("\n" + "=" * 64)
        print(f"  {self.passed} passed, {len(self.failures)} failed")
        for failure in self.failures:
            print("   FAILED:", failure)
        return 1 if self.failures else 0


async def mark_results(browser: Browser) -> None:
    """Drop a sentinel into the results list.

    The app re-renders by replacing innerHTML, so the sentinel disappearing is
    proof that *this* action produced a fresh render. Waiting for "results
    exist" instead would pass instantly against the previous render, and every
    assertion after a mode switch would silently read stale DOM.
    """
    await browser.js(
        """(() => {const r = document.querySelector('#results');
        const s = document.createElement('span'); s.id = '__sentinel'; r.appendChild(s);})()"""
    )


async def settle(browser: Browser, label: str, timeout: float = 180) -> None:
    await browser.wait_for(
        "(() => !document.querySelector('#__sentinel') && "
        f"({COUNT} > 0 || !!document.querySelector('#status.error')))()",
        timeout,
        label,
    )
    await asyncio.sleep(0.4)


async def search(browser: Browser, text: str) -> None:
    await mark_results(browser)
    await browser.type_text("#query", text)
    await browser.press_enter()
    await settle(browser, f"results for {text!r}")


async def click_and_settle(browser: Browser, selector: str, label: str) -> None:
    await mark_results(browser)
    await browser.click(selector)
    await settle(browser, label)


async def run(base_url: str, shots: Path | None) -> int:
    report = Report()

    async with await Browser.attach() as browser:

        async def shot(name: str) -> None:
            if shots is not None:
                await browser.screenshot(shots / name)

        report.section("1. load and bootstrap")
        await browser.goto(base_url)
        await browser.wait_for(
            "document.querySelectorAll('.chip').length > 0", 180, "example chips"
        )
        await browser.wait_for(
            "document.querySelector('#chapter').options.length > 1", 120, "chapter list"
        )
        report.check("page loads", True, str(await browser.js("document.title")))
        report.check(
            "stats hydrated from /api/stats",
            "98,186" in str(await browser.js("document.querySelector('#stat-codes').textContent")),
        )
        report.check(
            "chapter filter populated from the API",
            await browser.js("document.querySelector('#chapter').options.length") == 23,
            "1 + 22 chapters",
        )

        report.section("2. a lay query, typed with real keystrokes")
        await search(browser, "ringing in the ears")
        report.check(
            "'ringing in the ears' -> H93.19 at rank 1",
            await browser.js(TOP) == "H93.19",
            str(await browser.js(TOP)),
        )
        report.check("ten results rendered", await browser.js(COUNT) == 10)
        evidence = str(await browser.js(EVIDENCE))
        report.check(
            "evidence tags explain the ranking",
            "semantic" in evidence and "cos" in evidence,
            evidence[:55],
        )
        await shot("01_hybrid.png")

        report.section("3. keyword-only must LOSE the answer")
        await click_and_settle(browser, "input[name='mode'][value='lexical'] + span", "keyword")
        keyword_codes = await browser.js(CODES)
        report.check(
            "keyword radio is selected",
            await browser.js(
                "document.querySelector(\"input[name='mode'][value='lexical']\").checked"
            )
            is True,
        )
        report.check(
            "keyword-only cannot find H93.19 (the reason the semantic arm exists)",
            "H93.19" not in keyword_codes,
            str(keyword_codes[:3]),
        )
        await shot("02_keyword.png")

        report.section("4. semantic and hybrid restore it")
        await click_and_settle(browser, "input[name='mode'][value='dense'] + span", "semantic")
        report.check("semantic finds it again", await browser.js(TOP) == "H93.19")
        await click_and_settle(browser, "input[name='mode'][value='hybrid'] + span", "hybrid")
        report.check("hybrid finds it too", await browser.js(TOP) == "H93.19")

        report.section("5. example chips")
        await mark_results(browser)
        position = await browser.js(
            """(() => {const c = Array.from(document.querySelectorAll('.chip'))
                .find(c => c.textContent.trim() === 'nosebleed');
                c.scrollIntoView({block: 'center'});
                const r = c.getBoundingClientRect();
                return {x: r.left + r.width / 2, y: r.top + r.height / 2};})()"""
        )
        await browser.click_at(position["x"], position["y"])
        await settle(browser, "chip results")
        report.check("chip runs its query", await browser.js(TOP) == "R04.0")
        report.check(
            "search box reflects the chip",
            await browser.js("document.querySelector('#query').value") == "nosebleed",
        )

        report.section("6. code lookup and siblings")
        await search(browser, "E11.9")
        codes = await browser.js(CODES)
        report.check("'E11.9' pins the exact code first", codes[0] == "E11.9", str(codes[:3]))
        report.check(
            "labelled as an exact code match",
            "exact code match" in str(await browser.js(EVIDENCE)),
        )
        report.check(
            "siblings offered rather than semantic noise",
            sum(code.startswith("E11") for code in codes) >= 5,
            str(codes[:5]),
        )
        await shot("03_code.png")

        report.section("7. filters")
        await search(browser, "fracture")
        await click_and_settle(browser, "#billable-only", "billable filtered")
        badges = await browser.js(
            "Array.from(document.querySelectorAll('.result .badge'))"
            ".map(e => e.textContent.trim().toLowerCase())"
        )
        report.check(
            "'billable codes only' filters",
            all("not billable" not in badge for badge in badges),
            str(sorted(set(badges))),
        )
        report.check("the page stays full after filtering", await browser.js(COUNT) == 10)

        await mark_results(browser)
        await browser.js(
            """(() => {const s = document.querySelector('#chapter');
            s.value = '19'; s.dispatchEvent(new Event('change', {bubbles: true}));})()"""
        )
        await settle(browser, "chapter filtered")
        metas = await browser.js(
            "Array.from(document.querySelectorAll('.result-meta')).map(e => e.textContent.trim().slice(0, 11))"
        )
        report.check(
            "chapter filter applies", all(m.startswith("Chapter 19") for m in metas), str(metas[:2])
        )

        report.section("8. evaluation tables, served from the live API")
        await browser.js("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(1.0)
        rows = await browser.js("document.querySelectorAll('table.metrics tbody tr').length")
        best = await browser.js(
            """(() => {const r = document.querySelector('table.metrics tr.best');
            return r ? r.textContent.replace(/\\s+/g, ' ').trim() : null;})()"""
        )
        report.check("accuracy tables rendered", rows >= 6, f"{rows} rows")
        report.check("the hybrid row carries real numbers", bool(best), str(best)[:64])
        await shot("04_tables.png")

        report.section("9. theme preference survives a reload")
        before = await browser.js("document.documentElement.getAttribute('data-theme')")
        await browser.click("#theme-btn, #theme-toggle, .header-actions button")
        await asyncio.sleep(0.6)
        after = await browser.js("document.documentElement.getAttribute('data-theme')")
        report.check("theme toggle switches theme", before != after, f"{before} -> {after}")
        await browser.goto(base_url)
        await browser.wait_for("document.querySelectorAll('.chip').length > 0", 120, "reload")
        report.check(
            "theme survives a reload",
            await browser.js("document.documentElement.getAttribute('data-theme')") == after,
        )

        report.section("10. phone viewport, a true 390 px")
        await browser.emulate_phone()
        await browser.goto(f"{base_url}/?q=high+blood+pressure&chapter=9")
        await browser.wait_for(f"{COUNT} > 0", 180, "mobile results")
        await asyncio.sleep(1.0)
        inner = await browser.js("window.innerWidth")
        scroll = await browser.js("document.documentElement.scrollWidth")
        report.check(
            "no horizontal overflow on a phone",
            scroll <= inner + 1,
            f"viewport {inner}px, content {scroll}px",
        )
        report.check(
            "a shared link restores its chapter filter",
            await browser.js("document.querySelector('#chapter').value") == "9",
        )
        report.check("results are still correct on mobile", await browser.js(TOP) == "I10")
        await shot("05_mobile.png")
        await browser.clear_emulation()

        report.section("11. console and network health")
        report.check("no JavaScript errors", not browser.errors, "; ".join(browser.errors[:2]))
        network_failures = [f for f in browser.failed if "favicon" not in f.lower()]
        report.check("no failed requests", not network_failures, "; ".join(network_failures[:2]))

    return report.summary()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="instance to test")
    parser.add_argument("--shots", type=Path, default=None, help="directory for screenshots")
    parser.add_argument("--port", type=int, default=9222, help="DevTools port")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    print(f"Testing {base_url}")
    with ChromeProcess(port=args.port):
        return asyncio.run(run(base_url, args.shots))


if __name__ == "__main__":
    raise SystemExit(main())
