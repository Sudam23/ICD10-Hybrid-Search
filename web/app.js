/* ICD-10-CM Hybrid Code Finder - front end. */
(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);

  const form = $("#search-form");
  const queryInput = $("#query");
  const searchButton = $("#search-btn");
  const chapterSelect = $("#chapter");
  const kSelect = $("#k");
  const billableOnly = $("#billable-only");
  const statusEl = $("#status");
  const resultsEl = $("#results");
  const examplesEl = $("#examples");

  /* ---------------------------------------------------------------- utils */

  const escapeHtml = (value) =>
    String(value ?? "").replace(/[&<>"']/g, (char) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char])
    );

  const storage = {
    get(key) {
      try { return window.localStorage.getItem(key); } catch { return null; }
    },
    set(key, value) {
      try { window.localStorage.setItem(key, value); } catch { /* private mode */ }
    },
  };

  const selectedMode = () => form.querySelector('input[name="mode"]:checked').value;

  async function getJSON(url) {
    const response = await fetch(url, { headers: { Accept: "application/json" } });
    let payload = null;
    try { payload = await response.json(); } catch { /* non-JSON error page */ }
    if (!response.ok) {
      const detail = payload?.detail || payload?.error || `HTTP ${response.status}`;
      const error = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      error.status = response.status;
      throw error;
    }
    return payload;
  }

  /* --------------------------------------------------------------- theme */

  const themeToggle = $("#theme-toggle");
  const savedTheme = storage.get("icd10-theme");
  if (savedTheme === "dark" || savedTheme === "light") {
    document.documentElement.setAttribute("data-theme", savedTheme);
  }
  themeToggle.addEventListener("click", () => {
    const isDark = document.documentElement.getAttribute("data-theme") === "dark" ||
      (document.documentElement.getAttribute("data-theme") !== "light" &&
        window.matchMedia("(prefers-color-scheme: dark)").matches);
    const next = isDark ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    storage.set("icd10-theme", next);
  });

  /* ------------------------------------------------------------- rendering */

  function renderResults(payload) {
    if (!payload.results.length) {
      resultsEl.innerHTML =
        `<p class="hint">No codes matched &ldquo;${escapeHtml(payload.query)}&rdquo;. ` +
        `Try fewer words, or switch to Semantic mode.</p>`;
      return;
    }

    resultsEl.innerHTML = payload.results
      .map((item) => {
        const evidence = [];
        if (item.match_reason === "exact_code") {
          evidence.push(`<span class="tag pinned"><strong>exact code match</strong></span>`);
        } else if (item.match_reason === "code_prefix") {
          evidence.push(`<span class="tag pinned"><strong>code starts with your query</strong></span>`);
        } else if (item.match_reason === "code_sibling") {
          evidence.push(`<span class="tag pinned"><strong>same category</strong></span>`);
        }
        if (item.dense_rank) {
          evidence.push(
            `<span class="tag">semantic <strong>#${item.dense_rank}</strong>` +
            (item.dense_score !== null ? ` &middot; cos ${item.dense_score.toFixed(3)}` : "") +
            `</span>`
          );
        }
        if (item.lexical_rank) {
          evidence.push(
            `<span class="tag">keyword <strong>#${item.lexical_rank}</strong>` +
            (item.lexical_score !== null ? ` &middot; bm25 ${item.lexical_score.toFixed(2)}` : "") +
            `</span>`
          );
        }
        if (item.matched_terms.length) {
          evidence.push(
            `<span class="tag">matched: <strong>${escapeHtml(item.matched_terms.join(", "))}</strong></span>`
          );
        }

        const synonyms = item.synonyms.length
          ? `<p class="synonyms">Also called: <em>${escapeHtml(item.synonyms.join("; "))}</em></p>`
          : "";

        return `
          <article class="result">
            <div class="result-head">
              <span class="rank">${item.rank}</span>
              <button class="code" type="button" data-code="${escapeHtml(item.code_dotted)}"
                      title="Click to copy">${escapeHtml(item.code_dotted)}</button>
              <span class="badge ${item.billable ? "billable" : ""}">
                ${item.billable ? "billable" : "not billable on its own"}
              </span>
            </div>
            <p class="result-desc">${escapeHtml(item.description)}</p>
            <p class="result-meta">
              Chapter ${item.chapter} &middot; ${escapeHtml(item.chapter_title)}
              ${item.block ? ` &middot; ${escapeHtml(item.block)} ${escapeHtml(item.block_title)}` : ""}
            </p>
            ${synonyms}
            <div class="evidence">${evidence.join("")}</div>
          </article>`;
      })
      .join("");
  }

  resultsEl.addEventListener("click", async (event) => {
    const button = event.target.closest(".code");
    if (!button) return;
    try {
      await navigator.clipboard.writeText(button.dataset.code);
      const original = button.textContent;
      button.textContent = "copied";
      setTimeout(() => { button.textContent = original; }, 900);
    } catch { /* clipboard unavailable */ }
  });

  /* --------------------------------------------------------------- search */

  let activeRequest = 0;

  async function runSearch(query, { pushState = true } = {}) {
    const trimmed = (query || "").trim();
    if (!trimmed) return;

    const params = new URLSearchParams({
      q: trimmed,
      k: kSelect.value,
      mode: selectedMode(),
    });
    if (billableOnly.checked) params.set("billable_only", "true");
    if (chapterSelect.value) params.set("chapter", chapterSelect.value);

    const requestId = ++activeRequest;
    searchButton.disabled = true;
    statusEl.className = "status";
    statusEl.textContent = "Searching…";

    try {
      const payload = await getJSON(`/api/search?${params.toString()}`);
      if (requestId !== activeRequest) return;
      renderResults(payload);
      const timings = payload.timings_ms || {};
      statusEl.textContent =
        `${payload.results.length} of ${payload.total_candidates} candidates · ` +
        `${timings.total ?? "?"} ms (embed ${timings.embed ?? 0} · semantic ${timings.dense ?? 0}` +
        ` · keyword ${timings.lexical ?? 0})`;
      if (pushState) {
        const url = new URL(window.location.href);
        url.search = params.toString();
        window.history.replaceState({}, "", url);
      }
    } catch (error) {
      if (requestId !== activeRequest) return;
      statusEl.className = "status error";
      statusEl.textContent =
        error.status === 503
          ? "The search index is still loading. This is a free-tier container waking up — try again in a few seconds."
          : error.status === 429
          ? "Too many requests from this address. Please wait a moment."
          : error.status === 422
          ? "That query could not be understood. Try different words, or reload the page."
          : `Search failed: ${error.message}`;
      resultsEl.innerHTML = "";
    } finally {
      if (requestId === activeRequest) searchButton.disabled = false;
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    runSearch(queryInput.value);
  });

  form.querySelectorAll('input[name="mode"]').forEach((radio) =>
    radio.addEventListener("change", () => queryInput.value.trim() && runSearch(queryInput.value))
  );
  [chapterSelect, kSelect, billableOnly].forEach((control) =>
    control.addEventListener("change", () => queryInput.value.trim() && runSearch(queryInput.value))
  );

  /* ------------------------------------------------------------ bootstrap */

  async function loadExamples() {
    try {
      const { examples } = await getJSON("/api/examples");
      examplesEl.innerHTML = examples
        .map((text) => `<button class="chip" type="button">${escapeHtml(text)}</button>`)
        .join("");
      examplesEl.addEventListener("click", (event) => {
        const chip = event.target.closest(".chip");
        if (!chip) return;
        queryInput.value = chip.textContent.trim();
        runSearch(queryInput.value);
      });
    } catch { /* examples are optional */ }
  }

  async function loadMeta() {
    try {
      const stats = await getJSON("/api/stats");
      $("#stat-codes").textContent = stats.total_codes.toLocaleString();
      $("#stat-model").textContent = String(stats.embedding_model).split("/").pop();
      const megabytes = Object.values(stats.index_bytes || {}).reduce((a, b) => a + b, 0) / 1e6;
      $("#footer-stats").textContent =
        `${stats.total_codes.toLocaleString()} codes · ${stats.billable_codes.toLocaleString()} billable · ` +
        `${stats.embedding_dim}-dim ${stats.quantization} index (${megabytes.toFixed(1)} MB) · ` +
        `built ${String(stats.index_built_at || "").slice(0, 10)}`;
    } catch { /* stats are optional */ }

    try {
      const chapters = await getJSON("/api/chapters");
      chapterSelect.insertAdjacentHTML(
        "beforeend",
        chapters
          .map((c) => `<option value="${c.chapter}">${c.chapter}. ${escapeHtml(c.title)}</option>`)
          .join("")
      );
      const wanted = new URLSearchParams(window.location.search).get("chapter");
      if (wanted && chapterSelect.querySelector(`option[value="${wanted}"]`)) {
        chapterSelect.value = wanted;
      }
    } catch { /* filter stays "all chapters" */ }
  }

  function metricsTable(title, caption, rows) {
    const best = Math.max(...rows.map((r) => r.mrr));
    const body = rows
      .map(
        (row) => `
        <tr class="${row.mrr === best ? "best" : ""}">
          <td>${escapeHtml(row.label)}</td>
          <td>${(row.r1 * 100).toFixed(1)}%</td>
          <td>${(row.r5 * 100).toFixed(1)}%</td>
          <td>${(row.r10 * 100).toFixed(1)}%</td>
          <td>${row.mrr.toFixed(3)}</td>
          <td>${row.latency ? row.latency.toFixed(0) + " ms" : "—"}</td>
        </tr>`
      )
      .join("");
    return `
      <div class="metrics-scroll">
      <table class="metrics">
        <caption>${escapeHtml(title)}<br><span class="hint">${caption}</span></caption>
        <thead><tr>
          <th>Mode</th><th>Recall@1</th><th>Recall@5</th><th>Recall@10</th>
          <th>MRR@10</th><th>Median latency</th>
        </tr></thead>
        <tbody>${body}</tbody>
      </table>
      </div>`;
  }

  async function loadEvaluation() {
    const caption = $("#eval-caption");
    try {
      const report = await getJSON("/api/evaluation");
      const labels = {
        cms_index_test: {
          title: "Gold set 1 — CMS Alphabetic Index (held-out test split)",
          caption:
            "Queries and correct codes both come from the official CMS index, not from me. " +
            "Disjoint from the split used to tune the fusion weights.",
        },
        hand_written_lay: {
          title: "Gold set 2 — hand-written plain-English queries",
          caption:
            "Written by the author. Small, so treat it as a smoke test of lay phrasing rather " +
            "than a benchmark.",
        },
      };

      $("#eval-tables").innerHTML = Object.entries(report.gold_sets || {})
        .map(([name, entry]) => {
          const modes = entry.modes || {};
          const rows = ["lexical", "dense", "hybrid"]
            .filter((mode) => modes[mode])
            .map((mode) => ({
              label: { lexical: "Keyword (BM25)", dense: "Semantic", hybrid: "Hybrid (fused)" }[mode],
              r1: modes[mode].recall_at["1"],
              r5: modes[mode].recall_at["5"],
              r10: modes[mode].recall_at["10"],
              mrr: modes[mode].mrr_at_10,
              latency: modes[mode].median_latency_ms,
            }));
          const meta = labels[name] || { title: name, caption: "" };
          return metricsTable(
            `${meta.title} — ${entry.num_queries.toLocaleString()} queries`,
            meta.caption,
            rows
          );
        })
        .join("");

      caption.textContent =
        `Produced by \`make eval\` on ${String(report.generated_at).slice(0, 10)} ` +
        `with ${report.index?.embedding_model}. Highlighted row = best MRR@10.`;
    } catch (error) {
      caption.textContent =
        error.status === 404
          ? "No evaluation report has been generated for this deployment yet."
          : `Could not load the evaluation report (${error.message}).`;
    }
  }

  function restoreFromUrl() {
    const params = new URLSearchParams(window.location.search);
    const query = params.get("q");
    if (!query) return false;
    queryInput.value = query;
    const mode = params.get("mode");
    const modeRadio = mode && form.querySelector(`input[name="mode"][value="${mode}"]`);
    if (modeRadio) modeRadio.checked = true;
    const k = params.get("k");
    const kOption = k && Array.from(kSelect.options).find((o) => o.value === k);
    if (kOption) kSelect.value = kOption.value;
    if (params.get("billable_only") === "true") billableOnly.checked = true;
    runSearch(query, { pushState: false });
    return true;
  }

  loadExamples();
  loadEvaluation();
  loadMeta().then(() => {
    if (!restoreFromUrl()) queryInput.focus();
  });
})();
