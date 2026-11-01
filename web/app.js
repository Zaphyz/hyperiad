/* Hyperiad notebook front end.
 *
 * Cells hold Hyperiad code or text. Code runs in a kernel: the browser kernel (Pyodide running the
 * hyperiad Python package) or a server kernel (POST /api/run). A cell whose estimated cost exceeds
 * the browser budget comes back "too_expensive" and can be sent to the server, which rebuilds the
 * notebook's state by replaying the cells run so far.
 */
(() => {
  "use strict";

  const CONFIG = Object.assign(
    {
      pyodide: "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/",
      pysrc: "pysrc.json",
      server: null,
      budget: 2e7,
      builtins: [],
      storage: "hyperiad.notebook.v1",
      // Shown when a cell is over every budget available to it: no server, or the server refused it too.
      heavyMessage:
        "This task is estimated to be too heavy to be run here: about {estimate} operations, over the limit of {budget}. Try a smaller instance.",
      footer: null, // HTML for the footer, set by the site that hosts the notebook
      home: null, // {href, label}: a link back to the hosting site
    },
    window.HYPERIAD_CONFIG || {}
  );

  const KEYWORDS = new Set(["and", "or", "not", "true", "false", "null"]);
  const $ = (sel, root = document) => root.querySelector(sel);
  const el = (tag, attrs = {}, ...children) => {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else if (v !== null && v !== undefined && v !== false) node.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children) if (c) node.append(c);
    return node;
  };
  const nextFrame = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
  const escapeHtml = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  let uidCounter = 0;
  const uid = () => `c${Date.now().toString(36)}${(uidCounter++).toString(36)}`;

  // -- kernels ---------------------------------------------------------------

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = src;
      s.onload = resolve;
      s.onerror = () => reject(new Error(`could not load ${src}`));
      document.head.append(s);
    });
  }

  class BrowserKernel {
    constructor() {
      this.name = "browser";
    }
    async start(report) {
      report("loading", "Loading Python in the browser…");
      if (typeof window.loadPyodide !== "function") await loadScript(CONFIG.pyodide + "pyodide.js");
      this.py = await window.loadPyodide({ indexURL: CONFIG.pyodide });
      report("loading", "Loading NumPy, SymPy and NetworkX…");
      await this.py.loadPackage(["numpy", "sympy", "networkx"]);
      report("loading", "Loading Hyperiad…");
      const bundle = await (await fetch(CONFIG.pysrc)).json();
      const FS = this.py.FS;
      for (const [path, text] of Object.entries(bundle.files)) {
        const parts = `/hyd/${path}`.split("/").slice(1, -1);
        let dir = "";
        for (const p of parts) {
          dir += "/" + p;
          if (!FS.analyzePath(dir).exists) FS.mkdir(dir);
        }
        FS.writeFile(`/hyd/${path}`, text);
      }
      this.py.runPython("import sys; sys.path.insert(0, '/hyd')");
      this.mod = this.py.pyimport("hyperiad.web_kernel");
      this.mod.start(CONFIG.budget);
      return JSON.parse(this.mod.builtins());
    }
    async run(code) {
      return JSON.parse(this.mod.run(code));
    }
  }

  class ServerKernel {
    constructor(base) {
      this.base = base.replace(/\/$/, "");
      this.name = "server";
    }
    async info() {
      const r = await fetch(`${this.base}/info`);
      if (!r.ok) throw new Error(`server answered ${r.status}`);
      return r.json();
    }
    async run(code, replay) {
      const r = await fetch(`${this.base}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code, replay }),
      });
      if (!r.ok) {
        const body = await r.json().catch(() => null);
        throw new Error(body?.error || `server answered ${r.status}`);
      }
      return r.json();
    }
  }

  // -- state -------------------------------------------------------------------

  const state = {
    cells: [],
    counter: 0,
    selected: null,
    kernel: null, // the kernel cells run in
    server: null, // a server kernel, if one answers
    history: [], // sources of cells run successfully, in order: what a server replays
    builtins: [],
    builtinNames: new Set(CONFIG.builtins),
    ready: null,
  };

  const newCell = (type = "code", source = "") => ({ id: uid(), type, source, outputs: [], n: null, status: "idle", where: null });

  const PREVIEW = CONFIG.kernel === "none";

  function load() {
    if (PREVIEW) return window.HYPERIAD_NOTEBOOK;
    try {
      const saved = JSON.parse(localStorage.getItem(CONFIG.storage) || "null");
      if (saved && Array.isArray(saved.cells) && saved.cells.length) return saved;
    } catch (e) {
      /* storage unavailable: fall back to the bundled notebook */
    }
    return window.HYPERIAD_NOTEBOOK;
  }

  let saveTimer = null;
  function save() {
    if (PREVIEW) return;
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      try {
        const cells = state.cells.map(({ type, source, outputs, n, status, where }) => ({ type, source, outputs, n, status, where }));
        localStorage.setItem(CONFIG.storage, JSON.stringify({ cells }));
      } catch (e) {
        /* best effort only */
      }
    }, 400);
  }

  // -- syntax highlighting -----------------------------------------------------

  const TOKEN_RE =
    /(#[^\n]*)|("(?:[^"\\\n]|\\.)*"?|'(?:[^'\\\n]|\\.)*'?)|(\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?)|([A-Za-z_][A-Za-z0-9_]*)|(->|=>|==|!=|<=|>=|\*\*|[-+*/^<>=])/g;

  function highlight(code) {
    let out = "";
    let last = 0;
    for (const m of code.matchAll(TOKEN_RE)) {
      out += escapeHtml(code.slice(last, m.index));
      const t = escapeHtml(m[0]);
      if (m[1]) out += `<span class="t-com">${t}</span>`;
      else if (m[2]) out += `<span class="t-str">${t}</span>`;
      else if (m[3]) out += `<span class="t-num">${t}</span>`;
      else if (m[4]) {
        if (KEYWORDS.has(m[4])) out += `<span class="t-kw">${t}</span>`;
        else if (state.builtinNames.has(m[4])) out += `<span class="t-fn">${t}</span>`;
        else out += t;
      } else out += `<span class="t-op">${t}</span>`;
      last = m.index + m[0].length;
    }
    return out + escapeHtml(code.slice(last)) + "\n";
  }

  // -- markdown (text cells and registry definitions) ------------------------------

  function inlineMd(text) {
    const maths = [];
    const codes = [];
    let s = text.replace(/`([^`]+)`/g, (_, c) => `\u0001${codes.push(c) - 1}\u0001`);
    s = s.replace(/\$\$[\s\S]+?\$\$|\$[^$\n]+?\$/g, (m) => `\u0000${maths.push(m) - 1}\u0000`);
    s = escapeHtml(s)
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\*([^*]+)\*/g, "<em>$1</em>")
      .replace(/\[([^\]]+)\]\((https:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    return s
      .replace(/\u0000(\d+)\u0000/g, (_, i) => escapeHtml(maths[+i]))
      .replace(/\u0001(\d+)\u0001/g, (_, i) => `<code>${escapeHtml(codes[+i])}</code>`);
  }

  const slug = (text) =>
    "s-" +
    text
      .toLowerCase()
      .replace(/\$[^$]*\$/g, "")
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "");

  const CALLOUTS = {
    definition: "definition", claim: "claim", proposition: "claim", theorem: "claim", observation: "claim",
    proposal: "proposal", choice: "choice", settled: "choice", question: "question", open: "question",
    example: "example", remark: "note", note: "note", device: "device", interpretation: "question",
  };

  function markdown(text) {
    const lines = text.replace(/\r/g, "").trim().split("\n");
    const isBlockStart = (l) => /^(#{1,3}\s|>|```|---+\s*$|\s*([-*]|\d+\.)\s+|\|)/.test(l);
    let html = "";
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) {
        i++;
        continue;
      }
      if (/^```/.test(line)) {
        const code = [];
        for (i++; i < lines.length && !/^```/.test(lines[i]); i++) code.push(lines[i]);
        i++;
        html += `<pre class="md-code"><code>${highlight(code.join("\n"))}</code></pre>`;
        continue;
      }
      const h = line.match(/^(#{1,3})\s+(.*)$/);
      if (h) {
        const n = h[1].length;
        html += `<h${n} id="${slug(h[2])}">${inlineMd(h[2])}</h${n}>`;
        i++;
        continue;
      }
      if (/^---+\s*$/.test(line)) {
        html += "<hr>";
        i++;
        continue;
      }
      if (/^>/.test(line)) {
        const inner = [];
        for (; i < lines.length && /^>/.test(lines[i]); i++) inner.push(lines[i].replace(/^>\s?/, ""));
        const word = (inner[0].match(/^\*\*([A-Za-z]+)/) || [])[1];
        const kind = CALLOUTS[(word || "").toLowerCase()] || "note";
        html += `<aside class="callout ${kind}">${markdown(inner.join("\n"))}</aside>`;
        continue;
      }
      if (/^\s*([-*]|\d+\.)\s+/.test(line)) {
        const ordered = /^\s*\d+\./.test(line);
        const items = [];
        for (; i < lines.length; i++) {
          const l = lines[i];
          if (/^\s*([-*]|\d+\.)\s+/.test(l)) items.push(l.replace(/^\s*([-*]|\d+\.)\s+/, ""));
          else if (/^\s{2,}\S/.test(l) && items.length) items[items.length - 1] += " " + l.trim();
          else break;
        }
        const tag = ordered ? "ol" : "ul";
        html += `<${tag}>${items.map((it) => `<li>${inlineMd(it)}</li>`).join("")}</${tag}>`;
        continue;
      }
      if (/^\|/.test(line)) {
        const rows = [];
        for (; i < lines.length && /^\|/.test(lines[i]); i++) rows.push(lines[i]);
        const cells = (r) => r.replace(/^\||\|\s*$/g, "").split("|").map((c) => c.trim());
        const body = rows.filter((r) => !/^\|\s*:?-{3,}/.test(r));
        const [head, ...rest] = body;
        html +=
          `<div class="md-table"><table><thead><tr>${cells(head).map((c) => `<th>${inlineMd(c)}</th>`).join("")}</tr></thead>` +
          `<tbody>${rest.map((r) => `<tr>${cells(r).map((c) => `<td>${inlineMd(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
        continue;
      }
      const para = [];
      for (; i < lines.length && lines[i].trim() && !(para.length && isBlockStart(lines[i])); i++) para.push(lines[i]);
      html += `<p>${inlineMd(para.join("\n")).replace(/\n/g, " ")}</p>`;
    }
    return html;
  }

  // MathJax measures the page, so typeset only once the node is attached.
  function typeset(node) {
    const go = () =>
      requestAnimationFrame(() => {
        if (node.isConnected) window.MathJax.typesetPromise([node]).catch(() => {});
      });
    if (window.MathJax?.startup?.promise) window.MathJax.startup.promise.then(go);
    else window.addEventListener("load", () => window.MathJax?.startup?.promise?.then(go), { once: true });
  }

  function latex(tex, fallback) {
    const box = el("div", { class: "out-latex", title: fallback });
    const show = () => {
      if (window.MathJax?.tex2svgPromise) {
        window.MathJax.tex2svgPromise(tex, { display: true, em: 16, ex: 8, containerWidth: 960 })
          .then((node) => box.replaceChildren(node))
          .catch(() => box.replaceChildren(el("pre", { class: "out-text", text: fallback })));
      } else box.replaceChildren(el("pre", { class: "out-text", text: fallback }));
    };
    box.append(el("pre", { class: "out-text", text: fallback }));
    if (window.MathJax?.startup?.promise) window.MathJax.startup.promise.then(show);
    else window.addEventListener("load", () => window.MathJax?.startup?.promise?.then(show), { once: true });
    return box;
  }

  // -- rendering -----------------------------------------------------------------

  const nb = $("#notebook");

  function renderOutputs(cell) {
    const box = el("div", { class: "outputs", "aria-live": "polite" });
    if (cell.where === "server") box.append(el("span", { class: "badge", text: "computed on the server" }));
    for (const out of cell.outputs) {
      if (out.type === "svg") {
        const pic = el("div", { class: "out-svg" });
        pic.innerHTML = out.data; // produced by our own renderer, with entries escaped
        box.append(pic);
        if (out.caption) box.append(el("div", { class: "caption", text: out.caption }));
      } else if (out.type === "frames") {
        box.append(framesWidget(out));
      } else if (out.type === "latex") {
        box.append(latex(out.data, out.text || out.data));
      } else if (out.type === "markdown") {
        const md = el("div", { class: "md" });
        md.innerHTML = markdown(out.data);
        box.append(md);
        typeset(md);
      } else if (out.type === "error") {
        const err = el("div", { class: `out-error ${out.kind || "error"}` });
        if (out.kind === "too_expensive" && !(state.server && cell.where !== "server")) {
          err.append(document.createTextNode(heavyMessage(out)));
        } else {
          err.append(document.createTextNode(out.data));
        }
        if (out.kind === "too_expensive") {
          if (state.server && cell.where !== "server") {
            err.append(
              el("br"),
              el("button", { class: "run-server", type: "button", text: "Run on the server", onclick: () => runOnServer(cell) })
            );
          }
        }
        box.append(err);
      } else {
        box.append(el("pre", { class: "out-text", text: out.data }));
      }
    }
    return box;
  }

  function heavyMessage(out) {
    const num = (x) => (typeof x === "number" ? Number(x.toPrecision(2)).toExponential().replace("e+", "e") : "?");
    if (typeof out.estimate !== "number") return `${out.data}.\nThis task is estimated to be too heavy to be run here.`;
    return CONFIG.heavyMessage.replace("{estimate}", num(out.estimate)).replace("{budget}", num(out.budget));
  }

  // Frames: an animation or a stepper. Every frame is built once; switching shows one, hides the rest.
  const reducedMotion = () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

  function framesWidget(out) {
    const n = out.frames.length;
    const stage = el("div", { class: "frames-stage", role: "img", "aria-label": out.caption || "animation" });
    const frames = out.frames.map((svg, k) => {
      const f = el("div", { class: "frame" });
      f.innerHTML = svg; // produced by our own renderer, with entries escaped
      f.hidden = k !== 0;
      stage.append(f);
      return f;
    });
    const label = el("span", { class: "frames-label", "aria-live": "polite" });
    const slider = el("input", { type: "range", min: "0", max: String(n - 1), value: "0", step: "1", "aria-label": "Frame" });
    const play = el("button", { type: "button", class: "frames-play" });
    const prev = el("button", { type: "button", class: "frames-step", "aria-label": "Previous frame", text: "‹" });
    const next = el("button", { type: "button", class: "frames-step", "aria-label": "Next frame", text: "›" });
    let at = 0;
    let timer = null;
    let wanted = !!out.autoplay && !reducedMotion();
    let visible = false;
    const show = (k) => {
      at = ((k % n) + n) % n;
      frames.forEach((f, j) => (f.hidden = j !== at));
      slider.value = String(at);
      label.textContent = out.labels?.[at] || `${at + 1} / ${n}`;
    };
    const setPlay = () => {
      const on = timer !== null;
      play.textContent = on ? "❚❚" : "▶";
      play.setAttribute("aria-label", on ? "Pause" : "Play");
    };
    const tick = () => {
      const hold = at === n - 1 ? 2.2 : 1;
      timer = setTimeout(() => {
        show(at + 1);
        tick();
      }, (out.interval || 900) * hold);
    };
    const start = () => {
      if (timer === null && n > 1) tick();
      setPlay();
    };
    const stop = () => {
      clearTimeout(timer);
      timer = null;
      setPlay();
    };
    play.addEventListener("click", () => {
      wanted = timer === null;
      wanted ? start() : stop();
    });
    stage.addEventListener("click", () => play.click());
    prev.addEventListener("click", () => (stop(), (wanted = false), show(at - 1)));
    next.addEventListener("click", () => (stop(), (wanted = false), show(at + 1)));
    slider.addEventListener("input", () => (stop(), (wanted = false), show(+slider.value)));
    if ("IntersectionObserver" in window) {
      new IntersectionObserver((entries) => {
        visible = entries.some((e) => e.isIntersecting);
        visible && wanted ? start() : stop();
      }).observe(stage);
    }
    show(0);
    setPlay();
    const controls = el("div", { class: "frames-controls" }, play, prev, slider, next, label);
    const fig = el("figure", { class: "frames" }, stage, controls);
    if (out.caption) fig.append(el("figcaption", { class: "caption", text: out.caption }));
    return fig;
  }

  function editor(cell) {
    const pre = el("pre", { "aria-hidden": "true" });
    const area = el("textarea", {
      id: `src-${cell.id}`,
      spellcheck: "false",
      autocapitalize: "off",
      autocomplete: "off",
      "aria-label": cell.type === "code" ? "Hyperiad code" : "Text (Markdown with $math$)",
    });
    area.value = cell.source;
    const paint = () => {
      pre.innerHTML = cell.type === "code" ? highlight(area.value) : escapeHtml(area.value) + "\n";
    };
    paint();
    area.addEventListener("input", () => {
      cell.source = area.value;
      paint();
      save();
    });
    area.addEventListener("focus", () => select(cell, false));
    area.addEventListener("keydown", (e) => onKey(e, cell, area));
    if (cell.type === "text") area.addEventListener("blur", () => setTimeout(() => finishText(cell), 120));
    return el("div", { class: "editor" }, pre, area);
  }

  function tools(cell) {
    const b = (label, title, fn) => el("button", { type: "button", title, "aria-label": title, text: label, onclick: fn });
    return el(
      "div",
      { class: "cell-tools" },
      cell.type === "code" ? b("Run", "Run this cell", () => enqueue(cell)) : null,
      b("↑", "Move up", () => move(cell, -1)),
      b("↓", "Move down", () => move(cell, 1)),
      b("+", "Insert a code cell below", () => insertAfter(cell, newCell("code"))),
      b("×", "Delete this cell", () => remove(cell))
    );
  }

  function renderCell(cell) {
    const node = el("section", { class: `cell ${cell.type === "code" ? "code" : "text-cell"}`, "data-id": cell.id });
    if (cell.status === "running") node.classList.add("running");
    if (cell.status === "queued") node.classList.add("queued");
    if (state.selected === cell.id) node.classList.add("selected");
    const label = cell.type === "code" ? `In[${cell.n ?? " "}]:=` : "";
    const body = el("div", { class: "cell-body" }, tools(cell));
    if (cell.type === "text" && !cell.editing) {
      const r = el("div", { class: "rendered", tabindex: "0", "aria-label": "Text cell; press Enter to edit" });
      r.innerHTML = markdown(cell.source || "*Empty text cell*");
      const edit = () => {
        cell.editing = true;
        replace(cell);
        $(`#src-${cell.id}`)?.focus();
      };
      r.addEventListener("dblclick", edit);
      r.addEventListener("click", () => select(cell));
      r.addEventListener("keydown", (e) => e.key === "Enter" && (e.preventDefault(), edit()));
      body.append(r);
      typeset(r);
    } else {
      body.append(editor(cell));
    }
    node.append(el("div", { class: "label in", text: label }), body);
    if (cell.type === "code" && cell.outputs.length) {
      node.append(el("div", { class: "label out", text: cell.n ? `Out[${cell.n}]=` : "" }), renderOutputs(cell));
    }
    return node;
  }

  function replace(cell) {
    const old = nb.querySelector(`[data-id="${cell.id}"]`);
    const fresh = renderCell(cell);
    if (old) old.replaceWith(fresh);
    else nb.append(fresh);
  }

  function renderAll() {
    nb.replaceChildren(...state.cells.map(renderCell));
    renderToc();
  }

  function renderToc() {
    const toc = $("#toc");
    if (!toc || !CONFIG.toc) return;
    const items = [];
    for (const cell of state.cells) {
      if (cell.type !== "text") continue;
      for (const m of (cell.source || "").matchAll(/^(#{1,2})\s+(.*)$/gm)) items.push({ level: m[1].length, text: m[2] });
    }
    const list = el("ol", { class: "toc-list" });
    for (const it of items) {
      const a = el("a", { href: `#${slug(it.text)}`, class: `toc-l${it.level}` });
      a.innerHTML = inlineMd(it.text);
      a.addEventListener("click", () => document.body.classList.remove("toc-open"));
      list.append(el("li", {}, a));
    }
    toc.replaceChildren(el("h2", { text: "Contents" }), list);
    toc.hidden = false;
    typeset(toc);
  }

  function select(cell, rerender = true) {
    if (state.selected === cell.id) return;
    const prev = state.cells.find((c) => c.id === state.selected);
    state.selected = cell.id;
    for (const node of nb.querySelectorAll(".cell.selected")) node.classList.remove("selected");
    nb.querySelector(`[data-id="${cell.id}"]`)?.classList.add("selected");
    if (prev && prev.type === "text" && prev.editing && rerender) finishText(prev);
  }

  function finishText(cell) {
    if (cell.type !== "text" || !cell.editing) return;
    if (document.activeElement && document.activeElement.id === `src-${cell.id}`) return;
    cell.editing = false;
    replace(cell);
  }

  // -- editing -------------------------------------------------------------------

  function focusCell(cell) {
    if (cell.type === "text") cell.editing = true;
    replace(cell);
    const area = $(`#src-${cell.id}`);
    if (area) {
      area.focus();
      area.setSelectionRange(area.value.length, area.value.length);
    }
  }

  function insertAfter(cell, fresh) {
    const i = state.cells.indexOf(cell);
    state.cells.splice(i + 1, 0, fresh);
    renderAll();
    focusCell(fresh);
    save();
  }

  function move(cell, d) {
    const i = state.cells.indexOf(cell);
    const j = i + d;
    if (j < 0 || j >= state.cells.length) return;
    [state.cells[i], state.cells[j]] = [state.cells[j], state.cells[i]];
    renderAll();
    save();
  }

  function remove(cell) {
    state.cells = state.cells.filter((c) => c !== cell);
    if (!state.cells.length) state.cells.push(newCell("code"));
    renderAll();
    save();
  }

  function onKey(e, cell, area) {
    if (e.key === "Enter" && (e.shiftKey || e.ctrlKey || e.metaKey || e.altKey)) {
      e.preventDefault();
      cell.source = area.value;
      if (cell.type === "text") {
        cell.editing = false;
        area.blur();
      } else enqueue(cell);
      if (e.shiftKey) {
        const i = state.cells.indexOf(cell);
        const next = state.cells[i + 1] || null;
        if (next) focusCell(next);
        else insertAfter(cell, newCell("code"));
      } else if (e.altKey) insertAfter(cell, newCell("code"));
      else if (cell.type === "text") replace(cell);
    } else if (e.key === "Escape") {
      area.blur();
    } else if (e.key === "Tab" && !e.shiftKey && !e.ctrlKey && !e.metaKey) {
      e.preventDefault();
      const { selectionStart: a, selectionEnd: b } = area;
      area.setRangeText("  ", a, b, "end");
      area.dispatchEvent(new Event("input"));
    }
  }

  // -- running -------------------------------------------------------------------

  let chain = Promise.resolve();

  function enqueue(cell) {
    if (PREVIEW) return; // a preview keeps the outputs it was built with
    if (cell.type !== "code") return;
    cell.status = "queued";
    replace(cell);
    chain = chain.then(() => runCell(cell)).catch((err) => console.error(err));
    return chain;
  }

  async function runCell(cell) {
    try {
      await state.ready;
    } catch (e) {
      /* reported in the status bar; the run below reports it on the cell */
    }
    cell.status = "running";
    replace(cell);
    setKernel("busy", "Running…");
    await nextFrame();
    let result;
    try {
      if (!state.kernel) throw new Error("no kernel is available to run this cell");
      result =
        state.kernel.name === "server"
          ? await state.kernel.run(cell.source, state.history)
          : await state.kernel.run(cell.source);
    } catch (err) {
      result = { status: "error", outputs: [{ type: "error", data: `Could not run the cell: ${err.message}` }] };
    }
    finish(cell, result, state.kernel ? state.kernel.name : null);
  }

  async function runOnServer(cell) {
    if (!state.server) return;
    cell.status = "running";
    replace(cell);
    setKernel("busy", "Running on the server…");
    let result;
    try {
      result = await state.server.run(cell.source, state.history);
    } catch (err) {
      result = { status: "error", outputs: [{ type: "error", data: `The server could not run the cell: ${err.message}` }] };
    }
    finish(cell, result, "server");
  }

  function finish(cell, result, where) {
    state.counter += 1;
    cell.n = state.counter;
    cell.outputs = result.outputs || [];
    cell.status = result.status;
    cell.where = where;
    if (result.status === "ok") state.history.push(cell.source);
    replace(cell);
    setKernel("ready", readyText());
    save();
  }

  function runAll() {
    for (const cell of state.cells) if (cell.type === "code") enqueue(cell);
  }

  // -- status and reference --------------------------------------------------------

  function setKernel(dotState, text) {
    $("#kernel-status .dot").dataset.state = dotState;
    $("#kernel-text").textContent = text;
  }

  function readyText() {
    if (!state.kernel) return "No kernel";
    const where = state.kernel.name === "browser" ? "Browser kernel ready" : "Server kernel ready";
    return state.server && state.kernel.name === "browser" ? `${where} · server available` : where;
  }

  function renderReference() {
    const q = $("#ref-search").value.trim().toLowerCase();
    const items = state.builtins.filter((b) => !q || b.name.toLowerCase().includes(q) || b.usage.toLowerCase().includes(q));
    $("#ref-list").replaceChildren(
      ...items.map((b) =>
        el(
          "li",
          {},
          el("button", { type: "button", text: b.name, title: `Insert ${b.name}`, onclick: () => insertName(b.name) }),
          el("p", { text: b.usage })
        )
      )
    );
  }

  function insertName(name) {
    let cell = state.cells.find((c) => c.id === state.selected && c.type === "code");
    if (!cell) cell = [...state.cells].reverse().find((c) => c.type === "code");
    if (!cell) return;
    focusCell(cell);
    const area = $(`#src-${cell.id}`);
    area.setRangeText(`${name}()`, area.selectionStart, area.selectionEnd, "end");
    area.setSelectionRange(area.selectionStart - 1, area.selectionStart - 1);
    area.dispatchEvent(new Event("input"));
  }

  // -- start ------------------------------------------------------------------------

  async function startKernels() {
    if (CONFIG.kernel === "none") {
      setKernel("preview", "Preview: every output was computed ahead of time. To run cells, open the notebook in Hyperiad.");
      return;
    }
    if (CONFIG.server) {
      const server = new ServerKernel(CONFIG.server);
      try {
        await server.info();
        state.server = server;
      } catch (e) {
        state.server = null;
      }
    }
    const browser = new BrowserKernel();
    try {
      state.builtins = await browser.start((s, t) => setKernel(s, t));
      state.kernel = browser;
    } catch (err) {
      console.error(err);
      if (state.server) {
        state.kernel = state.server;
        state.builtins = state.builtins.length ? state.builtins : [];
      } else {
        setKernel("failed", `The browser kernel could not start: ${err.message}`);
        throw err;
      }
    }
    for (const b of state.builtins) state.builtinNames.add(b.name);
    setKernel("ready", readyText());
    renderAll();
    renderReference();
  }

  function init() {
    const saved = load();
    if (window.HYPERIAD_NOTEBOOK?.title) {
      document.title = window.HYPERIAD_NOTEBOOK.title;
      const t = $("#nb-title");
      if (t) t.textContent = window.HYPERIAD_NOTEBOOK.title;
    }
    if (PREVIEW) {
      document.body.classList.add("preview");
      const foot = $("#foot-text");
      if (foot) foot.textContent = "Hyperiad 0.1 · a preview: every output was computed ahead of time from the notebook's code.";
    }
    if (CONFIG.footer) {
      const foot = $("#foot-text");
      if (foot) foot.innerHTML = CONFIG.footer; // trusted: written by the site that builds the page
    }
    if (CONFIG.home) {
      const brand = $(".brand");
      const link = el("a", { class: "home-link", href: CONFIG.home.href, text: `← ${CONFIG.home.label}` });
      if (brand) brand.before(link);
    }
    if (CONFIG.toc) {
      const btn = $("#btn-toc");
      btn.hidden = false;
      btn.addEventListener("click", () => {
        const open = document.body.classList.toggle("toc-open");
        btn.setAttribute("aria-expanded", String(open));
      });
    }
    state.cells = (saved?.cells || [newCell("code")]).map((c) => ({ ...newCell(c.type, c.source), ...c, id: uid() }));
    state.counter = Math.max(0, ...state.cells.map((c) => c.n || 0));
    state.builtins = (CONFIG.builtinDocs || []).slice();
    for (const b of state.builtins) state.builtinNames.add(b.name);
    renderAll();
    renderReference();

    $("#btn-run-all").addEventListener("click", runAll);
    $("#btn-add-code").addEventListener("click", () => insertAfter(state.cells[state.cells.length - 1], newCell("code")));
    $("#btn-add-text").addEventListener("click", () => insertAfter(state.cells[state.cells.length - 1], newCell("text")));
    $("#btn-clear").addEventListener("click", () => {
      for (const c of state.cells) Object.assign(c, { outputs: [], n: null, status: "idle", where: null });
      state.counter = 0;
      renderAll();
      save();
    });
    const refBtn = $("#btn-reference");
    refBtn.addEventListener("click", () => {
      const panel = $("#reference");
      panel.hidden = !panel.hidden;
      refBtn.setAttribute("aria-expanded", String(!panel.hidden));
      if (!panel.hidden) $("#ref-search").focus();
    });
    $("#ref-search").addEventListener("input", renderReference);

    state.ready = startKernels();
  }

  init();
})();
