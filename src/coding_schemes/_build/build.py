#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the coding-scheme dossiers from the single source of truth in content.py.

For every scheme this writes:
  scheme_N/scheme_N.tex   (formal dossier; compiled to PDF with tectonic)
  scheme_N/scheme_N.html  (interactive expert-evaluation surface)
  scheme_N/README.md      (plain explanatory note)

Directory-level outputs:
  index.html   (aggregated interactive dashboard + feedback merge console)
  README.md    (directory index)
  common_preamble.tex (shared LaTeX preamble, refreshed)

Usage:
  python3 build.py            # write all surfaces and compile PDFs
  python3 build.py --no-pdf   # skip PDF compilation

No em dashes are used in any generated output.
"""

from __future__ import annotations

import html
import os
import subprocess
import sys
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import content as C            # noqa: E402
from assets import CSS, JS_BASE, JS_INDEX  # noqa: E402

ROOT = os.path.dirname(HERE)                      # .../src/coding_schemes
DOMAIN_COLORS = C.DOMAIN_COLORS


# ==========================================================================
# Escaping helpers
# ==========================================================================

def h(s):
    """HTML-escape."""
    return html.escape(str(s), quote=True)


_LATEX = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def lx(s):
    """LaTeX-escape general prose."""
    out = []
    for ch in str(s):
        out.append(_LATEX.get(ch, ch))
    return "".join(out)


def tt(s):
    """LaTeX \\texttt with escaping (for code-like tokens)."""
    return r"\texttt{" + lx(s) + "}"


def fp(s):
    """LaTeX file path (verbatim-safe)."""
    return r"\filepath{" + str(s) + "}"


# ==========================================================================
# HTML rendering
# ==========================================================================

def fb_widget(fb_id, label):
    return (
        f'<div class="fb" data-fb-id="{h(fb_id)}" data-fb-label="{h(label)}">'
        '<div class="fb-head"><span class="fb-title">Expert feedback</span>'
        '<span class="fb-status" data-role="status">Not reviewed</span></div>'
        '<div class="fb-verdicts" role="group" aria-label="Verdict">'
        '<button class="fb-chip" data-verdict="approve" aria-pressed="false">Approve</button>'
        '<button class="fb-chip" data-verdict="revise" aria-pressed="false">Approve with revisions</button>'
        '<button class="fb-chip" data-verdict="discuss" aria-pressed="false">Needs discussion</button>'
        '<button class="fb-chip" data-verdict="reject" aria-pressed="false">Reject</button>'
        '</div>'
        '<div class="fb-clarity">Clarity of specification:'
        '<span class="stars" role="group" aria-label="Clarity rating">'
        + "".join(f'<span class="star" data-n="{n}" title="{n} of 5">&#9733;</span>' for n in range(1, 6))
        + '</span></div>'
        '<textarea class="fb-note" placeholder="Comments, suggested wording, missing anchors, edge cases, or reasons for the verdict..."></textarea>'
        '</div>'
    )


def render_values(values):
    if not values:
        return ""
    out = ['<div class="ladder">']
    for val in values:
        has_body = bool(val["pos"] or val["neg"] or val["boundary"])
        out.append('<div class="val has-body">' if has_body else '<div class="val">')
        caret = '<span class="caret">&#9656;</span>' if has_body else '<span class="caret" style="visibility:hidden">&#9656;</span>'
        out.append(
            '<div class="vhead"' + (' role="button" tabindex="0" aria-expanded="false"' if has_body else '') + '>'
            f'{caret}<span class="vv">{h(val["value"])}</span>'
            f'<span class="va">{h(val["anchor"])}</span></div>'
        )
        if has_body:
            body = ['<div class="vbody">']
            if val["pos"] or val["neg"]:
                body.append('<div class="indi">')
                if val["pos"]:
                    body.append('<div class="box pos"><div class="lab">Choose when</div><ul>'
                                + "".join(f"<li>{h(x)}</li>" for x in val["pos"]) + '</ul></div>')
                if val["neg"]:
                    body.append('<div class="box neg"><div class="lab">Not this if</div><ul>'
                                + "".join(f"<li>{h(x)}</li>" for x in val["neg"]) + '</ul></div>')
                body.append('</div>')
            if val["boundary"]:
                body.append(f'<div class="boundary"><b>Boundary.</b> {h(val["boundary"])}</div>')
            body.append('</div>')
            out.append("".join(body))
        out.append('</div>')
    out.append('</div>')
    return "".join(out)


def render_field(f):
    out = ['<div class="field">']
    out.append(f'<span class="fname">{h(f["name"])}</span>')
    ft = ' <span class="chip" style="font-size:11px">free text</span>' if f["free_text"] else ""
    out.append(f'<div class="fconstruct">{h(f["construct"])}{ft}</div>')
    out.append(render_values(f["values"]))
    if f["notes"]:
        out.append(f'<div class="fnotes">{h(f["notes"])}</div>')
    out.append('</div>')
    return "".join(out)


def render_section(sec):
    kind = sec["kind"]
    enh = sec.get("enhancement")
    cls = "card enh" if enh else "card"
    title = sec["title"]
    htag = '<span class="tag">Proposed</span>' if enh else ""
    parts = [f'<section id="{h(sec["id"])}" class="{cls}">',
             f'<h2>{htag}{h(title)}</h2>']
    if sec.get("intro"):
        parts.append(f'<p class="intro">{h(sec["intro"])}</p>')

    if kind == "prose":
        body = sec["body"]
        for i, para in enumerate(body):
            lead = ""
            if enh and i == 0:
                lead = ""
            parts.append(f'<p>{h(para)}</p>')
    elif kind == "list":
        dm = sec.get("domain")
        if dm:
            color = DOMAIN_COLORS[dm]
            parts.append(f'<div class="subgrid">')
            for it in sec["items"]:
                parts.append(f'<div class="item" style="--dm:{color}">{h(it)}</div>')
            parts.append('</div>')
        else:
            parts.append('<ul class="plain">' + "".join(f"<li>{h(it)}</li>" for it in sec["items"]) + '</ul>')
    elif kind == "keyvals":
        parts.append('<div class="kvlist">')
        for k, val in sec["items"]:
            parts.append(f'<div class="kvrow"><div class="kk">{h(k)}</div><div class="vv">{h(val)}</div></div>')
        parts.append('</div>')
    elif kind == "fields":
        for f in sec["fields"]:
            parts.append(render_field(f))
    elif kind == "taxonomy":
        parts.append('<div class="taxo">')
        for fam in sec["families"]:
            parts.append('<div class="tfam">')
            parts.append(f'<h4>{h(fam["family"])}</h4>')
            parts.append(f'<div class="tsub">Aligns to ontology subdomain: {h(fam["subdomain"])}</div>')
            parts.append(f'<div class="tdef">{h(fam["definition"])}</div>')
            parts.append('<div class="tgrouplabel">Representative constructs</div><div class="tchips">')
            parts.extend(f'<span class="tchip">{h(m)}</span>' for m in fam["members"])
            parts.append('</div>')
            parts.append('<div class="tgrouplabel">Candidate frameworks</div><div class="tchips">')
            parts.extend(f'<span class="tchip fw">{h(fw)}</span>' for fw in fam["frameworks"])
            parts.append('</div></div>')
        parts.append('</div>')
    elif kind == "examples":
        for ex in sec["examples"]:
            parts.append('<div class="example">')
            parts.append(f'<div class="exrec">{h(ex["record"])}</div>')
            parts.append(f'<div class="exmeta">{h(ex["meta"])}</div>')
            parts.append('<div class="excode">')
            for k, val in ex["coding"]:
                parts.append(f'<span class="kv"><b>{h(k)}</b> = {h(val)}</span>')
            parts.append('</div>')
            parts.append(f'<div class="exwhy"><span class="lab">Why</span>{h(ex["rationale"])}</div>')
            parts.append('</div>')

    if sec.get("feedback"):
        parts.append(fb_widget(sec["id"], title))
    parts.append('</section>')
    return "".join(parts)


def scheme_html(s):
    sid, num, title = s["id"], s["num"], s["title"]

    def toc_link(href, label, fb_target=None):
        dot = f'<span class="tdot" data-fb-target="{h(fb_target)}"></span>' if fb_target else ""
        return f'<a href="#{h(href)}"><span class="tlabel">{h(label)}</span>{dot}</a>'

    toc = ['<h4>On this page</h4>']
    toc.append(toc_link("top", "Overview"))
    toc.append(toc_link("overall", "Overall assessment", "overall"))
    for sec in s["sections"]:
        toc.append(toc_link(sec["id"], sec["title"], sec["id"] if sec.get("feedback") else None))

    meta_cells = "".join(
        f'<div class="cell"><div class="k">{h(k)}</div><div class="val">{h(v)}</div></div>'
        for k, v in s["meta"].items()
    )
    rq_chips = "".join(f'<span class="chip rq">{h(r)}</span>' for r in s["rqs"])
    src_items = "".join(f'<li>{h(p)}</li>' for p in s["sources"])
    out_items = "".join(f'<li>{h(p)}</li>' for p in s["outputs"])
    sections_html = "".join(render_section(sec) for sec in s["sections"])

    appbar = f'''<div class="appbar"><div class="appbar-in">
  <a class="brand" href="index.html"><span class="dot">BPS</span><span>Scheme {num}</span></a>
  <div class="progress-mini" title="Click to jump to the next unreviewed section"><span class="lab">0 reviewed</span><span class="bar"><span class="fill"></span></span></div>
  <div class="spacer"></div>
  <span class="savechip" id="savechip">All changes saved</span>
  <div class="sep"></div>
  <button class="btn primary" id="btn-export" title="Download all your feedback for this scheme as one JSON file (shortcut: s)">&#8595;&nbsp;Export feedback (JSON)</button>
  <button class="btn ghost sm" id="btn-print" title="Print or save as PDF">Print</button>
</div></div>'''

    hero = f'''<div id="top" class="hero">
  <div class="kicker">
    <span class="badge num">Scheme {num}</span>
    <span class="badge stage">{h(s["stage"])}</span>
    <span class="badge">{h(C.PROJECT["status"])}</span>
  </div>
  <h1>{h(title)}</h1>
  <p class="subtitle">{h(s["subtitle"])}</p>
  <p class="tagline">{h(s["tagline"])}</p>
  <div class="status"><span class="ic">!</span><div><b>Draft for expert evaluation.</b>
    {h(C.PROJECT["status_long"])} This surface lets you record a verdict and comments per section and export them as JSON.</div></div>
  <div class="metagrid">{meta_cells}</div>
  <div style="margin-top:14px"><div class="k" style="font-size:11px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);font-weight:700;margin-bottom:6px">Research-question linkage</div>
    <div class="chips">{rq_chips}</div></div>
</div>'''

    howto = '''<div class="howto">
  <h2>How to give feedback</h2>
  <p class="intro">Each coding decision below has its own <b>Expert feedback</b> box. For each one, pick a single verdict, rate how clearly it is specified, and add comments. Your feedback autosaves in this browser as you go, so you can stop and return later. When you are finished, click <b>Export feedback (JSON)</b> (top bar, or the button at the end of the page) to download one file and send it to the review team. That single export is the only step needed to share your review.</p>
  <div class="legend">
    <div class="lrow"><span class="pill approve">Approve</span><span class="ld"><b>Ready as written</b><span>Adopt this section without changes.</span></span></div>
    <div class="lrow"><span class="pill revise">Approve with revisions</span><span class="ld"><b>Adopt after edits</b><span>Fine in principle; note the changes in the comment.</span></span></div>
    <div class="lrow"><span class="pill discuss">Needs discussion</span><span class="ld"><b>Raise at consensus</b><span>Unresolved; flag for the team meeting.</span></span></div>
    <div class="lrow"><span class="pill reject">Reject</span><span class="ld"><b>Do not adopt</b><span>Explain why in the comment.</span></span></div>
  </div>
  <div class="toolbar">
    <span class="tlbl">Reading tools</span>
    <button class="btn sm" id="btn-expand" title="Open every value definition">Expand all<span class="kbd">e</span></button>
    <button class="btn sm" id="btn-collapse" title="Close every value definition">Collapse all<span class="kbd">c</span></button>
    <button class="btn sm" id="btn-filter" aria-pressed="false" title="Show only the sections marked Proposed">Show only proposed<span class="kbd">p</span></button>
    <button class="btn sm" id="btn-next" title="Scroll to the next section without a verdict">Next unreviewed<span class="kbd">n</span></button>
  </div>
</div>'''

    overall = f'''<div id="overall" class="overall">
  <h2>Overall assessment</h2>
  <p class="intro">Identify yourself, then give an overall verdict on this scheme. Per-section feedback is collected in each card below and included in the export.</p>
  <div class="reviewer">
    <div><label for="rev-name">Reviewer name</label><input id="rev-name" type="text" placeholder="e.g. G. Crombez"></div>
    <div><label for="rev-role">Role or expertise</label><input id="rev-role" type="text" placeholder="e.g. pain psychology, methodology"></div>
  </div>
  {fb_widget("overall", "Overall assessment of Scheme " + str(num))}
  <div class="endbar">
    <button class="btn primary" id="btn-export2" title="Download all your feedback for this scheme as one JSON file" onclick="document.getElementById('btn-export').click()">&#8595;&nbsp;Export my feedback (JSON)</button>
    <span style="font-size:12.5px;color:var(--muted)"><b>Final step.</b> Downloads one file with every verdict and comment on this page. Send it to the review team, or drop it into the console on the home page to consolidate.</span>
    <span class="spacer" style="flex:1"></span>
    <button class="btn ghost sm" id="btn-reset" title="Clear all saved feedback for this scheme">Clear all feedback</button>
  </div>
</div>'''

    sources_card = f'''<section id="sources-card" class="card"><h2>Canonical Source Paths</h2>
  <p class="intro">Where this scheme is implemented and specified in the repository. Paths reflect the original project layout.</p>
  <ul class="paths">{src_items}</ul>
  <h2 style="margin-top:18px;font-size:16px">Primary Outputs</h2>
  <ul class="paths">{out_items}</ul>
</section>'''

    meta_js = ('<script>window.__SCHEME__=' +
               '{"id":"' + sid + '","title":' + _jsstr(title) + '};</script>')

    doc = f'''<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scheme {num} | {h(title)}</title>
<meta name="description" content="{h(s["subtitle"])}">
<style>{CSS}</style>
</head><body>
{appbar}
<div class="wrap">
  <nav class="toc">{"".join(toc)}</nav>
  <main class="main">
    {hero}
    {howto}
    {overall}
    {sources_card}
    {sections_html}
  </main>
</div>
<div class="foot">
  {h(C.PROJECT["title"])} &middot; OSF {h(C.PROJECT["osf_doi"])} &middot;
  Coding scheme dossier v{h(C.PROJECT["version"])} &middot; {h(C.PROJECT["release_date"])}<br>
  Feedback is stored locally in your browser and only leaves your machine when you export it.
</div>
{meta_js}
<script>{JS_BASE}</script>
</body></html>'''
    return doc


def _jsstr(s):
    """Safe JSON string literal for inline script."""
    import json
    return json.dumps(str(s))


# ==========================================================================
# Aggregated index.html
# ==========================================================================

def pipeline_svg():
    # Logical pipeline order with clickable scheme nodes.
    nodes = [
        ("search", "Search and\ndeduplication", None, "#8a94a6"),
        ("scheme_1", "Scheme 1\nStage 1 screening", "scheme_1/scheme_1.html", "#3b4ce0"),
        ("scheme_2", "Scheme 2\nStage 2 abstract coding", "scheme_2/scheme_2.html", "#3b4ce0"),
        ("scheme_4", "Scheme 4\nStage 3 triage", "scheme_4/scheme_4.html", "#0E8F80"),
        ("scheme_3", "Scheme 3\nStage 3 full-text", "scheme_3/scheme_3.html", "#0E8F80"),
        ("scheme_5", "Scheme 5\nConcept mapping", "scheme_5/scheme_5.html", "#6D5AE0"),
        ("scheme_6", "Scheme 6\nSemantic ontology", "scheme_6/scheme_6.html", "#D98016"),
        ("synth", "Synthesis and\nreporting", None, "#8a94a6"),
    ]
    w, gap, bw, bh, y = 178, 26, 150, 66, 40
    total_w = len(nodes) * bw + (len(nodes) - 1) * gap + 40
    svg = [f'<svg viewBox="0 0 {total_w} 150" width="{total_w}" height="150" xmlns="http://www.w3.org/2000/svg" font-family="var(--sans)">']
    x = 20
    centers = []
    for i, (nid, label, href, color) in enumerate(nodes):
        cx = x + bw / 2
        centers.append((cx, color))
        # connector
        if i > 0:
            px = centers[i - 1][0] + bw / 2
            svg.append(f'<line x1="{px}" y1="{y+bh/2}" x2="{x}" y2="{y+bh/2}" stroke="#cdd3db" stroke-width="2" marker-end="url(#arr)"/>')
        # box
        lines = label.split("\n")
        rect = f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="12" fill="#fff" stroke="{color}" stroke-width="1.6"/>'
        txt = f'<text x="{cx}" y="{y+26}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1c2530">{h(lines[0])}</text>'
        txt += f'<text x="{cx}" y="{y+44}" text-anchor="middle" font-size="11" fill="#5b6675">{h(lines[1]) if len(lines)>1 else ""}</text>'
        group = rect + txt
        if href:
            svg.append(f'<a href="{href}">{group}<title>Open {h(nid)}</title></a>')
        else:
            svg.append(group)
        x += bw + gap
    svg.append('<defs><marker id="arr" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">'
               '<path d="M0,0 L9,4.5 L0,9 z" fill="#cdd3db"/></marker></defs>')
    svg.append('</svg>')
    return "".join(svg)


def index_html():
    proj = C.PROJECT
    cards = []
    for s in C.SCHEMES:
        purpose = ""
        for sec in s["sections"]:
            if sec["kind"] == "prose" and sec["id"] == "purpose":
                purpose = sec["body"][0]
                break
        nfields = sum(len(sec.get("fields", [])) for sec in s["sections"] if sec["kind"] == "fields")
        cards.append(f'''<div class="scard">
  <div class="snum">{s["num"]}</div>
  <span class="badge stage">{h(s["stage"])}</span>
  <h3>{h(s["title"])}</h3>
  <p class="sp">{h(purpose[:220])}{"..." if len(purpose)>220 else ""}</p>
  <div class="chips" style="margin-bottom:10px">
    <span class="chip">{nfields} coded fields</span>
    <span class="chip">{len(s["sources"])} sources</span>
  </div>
  <div class="sactions">
    <a class="btn primary" href="{s["id"]}/{s["id"]}.html">Open and evaluate</a>
    <a class="btn" href="{s["id"]}/{s["id"]}.pdf">PDF</a>
  </div>
</div>''')

    appbar = '''<div class="appbar"><div class="appbar-in">
  <a class="brand" href="index.html"><span class="dot">BPS</span><span>Coding Scheme Dossiers</span></a>
  <div class="spacer"></div>
  <a class="btn" href="README.md">README</a>
  <button class="btn ghost" id="btn-print">Print</button>
</div></div>'''

    reviewers = ", ".join(proj["reviewers"])
    hero = f'''<div id="top" class="hero">
  <div class="kicker">
    <span class="badge num">6 schemes</span>
    <span class="badge stage">Expert evaluation package</span>
    <span class="badge">{h(proj["status"])}</span>
  </div>
  <h1>Coding Scheme Dossiers for Expert Evaluation</h1>
  <p class="subtitle">{h(proj["title"])}</p>
  <p class="tagline">OSF {h(proj["osf_doi"])} &middot; classification of BPS reviews and the categorization pipeline</p>
  <div class="status"><span class="ic">!</span><div><b>Please read before evaluating.</b>
    {h(proj["status_long"])}</div></div>
</div>'''

    stat_strip = f'''<div class="stat-strip">
  <div class="stat"><div class="n">6</div><div class="l">coding schemes</div></div>
  <div class="stat"><div class="n">3</div><div class="l">review stages</div></div>
  <div class="stat"><div class="n">42</div><div class="l">ontology subdomains</div></div>
  <div class="stat"><div class="n">111</div><div class="l">test-run records</div></div>
</div>'''

    scope = f'''<section class="card"><h2>Scope: two reviews, one shared instrument</h2>
  <p class="intro">Why the same schemes serve two reviews.</p>
  <p>{h(proj["review_scope"])}</p>
  <div class="chips" style="margin-top:6px">
    <span class="chip" style="border-left:3px solid var(--bio)">Review A: musculoskeletal chronic pain</span>
    <span class="chip" style="border-left:3px solid var(--soc)">Review B: neuropathic chronic pain</span>
    <span class="chip rq">Varying input: pain-condition family</span>
    <span class="chip">Constant: fields, values, anchors</span>
  </div>
</section>'''

    pipe = f'''<section class="card"><h2>Where each scheme sits in the pipeline</h2>
  <p class="intro">The six schemes form one classification and categorization pipeline. Click a blue, teal, violet, or amber node to open its evaluation surface.</p>
  <div class="pipe-wrap">{pipeline_svg()}</div>
  {stat_strip}
</section>'''

    how = '''<section class="card"><h2>How to evaluate</h2>
  <ul class="plain">
    <li>Open each scheme with <b>Open and evaluate</b>. Read the anchored value definitions and the sections marked <b>Proposed</b>.</li>
    <li>On each section, pick a verdict (Approve, Approve with revisions, Needs discussion, Reject), rate clarity, and add comments.</li>
    <li>Your feedback autosaves in your browser. Use <b>Export feedback</b> to download one JSON file per scheme.</li>
    <li>Return here and load every exported file into the console below to see a consolidated view and export a single bundle.</li>
  </ul></section>'''

    console = '''<section class="card console"><h2>Consolidated feedback console</h2>
  <p class="intro">Load the per-scheme feedback JSON files that reviewers exported. Everything runs locally in your browser; nothing is uploaded.</p>
  <div id="merge-drop" style="border:2px dashed var(--line-strong);border-radius:12px;padding:22px;text-align:center;color:var(--muted);transition:.15s">
    Drag and drop <code>*_feedback.json</code> files here, or
    <button class="btn" id="btn-merge" style="margin-left:6px">Choose files</button>
    <button class="btn" id="btn-merge-export" style="margin-left:6px">Export bundle</button>
    <input type="file" id="merge-files" accept="application/json" multiple hidden>
  </div>
  <div id="merge-out" style="margin-top:14px"></div>
</section>'''

    inv_rows = "".join(
        f'<div class="kvrow"><div class="kk">Scheme {s["num"]}: {h(s["stage"])}</div>'
        f'<div class="vv">{h(s["title"])} &middot; '
        f'<a href="{s["id"]}/{s["id"]}.html">HTML</a> &middot; '
        f'<a href="{s["id"]}/{s["id"]}.pdf">PDF</a> &middot; '
        f'<a href="{s["id"]}/README.md">README</a></div></div>'
        for s in C.SCHEMES
    )
    inventory = f'''<section class="card"><h2>Inventory</h2>
  <div class="kvlist">{inv_rows}</div></section>'''

    doc = f'''<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Coding Scheme Dossiers | {h(proj["short"])}</title>
<style>{CSS}</style>
</head><body>
{appbar}
<div class="wrap" style="grid-template-columns:1fr">
  <main class="main">
    {hero}
    {scope}
    {pipe}
    <section class="card"><h2>The six coding schemes</h2>
      <p class="intro">Each card opens a dedicated, self-contained evaluation surface.</p>
      <div class="scheme-cards">{"".join(cards)}</div>
    </section>
    {how}
    {console}
    {inventory}
  </main>
</div>
<div class="foot">
  {h(proj["title"])} &middot; OSF {h(proj["osf_doi"])} &middot; v{h(proj["version"])} &middot; {h(proj["release_date"])}<br>
  Lead: {h(proj["lead"])} &middot; Reviewers: {h(reviewers)}
</div>
<script>{JS_BASE}</script>
<script>{JS_INDEX}</script>
</body></html>'''
    return doc


# ==========================================================================
# LaTeX rendering
# ==========================================================================

def latex_values(values, indent="  "):
    if not values:
        return ""
    lines = [indent + r"\begin{itemize}[leftmargin=1.3em,itemsep=2pt,topsep=2pt]"]
    for val in values:
        parts = [r"\textbf{" + tt(val["value"]) + r"} " + lx(val["anchor"])]
        detail = []
        if val["pos"]:
            detail.append(r"\textit{Choose when:} " + lx("; ".join(val["pos"])) + ".")
        if val["neg"]:
            detail.append(r"\textit{Not this if:} " + lx("; ".join(val["neg"])) + ".")
        if val["boundary"]:
            detail.append(r"\textit{Boundary:} " + lx(val["boundary"]))
        body = " ".join(parts)
        if detail:
            body += r" {\small " + " ".join(detail) + "}"
        lines.append(indent + r"\item " + body)
    lines.append(indent + r"\end{itemize}")
    return "\n".join(lines)


def latex_section(sec):
    kind = sec["kind"]
    title = sec["title"]
    out = []
    if sec.get("enhancement"):
        t = title
        if ":" in t:
            t = "Proposed Refinement: " + t.split(":", 1)[-1].strip()
        out.append(r"\section*{" + lx(t) + "}")
        out.append(r"\refbadge")
    else:
        out.append(r"\section*{" + lx(title) + "}")
    if sec.get("intro"):
        out.append(r"{\small\itshape " + lx(sec["intro"]) + r"}\par\smallskip")

    if kind == "prose":
        for para in sec["body"]:
            out.append(lx(para))
            out.append("")
    elif kind == "list":
        out.append(r"\begin{itemize}")
        for it in sec["items"]:
            out.append(r"\item " + lx(it))
        out.append(r"\end{itemize}")
    elif kind == "keyvals":
        out.append(r"\begin{description}")
        for k, v in sec["items"]:
            out.append(r"\item[" + lx(k) + r"] " + lx(v))
        out.append(r"\end{description}")
    elif kind == "fields":
        out.append(r"\begin{description}")
        for f in sec["fields"]:
            head = r"\item[" + tt(f["name"]) + r"] " + lx(f["construct"])
            if f["free_text"]:
                head += r" \textit{(free text)}"
            out.append(head)
            if f["values"]:
                out.append(latex_values(f["values"]))
            if f["notes"]:
                out.append(r"{\small " + lx(f["notes"]) + "}")
        out.append(r"\end{description}")
    elif kind == "taxonomy":
        out.append(r"\begin{description}")
        for fam in sec["families"]:
            out.append(r"\item[" + lx(fam["family"]) + r"] " + lx(fam["definition"]))
            out.append(r"{\small \textit{Ontology subdomain:} " + lx(fam["subdomain"])
                       + r". \textit{Constructs:} " + lx(", ".join(fam["members"]))
                       + r". \textit{Frameworks:} " + lx(", ".join(fam["frameworks"])) + ".}")
        out.append(r"\end{description}")
    elif kind == "examples":
        out.append(r"\begin{description}")
        for ex in sec["examples"]:
            coding = "; ".join(tt(k) + " = " + lx(v) for k, v in ex["coding"])
            out.append(r"\item[" + lx(ex["record"]) + r"] {\small \textit{" + lx(ex["meta"]) + r"}}\\")
            out.append(r"{\small Coding: " + coding + r".\\ \textit{Why:} " + lx(ex["rationale"]) + "}")
        out.append(r"\end{description}")
    out.append("")
    return "\n".join(out)


def scheme_tex(s):
    lines = [
        r"\documentclass[11pt]{article}",
        r"\input{../common_preamble.tex}",
        "",
        r"\begin{document}",
        r"\schemeheader",
        "  {Scheme " + str(s["num"]) + "}",
        "  {" + lx(s["title"]) + "}",
        "  {" + lx(s["subtitle"]) + "}",
        "",
        r"\statusnote{" + lx(C.PROJECT["status"]) + "}{" + lx(C.PROJECT["status_long"]) + "}",
        "",
        r"\section*{At a Glance}",
        r"\begin{description}",
    ]
    for k, v in s["meta"].items():
        lines.append(r"\item[" + lx(k) + r"] " + lx(v))
    lines.append(r"\item[Research questions] " + lx("; ".join(s["rqs"])))
    lines.append(r"\item[Tagline] " + lx(s["tagline"]))
    lines.append(r"\end{description}")
    lines.append("")

    lines.append(r"\section*{Canonical Source Paths}")
    lines.append(r"\begin{itemize}")
    for p in s["sources"]:
        lines.append(r"\item " + fp(p))
    lines.append(r"\end{itemize}")
    lines.append("")

    for sec in s["sections"]:
        lines.append(latex_section(sec))

    lines.append(r"\section*{Primary Outputs Using This Scheme}")
    lines.append(r"\begin{itemize}")
    for p in s["outputs"]:
        lines.append(r"\item " + (fp(p) if "/" in p else lx(p)))
    lines.append(r"\end{itemize}")
    lines.append("")
    lines.append(r"\end{document}")
    lines.append("")
    return "\n".join(lines)


PREAMBLE = r"""\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{array}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{xcolor}
\usepackage{hyperref}
\definecolor{bpsink}{HTML}{1C2530}
\definecolor{bpsaccent}{HTML}{2534B8}
\definecolor{bpsmuted}{HTML}{5B6675}
\definecolor{bpsenh}{HTML}{7A4BD0}
\definecolor{bpsline}{HTML}{CDD3DB}
\definecolor{bpssoft}{HTML}{F0F2F5}
\definecolor{bpswarnbg}{HTML}{FDF6E6}
\definecolor{bpswarnink}{HTML}{7A5A00}
\hypersetup{
  colorlinks=true,
  linkcolor=bpsaccent,
  urlcolor=bpsaccent,
  pdftitle={BPS Coding Scheme Dossier}
}
\setlist[itemize]{leftmargin=1.5em,itemsep=2pt,topsep=3pt}
\setlist[description]{style=nextline,leftmargin=0em,labelsep=0.6em,itemsep=4pt}
\titleformat{\section}{\color{bpsink}\large\bfseries}{}{0em}{}[{\color{bpsline}\titlerule}]
\titlespacing*{\section}{0pt}{1.1em}{0.5em}
\newcommand{\field}[1]{\nolinkurl{#1}}
\newcommand{\filepath}[1]{{\small\ttfamily\color{bpsmuted}\nolinkurl{#1}}}
\newcommand{\refbadge}{{\small\colorbox{bpsenh}{\textcolor{white}{\bfseries\ PROPOSED REFINEMENT \ }}\quad\itshape\color{bpsenh}Awaiting expert evaluation. Not yet applied to the pipeline.\par\smallskip}}
\newcommand{\schemeheader}[3]{%
  \begin{center}
    {\LARGE \bfseries\color{bpsink} #1}\\[0.45em]
    {\large #2}\\[0.35em]
    {\normalsize \itshape\color{bpsmuted} #3}
  \end{center}
  \vspace{0.4em}\hrule\vspace{0.9em}
}
\newcommand{\statusnote}[2]{%
  \begin{center}
  \fcolorbox{bpsline}{bpswarnbg}{%
    \parbox{0.94\linewidth}{\small\color{bpswarnink}\textbf{#1.}\ #2}%
  }
  \end{center}
  \vspace{0.6em}
}
"""


# ==========================================================================
# README rendering
# ==========================================================================

def scheme_readme(s):
    L = []
    L.append(f"# Scheme {s['num']}: {s['title']}")
    L.append("")
    L.append(f"> **Status: {C.PROJECT['status']}.** {C.PROJECT['status_long']}")
    L.append("")
    L.append(f"*{s['subtitle']}*")
    L.append("")
    L.append(f"{s['tagline']}.")
    L.append("")
    L.append("## What this scheme does")
    L.append("")
    for sec in s["sections"]:
        if sec["kind"] == "prose" and sec["id"] == "purpose":
            for para in sec["body"]:
                L.append(para)
                L.append("")
            break
    L.append("## At a glance")
    L.append("")
    L.append("| Property | Value |")
    L.append("| --- | --- |")
    for k, v in s["meta"].items():
        L.append(f"| {k} | {v} |")
    L.append(f"| Research questions | {'; '.join(s['rqs'])} |")
    L.append("")
    L.append("## Files in this folder")
    L.append("")
    L.append(f"- [`{s['id']}.html`]({s['id']}.html) is the interactive evaluation surface. Open it in a browser, record a verdict and comments per section, then export your feedback as JSON.")
    L.append(f"- [`{s['id']}.pdf`]({s['id']}.pdf) is the formal dossier for sharing and printing.")
    L.append(f"- [`{s['id']}.tex`]({s['id']}.tex) is the LaTeX source (generated from `_build/content.py`).")
    L.append("")
    # Proposed refinements summary
    enh = [sec for sec in s["sections"] if sec.get("enhancement")]
    if enh:
        L.append("## Proposed refinements awaiting expert sign-off")
        L.append("")
        L.append("These are the enhancements that raise semantic resolution. They are proposals only and are not yet applied to the pipeline:")
        L.append("")
        for sec in enh:
            first = sec["body"][0] if sec.get("body") else sec.get("intro", "")
            # strip the standard lead sentence for brevity
            first = first.replace("Refinement proposed for expert evaluation. ", "")
            L.append(f"- **{sec['title'].split(':',1)[-1].strip()}.** {first}")
        L.append("")
    L.append("## Coded fields")
    L.append("")
    any_fields = False
    for sec in s["sections"]:
        if sec["kind"] == "fields":
            any_fields = True
            L.append(f"### {sec['title']}")
            L.append("")
            for f in sec["fields"]:
                vals = ", ".join(v["value"] for v in f["values"]) if f["values"] else ("free text" if f["free_text"] else "")
                if vals:
                    L.append(f"- `{f['name']}` ({vals}): {f['construct']}")
                else:
                    L.append(f"- `{f['name']}`: {f['construct']}")
            L.append("")
    if not any_fields:
        L.append("This scheme is specified through its prompts, seeds, and ontology rather than a single coded-field table. See the HTML or PDF for the full specification.")
        L.append("")
    tax = [sec for sec in s["sections"] if sec["kind"] == "taxonomy"]
    for sec in tax:
        L.append(f"## {sec['title']} (proposed)")
        L.append("")
        L.append("Each family aligns one to one with a Scheme 6 psychological subdomain.")
        L.append("")
        for fam in sec["families"]:
            L.append(f"- **{fam['family']}** (subdomain: {fam['subdomain']}). {fam['definition']} "
                     f"Constructs: {', '.join(fam['members'])}. Frameworks: {', '.join(fam['frameworks'])}.")
        L.append("")
    L.append("## Canonical source paths")
    L.append("")
    for p in s["sources"]:
        L.append(f"- `{p}`")
    L.append("")
    L.append("## Regenerating this dossier")
    L.append("")
    L.append("All three surfaces (PDF, HTML, README) are generated from one source of truth:")
    L.append("")
    L.append("```bash")
    L.append("cd src/coding_schemes/_build")
    L.append("python3 build.py")
    L.append("```")
    L.append("")
    L.append(f"Edit the scheme content in `_build/content.py`, not the generated files.")
    L.append("")
    return "\n".join(L)


def index_readme():
    proj = C.PROJECT
    L = []
    L.append("# Coding Scheme Dossiers")
    L.append("")
    L.append(f"> **Status: {proj['status']}.** {proj['status_long']}")
    L.append("")
    L.append("This directory contains one communication-ready dossier for each distinct coding scheme in the systematic review workflow. Every scheme is provided in three synchronized surfaces:")
    L.append("")
    L.append("- an **interactive HTML** evaluation surface with per-section feedback boxes and JSON export or import,")
    L.append("- a compiled **PDF** for sharing and printing,")
    L.append("- an explanatory **README**.")
    L.append("")
    L.append("Open [`index.html`](index.html) for the aggregated dashboard: a pipeline map, links to every scheme, and a console that merges exported feedback files into one consolidated view.")
    L.append("")
    L.append("## Why these schemes are circulated now")
    L.append("")
    L.append("The current manuscript is a **test run** (it exercised an earlier, coarser generation of these schemes in the Python workflow with an LLM, gemini-2.5-flash). This release raises their semantic quality and resolution (operational anchors for every value, positive and negative indicators, explicit boundary rules, worked examples from the real corpus, a comprehensive psychological concept taxonomy, and clearly labelled proposed refinements). Nothing here has been applied to a final corpus yet. The schemes are being circulated for expert evaluation first; the pipeline will be re-run only after sign-off.")
    L.append("")
    L.append("## Scope: two reviews, one shared instrument")
    L.append("")
    L.append(proj["review_scope"])
    L.append("")
    L.append("## Inventory")
    L.append("")
    for s in C.SCHEMES:
        L.append(f"### Scheme {s['num']}: {s['title']}")
        L.append("")
        purpose = ""
        for sec in s["sections"]:
            if sec["kind"] == "prose" and sec["id"] == "purpose":
                purpose = sec["body"][0]
                break
        L.append(f"- **Stage:** {s['stage']}")
        L.append(f"- **Purpose:** {purpose}")
        L.append(f"- **Files:** [`{s['id']}/{s['id']}.html`]({s['id']}/{s['id']}.html), [`{s['id']}/{s['id']}.pdf`]({s['id']}/{s['id']}.pdf), [`{s['id']}/README.md`]({s['id']}/README.md)")
        L.append("")
    L.append("## Evaluation workflow")
    L.append("")
    L.append("1. Open [`index.html`](index.html) and read the status note.")
    L.append("2. Open each scheme, read the anchored definitions and the sections marked **Proposed**, and record a verdict and comments per section.")
    L.append("3. Export one JSON file per scheme (the button is in the top bar).")
    L.append("4. Load every exported file into the console on `index.html` to see a consolidated view and export a single bundle for the team.")
    L.append("")
    L.append("## Regeneration")
    L.append("")
    L.append("All surfaces are generated from `_build/content.py` by `_build/build.py`. Do not hand-edit the generated `.tex`, `.html`, or `README.md` files; edit the content model and rebuild:")
    L.append("")
    L.append("```bash")
    L.append("cd src/coding_schemes/_build")
    L.append("python3 build.py")
    L.append("```")
    L.append("")
    L.append("## Notes")
    L.append("")
    L.append("- The dossiers prioritize the operational implementation used by the pipeline. Where protocol prose, codebooks, and generated outputs diverge, the dossiers state that explicitly.")
    L.append("- The underlying source files and outputs remain in their original project paths so reviewers can inspect the raw materials directly.")
    L.append("- No em dashes are used in any generated file.")
    L.append("")
    return "\n".join(L)


def build_readme():
    L = []
    L.append("# Coding scheme build system")
    L.append("")
    L.append("This folder holds the single source of truth for all six coding-scheme dossiers and the generator that renders them.")
    L.append("")
    L.append("- `content.py` is the structured, enriched specification of every scheme. **Edit content here.**")
    L.append("- `assets.py` holds the shared CSS and JavaScript for the HTML surfaces.")
    L.append("- `build.py` renders each scheme to LaTeX (compiled to PDF), interactive HTML, and README, plus the aggregated `index.html` and the directory `README.md`.")
    L.append("")
    L.append("## Build")
    L.append("")
    L.append("```bash")
    L.append("python3 build.py            # render everything and compile PDFs with tectonic")
    L.append("python3 build.py --no-pdf   # render text surfaces only")
    L.append("```")
    L.append("")
    L.append("PDF compilation uses `tectonic`. If it is not installed, run with `--no-pdf` and compile the `.tex` files with any LaTeX engine.")
    L.append("")
    return "\n".join(L)


# ==========================================================================
# Orchestration
# ==========================================================================

def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def check_no_emdash(paths):
    emdash = chr(0x2014)  # em dash, referenced by code point to avoid a literal
    bad = []
    for p in paths:
        try:
            with open(p, encoding="utf-8") as fh:
                if emdash in fh.read():
                    bad.append(p)
        except Exception:
            pass
    return bad


def main():
    no_pdf = "--no-pdf" in sys.argv
    written = []

    # shared preamble
    written.append(write(os.path.join(ROOT, "common_preamble.tex"), PREAMBLE))

    for s in C.SCHEMES:
        d = os.path.join(ROOT, s["id"])
        written.append(write(os.path.join(d, s["id"] + ".tex"), scheme_tex(s)))
        written.append(write(os.path.join(d, s["id"] + ".html"), scheme_html(s)))
        written.append(write(os.path.join(d, "README.md"), scheme_readme(s)))

    written.append(write(os.path.join(ROOT, "index.html"), index_html()))
    written.append(write(os.path.join(ROOT, "README.md"), index_readme()))
    written.append(write(os.path.join(HERE, "README.md"), build_readme()))

    print("Wrote %d files." % len(written))
    for p in written:
        print("  " + os.path.relpath(p, ROOT))

    bad = check_no_emdash(written)
    if bad:
        print("\nWARNING: em dash found in:")
        for p in bad:
            print("  " + p)
    else:
        print("\nEm-dash check passed (none found).")

    if no_pdf:
        print("\nSkipping PDF compilation (--no-pdf).")
        return

    print("\nCompiling PDFs with tectonic...")
    for s in C.SCHEMES:
        d = os.path.join(ROOT, s["id"])
        tex = s["id"] + ".tex"
        try:
            r = subprocess.run(
                ["tectonic", "--chatter", "minimal", tex],
                cwd=d, capture_output=True, text=True,
            )
            ok = (r.returncode == 0) and os.path.exists(os.path.join(d, s["id"] + ".pdf"))
            print(("  OK   " if ok else "  FAIL ") + s["id"] + ".pdf")
            if not ok:
                sys.stderr.write(r.stdout[-1500:] + "\n" + r.stderr[-1500:] + "\n")
        except FileNotFoundError:
            print("  tectonic not found; skip " + s["id"])


if __name__ == "__main__":
    main()
