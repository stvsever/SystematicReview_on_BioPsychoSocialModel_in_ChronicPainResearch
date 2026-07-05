# -*- coding: utf-8 -*-
"""
Static CSS and JS for the coding-scheme HTML surfaces (laptop and desktop first).

Kept as plain (non f-string) triple-quoted strings so that the many CSS and JS
braces are treated literally. build.py concatenates these into each document.
No em dashes are used anywhere.
"""

CSS = r"""
:root{
  --bg:#f5f6f8; --panel:#ffffff; --ink:#1c2530; --muted:#5b6675;
  --line:#e4e7ec; --line-strong:#cdd3db; --soft:#f0f2f5;
  --accent:#3b4ce0; --accent-ink:#2534b8;
  --bio:#0E8F80; --psy:#6D5AE0; --soc:#D98016;
  --ok:#12855a; --ok-bg:#e6f5ee; --revise:#9a6a00; --revise-bg:#fbf1dc;
  --discuss:#1f5fae; --discuss-bg:#e6f0fb; --reject:#b23b41; --reject-bg:#fbe9ea;
  --enh:#7a4bd0; --enh-bg:#f2ecfc; --enh-line:#d9c8f5;
  --shadow:0 1px 2px rgba(16,24,40,.06),0 8px 24px rgba(16,24,40,.06);
  --radius:14px; --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  font-size:16px;line-height:1.62;-webkit-font-smoothing:antialiased}
a{color:var(--accent-ink);text-decoration:none}
a:hover{text-decoration:underline}
code,.mono{font-family:var(--mono);font-size:.9em}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:6px}

/* ---- top action bar ---- */
.appbar{position:sticky;top:0;z-index:40;background:rgba(255,255,255,.93);
  backdrop-filter:saturate(140%) blur(8px);border-bottom:1px solid var(--line)}
.appbar-in{max-width:1240px;margin:0 auto;display:flex;align-items:center;
  gap:12px;padding:9px 22px;flex-wrap:wrap}
.appbar .brand{display:flex;align-items:center;gap:10px;font-weight:700;font-size:15px}
.appbar .brand .dot{width:26px;height:26px;border-radius:8px;display:grid;
  place-items:center;color:#fff;font-size:12px;font-weight:800;
  background:linear-gradient(135deg,var(--bio),var(--psy) 55%,var(--soc))}
.appbar .spacer{flex:1}
.sep{width:1px;height:22px;background:var(--line);margin:0 2px}
.btn{border:1px solid var(--line-strong);background:#fff;color:var(--ink);
  border-radius:10px;padding:7px 12px;font-size:13.5px;font-weight:600;cursor:pointer;
  display:inline-flex;align-items:center;gap:7px;transition:.15s;white-space:nowrap}
.btn:hover{border-color:var(--accent);color:var(--accent-ink);background:#fbfbff}
.btn.primary{background:var(--accent);border-color:var(--accent);color:#fff}
.btn.primary:hover{background:var(--accent-ink);color:#fff}
.btn.ghost{border-color:transparent;background:transparent}
.btn.sm{padding:5px 9px;font-size:12.5px}
.btn .ic{font-size:14px;line-height:1}
.savechip{font-size:12px;font-weight:700;color:var(--muted);padding:3px 9px;border-radius:99px;
  background:var(--soft);border:1px solid var(--line);transition:.2s;white-space:nowrap}
.savechip.flash{background:var(--ok-bg);color:var(--ok);border-color:#bfe6d2}
.progress-mini{display:flex;align-items:center;gap:8px;font-size:12.5px;color:var(--muted);cursor:pointer}
.progress-mini .bar{width:120px;height:7px;border-radius:99px;background:var(--soft);overflow:hidden}
.progress-mini .fill{height:100%;width:0;border-radius:99px;
  background:linear-gradient(90deg,var(--bio),var(--psy),var(--soc));transition:width .3s}

/* ---- layout ---- */
.wrap{max-width:1240px;margin:0 auto;padding:24px 22px 100px;
  display:grid;grid-template-columns:266px 1fr;gap:36px}
.toc{position:sticky;top:70px;align-self:start;max-height:calc(100vh - 92px);
  overflow:auto;font-size:13.5px;padding-right:6px}
.toc h4{margin:0 0 10px;font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}
.toc a{display:flex;align-items:center;gap:8px;color:var(--muted);padding:5px 10px;border-radius:8px;
  border-left:2px solid transparent;margin:1px 0}
.toc a:hover{background:var(--soft);text-decoration:none;color:var(--ink)}
.toc a.active{color:var(--accent-ink);background:#eef0fe;border-left-color:var(--accent);font-weight:600}
.toc a .tlabel{flex:1}
.toc a .tdot{width:9px;height:9px;border-radius:99px;border:1.5px solid var(--line-strong);flex:none}
.toc a .tdot[data-v]{border:none}
.toc a .tdot[data-v="approve"]{background:var(--ok)}
.toc a .tdot[data-v="revise"]{background:var(--revise)}
.toc a .tdot[data-v="discuss"]{background:var(--discuss)}
.toc a .tdot[data-v="reject"]{background:var(--reject)}
.main{min-width:0}

/* ---- hero ---- */
.hero{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);
  padding:26px 30px;box-shadow:var(--shadow);position:relative;overflow:hidden}
.hero::before{content:"";position:absolute;inset:0 0 auto 0;height:5px;
  background:linear-gradient(90deg,var(--bio),var(--psy) 55%,var(--soc))}
.hero .kicker{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px}
.badge{display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:700;
  padding:4px 10px;border-radius:99px;background:var(--soft);color:var(--muted);border:1px solid var(--line)}
.badge.num{background:var(--ink);color:#fff;border-color:var(--ink)}
.badge.stage{background:#eef0fe;color:var(--accent-ink);border-color:#d7dcff}
.hero h1{margin:.1em 0 .15em;font-size:27px;line-height:1.2;letter-spacing:-.01em}
.hero .subtitle{color:var(--ink);font-size:16.5px;margin:0 0 4px}
.hero .tagline{font-size:14px;color:var(--muted);font-style:italic}

/* ---- status banner ---- */
.status{margin-top:16px;border:1px solid #f0d9a6;background:#fdf6e6;border-radius:12px;
  padding:14px 16px;display:flex;gap:12px;align-items:flex-start;font-size:14px}
.status .ic{flex:none;width:26px;height:26px;border-radius:8px;background:#f2b705;color:#3a2b00;
  display:grid;place-items:center;font-weight:800}
.status b{color:#7a5a00}

/* ---- meta grid ---- */
.metagrid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-top:18px}
.metagrid .cell{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:13px 15px}
.metagrid .k{font-size:11px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);font-weight:700;margin-bottom:4px}
.metagrid .val{font-size:14px}
.eyebrow{font-size:11px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);font-weight:700;margin-bottom:6px}
.chips{display:flex;flex-wrap:wrap;gap:7px;margin-top:6px}
.chip{font-size:12.5px;padding:4px 10px;border-radius:99px;background:var(--soft);
  border:1px solid var(--line);color:var(--ink)}
.chip.rq{background:#eef0fe;border-color:#d7dcff;color:var(--accent-ink)}

/* ---- how-to / legend / toolbar ---- */
.howto{background:linear-gradient(180deg,#f7f8ff,#fff 90px);border:1px solid #dfe3fb;
  border-radius:var(--radius);padding:18px 22px;margin-top:20px;box-shadow:var(--shadow)}
.howto h2{margin:0 0 4px;font-size:16px}
.howto .intro{color:var(--muted);font-size:13.5px;margin:.2em 0 .9em}
.legend{display:grid;grid-template-columns:repeat(2,1fr);gap:9px}
.legend .lrow{display:flex;gap:10px;align-items:flex-start;font-size:13px;border:1px solid var(--line);
  border-radius:10px;padding:8px 11px;background:#fff}
.legend .pill{flex:none;font-size:11px;font-weight:800;padding:3px 9px;border-radius:99px;margin-top:1px}
.legend .pill.approve{background:var(--ok-bg);color:var(--ok)}
.legend .pill.revise{background:var(--revise-bg);color:var(--revise)}
.legend .pill.discuss{background:var(--discuss-bg);color:var(--discuss)}
.legend .pill.reject{background:var(--reject-bg);color:var(--reject)}
.legend .ld b{display:block;font-size:12.5px}
.legend .ld span{color:var(--muted);font-size:12.5px}
.toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-top:14px;
  border-top:1px dashed var(--line);padding-top:13px}
.toolbar .tlbl{font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-right:2px}
.kbd{font-family:var(--mono);font-size:10.5px;background:var(--soft);border:1px solid var(--line-strong);
  border-bottom-width:2px;border-radius:5px;padding:0 5px;color:var(--muted);margin-left:5px}

/* ---- section cards ---- */
section.card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);
  padding:22px 28px;margin-top:20px;box-shadow:var(--shadow);scroll-margin-top:76px}
section.card>h2{margin:0 0 6px;font-size:19px;letter-spacing:-.01em;display:flex;align-items:center;gap:10px}
.card .intro{color:var(--muted);margin:.2em 0 1em;font-size:14.5px}
.card p{margin:.55em 0}
.card ul.plain{margin:.4em 0;padding-left:1.2em}
.card ul.plain li{margin:.3em 0}

/* enhancement / proposed refinement */
section.card.enh{border-color:var(--enh-line);background:linear-gradient(180deg,var(--enh-bg),#fff 120px)}
section.card.enh>h2 .tag{font-size:11px;font-weight:800;color:#fff;background:var(--enh);
  padding:3px 9px;border-radius:99px;letter-spacing:.03em;text-transform:uppercase}
body.only-proposed section.card:not(.enh){display:none}
body.only-proposed #sources-card{display:none}

/* fields */
.field{border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:12px 0;background:#fdfdfe}
.field>.fname{font-family:var(--mono);font-size:13.5px;font-weight:700;color:var(--ink);
  background:var(--soft);border:1px solid var(--line);border-radius:7px;padding:2px 8px;display:inline-block}
.field>.fconstruct{margin:9px 0 4px;font-size:14.5px}
.field .fnotes{font-size:13px;color:var(--muted);margin-top:8px;border-top:1px dashed var(--line);padding-top:8px}
.ladder{margin-top:10px;display:grid;gap:8px}
.val{border:1px solid var(--line);border-radius:10px;overflow:hidden}
.val>.vhead{display:flex;gap:9px;align-items:center;padding:8px 11px;background:var(--soft)}
.val.has-body>.vhead{cursor:pointer}
.val>.vhead .vv{font-family:var(--mono);font-size:12.5px;font-weight:700;color:var(--accent-ink)}
.val>.vhead .va{font-size:13.5px;color:var(--ink);flex:1}
.val>.vhead .caret{color:var(--muted);transition:.2s;font-size:12px}
.val.open>.vhead .caret{transform:rotate(90deg)}
.val>.vbody{display:none;padding:0 12px 12px;font-size:13px}
.val.open>.vbody{display:block}
.indi{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:9px}
.indi .box{border-radius:8px;padding:8px 10px}
.indi .pos{background:var(--ok-bg)}
.indi .neg{background:var(--reject-bg)}
.indi .box .lab{font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;font-weight:800;margin-bottom:4px}
.indi .pos .lab{color:var(--ok)}
.indi .neg .lab{color:var(--reject)}
.indi ul{margin:0;padding-left:1.05em}
.indi li{margin:2px 0}
.boundary{margin-top:9px;font-size:12.5px;background:#eef0fe;border:1px solid #d7dcff;
  border-radius:8px;padding:7px 10px}
.boundary b{color:var(--accent-ink)}

/* examples */
.example{border:1px solid var(--line);border-left:3px solid var(--psy);border-radius:10px;
  padding:13px 15px;margin:11px 0;background:#fdfdfe}
.example .exrec{font-weight:600;font-size:14.5px}
.example .exmeta{font-size:12.5px;color:var(--muted);margin:2px 0 8px}
.example .excode{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px}
.example .excode .kv{font-size:12px;font-family:var(--mono);background:var(--soft);
  border:1px solid var(--line);border-radius:7px;padding:2px 8px}
.example .excode .kv b{color:var(--accent-ink)}
.example .exwhy{font-size:13.5px;color:var(--ink)}
.example .exwhy .lab{font-weight:700;color:var(--muted);font-size:11px;letter-spacing:.05em;text-transform:uppercase;margin-right:6px}

/* key value list */
.kvlist{display:grid;gap:8px}
.kvrow{display:grid;grid-template-columns:230px 1fr;gap:12px;padding:9px 12px;border:1px solid var(--line);border-radius:10px;background:#fdfdfe}
.kvrow .kk{font-weight:700;font-size:13.5px}
.kvrow .vv{font-size:14px}

/* subdomain lists (scheme 6) */
.subgrid{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-top:6px}
.subgrid .item{font-size:13.5px;padding:8px 11px;border:1px solid var(--line);border-radius:9px;background:#fdfdfe;
  border-left:3px solid var(--dm,#bbb)}
.paths{list-style:none;margin:0;padding:0;display:grid;gap:6px}
.paths li{font-family:var(--mono);font-size:12.5px;color:var(--muted);background:var(--soft);
  border:1px solid var(--line);border-radius:8px;padding:6px 10px;word-break:break-all}

/* ---- feedback widget ---- */
.fb{margin-top:16px;border:1px solid var(--line-strong);border-left:3px solid var(--accent);
  border-radius:12px;padding:13px 16px;background:#fbfbfe}
.fb-head{display:flex;align-items:center;gap:10px;margin-bottom:9px}
.fb-title{font-size:11px;font-weight:800;letter-spacing:.07em;text-transform:uppercase;color:var(--accent-ink)}
.fb-status{font-size:12px;font-weight:700;margin-left:auto;padding:2px 9px;border-radius:99px;background:var(--soft);color:var(--muted)}
.fb-status[data-v="approve"]{background:var(--ok-bg);color:var(--ok)}
.fb-status[data-v="revise"]{background:var(--revise-bg);color:var(--revise)}
.fb-status[data-v="discuss"]{background:var(--discuss-bg);color:var(--discuss)}
.fb-status[data-v="reject"]{background:var(--reject-bg);color:var(--reject)}
.fb-verdicts{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:9px}
.fb-chip{border:1px solid var(--line-strong);background:#fff;border-radius:99px;padding:5px 12px;
  font-size:12.5px;font-weight:600;cursor:pointer;transition:.12s;color:var(--muted)}
.fb-chip:hover{border-color:var(--accent)}
.fb-chip[aria-pressed="true"][data-verdict="approve"]{background:var(--ok);border-color:var(--ok);color:#fff}
.fb-chip[aria-pressed="true"][data-verdict="revise"]{background:var(--revise);border-color:var(--revise);color:#fff}
.fb-chip[aria-pressed="true"][data-verdict="discuss"]{background:var(--discuss);border-color:var(--discuss);color:#fff}
.fb-chip[aria-pressed="true"][data-verdict="reject"]{background:var(--reject);border-color:var(--reject);color:#fff}
.fb-clarity{display:flex;align-items:center;gap:8px;font-size:12.5px;color:var(--muted);margin-bottom:9px}
.stars{display:inline-flex;gap:3px}
.star{cursor:pointer;font-size:18px;line-height:1;color:#d3d7de;user-select:none}
.star.on{color:#f2b705}
.fb-note{width:100%;min-height:64px;border:1px solid var(--line-strong);border-radius:10px;
  padding:10px 12px;font-family:inherit;font-size:14px;resize:vertical;background:#fff;color:var(--ink)}
.fb-note:focus{outline:2px solid #d7dcff;border-color:var(--accent)}

/* overall panel */
.overall{background:linear-gradient(180deg,#fff, #fbfbff);border:1px solid var(--line);border-radius:var(--radius);
  padding:20px 26px;margin-top:20px;box-shadow:var(--shadow)}
.overall h2{margin:0 0 4px;font-size:18px}
.reviewer{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:12px 0}
.reviewer label{display:block;font-size:12px;font-weight:700;color:var(--muted);margin-bottom:4px;text-transform:uppercase;letter-spacing:.05em}
.reviewer input{width:100%;border:1px solid var(--line-strong);border-radius:9px;padding:8px 11px;font-size:14px;font-family:inherit}
.overall .endbar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-top:14px;border-top:1px solid var(--line);padding-top:14px}

/* footer */
.foot{max-width:1240px;margin:0 auto;padding:20px 22px;color:var(--muted);font-size:12.5px;text-align:center}
.foot a{color:var(--muted);text-decoration:underline}

/* toast */
.toast{position:fixed;bottom:22px;left:50%;transform:translateX(-50%) translateY(20px);
  background:var(--ink);color:#fff;padding:11px 18px;border-radius:11px;font-size:13.5px;font-weight:600;
  box-shadow:var(--shadow);opacity:0;pointer-events:none;transition:.25s;z-index:60}
.toast.show{opacity:1;transform:translateX(-50%) translateY(0)}

/* index specifics */
.scheme-cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px;margin-top:8px}
.scard{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:18px 20px;
  box-shadow:var(--shadow);display:flex;flex-direction:column;transition:.18s;position:relative;overflow:hidden}
.scard:hover{transform:translateY(-2px);border-color:var(--line-strong)}
.scard .snum{position:absolute;top:-10px;right:-6px;font-size:76px;font-weight:800;color:var(--soft);z-index:0;line-height:1}
.scard>*{position:relative;z-index:1}
.scard h3{margin:.3em 0 .1em;font-size:16px}
.scard .sp{font-size:13.5px;color:var(--muted);flex:1;margin:.3em 0 .8em}
.scard .sactions{display:flex;gap:8px}
.stat-strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin-top:16px}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;text-align:center}
.stat .n{font-size:24px;font-weight:800;letter-spacing:-.02em}
.stat .l{font-size:12px;color:var(--muted);margin-top:2px}
.pipe-wrap{overflow-x:auto;margin-top:8px;padding-bottom:6px}
.console table{width:100%;border-collapse:collapse;font-size:13px;margin-top:10px}
.console th,.console td{border:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top}
.console th{background:var(--soft);font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.console td .pill{font-size:11.5px;font-weight:700;padding:2px 8px;border-radius:99px;background:var(--soft);color:var(--muted)}
.console td .pill[data-v="approve"]{background:var(--ok-bg);color:var(--ok)}
.console td .pill[data-v="revise"]{background:var(--revise-bg);color:var(--revise)}
.console td .pill[data-v="discuss"]{background:var(--discuss-bg);color:var(--discuss)}
.console td .pill[data-v="reject"]{background:var(--reject-bg);color:var(--reject)}

@media (max-width:980px){
  .wrap{grid-template-columns:1fr;gap:18px}
  .toc{display:none}
  .metagrid,.reviewer,.indi,.subgrid,.legend{grid-template-columns:1fr}
  .kvrow{grid-template-columns:1fr}
}
@media print{
  .appbar,.toc,.fb,.howto,.overall .reviewer,.overall .endbar,.btn,.toast{display:none !important}
  body{background:#fff}
  .wrap{display:block;max-width:100%;padding:0}
  section.card,.hero{box-shadow:none;break-inside:avoid}
  .val>.vbody{display:block !important}
  body.only-proposed section.card:not(.enh){display:block !important}
}
"""

# Shared feedback engine + interactions used by both scheme pages and index.
JS_BASE = r"""
(function(){
  "use strict";
  const $ = (s,r)=> (r||document).querySelector(s);
  const $$ = (s,r)=> Array.from((r||document).querySelectorAll(s));
  const META = window.__SCHEME__ || {id:"aggregate",title:"Aggregate"};
  const KEY = "bpsfb::"+META.id;
  const VERDICT_LABEL = {approve:"Approved",revise:"Revise",discuss:"Discuss",reject:"Reject"};

  function toast(msg){
    let t=$(".toast"); if(!t){t=document.createElement("div");t.className="toast";document.body.appendChild(t);}
    t.textContent=msg; t.classList.add("show"); clearTimeout(t._h); t._h=setTimeout(()=>t.classList.remove("show"),1900);
  }
  function flashSaved(){
    const c=$("#savechip"); if(!c) return; c.textContent="Saved"; c.classList.add("flash");
    clearTimeout(c._h); c._h=setTimeout(()=>{c.classList.remove("flash");c.textContent="All changes saved";},1100);
  }

  // ---------- state ----------
  function load(){ try{return JSON.parse(localStorage.getItem(KEY))||{};}catch(e){return {};} }
  function persist(){ try{localStorage.setItem(KEY,JSON.stringify(state));}catch(e){} flashSaved(); }
  let state = load();

  // ---------- TOC review dots ----------
  function updateToc(){
    $$(".toc a .tdot").forEach(d=>{ const id=d.dataset.fbTarget; const v=(state[id]||{}).verdict;
      if(v) d.setAttribute("data-v",v); else d.removeAttribute("data-v"); });
  }

  function applyToWidget(fb){
    const id = fb.dataset.fbId; const d = state[id]||{};
    fb.dataset.verdict = d.verdict||"";
    fb.dataset.clarity = d.clarity||"";
    $$(".fb-chip",fb).forEach(c=> c.setAttribute("aria-pressed", String(c.dataset.verdict===d.verdict)));
    const st = $(".fb-status",fb);
    if(st){ if(d.verdict){st.textContent=VERDICT_LABEL[d.verdict];st.dataset.v=d.verdict;} else {st.textContent="Not reviewed";st.removeAttribute("data-v");} }
    $$(".star",fb).forEach(s=> s.classList.toggle("on", Number(s.dataset.n)<=Number(d.clarity||0)));
    const note=$(".fb-note",fb); if(note && document.activeElement!==note) note.value = d.comment||"";
  }
  function readReviewer(){
    return {name:(($("#rev-name")||{}).value||"").trim(), role:(($("#rev-role")||{}).value||"").trim()};
  }
  function writeReviewer(r){
    if($("#rev-name")) $("#rev-name").value=r.name||"";
    if($("#rev-role")) $("#rev-role").value=r.role||"";
  }

  function updateProgress(){
    const all=$$(".fb"); const done=all.filter(fb=>fb.dataset.verdict).length;
    const pct = all.length? Math.round(100*done/all.length):0;
    const fill=$(".progress-mini .fill"); if(fill) fill.style.width=pct+"%";
    const lab=$(".progress-mini .lab"); if(lab) lab.textContent=done+" / "+all.length+" reviewed";
    updateToc();
  }

  function setVerdict(fb,val){
    const id=fb.dataset.fbId; state[id]=state[id]||{label:fb.dataset.fbLabel};
    state[id].verdict = (state[id].verdict===val)? "" : val;
    persist(); applyToWidget(fb); updateProgress();
  }
  function setClarity(fb,n){
    const id=fb.dataset.fbId; state[id]=state[id]||{label:fb.dataset.fbLabel};
    state[id].clarity = (Number(state[id].clarity)===n)? 0 : n;
    persist(); applyToWidget(fb);
  }
  function setNote(fb,txt){
    const id=fb.dataset.fbId; state[id]=state[id]||{label:fb.dataset.fbLabel};
    state[id].comment=txt; persist();
  }

  // ---------- wire widgets ----------
  $$(".fb").forEach(fb=>{
    if(!state[fb.dataset.fbId]) state[fb.dataset.fbId]={label:fb.dataset.fbLabel};
    $$(".fb-chip",fb).forEach(c=> c.addEventListener("click",()=>setVerdict(fb,c.dataset.verdict)));
    $$(".star",fb).forEach(s=>{
      s.addEventListener("click",()=>setClarity(fb,Number(s.dataset.n)));
      s.addEventListener("keydown",e=>{ if(e.key==="Enter"||e.key===" "){e.preventDefault();setClarity(fb,Number(s.dataset.n));} });
    });
    const note=$(".fb-note",fb); if(note){ note.addEventListener("input",()=>setNote(fb,note.value)); }
    applyToWidget(fb);
  });
  if(state.__reviewer) writeReviewer(state.__reviewer);
  $$("#rev-name,#rev-role").forEach(inp=> inp && inp.addEventListener("input",()=>{
    state.__reviewer=readReviewer(); persist();
  }));
  updateProgress();

  // ---------- collapsible value ladders ----------
  $$(".val.has-body>.vhead").forEach(h=> h.addEventListener("click",()=> h.parentElement.classList.toggle("open")));
  function expandAll(open){ $$(".val.has-body").forEach(v=> v.classList.toggle("open",open)); }
  const be=$("#btn-expand"); if(be) be.addEventListener("click",()=>expandAll(true));
  const bc=$("#btn-collapse"); if(bc) bc.addEventListener("click",()=>expandAll(false));

  // ---------- filter to proposed refinements ----------
  const bf=$("#btn-filter");
  if(bf) bf.addEventListener("click",()=>{
    const on=document.body.classList.toggle("only-proposed");
    bf.setAttribute("aria-pressed",String(on));
    bf.textContent = on? "Show all sections" : "Show only proposed";
    toast(on? "Showing proposed refinements only" : "Showing all sections");
  });

  // ---------- jump to next unreviewed ----------
  function nextUnreviewed(){
    const target=$$(".fb").find(fb=>!fb.dataset.verdict);
    if(!target){ toast("Every section has a verdict"); return; }
    const sec=target.closest("section, .overall"); (sec||target).scrollIntoView({behavior:"smooth",block:"start"});
    const note=$(".fb-note",target); setTimeout(()=>note&&note.focus(),400);
  }
  const bn=$("#btn-next"); if(bn) bn.addEventListener("click",nextUnreviewed);
  const pm=$(".progress-mini"); if(pm) pm.addEventListener("click",nextUnreviewed);

  // ---------- scroll-spy TOC ----------
  const links=$$(".toc a"); const map={};
  links.forEach(a=>{ const id=a.getAttribute("href").slice(1); const el=document.getElementById(id); if(el) map[id]=a; });
  if(Object.keys(map).length){
    const ob=new IntersectionObserver((ents)=>{
      ents.forEach(e=>{ if(e.isIntersecting){ links.forEach(l=>l.classList.remove("active"));
        const a=map[e.target.id]; if(a) a.classList.add("active"); } });
    },{rootMargin:"-70px 0px -70% 0px",threshold:0});
    Object.keys(map).forEach(id=>{ const el=document.getElementById(id); if(el) ob.observe(el); });
  }

  // ---------- export / import / copy ----------
  function collect(){
    const items={};
    $$(".fb").forEach(fb=>{ const d=state[fb.dataset.fbId]||{};
      items[fb.dataset.fbId]={label:fb.dataset.fbLabel,verdict:d.verdict||"",clarity:d.clarity?Number(d.clarity):null,comment:d.comment||""};
    });
    return {schema:"bps-coding-scheme-feedback",version:"1.0",scheme_id:META.id,scheme_title:META.title,
      reviewer:readReviewer(),generated_at:new Date().toISOString(),items};
  }
  function download(obj,fname){
    const blob=new Blob([JSON.stringify(obj,null,2)],{type:"application/json"});
    const u=URL.createObjectURL(blob); const a=document.createElement("a");
    a.href=u; a.download=fname; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(()=>URL.revokeObjectURL(u),400);
  }
  function ingest(obj){
    if(!obj||!obj.items){ toast("No feedback items found in file"); return; }
    if(obj.scheme_id && obj.scheme_id!==META.id){ toast("Note: file is for "+obj.scheme_id); }
    Object.keys(obj.items).forEach(id=>{ const it=obj.items[id];
      state[id]=Object.assign(state[id]||{},{label:it.label||id,verdict:it.verdict||"",
        clarity:it.clarity||0,comment:it.comment||""}); });
    if(obj.reviewer){ state.__reviewer=obj.reviewer; writeReviewer(obj.reviewer); }
    persist(); $$(".fb").forEach(applyToWidget); updateProgress(); toast("Feedback imported");
  }

  const exp=$("#btn-export"); if(exp) exp.addEventListener("click",()=>{
    download(collect(), META.id+"_feedback.json"); toast("Exported "+META.id+"_feedback.json");
  });
  const cpy=$("#btn-copy"); if(cpy) cpy.addEventListener("click",async()=>{
    const txt=JSON.stringify(collect(),null,2);
    try{ await navigator.clipboard.writeText(txt); toast("Feedback JSON copied to clipboard"); }
    catch(e){ download(collect(),META.id+"_feedback.json"); toast("Clipboard blocked; downloaded instead"); }
  });
  const imp=$("#btn-import"); const impf=$("#file-import");
  if(imp&&impf){ imp.addEventListener("click",()=>impf.click());
    impf.addEventListener("change",e=>{ const f=e.target.files[0]; if(!f)return;
      const r=new FileReader(); r.onload=()=>{ try{ingest(JSON.parse(r.result));}catch(err){toast("Invalid JSON file");} };
      r.readAsText(f); impf.value=""; });
  }
  const pr=$("#btn-print"); if(pr) pr.addEventListener("click",()=>window.print());
  const rs=$("#btn-reset"); if(rs) rs.addEventListener("click",()=>{
    if(confirm("Clear all feedback saved in this browser for this scheme? Export first if you want a copy.")){
      state={}; persist(); $$(".fb").forEach(applyToWidget); writeReviewer({}); updateProgress(); toast("Feedback cleared");
    }
  });

  // ---------- keyboard shortcuts ----------
  document.addEventListener("keydown",e=>{
    const t=e.target.tagName; if(t==="INPUT"||t==="TEXTAREA"||e.metaKey||e.ctrlKey||e.altKey) return;
    if(e.key==="e"){expandAll(true);} else if(e.key==="c"){expandAll(false);}
    else if(e.key==="n"){nextUnreviewed();} else if(e.key==="p"&&bf){bf.click();}
    else if(e.key==="s"){e.preventDefault(); exp&&exp.click();}
  });

  window.__FB__ = {collect, ingest, state:()=>state};
})();
"""

# Index-only console: merge multiple per-scheme exports and show a matrix.
JS_INDEX = r"""
(function(){
  "use strict";
  const $=(s,r)=>(r||document).querySelector(s), $$=(s,r)=>Array.from((r||document).querySelectorAll(s));
  const box=$("#merge-drop"); const fin=$("#merge-files"); const btn=$("#btn-merge");
  const out=$("#merge-out"); if(!box) return;
  const merged={}; // scheme_id -> obj

  function esc(s){ return String(s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }
  function verdictPill(v){ return v? '<span class="pill" data-v="'+v+'">'+v+'</span>' : '<span class="pill">-</span>'; }
  function render(){
    const ids=Object.keys(merged);
    if(!ids.length){ out.innerHTML='<p class="intro">No files loaded yet. Add one or more exported <code>*_feedback.json</code> files.</p>'; return; }
    let counts={approve:0,revise:0,discuss:0,reject:0,none:0};
    let rows='';
    ids.sort().forEach(sid=>{ const o=merged[sid];
      const its=o.items||{}; const keys=Object.keys(its);
      keys.forEach(k=>{ const it=its[k]; const v=it.verdict||"none"; counts[v]=(counts[v]||0)+1;
        rows+='<tr><td><code>'+esc(sid)+'</code></td><td>'+esc(it.label||k)+'</td><td>'+verdictPill(it.verdict)+
              '</td><td>'+(it.clarity||"-")+'</td><td>'+esc(it.comment||"")+'</td></tr>';
      });
    });
    out.innerHTML=
      '<div class="stat-strip">'+
      '<div class="stat"><div class="n">'+ids.length+'</div><div class="l">schemes loaded</div></div>'+
      '<div class="stat"><div class="n" style="color:var(--ok)">'+counts.approve+'</div><div class="l">approve</div></div>'+
      '<div class="stat"><div class="n" style="color:var(--revise)">'+counts.revise+'</div><div class="l">revise</div></div>'+
      '<div class="stat"><div class="n" style="color:var(--discuss)">'+counts.discuss+'</div><div class="l">discuss</div></div>'+
      '<div class="stat"><div class="n" style="color:var(--reject)">'+counts.reject+'</div><div class="l">reject</div></div>'+
      '</div>'+
      '<table><thead><tr><th>Scheme</th><th>Section</th><th>Verdict</th><th>Clarity</th><th>Comment</th></tr></thead><tbody>'+rows+'</tbody></table>';
  }
  function add(obj){ if(obj&&obj.scheme_id&&obj.items){ merged[obj.scheme_id]=obj; } render(); }
  function readFiles(files){ Array.from(files).forEach(f=>{ const r=new FileReader();
    r.onload=()=>{ try{add(JSON.parse(r.result));}catch(e){} }; r.readAsText(f); }); }

  btn && btn.addEventListener("click",()=>fin.click());
  fin && fin.addEventListener("change",e=>{ readFiles(e.target.files); fin.value=""; });
  box.addEventListener("dragover",e=>{e.preventDefault();box.style.borderColor="var(--accent)";});
  box.addEventListener("dragleave",()=>box.style.borderColor="");
  box.addEventListener("drop",e=>{e.preventDefault();box.style.borderColor="";readFiles(e.dataTransfer.files);});
  const dl=$("#btn-merge-export"); dl && dl.addEventListener("click",()=>{
    const obj={schema:"bps-coding-scheme-feedback-bundle",version:"1.0",generated_at:new Date().toISOString(),schemes:merged};
    const b=new Blob([JSON.stringify(obj,null,2)],{type:"application/json"});
    const u=URL.createObjectURL(b); const a=document.createElement("a"); a.href=u;
    a.download="all_schemes_feedback_bundle.json"; document.body.appendChild(a); a.click(); a.remove();
  });
  render();
})();
"""
