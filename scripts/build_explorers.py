#!/usr/bin/env python3
"""Generate Distill-style HTML explorers for any environment from results JSON.

Usage:
    python scripts/build_explorers.py --results-dir results/
    python scripts/build_explorers.py --results-dir results/ --out-dir docs/

Expected JSON schema per file:
  {env_name, display_name, training_curves: {config: {epochs, train_loss,
   val_loss, train_ratio, val_ratio, sigreg}}, comparison: [{name,
   copy_ratio, pred_loss, copy_baseline, sigreg_loss, epochs}],
   rollout_episodes: [{id, success, n_frames, frames, cem_info}],
   dataset_episodes: [{ep_idx, n_frames, frames, length}]}
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

COLORS = dict(ink="#1a1a2e", paper="#faf8f4", surface="#ffffff",
              accent="#c44536", accent2="#2d6a4f", blue="#2563eb",
              muted="#6b7280", border="#d4d0c8", code_bg="#f0ede6")
PAL = ["#2563eb","#c44536","#2d6a4f","#d4a017","#60a5fa","#e879f9","#f97316","#14b8a6"]

# ── SVG helpers ───────────────────────────────────────────────────────────
def _polyline(xs, ys, w, h, pad, color, dash=False):
    if not xs or len(xs) < 2: return ""
    xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
    xr, yr = (xmax-xmin) or 1, (ymax-ymin) or 1e-9
    pts = " ".join(f"{pad+(x-xmin)/xr*(w-2*pad):.1f},{(h-pad)-(y-ymin)/yr*(h-2*pad):.1f}"
                   for x, y in zip(xs, ys))
    d = ' stroke-dasharray="5,3"' if dash else ""
    return f'<polyline points="{pts}" stroke="{color}" fill="none" stroke-width="1.8"{d}/>'

def _axes(w, h, pad, xl, yl, yv):
    B, M, MO = COLORS["border"], COLORS["muted"], "JetBrains Mono,monospace"
    s = (f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="{B}"/>'
         f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="{B}"/>'
         f'<text x="{w/2}" y="{h-2}" text-anchor="middle" font-size="9" fill="{M}" font-family="{MO}">{xl}</text>'
         f'<text x="4" y="{h/2}" text-anchor="middle" font-size="9" fill="{M}" font-family="{MO}" transform="rotate(-90,4,{h/2})">{yl}</text>')
    if yv:
        ymn, ymx = min(yv), max(yv); yr = (ymx-ymn) or 1e-9
        for i in range(1, 4):
            f_ = i/4; gy = (h-pad)-f_*(h-2*pad); v = ymn+f_*yr
            s += (f'<line x1="{pad}" y1="{gy:.1f}" x2="{w-pad}" y2="{gy:.1f}" stroke="{B}" '
                  f'stroke-width="0.5" stroke-dasharray="3,3"/>'
                  f'<text x="{pad-3}" y="{gy+3:.1f}" text-anchor="end" font-size="7" fill="{M}" font-family="{MO}">{v:.3g}</text>')
    return s

def svg_chart(series, w=700, h=200, pad=38, xl="epoch", yl="value", title=""):
    ay = [v for s in series for v in (s.get("ys") or [])]
    o = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:{w}px;height:auto;display:block;margin:0 auto">',
         f'<rect width="{w}" height="{h}" fill="{COLORS["surface"]}" rx="4"/>']
    if title:
        o.append(f'<text x="{w/2}" y="14" text-anchor="middle" font-size="10" font-weight="600" fill="{COLORS["ink"]}" font-family="Source Serif 4,Georgia,serif">{title}</text>')
    o.append(_axes(w, h, pad, xl, yl, ay))
    lx = pad+8
    for i, s in enumerate(series):
        ly = 24+i*13; c = s.get("color", PAL[i%len(PAL)])
        d = ' stroke-dasharray="4,2"' if s.get("dash") else ""
        o.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+14}" y2="{ly}" stroke="{c}" stroke-width="2"{d}/>')
        o.append(f'<text x="{lx+18}" y="{ly+3}" font-size="8" fill="{COLORS["muted"]}" font-family="JetBrains Mono,monospace">{s.get("label","")}</text>')
    for s in series:
        o.append(_polyline(s["xs"], s["ys"], w, h, pad, s.get("color", PAL[0]), s.get("dash", False)))
    o.append("</svg>"); return "\n".join(o)

def svg_bars(labels, values, colors=None, w=700, h=260, pad=50, title="Copy Ratio Comparison"):
    if not labels: return ""
    n = len(labels); bw = max(8, min(40, (w-2*pad)/(n*1.5))); gap = bw*0.5
    vm = max(max(values), 0.01)
    o = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:{w}px;height:auto;display:block;margin:0 auto">',
         f'<rect width="{w}" height="{h}" fill="{COLORS["surface"]}" rx="4"/>']
    if title:
        o.append(f'<text x="{w/2}" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="{COLORS["ink"]}" font-family="Source Serif 4,Georgia,serif">{title}</text>')
    o.append(f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="{COLORS["border"]}"/>')
    for i, (lb, v) in enumerate(zip(labels, values)):
        x = pad+i*(bw+gap); bh = (v/vm)*(h-pad-30); y = h-pad-bh
        c = colors[i] if colors else PAL[i%len(PAL)]
        o.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{c}" rx="2"/>')
        o.append(f'<text x="{x+bw/2:.1f}" y="{y-3:.1f}" text-anchor="middle" font-size="7" fill="{COLORS["ink"]}" font-family="JetBrains Mono,monospace">{v:.3f}</text>')
        o.append(f'<text x="{x+bw/2:.1f}" y="{h-pad+10:.1f}" text-anchor="end" font-size="7" fill="{COLORS["muted"]}" font-family="JetBrains Mono,monospace" transform="rotate(-40,{x+bw/2:.1f},{h-pad+10:.1f})">{lb}</text>')
    o.append("</svg>"); return "\n".join(o)

# ── CSS ───────────────────────────────────────────────────────────────────
CSS = """:root{--ink:#1a1a2e;--paper:#faf8f4;--surface:#fff;--accent:#c44536;--accent2:#2d6a4f;--blue:#2563eb;--muted:#6b7280;--border:#d4d0c8;--code-bg:#f0ede6;--serif:'Source Serif 4','Newsreader',Georgia,serif;--body:'Newsreader',Georgia,serif;--mono:'JetBrains Mono',monospace}
*{margin:0;padding:0;box-sizing:border-box}html{font-size:17px;scroll-behavior:smooth}
body{font-family:var(--body);color:var(--ink);background:var(--paper);line-height:1.72;-webkit-font-smoothing:antialiased}
.d-article{max-width:780px;margin:0 auto;padding:2rem 1.5rem 5rem}
.d-title{padding:3.5rem 0 1.8rem;border-bottom:1px solid var(--border);margin-bottom:2rem}
.d-title h1{font-family:var(--serif);font-weight:700;font-size:2.2rem;line-height:1.2;letter-spacing:-.02em}
.d-title .subtitle{font-family:var(--body);font-weight:300;font-size:1.05rem;color:var(--muted);margin-top:.6rem}
.d-byline{font-family:var(--mono);font-size:.75rem;color:var(--muted);margin-top:.8rem;letter-spacing:.02em}
h2{font-family:var(--serif);font-weight:600;font-size:1.45rem;margin:2.5rem 0 .8rem;padding-top:1.2rem;border-top:1px solid var(--border)}
p{margin-bottom:1rem}
.aside{background:var(--code-bg);border-left:3px solid var(--accent);padding:.8rem 1rem;margin:1.2rem 0;font-size:.85rem;border-radius:0 4px 4px 0}
.tabs{display:flex;gap:0;border:1px solid var(--border);border-radius:6px;overflow:hidden;margin:1.5rem 0;background:var(--surface)}
.tab{flex:1;padding:.6rem 1rem;font-family:var(--mono);font-size:.72rem;text-align:center;cursor:pointer;border-right:1px solid var(--border);color:var(--muted);transition:all .15s;letter-spacing:.03em}
.tab:last-child{border-right:none}.tab:hover{background:var(--code-bg);color:var(--ink)}.tab.active{background:var(--ink);color:var(--paper)}
.panel{display:none}.panel.active{display:block}
.summary{display:grid;grid-template-columns:repeat(4,1fr);gap:.8rem;margin-bottom:1.5rem}
.stat-card{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:.8rem 1rem;text-align:center}
.stat-val{font-family:var(--mono);font-size:1.5rem;font-weight:500}.stat-val.good{color:var(--accent2)}.stat-val.accent{color:var(--blue)}
.stat-label{font-size:.7rem;color:var(--muted);margin-top:.2rem}
.dataset-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:1rem}
.dataset-card{background:var(--surface);border:1px solid var(--border);border-radius:6px;overflow:hidden}
.video-area{position:relative;background:var(--code-bg);aspect-ratio:1}.video-area img{width:100%;height:100%;object-fit:contain}
.video-controls{position:absolute;bottom:0;left:0;right:0;background:linear-gradient(transparent,rgba(255,255,255,.95));padding:.6rem .8rem .4rem;display:flex;align-items:center;gap:.5rem}
.play-btn{width:26px;height:26px;border-radius:50%;background:var(--ink);border:none;color:var(--paper);font-size:.65rem;cursor:pointer;display:flex;align-items:center;justify-content:center}
.scrubber{flex:1;-webkit-appearance:none;background:var(--border);height:3px;border-radius:2px;outline:none}
.scrubber::-webkit-slider-thumb{-webkit-appearance:none;width:10px;height:10px;border-radius:50%;background:var(--ink);cursor:pointer}
.frame-counter{font-family:var(--mono);font-size:.65rem;color:var(--muted)}
.episode-header{padding:.6rem 1rem;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--border);background:var(--code-bg)}
.episode-header h3{font-family:var(--mono);font-size:.75rem;font-weight:500;color:var(--muted)}
.ratio-pill{font-family:var(--mono);font-size:.7rem;font-weight:500;padding:.15rem .5rem;border-radius:3px}
.ratio-pill.good{background:#d1fae5;color:#065f46}.ratio-pill.ok{background:#fef3c7;color:#92400e}.ratio-pill.bad{background:#fee2e2;color:#991b1b}
table.cmp{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:.78rem;margin:1rem 0}
table.cmp th{background:var(--code-bg);text-align:left;padding:.5rem .6rem;border-bottom:2px solid var(--border);font-weight:500;font-size:.68rem;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
table.cmp td{padding:.45rem .6rem;border-bottom:1px solid var(--border)}
table.cmp tr:hover{background:var(--code-bg)}td.best{background:#d1fae5;font-weight:600;color:#065f46}
.chart-wrap{margin:1.5rem 0;padding:1rem;background:var(--surface);border:1px solid var(--border);border-radius:6px}
.chart-title{font-family:var(--mono);font-size:.72rem;color:var(--muted);margin-bottom:.5rem;text-transform:uppercase;letter-spacing:.04em}
@media(max-width:700px){html{font-size:15px}.summary{grid-template-columns:repeat(2,1fr)}.d-title h1{font-size:1.7rem}}"""

FONTS = ('<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:'
         'ital,opsz,wght@0,8..60,300;0,8..60,400;0,8..60,600;0,8..60,700;'
         '1,8..60,400&family=JetBrains+Mono:wght@400;500&family=Newsreader:'
         'ital,opsz,wght@0,6..72,300;0,6..72,400;0,6..72,600;1,6..72,400'
         '&display=swap" rel="stylesheet">')

# ── Tab builders ──────────────────────────────────────────────────────────
def _stat(val, label, cls="accent"):
    return f'<div class="stat-card"><div class="stat-val {cls}">{val}</div><div class="stat-label">{label}</div></div>'

def _player_card(pfx, i, header_html, nf, onclick_fn):
    return (f'<div class="dataset-card">{header_html}<div class="video-area">'
            f'<img id="{pfx}-img"/><div class="video-controls">'
            f'<button class="play-btn" id="{pfx}-btn" onclick="{onclick_fn}({i})">&#9654;</button>'
            f'<input type="range" class="scrubber" id="{pfx}-scrub" min="0" max="{nf-1}" value="0" '
            f'oninput="seek{onclick_fn[6:]}({i},this.value)"/>'
            f'<span class="frame-counter" id="{pfx}-fc">0/{nf}</span></div></div></div>')

def tab_cem(data):
    eps = data.get("rollout_episodes") or []
    if not eps: return '<p style="color:var(--muted)">No rollout episodes available.</p>'
    ns = sum(1 for e in eps if e.get("success")); rate = ns/len(eps)*100 if eps else 0
    budget = eps[0].get("cem_info", "N/A") if eps else "N/A"
    h = ('<div class="summary">'
         + _stat(f"{rate:.0f}%", "Success Rate", "good" if rate > 0 else "accent")
         + _stat(len(eps), "Rollouts") + _stat(ns, "Successes") + _stat(budget, "Eval Budget")
         + '</div><div class="dataset-grid">')
    for i, ep in enumerate(eps):
        nf = ep.get("n_frames", len(ep.get("frames", [])))
        bc = "good" if ep.get("success") else "bad"
        bt = "Success" if ep.get("success") else "Failed"
        hdr = f'<div class="episode-header"><h3>CEM Rollout #{ep.get("id",i)}</h3><span class="ratio-pill {bc}">{bt}</span></div>'
        h += _player_card(f"cem{i}", i, hdr, nf, "toggleCem")
    return h + "</div>"

def tab_curves(data):
    curves = data.get("training_curves") or {}
    if not curves: return '<p style="color:var(--muted)">No training curves available.</p>'
    charts = []
    for chart_key, y_keys, title, yl in [
        ("loss", [("train_loss","train"),("val_loss","val")], "Prediction Loss", "loss"),
        ("ratio", [("train_ratio","train"),("val_ratio","val")], "Copy Ratio", "ratio"),
        ("sigreg", [("sigreg","")], "SIGReg Regularisation Loss", "sigreg"),
    ]:
        series = []
        for ci, (name, c) in enumerate(curves.items()):
            if c is None: continue
            col = PAL[ci % len(PAL)]
            for ykey, suffix in y_keys:
                if c.get(ykey):
                    lab = f"{name} {suffix}".strip()
                    series.append(dict(xs=c["epochs"], ys=c[ykey], color=col,
                                       label=lab, dash=(suffix == "val")))
        if series:
            charts.append(f'<div class="chart-wrap"><div class="chart-title">{title}</div>'
                          + svg_chart(series, xl="epoch", yl=yl, title=title) + "</div>")
    return "\n".join(charts) if charts else '<p style="color:var(--muted)">No curve data.</p>'

def tab_compare(data):
    rows = data.get("comparison") or []
    if not rows: return '<p style="color:var(--muted)">No comparison data available.</p>'
    best = {}
    for k in ("copy_ratio", "pred_loss", "sigreg_loss"):
        vs = [r[k] for r in rows if r.get(k) is not None]
        best[k] = min(vs) if vs else None
    def td(r, k, fmt=".4f"):
        v = r.get(k)
        if v is None: return "<td>--</td>"
        c = ' class="best"' if best.get(k) is not None and abs(v-best[k]) < 1e-9 else ""
        return f"<td{c}>{v:{fmt}}</td>"
    h = ('<table class="cmp"><thead><tr><th>Config</th><th>Copy Ratio</th><th>Pred Loss</th>'
         '<th>Copy Baseline</th><th>SIGReg Loss</th><th>Epochs</th></tr></thead><tbody>')
    for r in rows:
        h += f"<tr><td>{r['name']}</td>{td(r,'copy_ratio')}{td(r,'pred_loss')}{td(r,'copy_baseline')}{td(r,'sigreg_loss')}<td>{r.get('epochs','--')}</td></tr>"
    return h + "</tbody></table>"

def tab_dataset(data):
    eps = data.get("dataset_episodes") or []
    if not eps: return '<p style="color:var(--muted)">No dataset episodes available.</p>'
    h = ('<div style="margin-bottom:1rem">'
         '<button onclick="playAllDs()" style="font-family:var(--mono);font-size:.75rem;padding:.3rem .8rem;border:1px solid var(--border);border-radius:3px;background:var(--surface);cursor:pointer">Play All</button>'
         '<button onclick="stopAllDs()" style="font-family:var(--mono);font-size:.75rem;padding:.3rem .8rem;border:1px solid var(--border);border-radius:3px;background:var(--surface);cursor:pointer;margin-left:.3rem">Stop</button>'
         '</div><div class="dataset-grid">')
    for i, ep in enumerate(eps):
        nf = ep.get("n_frames", len(ep.get("frames", [])))
        foot = f'<div style="font-family:var(--mono);font-size:.65rem;color:var(--muted);padding:.3rem .8rem;border-top:1px solid var(--border)">Ep #{ep.get("ep_idx",i)} &middot; {ep.get("length",nf)} frames</div>'
        h += _player_card(f"ds{i}", i, "", nf, "toggleDs") + foot
    return h + "</div>"

# ── JS ────────────────────────────────────────────────────────────────────
def explorer_js(data):
    return f"""<script>
const D={json.dumps(data,separators=(',',':'))},PI={{}};
function _seek(pre,arr,i,f){{f=+f;const e=arr[i];if(!e||!e.frames)return;const p=pre+i;if(f<e.frames.length)document.getElementById(p+'-img').src='data:image/jpeg;base64,'+e.frames[f];document.getElementById(p+'-fc').textContent=f+'/'+e.n_frames;}}
function _toggle(pre,arr,i,ms){{const k=pre+i;if(PI[k]){{clearInterval(PI[k]);delete PI[k];document.getElementById(k+'-btn').innerHTML='&#9654;';return;}}let f=+document.getElementById(k+'-scrub').value;document.getElementById(k+'-btn').innerHTML='&#9646;&#9646;';PI[k]=setInterval(()=>{{f=(f+1)%arr[i].n_frames;document.getElementById(k+'-scrub').value=f;_seek(pre,arr,i,f);}},ms);}}
function seekCem(i,f){{_seek('cem',D.rollout_episodes,i,f)}}
function toggleCem(i){{_toggle('cem',D.rollout_episodes,i,150)}}
function seekDs(i,f){{_seek('ds',D.dataset_episodes,i,f)}}
function toggleDs(i){{_toggle('ds',D.dataset_episodes,i,150)}}
function playAllDs(){{(D.dataset_episodes||[]).forEach((_,i)=>{{if(!PI['ds'+i])toggleDs(i)}})}}
function stopAllDs(){{Object.keys(PI).forEach(k=>{{clearInterval(PI[k]);delete PI[k]}})}}
function switchTab(n){{document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));event.target.classList.add('active');document.getElementById('panel-'+n).classList.add('active')}}
addEventListener('DOMContentLoaded',()=>{{(D.rollout_episodes||[]).forEach((e,i)=>{{if(e.frames&&e.frames.length)document.getElementById('cem'+i+'-img').src='data:image/jpeg;base64,'+e.frames[0]}});(D.dataset_episodes||[]).forEach((e,i)=>{{if(e.frames&&e.frames.length)document.getElementById('ds'+i+'-img').src='data:image/jpeg;base64,'+e.frames[0]}})}});
</script>"""

# ── Per-environment HTML ──────────────────────────────────────────────────
def build_explorer(data):
    env = data.get("env_name", "unknown"); dn = data.get("display_name", env.title())
    t1, t2, t3, t4 = tab_cem(data), tab_curves(data), tab_compare(data), tab_dataset(data)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>LeWM {dn} -- Explorer</title>{FONTS}<style>{CSS}</style></head><body>
<article class="d-article">
<header class="d-title"><h1>{dn} Explorer</h1>
<p class="subtitle">LeWM world model results for {dn} -- CEM rollouts, training curves, architecture comparison, and dataset viewer.</p>
<div class="d-byline">Environment: {dn} &middot; Generated by build_explorers.py</div></header>
<div class="tabs">
<div class="tab active" onclick="switchTab('cem')">CEM Rollouts</div>
<div class="tab" onclick="switchTab('curves')">Training Curves</div>
<div class="tab" onclick="switchTab('compare')">Architecture</div>
<div class="tab" onclick="switchTab('dataset')">Dataset Viewer</div></div>
<div class="panel active" id="panel-cem"><h2>CEM Planning Rollouts</h2>
<p>The world model plans actions via CEM in embedding space, then executes them. Each card shows a playable rollout with success/fail status.</p>{t1}</div>
<div class="panel" id="panel-curves"><h2>Training Curves</h2>
<p>Loss, copy ratio, and SIGReg regularisation over training epochs for each configuration.</p>{t2}</div>
<div class="panel" id="panel-compare"><h2>Architecture Comparison</h2>
<p>Side-by-side comparison of all configurations tested on {dn}. Best values highlighted in green.</p>{t3}</div>
<div class="panel" id="panel-dataset"><h2>Dataset Viewer</h2>
<p>Playable episodes from the {dn} dataset. Scrub through frames to inspect the data.</p>{t4}</div>
</article>{explorer_js(data)}</body></html>"""

# ── Cross-environment dashboard ───────────────────────────────────────────
def build_dashboard(all_data):
    envs, all_cfg = [], set()
    for d in all_data:
        en = d.get("display_name", d.get("env_name", "?"))
        cm = {c["name"]: c.get("copy_ratio") for c in (d.get("comparison") or [])}
        all_cfg.update(cm.keys()); envs.append((en, cm))
    cfgs = sorted(all_cfg)
    best = {}
    for en, cm in envs:
        vs = [(n, v) for n, v in cm.items() if v is not None]
        if vs: best[en] = min(vs, key=lambda x: x[1])[0]
    th = "".join(f"<th>{c}</th>" for c in cfgs)
    tbl = f'<table class="cmp"><thead><tr><th>Environment</th>{th}</tr></thead><tbody>'
    for en, cm in envs:
        bn = best.get(en)
        tds = "".join(f'<td{" class=best" if c==bn else ""}>{cm[c]:.4f}</td>' if cm.get(c) is not None else "<td>--</td>" for c in cfgs)
        tbl += f"<tr><td><strong>{en}</strong></td>{tds}</tr>"
    tbl += "</tbody></table>"
    lb, vl, cl = [], [], []
    for en, cm in envs:
        for ci, c in enumerate(cfgs):
            v = cm.get(c)
            if v is not None: lb.append(f"{en}/{c[:12]}"); vl.append(v); cl.append(PAL[ci%len(PAL)])
    bar = svg_bars(lb, vl, cl, title="Copy Ratio Across Environments & Configs")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>LeWM -- Cross-Environment Comparison</title>{FONTS}<style>{CSS}</style></head><body>
<article class="d-article">
<header class="d-title"><h1>Cross-Environment Comparison</h1>
<p class="subtitle">Copy ratio and architecture comparison across all evaluated environments.</p>
<div class="d-byline">Environments: {len(envs)} &middot; Configurations: {len(cfgs)} &middot; Generated by build_explorers.py</div></header>
<h2>Copy Ratio Table</h2>
<p>Rows are environments, columns are architecture configurations. Best value per environment highlighted in green.</p>
{tbl}
<h2>Copy Ratio Chart</h2><div class="chart-wrap">{bar}</div>
<div class="aside"><strong>Lower is better.</strong> Copy ratio &lt; 1 means the model outperforms the copy baseline. Values closer to 0 indicate learned dynamics rather than memorising the last frame.</div>
</article></body></html>"""

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Generate Distill-style HTML explorers")
    ap.add_argument("--results-dir", required=True, help="Dir with per-env result JSONs")
    ap.add_argument("--out-dir", default="docs", help="Output dir (default: docs/)")
    args = ap.parse_args()
    rd, od = Path(args.results_dir), Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    jfs = sorted(rd.glob("*.json"))
    if not jfs: print(f"No JSON files in {rd}"); return
    ad = []
    for jf in jfs:
        print(f"Reading {jf} ...")
        data = json.loads(jf.read_text())
        ad.append(data)
        en = data.get("env_name", jf.stem)
        op = od / f"{en}-explorer.html"
        html = build_explorer(data)
        op.write_text(html, encoding="utf-8")
        print(f"  -> {op} ({len(html):,} bytes)")
    if ad:
        cp = od / "comparison.html"
        ch = build_dashboard(ad)
        cp.write_text(ch, encoding="utf-8")
        print(f"  -> {cp} ({len(ch):,} bytes)")
    print(f"Done. Generated {len(ad)} explorer(s) + 1 comparison dashboard.")

if __name__ == "__main__":
    main()
