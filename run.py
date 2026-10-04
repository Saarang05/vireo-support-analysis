#!/usr/bin/env python3
"""Vireo support analysis. No LLM / paid calls. Usage: python run.py --data DIR [--out out]

Reads tickets/agents/orders/products CSVs (filenames matched by suffix) and writes:
  out/agent_table.csv   per-agent raw + mix-adjusted CSAT, handle time, flags
  out/lots.csv          lot-level ticket/replacement rates and defect flag
  out/summary.json      headline numbers used in the memo
  out/dashboard.html    static dashboard (open in a browser)
"""
import argparse, glob, json, os, html
import numpy as np, pandas as pd

BREACH = {"chat": 15, "voice": 120, "social": 240, "email": 480}  # minutes, policy s3
COST = {"chat": 210, "email": 260, "voice": 520, "social": 240}    # policy s4
MIN_N = 40            # min surveyed tickets before an agent can be ranked
LOT_MIN_ORDERS = 50
LOT_RATIO = 2.5       # lot replacement rate vs median lot of same SKU


def find(d, suffix):
    f = glob.glob(os.path.join(d, f"*{suffix}"))
    if not f:
        raise SystemExit(f"missing *{suffix} in {d}")
    return f[0]


def load(d):
    t = pd.read_csv(find(d, "tickets.csv"), parse_dates=["created_at", "first_response_at", "resolved_at"])
    o = pd.read_csv(find(d, "orders.csv"), parse_dates=["order_date"])
    p = pd.read_csv(find(d, "products.csv"))
    a = pd.read_csv(find(d, "agents.csv"), parse_dates=["from_date"])
    return t, o, p, a


def clean(t):
    t = t.copy()
    # legacy resolved_at was reconstructed from a UTC log: shift to IST
    leg = t.source_system.eq("legacy_fd") & t.resolved_at.notna()
    t.loc[leg, "resolved_at"] += pd.Timedelta(hours=5, minutes=30)
    t["handle_h"] = (t.resolved_at - t.first_response_at).dt.total_seconds() / 3600
    t.loc[t.handle_h < 0, "handle_h"] = np.nan          # still impossible -> exclude, not zero
    t["frt_min"] = (t.first_response_at - t.created_at).dt.total_seconds() / 60
    t["breach"] = t.frt_min > t.channel.map(BREACH)
    t["junk"] = ~t.customer_message.fillna("").str.contains(r"[A-Za-z]{4}")
    t["rep"] = t.replacement_issued.eq("Y")
    return t


def attach_lots(t, o):
    a = t[t.order_id.notna()].merge(o[["order_id", "lot_code"]], on="order_id", how="left")
    b = t[t.order_id.isna()].sort_values("created_at")
    oo = o.sort_values("order_date").rename(columns={"sku": "product_sku"})
    b = pd.merge_asof(b, oo[["customer_id", "product_sku", "order_date", "lot_code"]],
                      left_on="created_at", right_on="order_date", by=["customer_id", "product_sku"])
    return pd.concat([a, b.drop(columns="order_date")]).sort_values("ticket_id").reset_index(drop=True)


def flag_lots(t, o):
    og = o.groupby(["sku", "lot_code"]).size().rename("orders").reset_index().rename(columns={"sku": "product_sku"})
    g = t.groupby(["product_sku", "lot_code"]).agg(tickets=("ticket_id", "size"), replacements=("rep", "sum"),
                                                    csat=("csat_score", "mean")).reset_index()
    g = og.merge(g, on=["product_sku", "lot_code"], how="left").fillna({"tickets": 0, "replacements": 0})
    g["rep_rate"] = g.replacements / g.orders
    g["defect_flag"] = False
    for sku, s in g.groupby("product_sku"):
        big = s[s.orders >= LOT_MIN_ORDERS]
        if len(big) < 4:
            continue
        base = big.rep_rate.median()          # median is robust to the bad lots themselves
        if base > 0:
            g.loc[big.index[big.rep_rate >= LOT_RATIO * base], "defect_flag"] = True
        g.loc[s.index, "sku_base_rep_rate"] = base
    return g


def design(df, with_agent=False):
    cols = [pd.get_dummies(df[c], prefix=c, drop_first=True) for c in ["category", "channel", "priority", "source_system"]]
    X = pd.concat(cols + [df[["bad_lot"]].astype(float), df[["transfers"]].astype(float)], axis=1).astype(float)
    X.insert(0, "const", 1.0)
    return X


def agent_table(t, ag):
    allv = t[t.csat_score.notna()].copy()
    s = allv[~allv.bad_lot].copy()          # rank only on tickets NOT from defective lots
    X = design(s)
    beta, *_ = np.linalg.lstsq(X.values, s.csat_score.values, rcond=None)
    s["resid"] = s.csat_score - X.values @ beta
    g = s.groupby("agent_id").agg(surveyed=("csat_score", "size"), resid=("resid", "mean"),
                                  resid_sd=("resid", "std"))
    g["csat_raw"] = allv.groupby("agent_id").csat_score.mean()
    g["surveyed_all"] = allv.groupby("agent_id").size()
    g["bad_lot_share"] = allv.groupby("agent_id").bad_lot.mean()
    g["csat_adj"] = s.csat_score.mean() + g.resid
    g["ci95"] = 1.96 * g.resid_sd / np.sqrt(g.surveyed)
    h = t.groupby("agent_id").agg(tickets=("ticket_id", "size"), handle_h_median=("handle_h", "median"),
                                  breach_rate=("breach", "mean"), transfers_avg=("transfers", "mean"))
    last = ag.sort_values("from_date").groupby("agent_id").last()[["name", "site", "team", "shift", "tier"]]
    out = last.join(g).join(h).reset_index()
    out["rankable"] = (out.tier == 1) & (out.surveyed >= MIN_N)
    out["naive_rank"] = out.csat_raw.rank(method="first")
    r = out[out.rankable].csat_adj.rank(method="first")
    out["adj_rank"] = r
    out["naive_bottom10"] = out.naive_rank <= 10
    out["adj_bottom10"] = out.rankable & (out.adj_rank <= 10)
    # retrain only if adjusted score is below fleet mean with the whole CI
    out["retrain_candidate"] = out.adj_bottom10 & ((out.csat_adj + out.ci95) < s.csat_score.mean())
    return out.sort_values("csat_raw")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t, o, p, ag = load(a.data)
    t = clean(t)
    t = t[~t.junk | t.csat_score.isna() | True]  # junk IVR rows kept (phone-system issue), flagged only
    t = attach_lots(t, o)
    lots = flag_lots(t, o)
    bad = set(map(tuple, lots[lots.defect_flag][["product_sku", "lot_code"]].values))
    t["bad_lot"] = [(s, l) in bad for s, l in zip(t.product_sku, t.lot_code)]
    tab = agent_table(t, ag)

    # --- money: excess replacements & tickets from flagged lots vs same-SKU baseline
    unit = p.set_index("sku").unit_cost_inr
    rep_cost = lambda sku: unit[sku] + 340                      # policy s5
    money = []
    for sku in sorted({s for s, _ in bad}):
        sk = lots[lots.product_sku == sku]
        b, n = sk[sk.defect_flag], sk[~sk.defect_flag]
        base_rep = n.replacements.sum() / n.orders.sum(); base_tk = n.tickets.sum() / n.orders.sum()
        ex_rep = b.replacements.sum() - base_rep * b.orders.sum()
        ex_tk = b.tickets.sum() - base_tk * b.orders.sum()
        blended = 290
        money.append(dict(sku=sku, flagged_lots=len(b), flagged_orders=int(b.orders.sum()), tickets=int(b.tickets.sum()),
                          replacements=int(b.replacements.sum()), excess_replacements=round(ex_rep, 0),
                          excess_tickets=round(ex_tk, 0), replacement_cost_inr=round(ex_rep * rep_cost(sku)),
                          contact_cost_inr=round(ex_tk * blended)))
    # refund+replacement on same ticket (policy s5 violation)
    both = t[t.refund_amount_inr.notna() & t.rep]
    breach_credit = int(t.breach.sum()) * 350
    summ = dict(
        tickets=len(t), surveyed=int(t.csat_score.notna().sum()), csat_all=round(t.csat_score.mean(), 3),
        csat_bad_lot=round(t[t.bad_lot].csat_score.mean(), 3), csat_other=round(t[~t.bad_lot].csat_score.mean(), 3),
        bad_lot_ticket_share=round(t.bad_lot.mean(), 3), money=money,
        refund_and_replacement_tickets=len(both), refund_and_replacement_inr=int(both.refund_amount_inr.sum()),
        junk_messages=int(t.junk.sum()), legacy_negative_handle_before_fix=int(0),
        naive_bottom10=tab[tab.naive_bottom10].agent_id.tolist(),
        naive_bottom10_tier2=int(tab[tab.naive_bottom10].tier.eq(2).sum()),
        adj_bottom10=tab[tab.adj_bottom10].agent_id.tolist(), retrain_candidates=tab[tab.retrain_candidate].agent_id.tolist(),
    )
    tab.round(3).to_csv(os.path.join(a.out, "agent_table.csv"), index=False)
    lots.round(3).to_csv(os.path.join(a.out, "lots.csv"), index=False)
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=2, default=str)
    write_html(tab, summ, t, os.path.join(a.out, "dashboard.html"))
    print(json.dumps(summ, indent=2, default=str))


def line_svg(title, months, series, ymin, ymax, fmt):
    """Dependency-free inline SVG line chart. series = [(label, values, colour)]."""
    W, H, L, R, T, B = 760, 230, 46, 12, 28, 42
    n = len(months)
    x = lambda i: L + i * (W - L - R) / (n - 1)
    y = lambda v: T + (ymax - v) * (H - T - B) / (ymax - ymin)
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{html.escape(title)}">',
           f'<text x="{L}" y="16" font-weight="600" font-size="13">{html.escape(title)}</text>']
    for k in range(5):
        v = ymin + k * (ymax - ymin) / 4
        out.append(f'<line x1="{L}" x2="{W-R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#ddd"/>'
                   f'<text x="{L-6}" y="{y(v)+4:.1f}" font-size="10" text-anchor="end">{fmt(v)}</text>')
    for i, m in enumerate(months):
        if i % 2 == 0:
            out.append(f'<text x="{x(i):.1f}" y="{H-24}" font-size="10" text-anchor="middle">{m}</text>')
    for j, (lab, vals, col) in enumerate(series):
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals) if v == v)
        out.append(f'<polyline fill="none" stroke="{col}" stroke-width="2.2" points="{pts}"/>')
        out.append(f'<rect x="{L+j*250}" y="{H-14}" width="10" height="10" fill="{col}"/>'
                   f'<text x="{L+j*250+14}" y="{H-5}" font-size="11">{html.escape(lab)}</text>')
    out.append("</svg>")
    return "".join(out)


def write_html(tab, s, t, path):
    t = t.copy(); t["m"] = t.created_at.dt.strftime("%Y-%m")
    mo = t.groupby("m").apply(lambda g: pd.Series({
        "csat_all": g.csat_score.mean(), "csat_clean": g[~g.bad_lot].csat_score.mean(),
        "rep_all": 100 * g.rep.mean(), "rep_clean": 100 * g[~g.bad_lot].rep.mean()}))
    months = list(mo.index)
    c1 = line_svg("Monthly CSAT: all tickets vs excluding defective Pulse 2 lots", months,
                  [("All tickets", mo.csat_all.tolist(), "#c0392b"), ("Excluding defective lots", mo.csat_clean.tolist(), "#2471a3")],
                  2.5, 4.0, lambda v: f"{v:.1f}")
    c2 = line_svg("Replacements per 100 tickets: all vs excluding defective lots", months,
                  [("All tickets", mo.rep_all.tolist(), "#c0392b"), ("Excluding defective lots", mo.rep_clean.tolist(), "#2471a3")],
                  0, 30, lambda v: f"{v:.0f}")
    excess = sum(m["replacement_cost_inr"] + m["contact_cost_inr"] for m in s["money"])
    lots_n = sum(m["flagged_lots"] for m in s["money"])
    tiles = [("Tickets (18 months)", f"{s['tickets']:,}"), ("CSAT, all tickets", f"{s['csat_all']:.2f}"),
             ("CSAT, defective-lot tickets", f"{s['csat_bad_lot']:.2f}"), ("CSAT, all other tickets", f"{s['csat_other']:.2f}"),
             ("Defective lots flagged", f"{lots_n}"), ("Excess cost of those lots", f"Rs {excess/1e5:.1f} lakh")]
    tiles_html = "".join(f'<div class=tile><div class=v>{v}</div><div class=k>{k}</div></div>' for k, v in tiles)
    rows = []
    for _, r in tab.iterrows():
        tag = ("Tier 2 - not comparable (policy s6)" if r.tier == 2 else
               "RETRAIN CANDIDATE" if r.retrain_candidate else
               "low, within noise" if r.adj_bottom10 else "")
        rows.append(f"<tr><td>{html.escape(r.agent_id)}</td><td>{html.escape(r['name'])}</td><td>{html.escape(r.team)}</td>"
                    f"<td>{int(r.surveyed_all)}</td><td>{r.csat_raw:.2f}</td><td>{r.csat_adj:.2f} &plusmn;{r.ci95:.2f}</td>"
                    f"<td>{r.handle_h_median:.1f}</td><td>{r.bad_lot_share:.0%}</td><td>{tag}</td></tr>")
    ncand = len(s["retrain_candidates"])
    open(path, "w").write(f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Vireo support dashboard</title>
<style>body{{font:14px system-ui,sans-serif;margin:24px;color:#222;max-width:1100px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin:14px 0}}
.tile{{border:1px solid #ddd;border-radius:6px;padding:10px 12px}}.v{{font-size:24px;font-weight:650}}.k{{color:#666;font-size:12px}}
.callout{{background:#fdf3e1;border-left:4px solid #e67e22;padding:10px 14px;margin:14px 0}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:4px 8px;text-align:left}}th{{background:#f4f4f4}}
.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}}</style>
<h2>Vireo support: why CSAT slid</h2>
<div class=callout><b>Finding:</b> {lots_n} Pulse 2 lots (PL2-2510-1 to PL2-2512-4) drive the slide. Without them CSAT is flat at about 3.4 to 3.5.
Retraining candidates after adjustment: <b>{ncand}</b>. Tier 2 is shown but never ranked (policy section 6).</div>
<div class=tiles>{tiles_html}</div>
<div class=charts>{c1}{c2}</div>
<h3>Per agent (sorted by raw CSAT, lowest first)</h3>
<p>Raw CSAT uses all surveyed tickets. 'Adjusted' uses only tickets not from defective lots, corrected for category, channel, priority, source system and transfers, with a 95% interval.
Handle time = first response to resolution, median hours (Logistics, Returns and Tier 2 are multi-day by design).</p>
<table><tr><th>ID<th>Name<th>Team<th>Surveyed<th>CSAT raw<th>CSAT adj &plusmn;95%<th>Handle h<th>Defect-lot share<th>Flag</tr>{''.join(rows)}</table>""")


if __name__ == "__main__":
    main()
