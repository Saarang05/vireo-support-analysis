"""Checks of the tool's own output. python validate.py --data DIR"""
import argparse, importlib.util, numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); A = ap.parse_args()
spec = importlib.util.spec_from_file_location("r", "run.py"); r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
t, o, p, ag = r.load(A.data); t = r.clean(t)

# 1. timestamp fix: handle time must be non-negative after shifting legacy UTC -> IST
print("negative handle time after fix:", int((t.handle_h < 0).sum()), "of", int(t.handle_h.notna().sum()))
raw = r.load(A.data)[0]; raw["h"] = (raw.resolved_at - raw.first_response_at).dt.total_seconds() / 3600
print("negative before fix:", int((raw.h < 0).sum()))

# 2. lot-join accuracy: hide order_id where known, re-join by customer+sku fallback, compare to truth
k = t[t.order_id.notna()].copy(); truth = k.merge(o[["order_id", "lot_code"]], on="order_id").set_index("ticket_id").lot_code
h = k.assign(order_id=np.nan); h = r.attach_lots(h, o).set_index("ticket_id").lot_code
common = truth.index.intersection(h.index); acc = (truth[common] == h[common]).mean()
print(f"fallback join accuracy on {len(common)} tickets: {acc:.3f} (error {1-acc:.3f})")

# 3. split-half reliability of agent ranking (are rankings signal or noise?)
t = r.attach_lots(t, o); lots = r.flag_lots(t, o); bad = set(map(tuple, lots[lots.defect_flag][["product_sku", "lot_code"]].values))
t["bad_lot"] = [(s, l) in bad for s, l in zip(t.product_sku, t.lot_code)]
s = t[t.csat_score.notna()]
rs = []
for seed in range(200):
    rng = np.random.default_rng(seed); m = rng.random(len(s)) < .5
    a = s[m].groupby("agent_id").csat_score.mean(); b = s[~m].groupby("agent_id").csat_score.mean()
    rs.append(a.corr(b))
print("split-half corr of raw agent CSAT (all tickets): mean %.2f" % np.mean(rs))
c = s[~s.bad_lot]; ids = c.groupby("agent_id").size(); ids = ids[ids >= 40].index; c = c[c.agent_id.isin(ids)]
rs = []
for seed in range(200):
    rng = np.random.default_rng(seed); m = rng.random(len(c)) < .5
    a = c[m].groupby("agent_id").csat_score.mean(); b = c[~m].groupby("agent_id").csat_score.mean()
    rs.append(a.corr(b))
print("split-half corr, Tier-1 agents, defect-lot tickets removed: mean %.2f" % np.mean(rs))

# 4. lot flag is not marginal: smallest flagged rate vs largest unflagged rate (per SKU)
l = lots[lots.orders >= r.LOT_MIN_ORDERS]
for sku, g in l.groupby("product_sku"):
    if g.defect_flag.any(): print(sku, "min flagged rate %.2f, max unflagged %.2f" % (g[g.defect_flag].rep_rate.min(), g[~g.defect_flag].rep_rate.max()))
