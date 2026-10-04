# Vireo support analysis

Per-agent CSAT and handle time, a defective-lot detector, and a mix-adjusted "who is actually a retraining candidate" list.
No LLM or paid API calls. One run is about 3 seconds and costs Rs 0.

## Run (clean machine, Python 3.9+)
    pip install -r requirements.txt
    python run.py --data PATH_TO_FOLDER_WITH_CSVS --out out
    python validate.py --data PATH_TO_FOLDER_WITH_CSVS
The data folder needs files ending in tickets.csv, agents.csv, orders.csv, products.csv (customers.csv is not used).
Open `out/dashboard.html` in a browser. Other outputs: `out/agent_table.csv`, `out/lots.csv`, `out/summary.json`.

## What it does
1. Fixes legacy timestamps (legacy resolved_at is UTC, shifted to IST; 2,309 impossible negative handle times become 0).
2. Joins tickets to lot codes (order_id, else customer+SKU latest prior order; 99% resolved).
3. Flags lots whose replacement rate is at least 2.5x the SKU's median lot (min 50 orders).
4. Ranks Tier-1 agents on CSAT from non-defective-lot tickets, adjusted for category, channel, priority, source system and transfers, with 95% intervals.
5. Tier 2 (Escalations & Warranty) is shown but never ranked (policy section 6).

## Decisions (no one to ask)
- Joined agents on agent_id (two agents are named Kavya Pandey).
- Blank CSAT is excluded, not zero. Open/pending tickets have no handle time.
- Junk IVR rows (13 found, not 40) are kept and flagged; they only affect the message text.
- Replacement cost = unit cost + Rs 340 (policy), not Arjun's Rs 2,500. Contact cost Rs 290 blended.

## Known limits
Limits: the lot flag thresholds (LOT_RATIO, LOT_MIN_ORDERS) and the MIN_N cutoff are judgement calls; adjusted CSAT is a linear model on a 1-5 score; about 5% of tickets without an order number may be joined to the wrong lot; the cause (a hardware fault in the lots) is inferred from replacement rates, not confirmed.
