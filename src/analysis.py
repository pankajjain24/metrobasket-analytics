"""
MetroBasket — analysis layer.

Answers five business questions and writes every result to output/metrics.json,
which is what the dashboard reads. Nothing here is hard-coded; change the data
and the dashboard changes with it.

  1. Where does the acquisition funnel leak, and on which device?
  2. How do monthly cohorts retain, and which channel buys customers worth keeping?
  3. Which customer segments hold the revenue? (RFM)
  4. Did the checkout redesign work? (A/B test, with segment read)
  5. What is late delivery actually costing us? (ops -> revenue link)

Run:  python src/analysis.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)

ANALYSIS_END = pd.Timestamp("2026-06-30")


# ----------------------------------------------------------------- load / clean
def load():
    cust = pd.read_csv(DATA / "customers.csv", parse_dates=["signup_date"])
    sess = pd.read_csv(DATA / "sessions.csv", parse_dates=["session_date"])
    orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
    reviews = pd.read_csv(DATA / "reviews.csv", parse_dates=["order_date"])
    ab = pd.read_csv(DATA / "ab_test.csv")

    log = []
    n0 = len(orders)
    orders = orders.drop_duplicates("order_id")
    log.append(f"dropped {n0 - len(orders)} duplicate order_ids")

    n0 = len(orders)
    orders = orders[orders.order_value_inr > 0]
    log.append(f"dropped {n0 - len(orders)} non-positive order values")

    orphans = (~orders.customer_id.isin(cust.customer_id)).sum()
    log.append(f"{orphans} orders with no matching customer (kept, flagged)")

    orders = orders[orders.order_date <= ANALYSIS_END]
    log.append(f"clipped orders to analysis window ending {ANALYSIS_END:%Y-%m-%d}")
    return cust, sess, orders, reviews, ab, log


# ------------------------------------------------------------------ 1. funnel
def funnel(sess):
    steps = ["viewed_product", "added_to_cart", "started_checkout", "completed_payment"]
    labels = ["Viewed a product", "Added to cart", "Started checkout", "Paid"]

    overall = [int(sess[s].sum()) for s in steps]
    by_device = {}
    for dev, g in sess.groupby("device"):
        counts = [int(g[s].sum()) for s in steps]
        by_device[dev] = {
            "counts": counts,
            "step_rate": [None] + [round(counts[i] / counts[i - 1] * 100, 1)
                                   for i in range(1, len(counts))],
            "overall_cvr": round(counts[-1] / counts[0] * 100, 2),
        }

    # Biggest single leak, measured in lost sessions vs the best device.
    best = max(by_device.items(), key=lambda kv: kv[1]["overall_cvr"])
    worst = min(by_device.items(), key=lambda kv: kv[1]["overall_cvr"])
    gap_sessions = int(by_device[worst[0]]["counts"][0]
                       * (best[1]["overall_cvr"] - worst[1]["overall_cvr"]) / 100)

    return {
        "labels": labels,
        "overall": overall,
        "overall_step_rate": [None] + [round(overall[i] / overall[i - 1] * 100, 1)
                                       for i in range(1, 4)],
        "overall_cvr": round(overall[-1] / overall[0] * 100, 2),
        "by_device": by_device,
        "best_device": best[0],
        "worst_device": worst[0],
        "recoverable_sessions": gap_sessions,
    }


# ------------------------------------------------------- 2. cohorts & channels
def cohorts(cust, orders, max_index=11):
    o = orders.merge(cust[["customer_id", "signup_month", "acquisition_channel",
                           "cac_inr"]], on="customer_id", how="inner")
    o["cohort"] = o["signup_month"]
    o["order_period"] = o["order_date"].dt.to_period("M")
    o["cohort_period"] = pd.PeriodIndex(o["cohort"], freq="M")
    o["month_index"] = (o["order_period"] - o["cohort_period"]).apply(lambda x: x.n)

    size = o.groupby("cohort")["customer_id"].nunique()
    active = (o.groupby(["cohort", "month_index"])["customer_id"].nunique()
              .reset_index(name="active"))
    active["cohort_size"] = active["cohort"].map(size)
    active["retention_pct"] = (active.active / active.cohort_size * 100).round(1)

    grid = (active[active.month_index <= max_index]
            .pivot(index="cohort", columns="month_index", values="retention_pct"))
    grid = grid.sort_index()

    rows = []
    for cohort, r in grid.iterrows():
        rows.append({
            "cohort": cohort,
            "size": int(size[cohort]),
            "values": [None if pd.isna(v) else float(v) for v in r.tolist()],
        })

    # Average retention curve, weighted by cohort size, only where mature.
    curve = []
    for i in range(max_index + 1):
        sub = active[active.month_index == i]
        curve.append(round(sub.active.sum() / sub.cohort_size.sum() * 100, 1)
                     if len(sub) else None)

    # Channel unit economics
    ch = []
    for name, g in cust.groupby("acquisition_channel"):
        go = orders[orders.customer_id.isin(g.customer_id)]
        buyers = go.customer_id.nunique()
        rev = go.order_value_inr.sum()
        spend = g.cac_inr.sum()
        repeat = (go.groupby("customer_id").size() > 1).mean() if buyers else 0
        ch.append({
            "channel": name,
            "customers": int(len(g)),
            "buyers": int(buyers),
            "buyer_rate": round(buyers / len(g) * 100, 1),
            "revenue": round(float(rev), 0),
            "spend": round(float(spend), 0),
            "aov": round(float(go.order_value_inr.mean()), 0) if buyers else 0,
            "orders_per_buyer": round(len(go) / buyers, 2) if buyers else 0,
            "repeat_rate": round(float(repeat) * 100, 1),
            "ltv": round(float(rev / len(g)), 0),
            "cac": round(float(g.cac_inr.mean()), 0),
            "ltv_cac": round(float(rev / spend), 2),
        })
    ch.sort(key=lambda d: -d["ltv_cac"])
    return {"grid": rows, "curve": curve, "max_index": max_index, "channels": ch}


# ---------------------------------------------------------------------- 3. RFM
def rfm(orders):
    snap = ANALYSIS_END + pd.Timedelta(days=1)
    t = orders.groupby("customer_id").agg(
        recency=("order_date", lambda s: (snap - s.max()).days),
        frequency=("order_id", "count"),
        monetary=("order_value_inr", "sum"))

    t["R"] = pd.qcut(t.recency, 4, labels=[4, 3, 2, 1]).astype(int)
    t["F"] = pd.qcut(t.frequency.rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    t["M"] = pd.qcut(t.monetary, 4, labels=[1, 2, 3, 4]).astype(int)

    def label(r):
        if r.R >= 3 and r.F >= 3:
            return "Champions"
        if r.R >= 3 and r.F == 2:
            return "Promising"
        if r.R >= 3:
            return "New / one-off"
        if r.F >= 3 and r.M >= 3:
            return "At risk — high value"
        if r.R <= 2 and r.F <= 2:
            return "Lapsed"
        return "Needs attention"

    t["segment"] = t.apply(label, axis=1)
    total_rev = t.monetary.sum()
    seg = (t.groupby("segment")
           .agg(customers=("segment", "size"),
                revenue=("monetary", "sum"),
                avg_orders=("frequency", "mean"),
                avg_recency=("recency", "mean"))
           .reset_index())
    seg["revenue_share"] = (seg.revenue / total_rev * 100).round(1)
    seg["customer_share"] = (seg.customers / len(t) * 100).round(1)
    seg["avg_value"] = (seg.revenue / seg.customers).round(0)
    seg = seg.sort_values("revenue", ascending=False)

    return {
        "segments": [{
            "segment": r.segment, "customers": int(r.customers),
            "revenue": round(float(r.revenue), 0),
            "revenue_share": float(r.revenue_share),
            "customer_share": float(r.customer_share),
            "avg_orders": round(float(r.avg_orders), 1),
            "avg_recency": int(r.avg_recency),
            "avg_value": float(r.avg_value),
        } for r in seg.itertuples()],
        "total_customers": int(len(t)),
        "total_revenue": round(float(total_rev), 0),
    }


# ----------------------------------------------------------------- 4. A/B test
def two_prop_test(c_conv, c_n, t_conv, t_n):
    p1, p2 = c_conv / c_n, t_conv / t_n
    pool = (c_conv + t_conv) / (c_n + t_n)
    se = np.sqrt(pool * (1 - pool) * (1 / c_n + 1 / t_n))
    z = (p2 - p1) / se if se else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    se_diff = np.sqrt(p1 * (1 - p1) / c_n + p2 * (1 - p2) / t_n)
    lo, hi = (p2 - p1) - 1.96 * se_diff, (p2 - p1) + 1.96 * se_diff
    return {
        "control_n": int(c_n), "treatment_n": int(t_n),
        "control_cvr": round(p1 * 100, 2), "treatment_cvr": round(p2 * 100, 2),
        "abs_lift_pp": round((p2 - p1) * 100, 2),
        "rel_lift_pct": round((p2 - p1) / p1 * 100, 1) if p1 else 0,
        "ci_low_pp": round(lo * 100, 2), "ci_high_pp": round(hi * 100, 2),
        "z": round(float(z), 2), "p_value": round(float(p), 4),
        "significant": bool(p < 0.05),
    }


def ab_test(ab):
    def agg(df):
        c = df[df.variant == "control"]
        t = df[df.variant == "treatment"]
        return two_prop_test(c.converted.sum(), len(c), t.converted.sum(), len(t))

    overall = agg(ab)
    by_device = {dev: agg(g) for dev, g in ab.groupby("device")}

    # Sample size needed to detect the observed lift at 80% power, for context.
    p1 = overall["control_cvr"] / 100
    delta = abs(overall["abs_lift_pp"]) / 100
    n_req = (int(np.ceil(((1.96 + 0.84) ** 2 * 2 * p1 * (1 - p1)) / delta ** 2))
             if delta > 0 else None)

    rev_c = ab.loc[ab.variant == "control", "revenue_inr"].sum() / (ab.variant == "control").sum()
    rev_t = ab.loc[ab.variant == "treatment", "revenue_inr"].sum() / (ab.variant == "treatment").sum()
    return {
        "overall": overall,
        "by_device": by_device,
        "required_n_per_arm": n_req,
        "rev_per_user_control": round(float(rev_c), 2),
        "rev_per_user_treatment": round(float(rev_t), 2),
    }


# ------------------------------------------------- 5. delivery -> revenue link
def delivery_impact(cust, orders, reviews):
    o = orders.sort_values(["customer_id", "order_date"]).copy()
    o["next_date"] = o.groupby("customer_id")["order_date"].shift(-1)
    o["reordered"] = o["next_date"].notna().astype(int)
    o["days_to_next"] = (o["next_date"] - o["order_date"]).dt.days

    grp = o.groupby("delivered_late").agg(
        orders=("order_id", "count"),
        reorder_rate=("reordered", "mean"),
        days_to_next=("days_to_next", "median"))
    on_time = grp.loc[0]
    late = grp.loc[1]

    gap = float(on_time.reorder_rate - late.reorder_rate)
    aov = float(orders.order_value_inr.mean())
    lost_orders = float(late.orders) * gap
    revenue_at_risk = lost_orders * aov

    themes = (reviews.review_theme.value_counts(normalize=True) * 100).round(1)
    theme_rows = [{"theme": k.replace("_", " ").title(), "pct": float(v),
                   "count": int(reviews.review_theme.value_counts()[k])}
                  for k, v in themes.items()]

    neg = reviews[reviews.rating <= 3]
    theme_by_rating = (neg.review_theme.value_counts(normalize=True) * 100).round(1)

    monthly = (orders.groupby("order_month")
               .agg(orders=("order_id", "count"),
                    revenue=("order_value_inr", "sum"),
                    late_rate=("delivered_late", "mean"))
               .reset_index())
    monthly["revenue"] = monthly.revenue.round(0)
    monthly["late_rate"] = (monthly.late_rate * 100).round(1)

    return {
        "on_time_reorder_rate": round(float(on_time.reorder_rate) * 100, 1),
        "late_reorder_rate": round(float(late.reorder_rate) * 100, 1),
        "reorder_gap_pp": round(gap * 100, 1),
        "on_time_days_to_next": float(on_time.days_to_next),
        "late_days_to_next": float(late.days_to_next),
        "late_orders": int(late.orders),
        "late_rate_overall": round(float(orders.delivered_late.mean()) * 100, 1),
        "aov": round(aov, 0),
        "lost_orders": int(lost_orders),
        "revenue_at_risk": round(revenue_at_risk, 0),
        "themes": theme_rows,
        "negative_themes": [{"theme": k.replace("_", " ").title(), "pct": float(v)}
                            for k, v in theme_by_rating.items()],
        "monthly": monthly.to_dict("records"),
        "reviews_analysed": int(len(reviews)),
    }


# ---------------------------------------------------------------------- headline
def headline(cust, orders, fn, dl):
    return {
        "customers": int(len(cust)),
        "buyers": int(orders.customer_id.nunique()),
        "orders": int(len(orders)),
        "revenue": round(float(orders.order_value_inr.sum()), 0),
        "aov": round(float(orders.order_value_inr.mean()), 0),
        "overall_cvr": fn["overall_cvr"],
        "repeat_rate": round(float((orders.groupby("customer_id").size() > 1).mean()) * 100, 1),
        "late_rate": dl["late_rate_overall"],
        "window": f"{orders.order_date.min():%b %Y} – {orders.order_date.max():%b %Y}",
    }


if __name__ == "__main__":
    cust, sess, orders, reviews, ab, cleaning_log = load()

    fn = funnel(sess)
    co = cohorts(cust, orders)
    seg = rfm(orders)
    test = ab_test(ab)
    dl = delivery_impact(cust, orders, reviews)

    metrics = {
        "headline": headline(cust, orders, fn, dl),
        "funnel": fn,
        "cohorts": co,
        "rfm": seg,
        "ab_test": test,
        "delivery": dl,
        "cleaning_log": cleaning_log,
    }

    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    h = metrics["headline"]
    print(f"\n{'MetroBasket — analysis summary':-^64}")
    print(f"  Window            {h['window']}")
    print(f"  Customers         {h['customers']:,}  ({h['buyers']:,} bought)")
    print(f"  Orders / revenue  {h['orders']:,} / ₹{h['revenue']:,.0f}")
    print(f"  Signup→pay CVR    {h['overall_cvr']}%   worst surface: {fn['worst_device']}")
    print(f"  Repeat rate       {h['repeat_rate']}%")
    print(f"  Late deliveries   {h['late_rate']}%  → ₹{dl['revenue_at_risk']:,.0f} at risk")
    print(f"  Best LTV:CAC      {co['channels'][0]['channel']} "
          f"({co['channels'][0]['ltv_cac']}x)  worst: "
          f"{co['channels'][-1]['channel']} ({co['channels'][-1]['ltv_cac']}x)")
    print(f"  A/B overall       {test['overall']['abs_lift_pp']:+}pp, "
          f"p={test['overall']['p_value']}")
    for d, r in test["by_device"].items():
        print(f"     {d:<12} {r['abs_lift_pp']:+}pp  p={r['p_value']}  "
              f"{'significant' if r['significant'] else 'not significant'}")
    print(f"\n  Wrote {OUT / 'metrics.json'}\n")
