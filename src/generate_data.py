"""
MetroBasket — synthetic dataset generator.

Builds a realistic 18-month transaction log for an Indian online grocery &
essentials marketplace. Realistic means: skewed order values, seasonal spikes,
channel-dependent customer quality, delivery failures that actually cost money,
and messy free-text reviews.

Writes five CSVs into ../data/. Everything is seeded, so the numbers are
reproducible.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
DATA = Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2025-01-01")
END = pd.Timestamp("2026-06-30")
N_CUSTOMERS = 24_000

CITIES = ["Bengaluru", "Mumbai", "Delhi NCR", "Hyderabad", "Pune", "Chennai", "Kolkata"]
CITY_W = [0.22, 0.19, 0.18, 0.13, 0.11, 0.10, 0.07]

# Channel quality is deliberately uneven. Discount affiliates buy volume that
# never comes back; referral is small but excellent. This is the finding.
CHANNELS = {
    #                  share  CAC(INR)  repeat_pref  aov_mult
    "Organic Search": (0.24, 180, 1.35, 1.05),
    "Paid Social": (0.26, 520, 0.85, 0.95),
    "Discount Affiliates": (0.21, 410, 0.45, 0.78),
    "Referral": (0.11, 120, 1.70, 1.18),
    "Email / CRM": (0.10, 90, 1.45, 1.10),
    "Marketplace Ads": (0.08, 640, 0.80, 1.00),
}

CATEGORIES = ["Fresh Produce", "Staples & Grains", "Dairy & Eggs",
              "Packaged Snacks", "Household Care", "Personal Care", "Beverages"]
CAT_W = [0.21, 0.18, 0.16, 0.15, 0.12, 0.10, 0.08]


def seasonal_multiplier(dates: pd.Series) -> np.ndarray:
    """Festive lift (Oct-Nov), summer beverage lift, Jan slump."""
    m = dates.dt.month.to_numpy()
    mult = np.ones(len(m))
    mult += np.where(np.isin(m, [10, 11]), 0.38, 0)      # Diwali / festive
    mult += np.where(np.isin(m, [4, 5]), 0.15, 0)        # summer
    mult -= np.where(m == 1, 0.12, 0)                    # post-festive slump
    return mult


def build_customers() -> pd.DataFrame:
    names = list(CHANNELS)
    shares = [CHANNELS[c][0] for c in names]
    channel = RNG.choice(names, size=N_CUSTOMERS, p=np.array(shares) / sum(shares))

    # Acquisition ramps over time; affiliates were pushed hard in H2.
    day_offset = RNG.beta(1.6, 1.9, N_CUSTOMERS) * (END - START).days
    signup = START + pd.to_timedelta(day_offset.astype(int), unit="D")
    affiliate_push = (channel == "Discount Affiliates") & (RNG.random(N_CUSTOMERS) < 0.55)
    signup = pd.Series(signup)
    signup[affiliate_push] = START + pd.to_timedelta(
        (270 + RNG.random(affiliate_push.sum()) * 270).astype(int), unit="D")

    df = pd.DataFrame({
        "customer_id": [f"C{100000 + i}" for i in range(N_CUSTOMERS)],
        "signup_date": signup.dt.normalize().clip(upper=END),
        "acquisition_channel": channel,
        "city": RNG.choice(CITIES, N_CUSTOMERS, p=CITY_W),
        "device": RNG.choice(["Mobile App", "Mobile Web", "Desktop"],
                             N_CUSTOMERS, p=[0.58, 0.27, 0.15]),
        "cac_inr": [CHANNELS[c][1] * RNG.normal(1, 0.18) for c in channel],
    })
    df["cac_inr"] = df["cac_inr"].clip(lower=40).round(0)
    df["signup_month"] = df["signup_date"].dt.to_period("M").astype(str)
    return df


def build_sessions(cust: pd.DataFrame) -> pd.DataFrame:
    """One acquisition-funnel session per customer, plus browsing sessions."""
    rows = []
    n = len(cust)
    # Every signup produces a first session. Drop-off is device dependent:
    # mobile web checkout is genuinely broken.
    device = cust["device"].to_numpy()
    p_cart = np.where(device == "Desktop", 0.62, np.where(device == "Mobile App", 0.58, 0.51))
    p_checkout = np.where(device == "Desktop", 0.74, np.where(device == "Mobile App", 0.69, 0.55))
    p_pay = np.where(device == "Desktop", 0.81, np.where(device == "Mobile App", 0.76, 0.58))

    viewed = np.ones(n, dtype=bool)
    carted = RNG.random(n) < p_cart
    checked = carted & (RNG.random(n) < p_checkout)
    paid = checked & (RNG.random(n) < p_pay)

    rows.append(pd.DataFrame({
        "session_id": [f"S{i:07d}" for i in range(n)],
        "customer_id": cust["customer_id"],
        "session_date": cust["signup_date"],
        "device": device,
        "viewed_product": viewed.astype(int),
        "added_to_cart": carted.astype(int),
        "started_checkout": checked.astype(int),
        "completed_payment": paid.astype(int),
    }))
    return pd.concat(rows, ignore_index=True)


def build_orders(cust: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    first_buyers = sessions.loc[sessions.completed_payment == 1, "customer_id"]
    buyers = cust[cust.customer_id.isin(first_buyers)].reset_index(drop=True)

    orders = []
    for _, c in buyers.iterrows():
        pref = CHANNELS[c.acquisition_channel][2]
        aov_mult = CHANNELS[c.acquisition_channel][3]
        # Lifetime order count: negative-binomial-ish, channel weighted.
        lam = max(0.6, RNG.gamma(1.9, 0.95) * pref)
        months_alive = max(1, (END - c.signup_date).days / 30.0)
        n_orders = 1 + RNG.poisson(min(lam * months_alive * 0.42, 26))

        t = c.signup_date
        bad_experience = 0
        for k in range(int(n_orders)):
            if k > 0:
                gap = max(2, RNG.gamma(2.2, 11 / max(pref, 0.4)))
                # A late delivery pushes the next order further out, or ends it.
                gap *= 1 + 0.8 * bad_experience
                t = t + pd.Timedelta(days=float(gap))
                if t > END:
                    break
                if bad_experience and RNG.random() < 0.42:
                    break  # churned after a bad delivery

            promised = int(RNG.choice([1, 2, 3], p=[0.45, 0.4, 0.15]))
            # Delivery reliability degrades in festive season and in Delhi NCR.
            stress = 1.0 + (0.55 if t.month in (10, 11) else 0) + (0.3 if c.city == "Delhi NCR" else 0)
            actual = promised + max(0, int(RNG.poisson(0.32 * stress) * RNG.choice([0, 1, 1, 2, 3])))
            late = int(actual > promised)
            bad_experience = late

            base = RNG.lognormal(6.45, 0.52) * aov_mult
            base *= seasonal_multiplier(pd.Series([t]))[0]
            orders.append((
                f"O{len(orders):07d}", c.customer_id, t.normalize(),
                round(float(np.clip(base, 149, 12000)), 2),
                RNG.choice(CATEGORIES, p=CAT_W),
                promised, actual, late,
                RNG.choice(["UPI", "Card", "Cash on Delivery", "Wallet"], p=[.52, .21, .18, .09]),
            ))

    df = pd.DataFrame(orders, columns=[
        "order_id", "customer_id", "order_date", "order_value_inr", "category",
        "promised_days", "actual_days", "delivered_late", "payment_method"])
    df["order_month"] = df["order_date"].dt.to_period("M").astype(str)
    return df


REVIEW_BANK = {
    "delivery_delay": [
        "Order was promised in 2 days but arrived on day 5, milk had turned.",
        "Third time this month the delivery slot was missed completely.",
        "Delivery partner kept rescheduling, no update on the app at all.",
        "Late again. Festive season is no excuse when I paid for express.",
    ],
    "product_quality": [
        "Tomatoes were soft and two of the mangoes were bruised.",
        "Bread was close to expiry date on arrival.",
        "Packaging was fine but the produce clearly was not fresh.",
    ],
    "pricing": [
        "Prices have quietly gone up compared to the local store.",
        "The coupon did not apply at checkout even though it showed as eligible.",
        "Delivery fee added at the last step, felt like a bait and switch.",
    ],
    "app_experience": [
        "Payment page froze twice on mobile before the order went through.",
        "Cart emptied itself when I switched tabs, had to redo everything.",
        "Search never finds the brand I want, filters are useless.",
    ],
    "positive": [
        "Arrived early and everything was fresh, very happy.",
        "Support resolved my missing item within an hour, no arguments.",
        "Good prices on staples and the app is quick to reorder from.",
        "Consistently on time for the last few months, well done.",
    ],
}


def build_reviews(orders: pd.DataFrame) -> pd.DataFrame:
    """~28% of orders reviewed. Theme depends on what actually went wrong."""
    sample = orders.sample(frac=0.28, random_state=7).copy()
    themes, texts, ratings = [], [], []
    for late in sample["delivered_late"]:
        r = RNG.random()
        if late and r < 0.62:
            theme = "delivery_delay"
        elif r < 0.70:
            theme = RNG.choice(["product_quality", "pricing", "app_experience"], p=[.45, .3, .25])
        else:
            theme = "positive"
        themes.append(theme)
        texts.append(RNG.choice(REVIEW_BANK[theme]))
        ratings.append(int(RNG.choice([4, 5], p=[.35, .65])) if theme == "positive"
                       else int(RNG.choice([1, 2, 3], p=[.42, .38, .20])))
    sample["review_theme"] = themes
    sample["review_text"] = texts
    sample["rating"] = ratings
    sample["review_id"] = [f"R{i:06d}" for i in range(len(sample))]
    return sample[["review_id", "order_id", "customer_id", "order_date",
                   "rating", "review_theme", "review_text"]]


def build_ab_test(cust: pd.DataFrame) -> pd.DataFrame:
    """
    Checkout redesign, 6-week run, 50/50 split, only users active in the window.
    True effect: +3.1pp on mobile web (the broken surface), ~0 on desktop.
    A naive overall read looks like a win; the segment read is the real story.
    """
    pool = cust.sample(18_000, random_state=11).copy()
    pool["variant"] = RNG.choice(["control", "treatment"], len(pool), p=[.5, .5])

    base = np.where(pool.device == "Desktop", 0.142,
                    np.where(pool.device == "Mobile App", 0.118, 0.081))
    lift = np.where(pool.device == "Desktop", 0.001,
                    np.where(pool.device == "Mobile App", 0.012, 0.031))
    p = base + np.where(pool.variant == "treatment", lift, 0)
    pool["converted"] = (RNG.random(len(pool)) < p).astype(int)
    pool["revenue_inr"] = np.where(
        pool.converted == 1, RNG.lognormal(6.5, 0.5, len(pool)).round(2), 0.0)
    return pool[["customer_id", "device", "city", "variant", "converted", "revenue_inr"]]


if __name__ == "__main__":
    print("Generating customers…")
    customers = build_customers()
    print("Generating sessions…")
    sessions = build_sessions(customers)
    print("Generating orders (this is the slow one)…")
    orders = build_orders(customers, sessions)
    print("Generating reviews…")
    reviews = build_reviews(orders)
    print("Generating A/B test…")
    ab = build_ab_test(customers)

    for name, df in [("customers", customers), ("sessions", sessions),
                     ("orders", orders), ("reviews", reviews), ("ab_test", ab)]:
        path = DATA / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"  {path.name:<16} {len(df):>8,} rows")
    print("\nDone.")
