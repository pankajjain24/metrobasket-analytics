-- =============================================================================
-- MetroBasket — analyst SQL
--
-- These are the same five questions the Python layer answers, written the way
-- you would actually write them against a warehouse. Runnable as-is with
-- DuckDB (`python src/run_sql.py`), and portable to Postgres / BigQuery /
-- Snowflake with only date-function changes (noted inline).
-- =============================================================================


-- -----------------------------------------------------------------------------
-- 0. Views over the raw files. In a warehouse these would be your staging models.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW customers AS SELECT * FROM read_csv_auto('data/customers.csv');
CREATE OR REPLACE VIEW sessions  AS SELECT * FROM read_csv_auto('data/sessions.csv');
CREATE OR REPLACE VIEW orders    AS SELECT * FROM read_csv_auto('data/orders.csv');
CREATE OR REPLACE VIEW reviews   AS SELECT * FROM read_csv_auto('data/reviews.csv');
CREATE OR REPLACE VIEW ab_test   AS SELECT * FROM read_csv_auto('data/ab_test.csv');


-- -----------------------------------------------------------------------------
-- Q1. Where does the signup -> payment funnel leak, and on which surface?
--
-- Unpivots the four step flags into rows so the funnel is one tidy result set
-- instead of four columns you have to eyeball. step_rate is the conversion from
-- the previous step; LAG does the work.
-- -----------------------------------------------------------------------------
WITH steps AS (
    SELECT device, 1 AS step_no, 'Viewed a product' AS step, SUM(viewed_product)    AS sessions FROM sessions GROUP BY device
    UNION ALL
    SELECT device, 2, 'Added to cart',    SUM(added_to_cart)     FROM sessions GROUP BY device
    UNION ALL
    SELECT device, 3, 'Started checkout', SUM(started_checkout)  FROM sessions GROUP BY device
    UNION ALL
    SELECT device, 4, 'Paid',             SUM(completed_payment) FROM sessions GROUP BY device
)
SELECT
    device,
    step,
    sessions,
    ROUND(100.0 * sessions
          / LAG(sessions) OVER (PARTITION BY device ORDER BY step_no), 1) AS step_rate_pct,
    ROUND(100.0 * sessions
          / FIRST_VALUE(sessions) OVER (PARTITION BY device ORDER BY step_no), 1) AS cumulative_pct
FROM steps
ORDER BY device, step_no;


-- -----------------------------------------------------------------------------
-- Q2a. Monthly cohort retention.
--
-- month_index = months between the customer's signup month and the order month.
-- The self-join to cohort_size is what turns counts into a comparable %.
-- Postgres: date_trunc('month', ...); BigQuery: DATE_TRUNC(d, MONTH).
-- -----------------------------------------------------------------------------
WITH base AS (
    SELECT
        c.customer_id,
        c.signup_month                                          AS cohort,
        STRFTIME(o.order_date, '%Y-%m')                          AS order_month,
        DATE_DIFF('month', DATE_TRUNC('month', c.signup_date),
                           DATE_TRUNC('month', o.order_date))    AS month_index
    FROM customers c
    JOIN orders    o ON o.customer_id = c.customer_id
),
cohort_size AS (
    SELECT cohort, COUNT(DISTINCT customer_id) AS n_customers
    FROM base
    WHERE month_index = 0
    GROUP BY cohort
),
active AS (
    SELECT cohort, month_index, COUNT(DISTINCT customer_id) AS n_active
    FROM base
    WHERE month_index BETWEEN 0 AND 11
    GROUP BY cohort, month_index
)
SELECT
    a.cohort,
    s.n_customers                                       AS cohort_size,
    a.month_index,
    a.n_active,
    ROUND(100.0 * a.n_active / s.n_customers, 1)        AS retention_pct
FROM active a
JOIN cohort_size s USING (cohort)
ORDER BY a.cohort, a.month_index;


-- -----------------------------------------------------------------------------
-- Q2b. Channel unit economics — the acquisition-spend question.
--
-- LTV here is revenue per acquired customer (not per buyer), because the
-- channel is charged for everyone it brings in, including the ones who never
-- convert. That choice is the whole point of the query.
-- -----------------------------------------------------------------------------
WITH spend AS (
    SELECT acquisition_channel, COUNT(*) AS customers, SUM(cac_inr) AS total_cac
    FROM customers
    GROUP BY acquisition_channel
),
revenue AS (
    SELECT
        c.acquisition_channel,
        COUNT(DISTINCT o.customer_id) AS buyers,
        COUNT(*)                      AS orders,
        SUM(o.order_value_inr)        AS revenue,
        AVG(o.order_value_inr)        AS aov
    FROM customers c
    JOIN orders    o ON o.customer_id = c.customer_id
    GROUP BY c.acquisition_channel
)
SELECT
    s.acquisition_channel                                   AS channel,
    s.customers,
    r.buyers,
    ROUND(100.0 * r.buyers / s.customers, 1)                AS buyer_rate_pct,
    ROUND(r.orders * 1.0 / r.buyers, 2)                     AS orders_per_buyer,
    ROUND(r.aov)                                            AS aov_inr,
    ROUND(r.revenue)                                        AS revenue_inr,
    ROUND(s.total_cac)                                      AS spend_inr,
    ROUND(r.revenue / s.customers)                          AS ltv_per_acquired,
    ROUND(s.total_cac / s.customers)                        AS cac_per_acquired,
    ROUND(r.revenue / NULLIF(s.total_cac, 0), 2)            AS ltv_cac_ratio
FROM spend s
JOIN revenue r USING (acquisition_channel)
ORDER BY ltv_cac_ratio DESC;


-- -----------------------------------------------------------------------------
-- Q3. RFM segmentation.
--
-- NTILE splits each dimension into quartiles; recency is reversed because
-- fewer days since the last order is better. The CASE ladder is ordered
-- deliberately — first match wins, so put the most specific rules on top.
-- -----------------------------------------------------------------------------
WITH snapshot AS (SELECT DATE '2026-07-01' AS as_of),
per_customer AS (
    SELECT
        o.customer_id,
        DATE_DIFF('day', MAX(o.order_date), (SELECT as_of FROM snapshot)) AS recency_days,
        COUNT(*)                                                          AS frequency,
        SUM(o.order_value_inr)                                            AS monetary
    FROM orders o
    GROUP BY o.customer_id
),
scored AS (
    SELECT *,
        NTILE(4) OVER (ORDER BY recency_days DESC) AS r_score,
        NTILE(4) OVER (ORDER BY frequency)         AS f_score,
        NTILE(4) OVER (ORDER BY monetary)          AS m_score
    FROM per_customer
),
labelled AS (
    SELECT *,
        CASE
            WHEN r_score >= 3 AND f_score >= 3               THEN 'Champions'
            WHEN r_score >= 3 AND f_score  = 2               THEN 'Promising'
            WHEN r_score >= 3                                THEN 'New / one-off'
            WHEN f_score >= 3 AND m_score >= 3               THEN 'At risk - high value'
            WHEN r_score <= 2 AND f_score <= 2               THEN 'Lapsed'
            ELSE 'Needs attention'
        END AS segment
    FROM scored
)
SELECT
    segment,
    COUNT(*)                                                             AS customers,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)                   AS customer_share_pct,
    ROUND(SUM(monetary))                                                 AS revenue_inr,
    ROUND(100.0 * SUM(monetary) / SUM(SUM(monetary)) OVER (), 1)         AS revenue_share_pct,
    ROUND(AVG(frequency), 1)                                             AS avg_orders,
    ROUND(AVG(recency_days))                                             AS avg_days_since_order
FROM labelled
GROUP BY segment
ORDER BY revenue_inr DESC;


-- -----------------------------------------------------------------------------
-- Q4. Checkout A/B test — overall and by device.
--
-- SQL gets you the rates and the lift; the significance test lives in Python
-- (scipy). Reporting the segment rows next to the overall row is what stops
-- someone shipping an average that hides a flat desktop result.
-- -----------------------------------------------------------------------------
WITH arm AS (
    SELECT device, variant,
           COUNT(*)                                  AS users,
           SUM(converted)                            AS conversions,
           100.0 * SUM(converted) / COUNT(*)         AS cvr_pct,
           SUM(revenue_inr) / COUNT(*)               AS rev_per_user
    FROM ab_test
    GROUP BY GROUPING SETS ((device, variant), (variant))   -- device + overall in one pass
)
SELECT
    COALESCE(device, 'ALL DEVICES')                                        AS surface,
    MAX(CASE WHEN variant = 'control'   THEN users END)                    AS control_users,
    MAX(CASE WHEN variant = 'treatment' THEN users END)                    AS treatment_users,
    ROUND(MAX(CASE WHEN variant = 'control'   THEN cvr_pct END), 2)        AS control_cvr_pct,
    ROUND(MAX(CASE WHEN variant = 'treatment' THEN cvr_pct END), 2)        AS treatment_cvr_pct,
    ROUND(MAX(CASE WHEN variant = 'treatment' THEN cvr_pct END)
        - MAX(CASE WHEN variant = 'control'   THEN cvr_pct END), 2)        AS abs_lift_pp
FROM arm
GROUP BY COALESCE(device, 'ALL DEVICES')
ORDER BY surface;


-- -----------------------------------------------------------------------------
-- Q5. What does a late delivery cost?
--
-- Uses LEAD to find each customer's next order, then compares the reorder rate
-- of late vs on-time deliveries. This is the query that turns an ops metric
-- into a rupee number a business owner will act on.
-- -----------------------------------------------------------------------------
WITH sequenced AS (
    SELECT
        customer_id,
        order_id,
        order_date,
        order_value_inr,
        delivered_late,
        LEAD(order_date) OVER (PARTITION BY customer_id ORDER BY order_date) AS next_order_date
    FROM orders
),
flagged AS (
    SELECT *,
        CASE WHEN next_order_date IS NOT NULL THEN 1 ELSE 0 END          AS reordered,
        DATE_DIFF('day', order_date, next_order_date)                     AS days_to_next
    FROM sequenced
)
SELECT
    CASE delivered_late WHEN 1 THEN 'Delivered late' ELSE 'On time' END   AS delivery_outcome,
    COUNT(*)                                                              AS orders,
    ROUND(100.0 * AVG(reordered), 1)                                      AS reorder_rate_pct,
    ROUND(MEDIAN(days_to_next), 1)                                        AS median_days_to_next,
    ROUND(AVG(order_value_inr))                                           AS aov_inr
FROM flagged
GROUP BY delivered_late
ORDER BY delivered_late;


-- -----------------------------------------------------------------------------
-- Q5b. Which complaint themes dominate negative reviews, and are they the ones
--      tied to late delivery? Joins free-text themes back to the order record.
-- -----------------------------------------------------------------------------
SELECT
    r.review_theme,
    COUNT(*)                                                       AS reviews,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)             AS share_pct,
    ROUND(AVG(r.rating), 2)                                        AS avg_rating,
    ROUND(100.0 * AVG(o.delivered_late), 1)                        AS pct_from_late_orders
FROM reviews r
JOIN orders  o ON o.order_id = r.order_id
GROUP BY r.review_theme
ORDER BY reviews DESC;
