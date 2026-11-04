# Stage 2 — Early levetiracetam and ICU delirium after acute brain injury (MIMIC-IV v3.1)
# Runs inside the authenticated Colab runtime from stage 1 (uses client, q, run, PROJECT_ID, WORK, HOSP, ICU).
# PRINTS AGGREGATE OUTPUT ONLY. Counts 1-9 are masked as "<10".

import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

DER = HOSP.replace("_hosp", "_derived")
C1 = f"`{PROJECT_ID}.{WORK}.cohort_stage1`"
C2 = f"`{PROJECT_ID}.{WORK}.cohort_stage2`"
DELIRIUM_ITEM = 228332   # "Delirium assessment": Positive / Negative / UTA
RASS_ITEM = 228096       # "Richmond-RAS Scale"


def mask(df, min_n=10):
    out = df.copy().astype(object)
    for c in out.columns:
        out[c] = out[c].map(lambda v: "<10" if isinstance(v, (int, np.integer)) and 0 < v < min_n else v)
    return out


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ---------------------------------------------------------------- 0. derived tables present?
section("0. Derived tables check")
have = set(q(f"SELECT table_name FROM `{DER}.INFORMATION_SCHEMA.TABLES`")["table_name"])
need = ["first_day_gcs", "first_day_sofa", "charlson", "ventilation"]
print({t: (t in have) for t in need})

# ---------------------------------------------------------------- 1. final cohort
run(f"""
CREATE OR REPLACE TABLE {C2} AS
WITH seiz AS (   -- seizure / epilepsy codes on the INDEX admission (treatment, not prophylaxis)
  SELECT DISTINCT hadm_id FROM `{HOSP}.diagnoses_icd`
  WHERE (icd_version = 9  AND (STARTS_WITH(icd_code, '345') OR STARTS_WITH(icd_code, '7803')))
     OR (icd_version = 10 AND REGEXP_CONTAINS(icd_code, r'^(G40|G41|R56)'))
),
aud AS (
  SELECT DISTINCT hadm_id FROM `{HOSP}.diagnoses_icd`
  WHERE (icd_version = 9  AND REGEXP_CONTAINS(icd_code, r'^(291|303|3050)'))
     OR (icd_version = 10 AND STARTS_WITH(icd_code, 'F10'))
)
SELECT c.*,
  (sz.hadm_id IS NOT NULL) AS seizure_index_adm,
  (au.hadm_id IS NOT NULL) AS alcohol_use_disorder
FROM {C1} c
LEFT JOIN seiz sz ON sz.hadm_id = c.hadm_id
LEFT JOIN aud  au ON au.hadm_id = c.hadm_id
WHERE c.in_cohort
""")

flow = q(f"""
SELECT
  COUNT(*)                                                                   AS eligible_stage1,
  COUNTIF(has_emar)                                                          AS with_emar,
  COUNTIF(has_emar AND los >= 2)                                             AS icu_ge_48h,
  COUNTIF(has_emar AND los >= 2 AND NOT seizure_index_adm)                   AS no_seizure_code,
  COUNTIF(has_emar AND los >= 2 AND NOT seizure_index_adm AND NOT other_asm_48h) AS final_cohort
FROM {C2}
""")
section("1. Flowchart")
print(mask(flow).T.to_string(header=False))

run(f"""
CREATE OR REPLACE TABLE {C2} AS
SELECT * FROM {C2}
WHERE has_emar AND los >= 2 AND NOT seizure_index_adm AND NOT other_asm_48h
""")

# ---------------------------------------------------------------- 2. analysis dataset (stays in Colab memory; never printed)
df = q(f"""
WITH base AS (SELECT * FROM {C2}),
delir AS (
  SELECT ce.stay_id,
    DATETIME_DIFF(ce.charttime, b.intime, HOUR) AS hr,
    ce.value
  FROM `{ICU}.chartevents` ce JOIN base b USING (stay_id)
  WHERE ce.itemid = {DELIRIUM_ITEM} AND ce.charttime <= b.outtime
),
rass AS (
  SELECT ce.stay_id,
    DATETIME_DIFF(ce.charttime, b.intime, HOUR) AS hr,
    ce.valuenum
  FROM `{ICU}.chartevents` ce JOIN base b USING (stay_id)
  WHERE ce.itemid = {RASS_ITEM} AND ce.valuenum IS NOT NULL AND ce.charttime <= b.outtime
),
d_agg AS (
  SELECT stay_id,
    LOGICAL_OR(hr <  48 AND value = 'Positive')                         AS delirium_0_48,
    LOGICAL_OR(hr >= 48 AND hr < 168 AND value = 'Positive')            AS delirium_48_168,
    LOGICAL_OR(hr >= 48 AND hr < 168 AND value IN ('Positive','Negative')) AS assessed_48_168
  FROM delir GROUP BY stay_id
),
r_agg AS (
  SELECT stay_id,
    LOGICAL_OR(hr <  48 AND valuenum <= -4)              AS coma_0_48,
    LOGICAL_OR(hr >= 48 AND hr < 168 AND valuenum <= -4) AS coma_48_168
  FROM rass GROUP BY stay_id
),
-- daily status, days 2-6 (hours 48-167)
days AS (SELECT stay_id, d FROM base, UNNEST(GENERATE_ARRAY(2, 6)) AS d),
day_status AS (
  SELECT dy.stay_id, dy.d,
    LOGICAL_OR(r.valuenum <= -4) AS coma_day,
    LOGICAL_OR(de.value = 'Positive') AS delir_day,
    LOGICAL_OR(de.value = 'Negative') AS neg_day
  FROM days dy
  LEFT JOIN rass  r  ON r.stay_id  = dy.stay_id AND DIV(r.hr, 24)  = dy.d
  LEFT JOIN delir de ON de.stay_id = dy.stay_id AND DIV(de.hr, 24) = dy.d
  GROUP BY dy.stay_id, dy.d
),
dcfd AS (
  SELECT ds.stay_id,
    COUNTIF(
      (DATETIME_ADD(b.intime, INTERVAL ds.d * 24 HOUR) >= b.outtime)            -- out of ICU that day
      OR (IFNULL(ds.neg_day, FALSE) AND NOT IFNULL(ds.coma_day, FALSE) AND NOT IFNULL(ds.delir_day, FALSE))
    ) AS free_days_raw
  FROM day_status ds JOIN base b USING (stay_id)
  GROUP BY ds.stay_id
),
vent AS (
  SELECT DISTINCT v.stay_id FROM `{DER}.ventilation` v JOIN base b USING (stay_id)
  WHERE v.ventilation_status IN ('InvasiveVent', 'Tracheostomy')
    AND v.starttime < DATETIME_ADD(b.intime, INTERVAL 48 HOUR) AND v.endtime > b.intime
),
sed AS (
  SELECT ie.stay_id,
    LOGICAL_OR(REGEXP_CONTAINS(LOWER(di.label), r'propofol'))                    AS propofol_48h,
    LOGICAL_OR(REGEXP_CONTAINS(LOWER(di.label), r'midazolam|lorazepam'))         AS benzo_48h,
    LOGICAL_OR(REGEXP_CONTAINS(LOWER(di.label), r'dexmedetomidine|precedex'))    AS dexmed_48h,
    LOGICAL_OR(REGEXP_CONTAINS(LOWER(di.label), r'fentanyl|hydromorphone|morphine')) AS opioid_48h
  FROM `{ICU}.inputevents` ie
  JOIN `{ICU}.d_items` di ON di.itemid = ie.itemid
  JOIN base b ON b.stay_id = ie.stay_id
  WHERE ie.starttime < DATETIME_ADD(b.intime, INTERVAL 48 HOUR)
  GROUP BY ie.stay_id
)
SELECT
  b.stay_id,
  CAST(b.lev_48h AS INT64) AS lev_48h,
  CAST(b.lev_late AS INT64) AS lev_late,
  b.age, IF(b.gender = 'F', 1, 0) AS female,
  CASE WHEN b.tbi = 1 AND b.sah + b.ich = 0 THEN 'TBI'
       WHEN b.sah = 1 AND b.tbi + b.ich = 0 THEN 'SAH'
       WHEN b.ich = 1 AND b.tbi + b.sah = 0 THEN 'ICH'
       ELSE 'Multiple' END AS injury,
  b.anchor_year_group, b.first_careunit,
  CAST(b.alcohol_use_disorder AS INT64) AS aud,
  g.gcs_min, s.sofa, ch.charlson_comorbidity_index AS charlson,
  IF(v.stay_id IS NOT NULL, 1, 0) AS vent_48h,
  CAST(IFNULL(sd.propofol_48h, FALSE) AS INT64) AS propofol_48h,
  CAST(IFNULL(sd.benzo_48h,    FALSE) AS INT64) AS benzo_48h,
  CAST(IFNULL(sd.dexmed_48h,   FALSE) AS INT64) AS dexmed_48h,
  CAST(IFNULL(sd.opioid_48h,   FALSE) AS INT64) AS opioid_48h,
  CAST(IFNULL(da.delirium_0_48,   FALSE) AS INT64) AS delirium_0_48,
  CAST(IFNULL(ra.coma_0_48,       FALSE) AS INT64) AS coma_0_48,
  CAST(IFNULL(da.delirium_48_168, FALSE) AS INT64) AS delirium,
  CAST(IFNULL(da.assessed_48_168, FALSE) AS INT64) AS assessed,
  CAST(IFNULL(ra.coma_48_168,     FALSE) AS INT64) AS coma,
  a.hospital_expire_flag AS hosp_death,
  IF(a.hospital_expire_flag = 1, 0, IFNULL(dc.free_days_raw, 0)) AS dcfd
FROM base b
JOIN `{HOSP}.admissions` a ON a.hadm_id = b.hadm_id
LEFT JOIN `{DER}.first_day_gcs`  g  ON g.stay_id  = b.stay_id
LEFT JOIN `{DER}.first_day_sofa` s  ON s.stay_id  = b.stay_id
LEFT JOIN `{DER}.charlson`       ch ON ch.hadm_id = b.hadm_id
LEFT JOIN vent  v  ON v.stay_id  = b.stay_id
LEFT JOIN sed   sd ON sd.stay_id = b.stay_id
LEFT JOIN d_agg da ON da.stay_id = b.stay_id
LEFT JOIN r_agg ra ON ra.stay_id = b.stay_id
LEFT JOIN dcfd  dc ON dc.stay_id = b.stay_id
""")
for c in df.columns:
    if str(df[c].dtype) in ("Int64", "boolean"):
        df[c] = df[c].astype("int64")
df["delirium_or_coma"] = ((df.delirium == 1) | (df.coma == 1)).astype(int)
for c in ["gcs_min", "sofa", "charlson", "age"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

section("2. Analysis dataset")
print(f"N = {len(df)}   LEV <=48h: {int(df.lev_48h.sum())}   No early LEV: {int((1 - df.lev_48h).sum())}")
print("Missing:", {c: int(df[c].isna().sum()) for c in ["gcs_min", "sofa", "charlson"]})

# ---------------------------------------------------------------- 3. covariate matrix
df["gcs_missing"] = df.gcs_min.isna().astype(int)
for c in ["gcs_min", "sofa", "charlson"]:
    df[c] = df[c].fillna(df[c].median())
cat = pd.get_dummies(df[["injury", "anchor_year_group", "first_careunit"]], drop_first=True, dtype=float)
num_cols = ["age", "female", "gcs_min", "gcs_missing", "sofa", "charlson", "aud", "vent_48h",
            "propofol_48h", "benzo_48h", "dexmed_48h", "opioid_48h", "delirium_0_48", "coma_0_48"]
X = pd.concat([df[num_cols].astype(float), cat], axis=1)
T = df.lev_48h.astype(int).values


def design(Tv, Xv):
    Xv = Xv.loc[:, Xv.std() > 0]
    return sm.add_constant(pd.concat([pd.DataFrame({"lev_48h": Tv}, index=Xv.index), Xv], axis=1),
                           has_constant="add")


def smd(x, t, w=None):
    w = np.ones_like(x) if w is None else w
    m1, m0 = np.average(x[t == 1], weights=w[t == 1]), np.average(x[t == 0], weights=w[t == 0])
    v1 = np.average((x[t == 1] - m1) ** 2, weights=w[t == 1])
    v0 = np.average((x[t == 0] - m0) ** 2, weights=w[t == 0])
    return (m1 - m0) / np.sqrt((v1 + v0) / 2) if (v1 + v0) > 0 else 0.0


# ---------------------------------------------------------------- 4. propensity score + IPTW
Xps = sm.add_constant(X.loc[:, X.std() > 0], has_constant="add")
ps_model = sm.GLM(T, Xps, family=sm.families.Binomial()).fit()
ps = np.clip(ps_model.predict(Xps).values, 0.01, 0.99)
p_t = T.mean()
w = np.where(T == 1, p_t / ps, (1 - p_t) / (1 - ps))           # stabilized ATE weights
lo, hi = np.percentile(w, [1, 99])
w = np.clip(w, lo, hi)

section("3. Table 1 (unweighted) and balance after IPTW")
rows = []
for c in X.columns:
    x = X[c].values
    binary = set(np.unique(x)) <= {0.0, 1.0}
    f = (lambda v: f"{100 * v:.1f}%") if binary else (lambda v: f"{v:.1f}")
    rows.append([c, f(x[T == 1].mean()), f(x[T == 0].mean()), round(smd(x, T), 3), round(smd(x, T, w), 3)])
t1 = pd.DataFrame(rows, columns=["variable", "LEV<=48h", "No early LEV", "SMD raw", "SMD IPTW"])
print(t1.to_string(index=False))
print(f"\nMax |SMD| after IPTW: {t1['SMD IPTW'].abs().max():.3f}   "
      f"PS range: {ps.min():.3f}-{ps.max():.3f}   weights clipped to {lo:.2f}-{hi:.2f}")


# ---------------------------------------------------------------- 5. outcomes
def or_ci(res, term="lev_48h"):
    b, se = res.params[term], res.bse[term]
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), res.pvalues[term]


def evalue(or_, prev):
    rr = np.sqrt(or_) if prev > 0.15 else or_
    rr = 1 / rr if rr < 1 else rr
    return rr + np.sqrt(rr * (rr - 1))


def fit_binary(y, Tv, Xv, wv, label):
    t_only = sm.add_constant(pd.DataFrame({"lev_48h": Tv}), has_constant="add")
    unadj = sm.GLM(y, t_only, family=sm.families.Binomial()).fit()
    iptw = sm.GLM(y, t_only, family=sm.families.Binomial(), var_weights=wv).fit(cov_type="HC0")
    dr = sm.GLM(y, design(Tv, Xv), family=sm.families.Binomial(), var_weights=wv).fit(cov_type="HC0")
    n1, n0 = int((Tv == 1).sum()), int((Tv == 0).sum())
    e1, e0 = int(y[Tv == 1].sum()), int(y[Tv == 0].sum())
    out = []
    for name, r in [("Unadjusted", unadj), ("IPTW", iptw), ("IPTW + covariates", dr)]:
        o, l, h, p = or_ci(r)
        out.append([label, name, f"{o:.2f} ({l:.2f}-{h:.2f})", f"{p:.3g}"])
    print(f"\n{label}: events LEV {e1 if e1 >= 10 else '<10'}/{n1} ({100 * e1 / max(n1, 1):.1f}%)  vs  "
          f"no early LEV {e0 if e0 >= 10 else '<10'}/{n0} ({100 * e0 / max(n0, 1):.1f}%)")
    print(pd.DataFrame(out, columns=["outcome", "model", "OR (95% CI)", "p"]).to_string(index=False))
    o = or_ci(dr)[0]
    print(f"E-value (point estimate, DR model): {evalue(o, y.mean()):.2f}")


section("4. Primary: delirium (CAM-ICU positive) hours 48-168, among patients assessed in that window")
m = df.assessed.values == 1
fit_binary(df.delirium.values[m], T[m], X[m], w[m], "Delirium | assessed")

section("5. Secondary outcomes (full cohort)")
fit_binary(df.delirium_or_coma.values, T, X, w, "Delirium or coma (RASS<=-4)")
fit_binary(df.coma.values, T, X, w, "Coma (RASS<=-4)")
fit_binary(df.hosp_death.values.astype(int), T, X, w, "In-hospital death")

lin = sm.WLS(df.dcfd.values.astype(float), design(T, X), weights=w).fit(cov_type="HC0")
b, se = lin.params["lev_48h"], lin.bse["lev_48h"]
print(f"\nDelirium/coma-free days (days 2-6, 0-5): mean LEV {df.dcfd[T == 1].mean():.2f} vs "
      f"{df.dcfd[T == 0].mean():.2f};  IPTW+cov difference {b:.2f} ({b - 1.96 * se:.2f} to {b + 1.96 * se:.2f}), "
      f"p={lin.pvalues['lev_48h']:.3g}")

section("6. Sensitivity analyses (primary outcome, IPTW + covariates)")
def dr_only(mask_, label):
    y = df.delirium.values[mask_]
    if mask_.sum() < 100 or y.sum() < 10:
        print(f"{label}: too few patients/events — skipped")
        return
    r = sm.GLM(y, design(T[mask_], X[mask_]), family=sm.families.Binomial(),
               var_weights=w[mask_]).fit(cov_type="HC0")
    o, l, h, p = or_ci(r)
    print(f"{label:<55} n={int(mask_.sum()):>5}  OR {o:.2f} ({l:.2f}-{h:.2f})  p={p:.3g}")

base_m = df.assessed.values == 1
dr_only(base_m & (df.lev_late.values == 0), "Exclude late LEV starters (>48h)")
dr_only(base_m & (df.delirium_0_48.values == 0), "Exclude delirium in first 48h")
dr_only(base_m & df.anchor_year_group.isin(["2014 - 2016", "2017 - 2019", "2020 - 2022"]).values,
        "Admissions 2014-2022 only")
for inj in ["TBI", "ICH", "SAH"]:
    dr_only(base_m & (df.injury.values == inj), f"Subgroup: {inj} only")

print("\nDone. All output above is aggregate.")
