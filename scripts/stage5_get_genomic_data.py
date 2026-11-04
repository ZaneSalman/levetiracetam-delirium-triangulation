# Stage 5 — Obtain public genomic data for drug-target Mendelian randomization (runs in Colab; public data only)
# Outcome (discovery):  Raptis et al. delirium GWAS meta-analysis (UKB, FinnGen, All of Us, MGI; 1,059,130 indiv, 11,931 cases)
#                       DOI 10.7488/ds/8014, CC-BY 4.0
# Outcome (replication): FinnGen R12 F5_DELIRIUM (second outcome dataset; FinnGen also contributes to the meta-analysis)
# Exposure:              cis-eQTLs for SV2A and CACNA1B from the eQTL Catalogue (GTEx v8 brain tissues + others)
# Saves region extracts (gene +/- 1 Mb) and genome-wide significant hits to Drive folder "LEV_delirium_genomics".

import os, io, gzip, time, requests, numpy as np, pandas as pd
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

OUT = "/content/lev_genomics"; os.makedirs(OUT, exist_ok=True)
WIN = 1_000_000
GENES = {"SV2A": None, "CACNA1B": None}


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def download(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 22):
                f.write(chunk)
    return path


# ---------------------------------------------------------------- gene coordinates (GRCh38)
section("0. Gene coordinates (Ensembl GRCh38)")
for g in GENES:
    j = requests.get(f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{g}",
                     headers={"Content-Type": "application/json"}, timeout=60).json()
    GENES[g] = dict(id=j["id"], chrom=str(j["seq_region_name"]), start=j["start"], end=j["end"])
    print(g, GENES[g])


def in_regions(chrom, pos):
    m = np.zeros(len(pos), dtype=bool); lab = np.array([""] * len(pos), dtype=object)
    for g, c in GENES.items():
        k = (chrom == c["chrom"]) & (pos >= c["start"] - WIN) & (pos <= c["end"] + WIN)
        m |= k; lab[k] = g
    return m, lab


# ---------------------------------------------------------------- 1. delirium meta-analysis
section("1. Delirium GWAS meta-analysis (Raptis & Tenesa 2025)")
meta_gz = download("https://datashare.ed.ac.uk/bitstreams/b3652f1a-38c0-4c7b-a135-e00522aa5bfe/download",
                   f"{OUT}/METAANALYSIS_DELIRIUM_QC.tbl.gz")
print(f"Downloaded {os.path.getsize(meta_gz)/1e9:.2f} GB")
reg, hits, n = [], [], 0
for ch in pd.read_csv(meta_gz, sep=" ", chunksize=2_000_000, dtype={"Chr": str}):
    n += len(ch)
    m, lab = in_regions(ch["Chr"].astype(str).str.replace("chr", ""), ch["Pos"].values)
    if m.any():
        r = ch[m].copy(); r["gene_region"] = lab[m]; reg.append(r)
    hits.append(ch[ch["P-value"] < 5e-8] if "P-value" in ch else ch[ch["P_sterr"] < 5e-8])
meta_reg = pd.concat(reg) if reg else pd.DataFrame()
meta_hits = pd.concat(hits)
print(f"Variants in file: {n:,}   columns: {list(ch.columns)}")
print(f"Genome-wide significant (p<5e-8): {len(meta_hits):,}")
print(f"Variants in SV2A/CACNA1B +/-1 Mb: {meta_reg.gene_region.value_counts().to_dict()}")
pcol = "P_sterr" if "P_sterr" in meta_reg else "P-value"
for g in GENES:
    s = meta_reg[meta_reg.gene_region == g]
    if len(s):
        b = s.loc[s[pcol].idxmin()]
        print(f"  {g}: min p in region = {b[pcol]:.2e} ({b['MarkerName']})")
meta_reg.to_csv(f"{OUT}/delirium_meta_SV2A_CACNA1B_region.tsv.gz", sep="\t", index=False)
meta_hits.to_csv(f"{OUT}/delirium_meta_gwsig_hits.tsv.gz", sep="\t", index=False)
os.remove(meta_gz)

# ---------------------------------------------------------------- 2. FinnGen replication
section("2. FinnGen F5_DELIRIUM (replication)")
fg = None
for rel in ["R12", "R11", "R10"]:
    url = f"https://storage.googleapis.com/finngen-public-data-{rel.lower()}/summary_stats/release/finngen_{rel}_F5_DELIRIUM.gz"
    if requests.head(url, timeout=60).status_code == 200:
        fg = (rel, url); break
if fg:
    rel, url = fg
    fg_gz = download(url, f"{OUT}/finngen_{rel}_F5_DELIRIUM.gz")
    print(f"Using FinnGen {rel}: {os.path.getsize(fg_gz)/1e9:.2f} GB")
    reg = []
    for ch in pd.read_csv(fg_gz, sep="\t", chunksize=2_000_000, dtype={"#chrom": str}):
        m, lab = in_regions(ch["#chrom"].astype(str), ch["pos"].values)
        if m.any():
            r = ch[m].copy(); r["gene_region"] = lab[m]; reg.append(r)
    fg_reg = pd.concat(reg)
    print(f"Variants in regions: {fg_reg.gene_region.value_counts().to_dict()}   columns: {list(fg_reg.columns)}")
    fg_reg.to_csv(f"{OUT}/finngen_{rel}_F5_DELIRIUM_SV2A_CACNA1B_region.tsv.gz", sep="\t", index=False)
    os.remove(fg_gz)
    try:
        man = requests.get(f"https://storage.googleapis.com/finngen-public-data-{rel.lower()}/summary_stats/{rel}_manifest.tsv", timeout=60)
        if man.ok:
            mf = pd.read_csv(io.StringIO(man.text), sep="\t")
            row = mf[mf.iloc[:, 0].astype(str).str.contains("F5_DELIRIUM")]
            print("FinnGen manifest row:\n", row.to_string(index=False))
    except Exception as ex:
        print("Manifest not read:", ex)
else:
    print("FinnGen F5_DELIRIUM file not found at public bucket URLs (R10-R12).")

# ---------------------------------------------------------------- 3. eQTL Catalogue cis-eQTLs (exposure)
section("3. eQTL Catalogue cis-eQTLs for SV2A and CACNA1B")
API = "https://www.ebi.ac.uk/eqtl/api/v2"
ds = pd.DataFrame(requests.get(f"{API}/datasets", params={"size": 1000, "quant_method": "ge"}, timeout=120).json())
keep = ds[ds.study_label.isin(["GTEx", "MetaBrain", "ROSMAP", "BrainSeq", "CommonMind", "Braineac2"]) |
          ds.tissue_label.str.contains("brain|cortex|cerebell|hippocamp|caudate|putamen|blood", case=False, na=False)]
print(f"Candidate datasets: {len(keep)} (of {len(ds)} gene-expression datasets)")
eq_rows = []
for _, d in keep.iterrows():
    for g, c in GENES.items():
        page, got = 0, []
        while True:
            r = requests.get(f"{API}/datasets/{d.dataset_id}/associations",
                             params={"gene_id": c["id"], "size": 1000, "start": page * 1000}, timeout=120)
            if r.status_code != 200:
                break
            js = r.json()
            if not js:
                break
            got.extend(js); page += 1
            if len(js) < 1000:
                break
        if got:
            df_ = pd.DataFrame(got)
            df_["dataset_id"], df_["study"], df_["tissue"], df_["gene"] = d.dataset_id, d.study_label, d.tissue_label, g
            eq_rows.append(df_)
    time.sleep(0.2)
eq = pd.concat(eq_rows) if eq_rows else pd.DataFrame()
if len(eq):
    eq["F"] = (eq["beta"] / eq["se"]) ** 2
    s = (eq.groupby(["gene", "study", "tissue"])
           .agg(n_variants=("variant", "size"), min_p=("pvalue", "min"), max_F=("F", "max"),
                n_F_gt10_p_lt5e8=("pvalue", lambda p: int(((p < 5e-8)).sum())))
           .reset_index().sort_values(["gene", "min_p"]))
    print(s.to_string(index=False))
    eq.to_csv(f"{OUT}/eqtl_catalogue_SV2A_CACNA1B.tsv.gz", sep="\t", index=False)
    s.to_csv(f"{OUT}/eqtl_instrument_strength_summary.tsv", sep="\t", index=False)
else:
    print("No eQTL associations returned.")

# ---------------------------------------------------------------- 4. save to Drive
section("4. Saving to Google Drive folder 'LEV_delirium_genomics'")
drv = build("drive", "v3")
q = "name='LEV_delirium_genomics' and mimeType='application/vnd.google-apps.folder' and trashed=false"
f = drv.files().list(q=q, fields="files(id)").execute().get("files", [])
fid = f[0]["id"] if f else drv.files().create(body={"name": "LEV_delirium_genomics",
                                                     "mimeType": "application/vnd.google-apps.folder"}, fields="id").execute()["id"]
for fn in sorted(os.listdir(OUT)):
    p = f"{OUT}/{fn}"
    drv.files().create(body={"name": fn, "parents": [fid]}, media_body=MediaFileUpload(p, resumable=True), fields="id").execute()
    print(f"  uploaded {fn} ({os.path.getsize(p)/1e6:.1f} MB)")
print(f"Drive folder id: {fid}")
print("\nDone (public data only).")
