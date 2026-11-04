# Stage 4 — Bioinformatics / network-pharmacology layer (public data only; no MIMIC data used here)
# 1) Drug targets (Open Targets / ChEMBL mechanisms of action) for levetiracetam and comparator ASMs
# 2) Delirium-associated genes (Open Targets association scores)
# 3) Network proximity of each drug's targets to the delirium module in the STRING v12 human interactome
#    (closest-distance d_c with degree-preserving randomisation; Guney et al., Nat Commun 2016)
# 4) Shortest-path "bridge" genes between SV2A and delirium genes + pathway enrichment (Enrichr)
# 5) Tissue specificity of SV2A (GTEx v8 median TPM)

import subprocess, sys, io, gzip, json, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gseapy"], check=False)
import requests, numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components

OT = "https://api.platform.opentargets.org/api/v4/graphql"
rng = np.random.default_rng(2026)
N_RAND = 1000


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def gql(query, variables=None):
    r = requests.post(OT, json={"query": query, "variables": variables or {}}, timeout=60)
    r.raise_for_status()
    return r.json()["data"]


def search(term, entity):
    d = gql("""query($q:String!,$e:[String!]!){search(queryString:$q,entityNames:$e,page:{index:0,size:5}){hits{id name entity}}}""",
            {"q": term, "e": [entity]})
    return d["search"]["hits"]


# ---------------------------------------------------------------- 1. drug targets
section("1. Drug targets (Open Targets / ChEMBL mechanism of action)")
DRUGS = ["levetiracetam", "brivaracetam", "phenytoin", "lacosamide", "valproic acid", "lamotrigine"]
drug_targets = {}
for name in DRUGS:
    try:
        hit = search(name, "drug")[0]
        d = gql("""query($id:String!){drug(chemblId:$id){name mechanismsOfAction{rows{mechanismOfAction targets{approvedSymbol}}}}}""",
                {"id": hit["id"]})
        rows = d["drug"]["mechanismsOfAction"]["rows"]
        syms = sorted({t["approvedSymbol"] for r in rows for t in r["targets"]})
        moa = sorted({r["mechanismOfAction"] for r in rows})
        drug_targets[name] = syms
        print(f"{name:<14} {hit['id']:<14} targets={len(syms):>2}  {', '.join(syms[:12])}{' ...' if len(syms) > 12 else ''}")
        print(f"{'':<14} MoA: {'; '.join(moa)[:150]}")
    except Exception as ex:
        print(f"{name}: lookup failed ({ex})")

# ---------------------------------------------------------------- 2. delirium genes
section("2. Delirium-associated genes (Open Targets)")
dis_hits = search("delirium", "disease")
print("Disease hits:", [(h["id"], h["name"]) for h in dis_hits])
disease_genes = {}
for h in dis_hits:
    if "delirium" not in h["name"].lower():
        continue
    d = gql("""query($id:String!){disease(efoId:$id){name associatedTargets(page:{index:0,size:500}){count rows{score target{approvedSymbol}}}}}""",
            {"id": h["id"]})
    for r in d["disease"]["associatedTargets"]["rows"]:
        s = r["target"]["approvedSymbol"]
        disease_genes[s] = max(disease_genes.get(s, 0), r["score"])
dg = pd.Series(disease_genes).sort_values(ascending=False)
print(f"Associated targets (all scores): {len(dg)}")
DELIRIUM = list(dg[dg >= 0.05].index)
if len(DELIRIUM) < 20:
    DELIRIUM = list(dg.head(100).index)
print(f"Delirium module used: {len(DELIRIUM)} genes (score >= 0.05, or top 100 if fewer than 20)")
print("Top 25:", ", ".join(f"{g}({dg[g]:.2f})" for g in DELIRIUM[:25]))

# ---------------------------------------------------------------- 3. STRING interactome
section("3. STRING v12 human interactome (combined score >= 700)")
base = "https://stringdb-downloads.org/download"
info = pd.read_csv(io.BytesIO(requests.get(f"{base}/protein.info.v12.0/9606.protein.info.v12.0.txt.gz", timeout=300).content),
                   sep="\t", compression="gzip", usecols=[0, 1])
info.columns = ["pid", "symbol"]
links = pd.read_csv(io.BytesIO(requests.get(f"{base}/protein.links.v12.0/9606.protein.links.v12.0.txt.gz", timeout=600).content),
                    sep=" ", compression="gzip")
links = links[links.combined_score >= 700]
p2s = dict(zip(info.pid, info.symbol))
links["a"], links["b"] = links.protein1.map(p2s), links.protein2.map(p2s)
nodes = pd.Index(sorted(set(links.a) | set(links.b)))
ai, bi = nodes.get_indexer(links.a), nodes.get_indexer(links.b)
A = csr_matrix((np.ones(len(ai)), (ai, bi)), shape=(len(nodes), len(nodes)))
A = ((A + A.T) > 0).astype(np.int8)
ncomp, lab = connected_components(A, directed=False)
lcc_label = np.bincount(lab).argmax()
keep = np.where(lab == lcc_label)[0]
A = A[keep][:, keep]; nodes = nodes[keep]
deg = np.asarray(A.sum(1)).ravel()
idx = {s: i for i, s in enumerate(nodes)}
print(f"Largest connected component: {len(nodes):,} proteins, {A.nnz // 2:,} interactions")

D_idx = np.array([idx[g] for g in DELIRIUM if g in idx])
print(f"Delirium genes mapped to interactome: {len(D_idx)}/{len(DELIRIUM)}")

# degree bins (>=100 nodes per bin) for degree-preserving randomisation
order = np.argsort(deg); bins, cur = [], []
for i in order:
    cur.append(i)
    if len(cur) >= 100 and (len(bins) == 0 or True):
        bins.append(np.array(cur)); cur = []
if cur:
    bins[-1] = np.concatenate([bins[-1], cur])
bin_of = np.empty(len(nodes), dtype=int)
for b, members in enumerate(bins):
    bin_of[members] = b


def rand_like(S):
    return np.array([rng.choice(bins[bin_of[s]]) for s in S])


def d_closest(T, D):
    dist = shortest_path(A, directed=False, unweighted=True, indices=T)
    return np.mean(dist[:, D].min(axis=1))


section(f"4. Network proximity to delirium module (closest distance, {N_RAND} degree-matched randomisations)")
rows = []
for name, tg in drug_targets.items():
    T_idx = np.array([idx[g] for g in tg if g in idx])
    if len(T_idx) == 0:
        rows.append([name, 0, np.nan, np.nan, np.nan, np.nan]); continue
    obs = d_closest(T_idx, D_idx)
    null = np.array([d_closest(rand_like(T_idx), rand_like(D_idx)) for _ in range(N_RAND)])
    z = (obs - null.mean()) / null.std(ddof=1)
    p = (np.sum(null <= obs) + 1) / (N_RAND + 1)
    rows.append([name, len(T_idx), round(obs, 3), round(null.mean(), 3), round(z, 2), round(p, 4)])
prox = pd.DataFrame(rows, columns=["drug", "targets_in_network", "d_c", "null_mean", "z", "p_empirical"])
print(prox.to_string(index=False))
print("Interpretation: z < -1.5 and p < 0.05 indicate targets closer to the delirium module than expected by chance.")

# ---------------------------------------------------------------- 5. bridge genes SV2A -> delirium
section("5. Shortest-path bridge genes from levetiracetam targets to delirium genes")
lev_T = [idx[g] for g in drug_targets.get("levetiracetam", []) if g in idx]
bridge = set()
if lev_T:
    dist, pred = shortest_path(A, directed=False, unweighted=True, indices=lev_T, return_predecessors=True)
    for k, t in enumerate(lev_T):
        dmin = dist[k, D_idx].min()
        for dgi in D_idx[dist[k, D_idx] == dmin]:
            cur = dgi
            while cur != t and cur >= 0:
                bridge.add(nodes[cur]); cur = pred[k, cur]
            bridge.add(nodes[t])
        print(f"{nodes[t]}: nearest delirium gene(s) at distance {int(dmin)} -> "
              f"{', '.join(nodes[D_idx[dist[k, D_idx] == dmin]][:15])}")
print(f"Bridge module ({len(bridge)} genes): {', '.join(sorted(bridge))}")

section("6. Pathway enrichment of bridge module (Enrichr)")
try:
    import gseapy as gp
    if len(bridge) >= 5:
        enr = gp.enrichr(gene_list=sorted(bridge),
                         gene_sets=["GO_Biological_Process_2023", "KEGG_2021_Human", "Reactome_2022"],
                         organism="human", outdir=None, cutoff=1.0)
        res = enr.results.sort_values("Adjusted P-value").head(15)
        res["Term"] = res["Term"].str.slice(0, 70)
        print(res[["Gene_set", "Term", "Overlap", "Adjusted P-value"]].to_string(index=False))
    else:
        print("Bridge module too small for enrichment.")
except Exception as ex:
    print(f"Enrichment failed: {ex}")

# ---------------------------------------------------------------- 7. GTEx SV2A expression
section("7. GTEx v8 median expression of levetiracetam targets (TPM)")
try:
    for sym in drug_targets.get("levetiracetam", []):
        g = requests.get("https://gtexportal.org/api/v2/reference/gene", params={"geneId": sym}, timeout=60).json()["data"][0]
        e = requests.get("https://gtexportal.org/api/v2/expression/medianGeneExpression",
                         params={"gencodeId": g["gencodeId"], "datasetId": "gtex_v8"}, timeout=60).json()["data"]
        ex = pd.DataFrame(e)[["tissueSiteDetailId", "median"]].sort_values("median", ascending=False)
        brain = ex[ex.tissueSiteDetailId.str.startswith("Brain")]
        other = ex[~ex.tissueSiteDetailId.str.startswith("Brain")]
        print(f"{sym}: brain median of medians {brain['median'].median():.1f} TPM vs non-brain {other['median'].median():.1f} TPM")
        print(ex.head(8).to_string(index=False))
except Exception as ex:
    print(f"GTEx lookup failed: {ex}")

print("\nDone (public data only).")
