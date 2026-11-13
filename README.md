# Levetiracetam and ICU delirium after acute brain injury — analysis code

Code for the article *Early Levetiracetam Prophylaxis and Delirium After Acute Brain Injury: Triangulating Clinical, Genetic and Network Evidence* (Salman ZG, Zeitouny R, Salman SG, Tripuraneni A; submitted to *Biomedicines*).

The study combines three evidence layers:

1. **Clinical cohorts** — MIMIC-IV v3.1 (discovery) and eICU-CRD v2.0 (external replication), queried on Google BigQuery.
2. **Drug-target genetics** — Mendelian randomization, SMR/HEIDI and colocalization of SV2A and CACNA1B brain expression with delirium, plus an APOE-region positive control.
3. **Network medicine** — interactome proximity of levetiracetam targets to delirium genes in STRING.

## What this repository contains — and what it does not

* **No patient-level data.** MIMIC-IV and eICU-CRD may not be redistributed. Every cell that touches them prints aggregate results only, and counts from 1 to 9 are masked as `<10`, as the PhysioNet data use agreement requires.
* **No notebook outputs.** All cell outputs were cleared before release.
* **No third-party summary statistics.** The public GWAS and eQTL files are downloaded by the code from their original sources (see *Data sources*).

## Layout

```
notebooks/LEV_delirium_analysis.ipynb   main Colab notebook, stages 1–12 in run order
scripts/stage2_lev_delirium.py          MIMIC-IV cohort, IPTW models, sensitivity analyses
scripts/stage3_ml_hte.py                doubly robust learner, GATES, effect modifiers
scripts/stage4_network_pharm.py         Open Targets, STRING proximity, bridge genes, GTEx
scripts/stage5_get_genomic_data.py      download and extract public GWAS / eQTL summary statistics
```

| Stage | Notebook section | Purpose |
|---|---|---|
| 1 | cells 1–7 | Authenticate, build MIMIC-IV stage-1 cohort, flowchart, delirium item checks |
| 2–3 | `run_script(...)` | Primary MIMIC-IV analyses; causal machine learning |
| 4, 4b | `run_script(...)`, Stage 4b | Network pharmacology (genetics-only delirium module is the primary module) |
| 5, 5b | `run_script(...)`, Stage 5b | Public GWAS and cis-eQTL extracts (written to `/content/lev_genomics`) |
| 6 | Stage 6 | Drug-target MR and colocalization |
| 7a–7d | Stage 7 | eICU-CRD replication and robustness |
| 8 | Stage 8 | Harmonized analysis and pooling (overlap weights) |
| 9, 9b | Stage 9 | Confounding tests: active comparator, dose–response, negative controls, neurosurgery |
| 10 | Stage 10 | Single primary model in both cohorts |
| 11 | Stage 11 | Regional association plot data |
| 12a–12d | Stage 12 | Cell-type eQTLs, SMR/HEIDI, multi-variant MR, APOE positive control |

## How to run

1. Obtain credentialed PhysioNet access to MIMIC-IV v3.1 and eICU-CRD v2.0, and link BigQuery access to your Google account.
2. Open the notebook in Google Colaboratory and clone this repository to `/content/levetiracetam-delirium-triangulation` (or edit `REPO` in the setup cell):

   ```bash
   git clone https://github.com/ZaneSalman/levetiracetam-delirium-triangulation.git /content/levetiracetam-delirium-triangulation
   ```
3. Set `PROJECT_ID` to your own Google Cloud billing project. The notebook creates a working dataset (`epi_lev_delirium`) in that project. BigQuery storage and queries may incur Google Cloud charges.
4. Run the cells in order. Stage 5 downloads public summary statistics. The delirium meta-analysis file on Edinburgh DataShare may need to be downloaded manually and placed in `/content/lev_genomics/` if automated download is refused.

Software versions are listed in `requirements.txt` (Python 3.13.18, Google Colaboratory).

## Data sources

| Resource | Access |
|---|---|
| MIMIC-IV v3.1 | https://physionet.org/content/mimiciv/3.1/ (credentialed) |
| eICU-CRD v2.0 | https://physionet.org/content/eicu-crd/2.0/ (credentialed) |
| Delirium GWAS meta-analysis (Raptis & Tenesa) | https://doi.org/10.7488/ds/8014 |
| FinnGen R12 (F5_DELIRIUM) | https://www.finngen.fi/en/access_results |
| eQTL Catalogue r8 | https://www.ebi.ac.uk/eqtl/ |
| Brain cell-type cis-eQTLs (Bryois et al. 2022) | https://zenodo.org/records/6104982 |
| 1000 Genomes Project (GRCh38 phased) | https://www.internationalgenome.org/ |
| Open Targets Platform | https://platform.opentargets.org/ |
| STRING v12 | https://string-db.org/ |
| GTEx | https://gtexportal.org/ |

Access to and use of each external dataset remain subject to the provider's license, data-use agreement and access restrictions.

## Notes

* The SMR and HEIDI tests are reimplemented in Python following Zhu et al., *Nat Genet* 2016, because the SMR binary and the BrainMeta eQTL resource could not be obtained when the analysis was run. Results may differ slightly from the reference implementation.
* Some cells were added after initial results, in response to them (eICU replication, overlap weighting, neurosurgery adjustment, dose–response, negative controls, SMR/HEIDI, positive control). The article reports which analyses were prespecified and which were post hoc.


## License

MIT (see `LICENSE`). Please cite the article if you use this code (see `CITATION.cff`).
