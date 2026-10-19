# Data card

| Dataset | Source and license | Version / checksum | Size | Synthetic? |
|---|---|---|---|---|
| ESCI shopping queries (relevance) | github.com/amazon-science/esci-data, Apache-2.0 | commit `7916cdf6`; examples `4a735b69…`, products `25124442…`, sources `a5fed8ec…` | US `small_version=1`: 29,844 queries, 601,354 judgments, 482,105 products | No |
| Slice `728e301ef1fb` | derived (`cartograph/esci.py`) | `experiments/gate_a/esci_slice/data_manifest.json` | 18,777 train / 2,089 dev / 8,956 test queries (+22 excluded) | No |
| Operational events `ops-sim-v1` | generated (`cartograph/simulator.py`, seed 20260901) | `experiments/gate_c/drills_v5/simulation_manifest.json` | 5,055,251 events over 30 days, 2 delivery batches | **Yes** |
| Feature snapshots `ingest_1`, `ingest_2` | Iceberg tags → Parquet exports | `experiments/gate_c/drills_v5/export_manifest_*.json` (snapshot IDs, SHA-256) | price 1.44M, stock 1.20M, popularity 2.85M rows | **Yes** |
| Bundle tasks `tasks_v1` | generated (`cartograph/tasks.py`, seeds 101 / 202) | `experiments/gate_d/tasks_manifest.json` | 3,000 train / 500 test, disjoint product halves | **Yes** |
| Verifier preference pairs | sampled from the SFT policy, labeled by the verifier | `experiments/gate_e/pairs_verifier_v1.jsonl` | 607 pairs | **Yes** (`verifier_v1`) |
| Human preference labels | owner, rubric `pref-rubric-v1` | — | **none collected yet** | — |
| Demo bundle | subset of the slice + export (`cartograph/demo_bundle.py`) | `data/demo_bundle_v1/bundle_manifest.json` | 14,678 categorized products | partly |

**Split and leakage rules.** The official ESCI test split is never used for training, negative mining, PCA fitting,
fusion or HNSW selection, or demonstrations. Dev is a hashed 10% of train query groups. Bundle-task products are split
by a hash of the product ID, so train and test tasks share none.

**What the data cannot support.** ESCI labels are query-product relevance, not purchases or user preference. All prices,
stock, and clicks are simulated, so nothing here supports claims about revenue, conversion, or real inventory. The
product categories are derived from title keyword rules (`CATEGORY_RULES` in `cartograph/simulator.py`) and are heuristic.
