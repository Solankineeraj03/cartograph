# Model card: Cartograph assistant models

## Generator (released: v1 `2406781e60ea`)
- **Base:** Qwen/Qwen2.5-1.5B-Instruct, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, Apache-2.0.
- **Adapter:** LoRA r=16, alpha 32, all linear layers. SFT on 3,000 synthetic train bundle tasks, 2 epochs, max length 1,536.
  Adapter SHA-256 `63d343e8…24c` (`experiments/gate_d/sft_full`).
- **Training data:** solver-derived demonstrations with a templated explanation (`label_source = solver_template_v1`).
  No human-written or human-preference data.
- **Intended use:** produce one JSON decision (a bundle of evidence product IDs with an exact total, or an abstention with
  a reason) from a request plus bounded catalog evidence. Its output is **always verified**; unverifiable bundles are withheld.
- **Measured** (500 held-out tasks on unseen products): 0.688 correct raw, 0.748 with verify-and-retry, 0.838 optimal
  among valid bundles, 0.8% fabricated-ID rate (raw). Exact solver: 1.000.
- **Known failures:** arithmetic on totals and budgets (most raw errors are over-budget or wrong-total bundles). The
  verify-and-retry wrapper turns many errors into unnecessary refusals (false abstention rises from 3% to 20%).
  Explanations aren't checked claim-by-claim.
- **Alignment variants (not released):** GRPO with binary verifier reward (no significant gain, optimality 0.59 —
  reward hacking); GRPO with graded reward (no significant gain, optimality preserved); DPO on verifier pairs (no change).
  See `experiments/gate_e/SUMMARY.md`. None is RLHF.
- **Serving:** vLLM 0.11.0. FP8 weight-only quantization is the best speed/memory tradeoff (−1.4 to −1.8 points).
  bnb 4-bit loses 10 points raw. See `experiments/gate_f/SUMMARY.md`.

## Retrieval encoder (b1)
- **Base:** sentence-transformers/all-MiniLM-L6-v2 `1110a243…`, Apache-2.0. Mean pooling, 384 dims, cosine.
- **Training:** contrastive loss (temperature 0.05) with in-batch negatives plus query-aware hard negatives (the query's own
  judged I/C products), with known E/S positives masked out. 1,000 steps × 32 queries, ESCI US `split == "train"` only.
- **Served form:** train-fitted PCA to 256 dims (basis in `data/vectors/b1_pca256/pca.npz` on the host), fused with BM25
  (weighted, α=0.5).
- **Measured** (ESCI US test, full catalog, NDCG@10): b1 0.308; b1+PCA-256 0.302; hybrid 0.353; BM25 0.302.
- **Limits:** English US queries; trained on 1,000 steps on CPU; judged coverage of top-10 is only 32% (b1) and 37% (hybrid), so the true
  quality of unjudged results is unknown.

## Ethical and usage notes
All prices, stock, and interaction data are synthetic. The models make no claims about real product availability.
Product titles come from ESCI and can contain brand names; no personal data is involved.
