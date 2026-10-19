# Sanitization of this public copy

This repository is a sanitized export of the private development repository at commit `c87ff6da53f2d4d8b3e11116d3220e81ac999ebc`, published
with a fresh history. Only private infrastructure identifiers were replaced; no numbers, outputs, or code logic changed.

| Replaced | With | Files affected |
|---|---|---|
| lab-specific value #1 | `/home/user` | 304 |
| lab-specific value #2 | `gpu-host` | 13 |
| lab-specific value #3 | `gpu-host` | 79 |
| lab-specific value #4 | `user` | 0 |
| lab-specific value #5 | `/Users/user` | 1 |

Consequences for verifying evidence:
- `outputs_sha256` and `inputs` checksums in `experiments/**/manifest.json` were computed on the original files.
  Files whose text contained a replaced value (mostly paths in logs and manifests) won't match their recorded hash.
  Data files (Parquet, model weights, vectors) were never in the repository and are unaffected.
- `git.commit` values in manifests refer to the private development history, not this repository's history.
