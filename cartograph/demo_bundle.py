"""Build a small, self-contained demo data bundle for the deterministic request path (containers, KubeRay).

Keeps only categorized products (the bundle tasks' catalog), their export feature rows, and the manifests, so a
container doesn't need the 1.1 GB ESCI product file. The checksums of every source file are recorded.
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from .provenance import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", type=Path, required=True)
    parser.add_argument("--sim", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cats = pd.read_parquet(args.sim / "product_categories.parquet")
    keep = set(cats.loc[cats.category != "other", "product_id"])
    for sub in ("slice", "sim", "export"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    products = pd.read_parquet(args.slice / "products.parquet")
    products[products.product_id.isin(keep)].to_parquet(args.out / "slice/products.parquet", index=False)
    shutil.copy(args.slice / "data_manifest.json", args.out / "slice/data_manifest.json")
    cats[cats.product_id.isin(keep)].to_parquet(args.out / "sim/product_categories.parquet", index=False)
    shutil.copy(args.sim / "simulation_manifest.json", args.out / "sim/simulation_manifest.json")
    for name in ("price", "stock", "popularity"):
        frame = pd.read_parquet(args.export / f"{name}.parquet")
        frame[frame.product_id.isin(keep)].to_parquet(args.out / f"export/{name}.parquet", index=False)
    shutil.copy(args.export / "export_manifest.json", args.out / "export/export_manifest.json")
    manifest = {"products": len(keep), "sources": {str(p): sha256_file(p) for p in
                [args.slice / "products.parquet", args.sim / "product_categories.parquet",
                 *[args.export / f"{n}.parquet" for n in ("price", "stock", "popularity")]]},
                "bundle": {str(p.relative_to(args.out)): sha256_file(p) for p in sorted(args.out.rglob("*")) if p.is_file()}}
    (args.out / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"products": len(keep), "files": len(manifest["bundle"])}))


if __name__ == "__main__":
    main()
