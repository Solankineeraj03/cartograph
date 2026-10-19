"""Deterministic structured bundle checks. Does not judge natural-language claims."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Product:
    product_id: str
    category: str
    price_cents: int
    in_stock: bool

    def __post_init__(self):
        if type(self.price_cents) is not int or self.price_cents < 0:
            raise ValueError("Price must be a nonnegative integer number of cents")
        if type(self.in_stock) is not bool:
            raise ValueError("Stock must be boolean")


def verify_bundle(output: object, catalog: dict[str, Product], budget_cents: int,
                  required_categories: set[str]) -> dict:
    if type(budget_cents) is not int or budget_cents < 0:
        raise ValueError("Budget must be nonnegative integer cents")
    if any(key != item.product_id for key, item in catalog.items()):
        raise ValueError("Catalog keys must match product IDs")

    def result(errors, total=None):
        return {"valid": not errors, "reward": float(not errors),
                "errors": errors, "computed_total_cents": total}

    if not isinstance(output, dict) or set(output) != {"product_ids", "total_cents"}:
        return result(["invalid_schema"])
    ids = output["product_ids"]
    if (not isinstance(ids, list) or not ids or
            any(not isinstance(item, str) for item in ids) or
            type(output["total_cents"]) is not int):
        return result(["invalid_schema"])
    if len(ids) != len(set(ids)):
        return result(["duplicate_product"])
    if any(item not in catalog for item in ids):
        return result(["unknown_product"])
    products = [catalog[item] for item in ids]
    total = sum(item.price_cents for item in products)
    errors = []
    if any(not item.in_stock for item in products):
        errors.append("out_of_stock")
    if not required_categories <= {item.category for item in products}:
        errors.append("missing_category")
    if total > budget_cents:
        errors.append("over_budget")
    if total != output["total_cents"]:
        errors.append("incorrect_total")
    return result(errors, total)
