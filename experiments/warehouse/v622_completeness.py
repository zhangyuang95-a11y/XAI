"""Protocol completeness for the four declared D32 frozen-template tables.

This checks receipt structure only. It does not fit, route, load assets, inspect
an environment, prove a table's arithmetic, or grant execution authority. The
native audit can correctly certify a smaller completed table family; such a
certificate does not complete the declared four-template comparison.
"""
from collections.abc import Mapping


SCHEMA = "warehouse-v622-D32-four-template-completeness.v1"
EXPECTED_TEMPLATES = 4


class D32ProtocolIncomplete(RuntimeError):
    """The declared comparison is incomplete, without a method-failure claim."""
    def __init__(self, message):
        super().__init__("D32 protocol incomplete: "+message)
        self.report = {"schema": SCHEMA, "passed": False,
            "reason": "four_template_protocol_incomplete", "violation": message,
            "scientific_method_failure_claimed": False, "admission_granted": False}


def _require(condition, message):
    if not condition:
        raise D32ProtocolIncomplete(message)


def _family(rows, label):
    _require(isinstance(rows, list) and len(rows) == EXPECTED_TEMPLATES,
        label+" must contain exactly four entries")
    _require(all(isinstance(row, Mapping) and type(row.get("template")) is int for row in rows),
        label+" has a missing or non-integer template ID")
    ids = [row["template"] for row in rows]
    _require(sorted(ids) == list(range(EXPECTED_TEMPLATES)),
        label+" IDs must be exactly 0,1,2,3 without duplicates")
    return rows


def require_complete_d32_search(search):
    """Return a completeness receipt or raise ``D32ProtocolIncomplete``.

    The caller must separately bind the D32 search bytes, verify its numerical
    audit, and enforce quality, legality and budget gates. Wall-clock flags and
    restricted optimality gaps are deliberately not substitutes for four
    complete structures, four complete frozen tables, and four complete DPs.
    No maximal-template or whole-function-class optimality claim is made here.
    """
    _require(isinstance(search, Mapping), "search receipt is missing")
    for key in ("templates_requested", "complete_templates", "complete_frozen_tables"):
        _require(type(search.get(key)) is int and search[key] == EXPECTED_TEMPLATES,
            key+" must equal four")
    certificate = search.get("certificate")
    _require(isinstance(certificate, Mapping)
        and certificate.get("all_requested_templates_completed_and_enumerated") is True,
        "the complete declared family is not certified as enumerated")
    generation = _family(search.get("template_generation_reports"), "generation family")
    for row in generation:
        for key in ("completed_structure", "frozen_table_completed", "dp_completed"):
            _require(row.get(key) is True, "template "+str(row["template"])+" lacks "+key)
    tables = _family(search.get("frozen_terminal_tables"), "frozen-table family")
    for row in tables:
        proof = row.get("certificate")
        _require(isinstance(proof, Mapping) and proof.get("complete_DP") is True,
            "template "+str(row["template"])+" lacks a completed DP receipt")
    return {"schema": SCHEMA, "passed": True, "expected_templates": EXPECTED_TEMPLATES,
        "completed_template_ids": list(range(EXPECTED_TEMPLATES)),
        "scope": "declared_four_template_protocol_only_not_numeric_fidelity_or_optimality_audit",
        "native_partial_family_arithmetic_audit_meaning_changed": False,
        "admission_granted": False}
