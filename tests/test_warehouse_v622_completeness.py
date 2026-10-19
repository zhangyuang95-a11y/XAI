"""Independent structural receipts only; no model, solver, file or RNG use."""
from copy import deepcopy

import pytest

from experiments.warehouse.v622_completeness import D32ProtocolIncomplete, require_complete_d32_search


def complete():
    return {"templates_requested": 4, "complete_templates": 4, "complete_frozen_tables": 4,
        "certificate": {"all_requested_templates_completed_and_enumerated": True},
        "template_generation_reports": [{"template": i, "completed_structure": True,
            "frozen_table_completed": True, "dp_completed": True} for i in range(4)],
        "frozen_terminal_tables": [{"template": i, "certificate": {"complete_DP": True}} for i in range(4)]}


def test_complete_four_is_structurally_valid_without_new_authority_or_mutation():
    receipt=complete();before=deepcopy(receipt)
    result=require_complete_d32_search(receipt)
    assert result["passed"] is True and result["completed_template_ids"] == [0,1,2,3]
    assert result["admission_granted"] is False
    assert result["native_partial_family_arithmetic_audit_meaning_changed"] is False
    assert receipt == before


@pytest.mark.parametrize("count", [1,3])
@pytest.mark.parametrize("false_full_flag", [False,True])
def test_partial_family_is_rejected_even_with_falsely_true_full_flag(count,false_full_flag):
    receipt=complete()
    for key in ("template_generation_reports","frozen_terminal_tables"):receipt[key]=receipt[key][:count]
    receipt["complete_templates"]=receipt["complete_frozen_tables"]=count
    receipt["certificate"]["all_requested_templates_completed_and_enumerated"]=false_full_flag
    with pytest.raises(D32ProtocolIncomplete) as error:require_complete_d32_search(receipt)
    assert error.value.report["reason"] == "four_template_protocol_incomplete"
    assert error.value.report["scientific_method_failure_claimed"] is False


@pytest.mark.parametrize("family", ["template_generation_reports","frozen_terminal_tables"])
@pytest.mark.parametrize("damage", ["missing_id","duplicate_id","missing_entry","out_of_range","bool_id"])
def test_four_counts_and_true_flag_cannot_hide_missing_or_duplicate_ids(family,damage):
    receipt=complete();row=receipt[family][3]
    if damage=="missing_id":row.pop("template")
    if damage=="duplicate_id":row["template"]=2
    if damage=="missing_entry":receipt[family].pop()
    if damage=="out_of_range":row["template"]=4
    if damage=="bool_id":row["template"]=True
    with pytest.raises(D32ProtocolIncomplete):require_complete_d32_search(receipt)


@pytest.mark.parametrize("field", ["completed_structure","frozen_table_completed","dp_completed"])
@pytest.mark.parametrize("value", [None,False,1])
def test_generation_requires_each_completed_stage_as_a_real_boolean(field,value):
    receipt=complete()
    if value is None:receipt["template_generation_reports"][2].pop(field)
    else:receipt["template_generation_reports"][2][field]=value
    with pytest.raises(D32ProtocolIncomplete):require_complete_d32_search(receipt)


@pytest.mark.parametrize("field", ["templates_requested","complete_templates","complete_frozen_tables"])
def test_integer_counts_must_really_be_four(field):
    receipt=complete();receipt[field]=4.0
    with pytest.raises(D32ProtocolIncomplete):require_complete_d32_search(receipt)


def test_table_and_family_completion_certificates_are_not_optional():
    for position in ("table","family"):
        receipt=complete()
        if position=="table":receipt["frozen_terminal_tables"][3]["certificate"]["complete_DP"]=False
        else:receipt["certificate"]["all_requested_templates_completed_and_enumerated"]=False
        with pytest.raises(D32ProtocolIncomplete):require_complete_d32_search(receipt)


def test_wall_flag_does_not_replace_or_negate_complete_table_evidence():
    receipt=complete();receipt["wall_time_budget_exhausted"]=True
    assert require_complete_d32_search(receipt)["passed"]
    receipt["complete_frozen_tables"]=3
    with pytest.raises(D32ProtocolIncomplete):require_complete_d32_search(receipt)
