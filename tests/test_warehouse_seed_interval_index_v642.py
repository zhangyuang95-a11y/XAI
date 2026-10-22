"""Exact original-predicate parity, without new random draws or scan skips."""
from itertools import product
import pytest
from experiments.warehouse import seed_metadata_audit as old
from experiments.warehouse.seed_interval_index_v642 import SeedIntervalIndex


def test_boundaries_gaps_overlap_order_and_integer_type_semantics():
    families=[dict(name='second',intervals=[(30,35),(10,20),(15,25)]),dict(name='first',intervals=[(0,1),(20,32)]),dict(name='empty',intervals=[])]
    index=SeedIntervalIndex.compile(families)
    for value in [*range(-5,40),False,True,.0,20.,'20',None,2**63-1]:assert index.matches(value)==old.matching_families(value,families)
    assert index.matches(22)==['second','first'] and not index.matches(35)


def test_all_small_interval_combinations_and_large_integer_boundaries():
    intervals=[(a,b) for a in range(4) for b in range(a+1,6)]
    for first,second in product(intervals,repeat=2):
        families=[dict(name='x',intervals=[first,second]),dict(name='y',intervals=[(2**63-3,2**63)])]
        index=SeedIntervalIndex.compile(families)
        for value in [*range(-1,8),2**63-4,2**63-3,2**63-1,2**63]:assert index.matches(value)==old.matching_families(value,families)


def test_actual_reserved_arguments_match_original_predicate_exactly():
    from experiments.warehouse.v626_protocol import ROOT,bound,pointer
    from experiments.warehouse.exact_seed_audit import compile_families
    p=ROOT/'output/warehouse_v642/training-seed-plans/warehouse-v642-calibration-02/exact-plan.json'
    families=compile_families(bound(pointer(p)))
    # metadata's normalized family representation is the actual lookup input.
    proposal=ROOT/'output/warehouse_v642/training-seed-plans/warehouse-v642-calibration-02/seed-ledger/proposal.json'
    _,families,_=old.read_proposal(proposal);index=SeedIntervalIndex.compile(families)
    values={0,1,2**63-1}
    for row in families:
        for a,b in row['intervals']:values.update((a-1,a,b-1,b))
    for value in sorted(values):assert index.matches(value)==old.matching_families(value,families)
