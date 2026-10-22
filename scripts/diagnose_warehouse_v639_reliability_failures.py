#!/usr/bin/env python3
"""Read-only gate attribution and explicitly non-operational support ceiling."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v630_protocol as v
from backend.training.warehouse_v622_program import _eligible
from backend.training.warehouse_v630_partition_proposals import allocate
from scripts.run_warehouse_v625 import write_new


def reasons(child):
    r=child['risk'];s=child['selection'];out=[]
    if not child['fit_count']:out.append('no_fit_exposure')
    if s['rows']==0:out.append('no_selection_rows')
    elif s['agreement']<.97:out.append('selection_agreement_below97')
    if s['kl'] is not None and s['kl']>.20:out.append('selection_KL_above020')
    if r['agreement']<.95:out.append('risk_agreement_below95')
    if r['kl'] is None or r['kl']>.25:out.append('risk_KL_above025_or_unknown')
    if r['illegal_count'] or r['path_uncertified_count']:out.append('risk_physical_or_path_failure')
    if not _eligible(r,.95) and not any(t.startswith('risk_') for t in out):out.append('risk_support_or_wilson_failure')
    if child['repeated_passes']['explanation']<27:out.append('risk_repeat_below27')
    return out


def main(folder,output):
    sr=v.pointer(folder/'result.json');summary=v.bound(sr);ar=v.pointer(folder/'independent-replay.json')
    v.require(summary['all_six_settled'] and v.bound(ar)['integrity_passed'],'need complete replayed six-unit source')
    output.mkdir(parents=True,exist_ok=False);source=v.pointer(__file__)
    plan=dict(source=source,summary=sr,audit=ar,created_utc=datetime.now(timezone.utc).isoformat(),new_fits=0,new_environment_resets=0,
        hypothetical_ceiling_is_not_an_admission_or_a_program=True)
    write_new(output/'plan.json',plan);rows=[]
    for ref in summary['units']:
        r=v.bound(ref);report=v.bound(r['search_report']);selected=report['candidates'][report['selected']];e=selected['metrics']['explanation']
        n=round(e['rows']/e['coverage']);primary={};overlap={};support_only_rows=0;children=[];actualE=0
        for group,name in selected['allocation']['selected'].items():
            table=report['tables'][group][name]
            for child in table['children']:
                accepted=child['reliable_selected']['explanation'];why=reasons(child);count=child['selection']['rows']
                if accepted:actualE+=count
                else:
                    v.require(bool(why),'unaccepted leaf lacks a reason');primary[why[0]]=primary.get(why[0],0)+count
                    for cause in why:overlap[cause]=overlap.get(cause,0)+count
                    if set(why)<={'risk_support_or_wilson_failure','risk_repeat_below27'}:support_only_rows+=count
                children.append(dict(group=group,option=name,leaf=child['leaf'],selection_rows=count,accepted=accepted,reasons=why))
        v.require(actualE==selected['reliable_counts']['E'],'selected reliable count differs')
        groups={}
        for group,options in report['groups'].items():
            groups[group]=[]
            for option in options:
                table=report['tables'][group][option['id']]
                # Diagnostic ONLY: ignore support/repeat failures, keep all
                # observed mean agreement/KL/physical/selection requirements.
                mass=sum(c['selection']['rows'] for c in table['children'] if not(set(reasons(c))-{'risk_support_or_wilson_failure','risk_repeat_below27'}))
                groups[group].append(dict(id=option['id'],leaves=option['leaves'],score=[mass]))
        optimistic=allocate(groups,256)
        row=dict(job=r['spec']['id'],input=ref,selection_frames=n,reliable_E=actualE/n,primary_excluded_reason_rows=primary,
            overlapping_excluded_reason_rows=overlap,selected_support_or_repeat_only_rows=support_only_rows,
            hypothetical_support_repeat_only_E_ceiling=optimistic['score'][0]/n,hypothetical_only_no_gate_changed=True,
            selected_children=children)
        rows.append(row);print(row['job'],json.dumps(dict(primary=primary,support_only=support_only_rows,optimisticE=row['hypothetical_support_repeat_only_E_ceiling'])),flush=True)
    v.require(v.pointer(__file__)==source,'diagnostic source changed')
    result=dict(schema='warehouse-v639-reliability-gate-attribution.v1',plan=v.pointer(output/'plan.json'),units=rows,new_fits=0,new_environment_resets=0,
        gate_or_program_changes=0,admission_authority=False,pilot_opened=False)
    write_new(output/'result.json',result);print(json.dumps(v.pointer(output/'result.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.run.resolve(),a.output.resolve())
