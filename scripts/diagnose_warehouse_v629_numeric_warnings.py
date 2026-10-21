#!/usr/bin/env python3
"""Read-only bounded-matrix diagnostics; no fitting, LP or environment calls.

Uses spent INNER FIT data and the frozen starting program, never a validation
partition for method selection. This does not alter running scientific jobs
or certify every intermediate matrix from their optimizers.
"""
from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
import argparse,json,sys,warnings
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v629_protocol as v,v628_data as data
from backend.training.warehouse_v626_domain import domain_table,transform,validate_rows
from backend.training.warehouse_v626_certificate import coefficient_data
from scripts.run_warehouse_v629 import initial_program
from scripts.run_warehouse_v625 import write_new


def array_hash(value):
    a=np.asarray(value);return sha256(a.dtype.str.encode()+str(a.shape).encode()+np.ascontiguousarray(a).tobytes()).hexdigest()


def compare(label,left,right,*,repeats=20):
    left,right=np.asarray(left,np.float64),np.asarray(right,np.float64)
    v.require(left.ndim==right.ndim==2 and left.shape[1]==right.shape[0]
        and np.isfinite(left).all() and np.isfinite(right).all(),'invalid bounded diagnostic inputs')
    before=(array_hash(left),array_hash(right))
    # Explicit contraction avoids delegating the reference back to a BLAS
    # matmul via optimize=True. It is a numerical reference, not exact math.
    with warnings.catch_warnings(record=True) as ref_warnings,np.errstate(over='warn',invalid='warn',divide='warn'):
        warnings.simplefilter('always');reference=np.einsum('ik,kj->ij',left,right,optimize=False)
    failures=[];records=[];maximum_error=0.
    for index in range(repeats):
        with warnings.catch_warnings(record=True) as seen,np.errstate(over='warn',invalid='warn',divide='warn'):
            warnings.simplefilter('always');actual=left@right
        finite=bool(np.isfinite(actual).all());error=float(np.max(np.abs(actual-reference))) if finite else None
        records.extend(dict(repeat=index,category=w.category.__name__,message=str(w.message)) for w in seen)
        if not finite or not np.allclose(actual,reference,rtol=1e-11,atol=1e-10):failures.append(index)
        if error is not None:maximum_error=max(maximum_error,error)
    v.require(before==(array_hash(left),array_hash(right)),'diagnostic changed its matrix inputs')
    limit=float(np.max(np.abs(left)))*float(np.max(np.abs(right)))*left.shape[1]
    return dict(label=label,left_shape=list(left.shape),right_shape=list(right.shape),dtype='float64',input_hashes=list(before),
        maximum_absolute_input=[float(np.abs(left).max()),float(np.abs(right).max())],
        elementary_absolute_dot_bound=limit,bound_is_finite=bool(np.isfinite(limit)),
        reference_is_finite=bool(np.isfinite(reference).all()),reference_warnings=[str(w.message) for w in ref_warnings],
        repetitions=repeats,matmul_warnings=records,numerical_failed_repetitions=failures,
        maximum_absolute_error_against_explicit_einsum=maximum_error)


def run(output):
    path,plan=v.checked_plan();sources=v.source_files()
    job=next(j for j in plan['jobs'] if j['job_id']=='C64-fold-0-robot_1')
    program,initial,_=initial_program(plan,job)
    parts=data.load_parts(v.bound(plan['previous_plan']),0,'robot_1',inner=True);a=parts['fit']
    x=validate_rows(a['x']);mask=a['m'];p=a['p']
    routes,z,q=program.distributions(x,mask)
    v.require(np.isfinite(p).all() and np.isfinite(q).all() and np.all((p>=0)&(p<=1))
        and np.all((q>=0)&(q<=1)),'starting distributions are not finite probability targets')
    valid=[i for i,item in enumerate(domain_table()['features']) if not item['constant']]
    u=transform(x,valid).astype(float);u-=u.mean(0);residual=p-q
    records=[compare('centered_public_features_T_times_full_probability_residual',u.T,residual)]
    candidates=[i for i,leaf in enumerate(program.leaves) if leaf.get('normalized') and int((routes==i).sum())>0]
    v.require(candidates,'no normalized occupied leaf for objective arithmetic check')
    leaf_id=max(candidates,key=lambda i:int((routes==i).sum()));leaf=program.leaves[leaf_id]
    support,bias,weights=coefficient_data(leaf);take=routes==leaf_id
    design=np.column_stack((np.ones(int(take.sum())),transform(x[take],support).astype(float)))
    coefficients=np.column_stack((bias,weights)).astype(float)
    records.append(compare('actual_stored_leaf_design_times_coefficients',design,coefficients.T))
    boundary=np.linspace(-32.,32.,5*design.shape[1]).reshape(5,design.shape[1])
    records.append(compare('same_design_registered_coefficient_box_probe',design,boundary.T))
    # Same shape as the SLSQP objective gradient, using observed probability
    # residuals; no objective optimization or finite-difference fit is run.
    records.append(compare('actual_probability_residual_T_times_leaf_design',(q[take]-p[take]).T,design))
    v.require(v.source_files()==sources,'scientific source changed during read-only diagnostic')
    output=Path(output).resolve();base=ROOT/'output/warehouse_v629/diagnostics'
    v.require(output.is_relative_to(base) and output!=base,'diagnostic output must be isolated')
    output.mkdir(parents=True,exist_ok=False)
    result=dict(schema='warehouse-v629-bounded-matrix-warning-diagnostic.v1',created_utc=datetime.now(timezone.utc).isoformat(),
        scope='spent_inner_fit_and_starting_program_only_not_optimizer_intermediate_capture',plan=v.pointer(path),initial=initial,
        diagnostic_source=v.pointer(__file__),scientific_source_sha256=v.digest(sources),numpy_version=np.__version__,
        input_rows=len(x),input_features=x.shape[1],variable_features=len(valid),selected_leaf=leaf_id,selected_leaf_rows=int(take.sum()),
        records=records,all_probed_products_finite_and_reference_consistent=all(r['reference_is_finite'] and not r['numerical_failed_repetitions'] for r in records),
        cannot_exonerate_all_live_optimizer_intermediates=True,scientific_fit_calls=0,LP_calls=0,NN_updates=0,environment_resets=0,
        modified_running_source=False,restarted_jobs=False,changes_scientific_admission=False,pilot_opened=False)
    write_new(output/'result.json',result);(output/'result.json').chmod(0o444)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    result=run(parser.parse_args().output)
    print(json.dumps(dict(result=v.pointer(parser.parse_args().output/'result.json'),
        products=len(result['records']),consistent=result['all_probed_products_finite_and_reference_consistent'],
        warning_count=sum(len(r['matmul_warnings']) for r in result['records'])),ensure_ascii=False))
