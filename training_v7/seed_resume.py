"""Resume exact-proposal triage without reparsing already validated large JSON.

The raw prefilter can ONLY prove no candidate numeral AND no reservation
syntax. It is used only with a SHA-identical, previously COMPLETE JSON parse.
Every possible hit, escape, long token or unknown format falls back to the
unchanged full parser. Previous validity does not imply new-seed clearance.
"""
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
from .contracts import file_sha, write_new
from .seed_audit import engine
from experiments.warehouse import seed_triage_indexed_v642 as parser


def numeric_patterns(families):
    intervals = [span for f in families for span in f['intervals']]
    terms = []
    for lo,hi in intervals:
        for digits in range(len(str(lo)),len(str(hi-1))+1):
            a=max(lo,0 if digits==1 else 10**(digits-1)); b=min(hi-1,10**digits-1)
            if a>b: continue
            sa,sb=str(a),str(b); prefix=''
            for x,y in zip(sa,sb):
                if x!=y:break
                prefix+=x
            terms.append(re.escape(prefix).encode()+rb'[0-9]{'+str(digits-len(prefix)).encode()+rb'}')
    # Exponent notation may represent ANY candidate interval, irrespective of
    # mantissa length or decimal placement. Do not restrict it by a prefix.
    terms.append(rb'-?[0-9]+(?:\.[0-9]+)?[eE][+-]?[0-9]+')
    return re.compile(rb'(?<![0-9A-Za-z_.])(?:'+b'|'.join(terms)+rb')(?![0-9A-Za-z_])'),intervals


def no_numeric_or_reservation_hit(path, families, *, chunk_bytes=8*1024*1024):
    """Conservative byte superset. False positives cost time, never clearance."""
    pattern,intervals=numeric_patterns(families); carry=b''
    reservation=re.compile(rb'"start"|seed_range|\\u',re.I)
    long_token=re.compile(rb'[0-9.eE+\-]{257,}')
    with Path(path).open('rb') as f:
        while chunk:=f.read(chunk_bytes):
            value=carry+chunk
            if reservation.search(value) or long_token.search(value): return False
            for hit in pattern.finditer(value):
                try: number=Decimal(hit.group().decode('ascii'))
                except InvalidOperation:return False
                if not number.is_finite() or any(lo<=number<hi for lo,hi in intervals):return False
            carry=value[-1024:]
    return True


def resume(output, previous_complete):
    output=Path(output); proposal_path=output/'seed-proposal.json'
    proposal,families,pointer=engine.read_proposal(proposal_path)
    bounded_path=output/'seed-inventory/inventory.json'
    bounded=json.loads(bounded_path.read_text())
    if bounded['proposal']!=pointer: raise ValueError('resume proposal differs')
    previous_path=Path(previous_complete).resolve(); previous=json.loads(previous_path.read_text())
    if (previous['auditor_sha256']!=file_sha(engine.__file__)
            or previous['triage_driver_sha256']!=file_sha(parser.__file__)):
        raise ValueError('previous complete parser source differs')
    prior={r['path']:r for r in previous['inventory'] if r.get('resolved')}
    partial={}
    for p in sorted((output/'seed-triage').glob('file-*.json')):
        try:r=json.loads(p.read_text())
        except json.JSONDecodeError:continue  # unfinished CACHE; its actual input is still parsed below
        partial[r['path']]=(r,dict(path=str(p.resolve()),sha256=file_sha(p)))
    result=dict(bounded); result['schema']='warehouse-seed-metadata-triage.v2'
    result['triage_driver_sha256']=file_sha(parser.__file__)
    result['resume_adapter']=dict(path=str(Path(__file__).resolve()),sha256=file_sha(__file__),
        previous_complete=dict(path=str(previous_path),sha256=file_sha(previous_path)),
        raw_prefilter_is_not_a_JSON_parser=True,exact_proposal_partial_reuse=True)
    rows={}; remaining=[]; evidence=[]; fast=0; reused=0
    for original in bounded['inventory']:
        path=Path(original['path']); row=dict(original)
        if (path.stat().st_size!=row['bytes'] or ('mtime_ns' in row and path.stat().st_mtime_ns!=row['mtime_ns'])):
            raise ValueError('bounded metadata changed before resume')
        h=file_sha(path)
        if row.get('sha256',h)!=h: raise ValueError('bounded metadata hash changed')
        row['sha256']=h
        if row.get('resolved'):
            rows[str(path)]=row;continue
        item=partial.get(str(path))
        if item and item[0].get('resolved') and item[0]['sha256']==h:
            rows[str(path)]=item[0]; evidence.append(item[1]);reused+=1;continue
        old=prior.get(str(path))
        if (old and old.get('sha256')==h and path.suffix in ('.json','.jsonl')
                and old.get('parser') in ('jq_complete_json_stream','bounded_complete_JSON_exact_integer_evidence')
                and no_numeric_or_reservation_hit(path,families)):
            row.update(resolved=True,classification=old['classification'],same_candidate_declaration=False,
                detail=engine.analyze_literal({},families),parser='v7_prior_complete_JSON_SHA_and_raw_numeric_reservation_exclusion',
                previous_JSON_validity=dict(report=str(previous_path),file_sha256=h))
            rows[str(path)]=row;fast+=1
        else: remaining.append(row)
    folder=output/'seed-triage-resumed';folder.mkdir(exist_ok=False)
    subset=dict(bounded,inventory=remaining)
    write_new(folder/'remaining-inventory.json',subset)
    detail=parser.triage(folder/'remaining-inventory.json',proposal_path,folder/'full-parser')
    rows.update({r['path']:r for r in detail['inventory']})
    result['inventory']=[rows[r['path']] for r in bounded['inventory']]
    result['resume_adapter'].update(reused_current_rows=reused,fast_negative_rows=fast,
        full_parser_rows=len(remaining),partial_evidence=evidence,
        resumed_full_parse=dict(path=str((folder/'full-parser/result.json').resolve()),sha256=file_sha(folder/'full-parser/result.json')))
    result['summary']=engine.summarize(result['inventory'],families,selection_stats=result['selection_stats'],
        missing_roots=result.get('missing_roots',()))
    # Context and final delta independently rehash the full selected population.
    write_new(folder/'result.json',result)
    return result
