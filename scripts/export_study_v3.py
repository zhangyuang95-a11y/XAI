"""Authenticated, read-only study export; the key stays out of URLs and output."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='https://policylens-warehouse-study.onrender.com')
    parser.add_argument('--release-id')
    parser.add_argument('--mode',choices=('pilot','preview','test'))
    parser.add_argument('--format',choices=('jsonl','csv'),default='jsonl')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    key=os.environ.get('POLICYLENS_ADMIN_TOKEN')
    if not key:parser.error('Set POLICYLENS_ADMIN_TOKEN in your environment first.')
    query={k:v for k,v in {'release_id':args.release_id,'mode':args.mode,'format':args.format}.items() if v}
    request=Request(args.base_url.rstrip('/')+'/api/study/admin/export?'+urlencode(query),headers={'Authorization':'Bearer '+key})
    destination=args.output or Path('output/study_v3/exports')/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.'+args.format)
    destination.parent.mkdir(parents=True,exist_ok=True)
    try:
        with urlopen(request,timeout=90) as response:
            content=response.read()
    except Exception as exc:
        parser.exit(1,'Export failed ('+type(exc).__name__+'); no completed export was written.\n')
    with destination.open('xb') as output:
        destination.chmod(0o600);output.write(content)
    print(json.dumps({'saved':str(destination.resolve()),'bytes':len(content),'format':args.format}))

if __name__=='__main__':main()
