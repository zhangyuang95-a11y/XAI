"""Bounded operational reconciler; never pays, rejects, messages, or buys places.

Uses an existing locally stored Prolific token and the deployed audited release
endpoint. Does not change the experiment or store the Prolific token on Render.
"""
import argparse
import collections
import datetime
import fcntl
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CELLS = [(d, g) for d in ('warehouse', 'pong', 'kitchen') for g in ('A', 'B')]
TERMINAL = {'RETURNED': 'RETURNED', 'TIMED-OUT': 'TIMED_OUT'}


def plan(records, submissions, target=10):
    by_id = {s['id']: s for s in submissions}
    counts = collections.Counter()
    completed = collections.Counter()
    releases = []
    unfinished = 0
    for row in records:
        cell = (row['domain'], row['group_code'])
        if cell not in CELLS:
            raise ValueError('Unknown cell')
        sub = by_id.get(row['submission_id'])
        if not sub or sub['participant_id'] != row['prolific_pid']:
            raise ValueError('Missing or mismatched Prolific submission')
        done = row['completed'] is not None or row['stage'] == 'completed'
        if done:
            if row.get('released'):
                raise ValueError('Completed session has been released')
            completed[cell] += 1
        if row.get('released'):
            continue
        if not done and sub['status'] in TERMINAL:
            releases.append({'instance_id': row['instance_id'],
                             'submission_id': row['submission_id'],
                             'submission_status': TERMINAL[sub['status']]})
            continue
        counts[cell] += 1
        unfinished += not done
    if any(counts[c] > target for c in CELLS):
        raise ValueError('Cell capacity exceeded')
    return {'releases': releases, 'vacancies': sum(target-counts[c] for c in CELLS),
            'unfinished': unfinished, 'complete': sum(completed.values()),
            'cells': [{'domain': d, 'group': g, 'occupied': counts[d, g],
                       'completed': completed[d, g], 'vacant': target-counts[d, g]}
                      for d, g in CELLS]}


class Sync:
    def __init__(self, directory):
        self.directory = directory
        settings = json.loads((directory/'settings.json').read_text())
        self.sid = settings['study_id']
        self.platform_places = settings.get('prolific_places', 60)
        if settings['places'] != 60 or settings['cell_target'] != 10:
            raise ValueError('Expected the authorized 60-person main study')
        self.token = (directory/'prolific-api-token').read_text().strip()
        config = json.loads((ROOT/'output/private_render/deployment.json').read_text())
        self.origin, self.admin = config['origin'], config['admin_token']
        self.study_url = 'https://api.prolific.com/api/v1/studies/'+self.sid+'/'

    def request(self, url, method='GET', body=None, site=False):
        headers = {'Authorization': ('Bearer '+self.admin) if site else ('Token '+self.token)}
        if body is not None:
            headers['Content-Type'] = 'application/json'
        request = urllib.request.Request(url, method=method, headers=headers,
            data=None if body is None else json.dumps(body).encode())
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    def submissions(self):
        # Explicit pagination avoids Prolific's limit=0 links; never accept a
        # partial response as evidence that an enrollment is no longer active.
        rows = []
        for offset in range(0, 10000, 200):
            page = self.request(self.study_url+'submissions/?limit=200&offset='+str(offset))
            rows.extend(page['results'])
            expected = page['meta']['count']
            if len(rows) >= expected:
                if len({r['id'] for r in rows}) != expected:
                    raise ValueError('Inconsistent submission pagination')
                return rows
            if not page['results']:
                break
        raise ValueError('Incomplete submission listing')

    def pause(self):
        study = self.request(self.study_url)
        if study['status'] == 'ACTIVE':
            self.request(self.study_url+'transition/', 'POST', {'action': 'PAUSE'})

    def cycle(self, apply=False, resume=False):
        study = self.request(self.study_url)
        if study['id'] != self.sid or study['total_available_places'] != self.platform_places or study['reward'] != 300:
            raise ValueError('Study identity, total places, or reward changed')
        submissions = self.submissions()
        records = self.request(self.origin+'/api/prolific/admin/payments', site=True)['records']
        if any(r['study_id'] != self.sid for r in records):
            raise ValueError('Website study ID mismatch')
        result = plan(records, submissions)
        released = 0
        if apply:
            for item in result['releases']:
                # Re-check the exact submission immediately before releasing.
                sub = self.request('https://api.prolific.com/api/v1/submissions/'+item['submission_id']+'/')
                if TERMINAL.get(sub['status']) != item['submission_status']:
                    raise ValueError('Submission changed during reconciliation')
                item['reason'] = 'Verified Prolific API terminal status; incomplete session; main-study capacity reconciliation at '+datetime.datetime.now(datetime.timezone.utc).isoformat()
                try:
                    self.request(self.origin+'/api/prolific/admin/release-slot', 'POST', item, site=True)
                    released += 1
                except urllib.error.HTTPError as exc:
                    # A concurrent completion is never overridden.
                    if exc.code != 409:
                        raise
            records = self.request(self.origin+'/api/prolific/admin/payments', site=True)['records']
            result = plan(records, self.submissions())
            if result['releases']:
                raise ValueError('Pending reconciliation; keep recruitment paused')
            if result['vacancies'] == 0 or (study.get('places_taken', self.platform_places) >= self.platform_places and result['unfinished'] == 0):
                self.pause()
            elif resume:
                # Never offer more simultaneous reservations than the website
                # can accommodate, while respecting the requested max of 20.
                cap = max(1, min(20, result['vacancies'] + result['unfinished']))
                config = study['submissions_config']
                if config['max_concurrent_submissions'] != cap:
                    self.request(self.study_url, 'PATCH', {'submissions_config': {**config, 'max_concurrent_submissions': cap}})
                study = self.request(self.study_url)
                if study['status'] == 'PAUSED' and study.get('places_taken', self.platform_places) < self.platform_places:
                    self.request(self.study_url+'transition/', 'POST', {'action': 'START'})
        study = self.request(self.study_url)
        result.pop('releases')
        result.update(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      status=study['status'], released_this_cycle=released,
                      places_taken=study.get('places_taken'),
                      concurrency=study['submissions_config']['max_concurrent_submissions'],
                      applied=apply)
        if result['vacancies'] and study.get('places_taken', self.platform_places) >= self.platform_places:
            result['blocked_by_prolific_capacity'] = True
        (self.directory/'sync-status.json').write_text(json.dumps(result, indent=2))
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--hours', type=float, default=8)
    args = parser.parse_args()
    directory = ROOT/'output/main-study'
    with (directory/'sync.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sync = Sync(directory)
        deadline = time.monotonic() + min(args.hours, 8)*3600
        previous = None
        while True:
            try:
                result = sync.cycle(args.apply, args.resume)
                comparable = {k: v for k, v in result.items() if k != 'at'}
                if comparable != previous:
                    print(json.dumps(result), flush=True)
                    previous = comparable
                if result['complete'] == 60 or not args.watch:
                    break
            except Exception as exc:
                pause_confirmed = False
                if args.apply:
                    try:
                        sync.pause()
                        pause_confirmed = True
                    except Exception:
                        pass
                # Do not log tokens, response bodies or participant identifiers.
                print(json.dumps({'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                  'error': type(exc).__name__, 'pause_confirmed': pause_confirmed}), flush=True)
                if not args.watch:
                    raise SystemExit(1)
            if time.monotonic() >= deadline:
                if args.apply:
                    sync.pause()
                print(json.dumps({'stopped': 'bounded eight-hour window expired'}), flush=True)
                break
            time.sleep(15)


if __name__ == '__main__':
    main()
