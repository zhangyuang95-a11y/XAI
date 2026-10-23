"""Training-only physical evidence and current-NN program feedback."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from .bc import collect_episode
from .contracts import MethodSpec, digest, write_new, file_sha
from .model import tensor_hash
from .program import (FitBudget, fit_program, refit_program, Program, qualify, metrics)
from .storage import append


class Evidence:
    def __init__(self, adapter, arrays, manifest=None):
        self.adapter = adapter; self.arrays = arrays; self.manifest = manifest
        for partitions in arrays.values():
            for data in partitions.values():
                for value in data.values(): value.flags.writeable = False

    def verify_static(self):
        if self.manifest is None: return  # Software fixtures have no scientific manifest.
        ref = self.manifest
        if file_sha(ref['path']) != ref['sha256']: raise ValueError('evidence manifest changed')
        manifest = json.loads(Path(ref['path']).read_text())
        for asset in [*manifest['arrays'], *manifest['traces']]:
            if file_sha(asset['path']) != asset['sha256']: raise ValueError('static physical evidence changed')

    @classmethod
    def collect(cls, adapter, model, run, *, seed):
        rng = torch.Generator().manual_seed(seed)
        records = {role.name: {p: [] for p in ('fit', 'calibration', 'eligibility', 'validation')}
                   for role in adapter.spec.roles}
        folder = run.folder / 'program-source'; folder.mkdir(parents=True, exist_ok=False)
        import gzip
        cases = [c for c in run.plan['cases'] if c['purpose'] == 'program']
        counts = tuple(sum(c['partition'] == p for c in cases) for p in ('fit', 'calibration', 'eligibility', 'validation'))
        if counts != MethodSpec().evidence_counts: raise ValueError('four finite evidence quotas differ')
        for case in cases:
            state = run.reset(adapter, case, purpose='program')
            rows, outcome = collect_episode(adapter, model, state, case, rng, mode='sampled', labelled=False)
            for role in adapter.spec.roles:
                records[role.name][case['partition']].extend(dict(**row['observations'][role.name],
                    scene=str(case['seed']), sampled=row['indices'][role.name]) for row in rows)
            with gzip.open(folder/(case['case_id'].replace(':', '_') + '.json.gz'), 'xt') as f:
                json.dump(dict(case=case, outcome=outcome, frames=[{k: v for k, v in row.items()
                    if k not in ('observations', 'reward', 'probabilities')} | dict(reward=row['reward'].tolist()) for row in rows]), f,
                    separators=(',', ':'), allow_nan=False)
        arrays = {}; source_assets = []
        for role in adapter.spec.roles:
            arrays[role.name] = {}
            for partition, rows in records[role.name].items():
                data = {key: np.array([r[key] for r in rows]) for key in ('x', 'mask', 'action_features', 'scene', 'sampled')}
                arrays[role.name][partition] = data
                path = folder/f'{role.name}-{partition}.npz'
                with path.open('xb') as f: np.savez_compressed(f, **data)
                source_assets.append(dict(path=str(path.resolve()), sha256=file_sha(path), role=role.name, partition=partition))
        manifest = dict(version='three-domain-unified-physical-program-source.v1', actor_role_sha256=model.role_hashes(),
                        source_cases_sha256=digest(cases), arrays=source_assets,
                        traces=[dict(path=str(p.resolve()), sha256=file_sha(p)) for p in sorted(folder.glob('*.json.gz'))],
                        NN_gradient_buffer=False, real_behavior='BC_NN_sampled', later_labels='counterfactual_current_NN_on_fixed_public_states')
        write_new(folder/'manifest.json', manifest)
        return cls(adapter, arrays, dict(path=str((folder/'manifest.json').resolve()), sha256=file_sha(folder/'manifest.json')))

    def labels(self, model, role, *, partitions=('fit', 'calibration', 'eligibility', 'validation')):
        if not partitions or set(partitions) - set(self.arrays[role]): raise ValueError('unknown evidence partition')
        result = {}
        for partition in partitions:
            source = self.arrays[role][partition]
            parts = []
            with torch.no_grad():
                for start in range(0, len(source['x']), 256):
                    sl = slice(start, start+256)
                    logits = model.actors[role](torch.from_numpy(source['x'][sl].copy()),
                        torch.from_numpy(source['action_features'][sl].copy()), torch.from_numpy(source['mask'][sl].copy()))
                    parts.append(logits.softmax(-1).numpy())
            p = np.concatenate(parts); p.flags.writeable = False
            result[partition] = {k: source[k] for k in ('x', 'mask', 'scene', 'sampled')} | dict(p=p)
        return result


class Envelope:
    def __init__(self, programs, hashes, reports):
        self.programs, self.hashes, self.reports = programs, hashes, reports
        self.invalidated = False

    def targets(self, name, x, mask, sampled, *, current_actor):
        program = self.programs[name]
        q, top, leaves = program.distributions(x, mask)
        valid = (not self.invalidated and tensor_hash(current_actor.state_dict()) == self.hashes[name]
                 and self.reports[name]['passed'])
        selected = np.isin(leaves, program.feedback) if valid else np.zeros(len(x), bool)
        return q, selected, top

    def invalidate_after_update(self, hashes):
        if hashes != self.hashes: self.invalidated = True


class ProgramProvider:
    def __init__(self, evidence, programs, journal):
        self.evidence, self.programs, self.journal = evidence, programs, journal

    @classmethod
    def initial(cls, evidence, model, folder, journal):
        evidence.verify_static()
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
        programs, reports = {}, {}
        for role in model.domain.roles:
            budget = FitBudget(lambda e: append(journal, dict(event='solver_predebit', role=role.name, **e)))
            program, report = fit_program(role, evidence.labels(model, role.name), model.role_hashes()[role.name], budget)
            programs[role.name], reports[role.name] = program, report
            write_new(folder/f'{role.name}.json', dict(program=program.as_dict(), metrics=report, solver_counts=budget.counts))
        write_new(folder/'result.json', dict(passed=all(r['passed'] for r in reports.values()), reports=reports,
                  model_role_sha256=model.role_hashes(), evidence=evidence.manifest, model_acceptance_passed=False, pilot_opened=False))
        # v7 admits the actual post-20k pair. A weak initial program disables
        # feedback; it is not an extra BC gate from an older Warehouse recipe.
        evidence.verify_static()
        return cls(evidence, programs, journal)

    def __call__(self, model, epoch):
        programs, reports = {}, {}
        hashes = model.role_hashes()
        for name, previous in self.programs.items():
            data = self.evidence.labels(model, name, partitions=('eligibility', 'validation'))
            bound = Program(previous.role, previous.tree, hashes[name], temperature=previous.temperature, protection=previous.protection,
                            anchor_actor_sha256=previous.anchor_actor_sha256)
            program = qualify(bound, data['eligibility'])
            report = metrics(program, data['validation'])
            programs[name], reports[name] = program, report
        append(self.journal, dict(event='current_epoch_program_check', epoch=epoch,
            actor_role_sha256=hashes, reports=reports))
        return Envelope(programs, hashes, reports)

    def refit(self, model, *, step, folder):
        self.evidence.verify_static()
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
        reports = {}
        for name, previous in list(self.programs.items()):
            budget = FitBudget(lambda e: append(self.journal, dict(event='solver_predebit', role=name, step=step, **e)))
            program, report = refit_program(previous, self.evidence.labels(model, name), model.role_hashes()[name], budget)
            self.programs[name] = program; reports[name] = report
            write_new(folder/f'{name}.json', dict(program=program.as_dict(), metrics=report, solver_counts=budget.counts))
        self.evidence.verify_static()
        return reports
