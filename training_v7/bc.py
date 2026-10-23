"""Common BC stage execution, with public-only labels and streamed datasets."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier
from .contracts import BC_STAGES, BC_FITS, MethodSpec, digest, write_new
from .losses import bc_loss
from .storage import append, export_actor, save_checkpoint


def choose(model, observations, generator, *, deterministic=False):
    indices, probabilities = {}, {}
    with torch.no_grad():
        for role, obs in observations.items():
            logits = model.actors[role](torch.from_numpy(obs['x'][None]),
                torch.from_numpy(obs['action_features'][None]), torch.from_numpy(obs['mask'][None]))[0]
            p = logits.softmax(-1)
            index = p.argmax() if deterministic else torch.multinomial(p, 1, generator=generator)[0]
            indices[role] = int(index); probabilities[role] = p.numpy()
    return indices, probabilities


def collect_episode(adapter, model, state, case, generator, *, mode, labelled=True):
    initial = adapter.snapshot(state); records = []
    while not adapter.terminal(state):
        public = adapter.public(state); observations = adapter.observe_public(public)
        labels = adapter.label_public(public) if labelled else None
        if mode == 'teacher':
            if labels is None: raise ValueError('teacher trajectory needs explicit public labels')
            indices, p = dict(labels), None
        else:
            indices, p = choose(model, observations, generator, deterministic=mode == 'argmax')
        before = adapter.snapshot(state)
        state, reward = adapter.step(state, indices)
        records.append(dict(public=public, observations=observations, labels=labels,
            indices=indices, probabilities=p, reward=reward, before=before,
            after=adapter.snapshot(state), ending=adapter.ending(state)))
    return records, adapter.outcome(state, adapter.restore(initial))


def _save_rows(path, records, roles):
    arrays = {}
    for role in roles:
        n = role.name
        arrays.update({n + '_x': np.stack([r['observations'][n]['x'] for r in records]),
            n + '_mask': np.stack([r['observations'][n]['mask'] for r in records]),
            n + '_af': np.stack([r['observations'][n]['action_features'] for r in records]),
            n + '_y': np.array([r['labels'][n] for r in records], np.int64)})
    with Path(path).open('xb') as f: np.savez(f, **arrays)


def _initialize_tree(actor, files, role):
    xs, ys = [], []
    for file in files:
        with np.load(file, allow_pickle=False) as data:
            xs.append(data[role + '_x']); ys.append(data[role + '_y'])
    classifier = DecisionTreeClassifier(max_depth=4, max_leaf_nodes=16, random_state=0)
    classifier.fit(np.concatenate(xs), np.concatenate(ys))
    tree = classifier.tree_
    with torch.no_grad():
        actor.tree.split.weight.zero_(); actor.tree.split.bias.zero_()
        def install(source, heap, depth):
            if depth == 4:
                values = tree.value[source].reshape(-1)
                p = np.full(len(actor.spec.actions), 1e-4)
                p[classifier.classes_] += values / max(values.sum(), 1e-12)
                actor.tree.leaf_logits[heap - 15].copy_(torch.tensor(np.log(p), dtype=torch.float32)); return
            if tree.children_left[source] == tree.children_right[source]:
                actor.tree.split.bias[heap] = -1.
                install(source, 2 * heap + 1, depth + 1); install(source, 2 * heap + 2, depth + 1)
            else:
                actor.tree.split.weight[heap, int(tree.feature[source])] = 1.
                actor.tree.split.bias[heap] = -float(tree.threshold[source])
                install(tree.children_left[source], heap * 2 + 1, depth + 1)
                install(tree.children_right[source], heap * 2 + 2, depth + 1)
        install(0, 0, 0)
    actor.tree.refresh(lock=False)


def train_bc(adapter, model, run, *, initialization_seed, shuffle_seed, action_seed):
    method = MethodSpec(); roles = adapter.spec.roles
    generator = torch.Generator().manual_seed(shuffle_seed)
    behavior_rng = torch.Generator().manual_seed(action_seed)
    stages = {}; stage_rows = []; reset_count = 0
    folder = run.folder / 'bc'; folder.mkdir(parents=True, exist_ok=False)
    initial_hashes = model.role_hashes()
    def collect(name):
        nonlocal reset_count
        if name in stages: return
        stage = next(row for row in BC_STAGES if row[0] == name)
        cases = [c for c in run.plan['cases'] if c['purpose'] == 'bc' and c['stage'] == name]
        if len(cases) != stage[1]: raise ValueError('BC finite stage episode quota differs')
        files = []
        behavior_actor = None
        if stage[5] != 'teacher':
            behavior_actor = export_actor(model, folder/f'{name}-behavior.actor.npz')
            append(run.journal, dict(event='bc_visitation_actor', stage=name, actor=behavior_actor,
                                    action_rng=behavior_rng.get_state().tolist()))
        for i, case in enumerate(cases):
            state = run.reset(adapter, case, purpose='bc'); reset_count += 1
            records, outcome = collect_episode(adapter, model, state, case, behavior_rng, mode=stage[5])
            path = folder / f'{name}-{i:04d}.npz'; _save_rows(path, records, roles); files.append(path)
            # Complete public traces support independent reconstruction and
            # physics replay. Teacher actions never label NN actions as executed.
            trace = [{k: v for k, v in r.items() if k not in ('observations', 'reward', 'probabilities')}
                     | dict(reward=r['reward'].tolist()) for r in records]
            with gzip.open(path.with_suffix('.trace.json.gz'), 'xt') as handle:
                json.dump(dict(case=case, mode=stage[5], behavior_actor=behavior_actor, outcome=outcome, frames=trace), handle, separators=(',', ':'), allow_nan=False)
            append(run.journal, dict(event='bc_episode_completed', case_id=case['case_id'], frames=len(records),
                                    path=str(path), outcome=outcome, teacher_actions_executed=stage[5] == 'teacher'))
        stages[name] = files
    # Teacher-only stages do not depend on changing NN visitation.
    for stage in BC_STAGES:
        if stage[5] == 'teacher': collect(stage[0])
    for r in roles: _initialize_tree(model.actors[r.name], stages['initial_public_teacher'], r.name)
    maximum_gradient = {r.name: 0. for r in roles}
    for fit_name, names, epochs, batch, learning_rate in BC_FITS:
        for name in names: collect(name)
        files = [p for name in names for p in stages[name]]
        optimizer = torch.optim.Adam(model.actors.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            weighted_loss, physical_frames = 0., 0
            for index in torch.randperm(len(files), generator=generator).tolist():
                with np.load(files[index], allow_pickle=False) as loaded:
                    arrays = {k: loaded[k] for k in loaded.files}
                rows = len(arrays[roles[0].name + '_x'])
                for selected in torch.randperm(rows, generator=generator).split(batch):
                    run.ensure_open()
                    optimizer.zero_grad(set_to_none=True); loss = None
                    for role in roles:
                        n = role.name; actor = model.actors[n]
                        x, af, mask, y = (torch.from_numpy(arrays[n + '_' + k][selected.numpy()]) for k in ('x', 'af', 'mask', 'y'))
                        components = actor.components(x, af)
                        part = bc_loss(*components, mask, y)
                        loss = part / 2 if loss is None else loss + part / 2
                    if not torch.isfinite(loss): raise ValueError('nonfinite BC objective')
                    loss.backward()
                    for role in roles:
                        norm = sum(float(p.grad.square().sum()) for p in model.actors[role.name].parameters() if p.grad is not None) ** .5
                        maximum_gradient[role.name] = max(maximum_gradient[role.name], norm)
                    torch.nn.utils.clip_grad_norm_(model.actors.parameters(), method.gradient_clip, error_if_nonfinite=True)
                    optimizer.step(); weighted_loss += float(loss.detach()) * len(selected); physical_frames += len(selected)
            report = dict(stage=fit_name, epoch=epoch, rows=physical_frames, loss=weighted_loss / physical_frames)
            append(run.journal, dict(event='bc_epoch_completed', **report)); print(json.dumps(report), flush=True)
        for actor in model.actors.values(): actor.tree.refresh(lock=fit_name == 'final')
        stage_rows.append(report)
        save_checkpoint(folder / 'checkpoint.pt', dict(version='three-domain-unified-bc-checkpoint.v1',
            method_sha256=method.sha256, domain_sha256=adapter.spec.sha256, model=model.state_dict(),
            completed_fit=fit_name, optimizer=optimizer.state_dict(),
            shuffle_rng=generator.get_state(), behavior_rng=behavior_rng.get_state(), used_cases=sorted(run.used)))
    if reset_count != 1270 or any(model.role_hashes()[r] == initial_hashes[r] or maximum_gradient[r] <= 0 for r in initial_hashes):
        raise ValueError('BC missing complete episodes or independent Actor updates')
    actor = export_actor(model, folder / 'final.actor.npz')
    result = dict(version='three-domain-unified-bc-result.v1', method_sha256=method.sha256,
        domain_sha256=adapter.spec.sha256, episodes=reset_count, actor=actor, stages=stage_rows,
        role_maximum_gradients=maximum_gradient, initial_role_sha256=initial_hashes,
        offline_collision_loss=0, nonuniform_sample_weights=0,
        model_acceptance_passed=False, pilot_opened=False)
    write_new(folder / 'result.json', result)
    return result
