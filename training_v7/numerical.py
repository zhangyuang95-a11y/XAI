"""Actual exported-NN checks on a declared, fixed training-frame cache."""
import numpy as np
import torch
from .contracts import MethodSpec, write_new
from .storage import load_frozen


def verify_export(model, evidence, asset, output):
    frozen = load_frozen(asset['path'], asset['sha256'], model.domain, asset['role_sha256'])
    reports = {}
    for role in model.domain.roles:
        source = evidence.arrays[role.name]['fit']
        indices = np.unique(np.linspace(0, len(source['x'])-1, min(4096, len(source['x'])), dtype=int))
        mismatches, maximum_probability_error, maximum_logit_error = 0, 0., 0.
        for start in range(0, len(indices), 128):
            selected = indices[start:start+128]
            x, af, mask = (source[k][selected] for k in ('x', 'action_features', 'mask'))
            with torch.no_grad():
                logits = model.actors[role.name](torch.from_numpy(x), torch.from_numpy(af), torch.from_numpy(mask))
                p = logits.softmax(-1).numpy(); original = logits.numpy()
            # Use the exact scalar frozen call used by participant execution,
            # not a changed matrix shape masquerading as the runtime kernel.
            actual = np.stack([frozen[role.name].logits(x[i], af[i], mask[i]) for i in range(len(x))])
            q = np.exp(actual.astype(float)-np.max(actual, axis=-1, keepdims=True)); q /= q.sum(-1, keepdims=True)
            maximum_logit_error = max(maximum_logit_error, float(np.max(np.abs(actual[mask]-original[mask]), initial=0)))
            maximum_probability_error = max(maximum_probability_error, float(np.max(np.abs(p-q), initial=0)))
            mismatches += int(np.count_nonzero(actual.argmax(-1) != original.argmax(-1)))
        reports[role.name] = dict(frames=len(indices), argmax_mismatches=mismatches,
            maximum_logit_error=maximum_logit_error, maximum_probability_error=maximum_probability_error,
            passed=mismatches == 0 and maximum_logit_error <= 1e-4 and maximum_probability_error <= 1e-5)
    result = dict(version='three-domain-unified-export-numerical-check.v1', actor=asset,
        method_sha256=MethodSpec().sha256, scope='at_most_4096_fixed_strided_fit_training_frames_per_role',
        roles=reports, passed=all(r['passed'] for r in reports.values()),
        scientific_quality_evaluation=False, model_acceptance_passed=False, pilot_opened=False)
    write_new(output, result)
    if not result['passed']: raise ValueError('Torch and actual frozen scalar inference differ')
    return result
