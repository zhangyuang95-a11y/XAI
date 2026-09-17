"""Single training entry for frozen current-NN extraction and gated KL feedback."""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import numpy as np
import torch
from core import RCPD, RCPDConfig

class Feedback:
    def __init__(self, config, action_names):
        self.config = dict(config)
        self.actions = tuple(action_names)
        self.engine = RCPD(RCPDConfig(**config['extraction']))
        self.weight = 0.
        self.last_step = -config['interval']
        self.report = {}
        self.updates = 0

    def maybe_extract(self, model, batch, step):
        if not self.config['enabled']:
            self.weight = 0.
            return None
        if step < self.config['warmup_steps'] or step-self.last_step < self.config['interval']:
            return None
        self.last_step = step
        self.weight = 0.
        # Stable content split: identical semantic states never cross partitions,
        # including across extraction rounds. Independent outer holdout is not used
        # for feature preselection or candidate tree selection.
        partitions = [[],[]]
        unique = [set(),set()]
        keys = []
        for i, features in enumerate(batch['features']):
            key = sha256(json.dumps(features,sort_keys=True).encode()).hexdigest()
            keys.append(key)
            split = int(int(key[:8],16)/2**32 < .25)
            unique[split].add(key)
            partitions[split].append(i)
        self.report = {'step':step,'training_samples':len(partitions[0]),'validation_samples':len(partitions[1]),
                       'source':'frozen_current_actor','split_overlap':0,'active':False,
                       'unique_training_states':len(unique[0]),'unique_validation_states':len(unique[1])}
        if min(map(len,unique)) < self.config['minimum_partition_samples']:
            self.report['reason'] = 'insufficient_disjoint_states'
            return self.report
        frozen = deepcopy(model).eval()
        device = next(frozen.parameters()).device
        with torch.no_grad():
            probabilities = frozen.probabilities(torch.as_tensor(batch['obs'],device=device),
                                                torch.as_tensor(batch['roles'],device=device)).cpu().numpy()
        oracle = lambda i: dict(zip(self.actions,map(float,probabilities[i])))
        try:
            result = self.engine.fit(partitions[0],oracle,lambda i:batch['features'][i],validation_states=partitions[1],split_group_provider=lambda i:keys[i])
            self.engine.program = result.program
            self.engine.last_result = result
            self.engine.last_extract_step = step
            self.engine.regularization_weight = self.engine.config.regularization_lambda if result.metrics.feedback_eligible else 0.
            self.weight = min(self.engine.regularization_weight,self.engine.config.maximum_regularization_lambda)
            self.report.update(metrics=result.metrics.to_dict(),complexity=result.program.complexity(),
                               program_sha256=self.program_hash, active=self.weight>0, weight=self.weight)
        except (ValueError, RuntimeError) as exc:
            self.report['reason'] = str(exc)
        return self.report

    @property
    def program_hash(self):
        return sha256(json.dumps(self.engine.program.to_dict(),sort_keys=True).encode()).hexdigest() if self.engine.program else None

    def loss(self, logits, features):
        if not self.weight or self.engine.program is None: return logits.sum()*0
        targets = np.asarray([[self.engine.program.predict_proba(row)[a] for a in self.actions] for row in features],dtype=np.float32)
        # Shared core KL implementation: targets detached, smoothing, NN || program.
        self.updates += 1
        return self.weight*self.engine.regularization_loss(logits,targets)

    def state_dict(self):
        return dict(config=self.config,actions=self.actions,engine=self.engine.training_state(),weight=self.weight,
                    last_step=self.last_step,report=self.report,updates=self.updates)

    def load_state_dict(self, state):
        if state['config'] != self.config or tuple(state['actions']) != self.actions:
            raise ValueError('Feedback configuration/action signature mismatch')
        self.engine.restore_training_state(state['engine'])
        self.weight,self.last_step,self.report,self.updates = (state[k] for k in ['weight','last_step','report','updates'])
