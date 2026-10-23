"""The same differentiable Actor and critics for every registered domain."""
from hashlib import sha256
import numpy as np
import torch
from torch import nn
from .contracts import MethodSpec, RoleSpec, source_support


def mask_logits(logits, masks):
    if (logits.shape != masks.shape or torch.isnan(logits).any() or torch.isposinf(logits).any()
            or not torch.isfinite(logits.masked_select(masks.bool())).all()
            or not torch.all((masks == 0) | (masks == 1)) or not masks.bool().any(-1).all()):
        raise ValueError('finite logits and nonempty physical binary masks required')
    return logits.masked_fill(~masks.bool(), -torch.inf)


class SparseTree(nn.Module):
    def __init__(self, role: RoleSpec):
        super().__init__(); self.role = role
        self.split = nn.Linear(len(role.features), 15)
        self.leaf_logits = nn.Parameter(torch.zeros(16, len(role.actions)))
        self.register_buffer('support_mask', torch.zeros(15, len(role.features), dtype=torch.bool))
        self.register_buffer('support_fixed', torch.tensor(False))
        self.refresh(lock=False)

    def refresh(self, *, lock):
        with torch.no_grad():
            support = torch.zeros_like(self.support_mask)
            for node, row in enumerate(self.split.weight):
                selected = []
                for i in torch.argsort(-row.abs(), stable=True).tolist():
                    if self.role.features[i].lower == self.role.features[i].upper:
                        continue
                    if len(source_support(self.role.features, [*selected, i])) <= 3:
                        selected.append(i)
                    if len(source_support(self.role.features, selected)) >= 3:
                        break
                support[node, selected] = True
            self.support_mask.copy_(support); self.support_fixed.fill_(lock)

    def effective_weights(self):
        weights = self.split.weight
        # Numerically zero straight-through term; hard support is unchanged.
        return weights * self.support_mask + .1 * (weights - weights.detach())

    def forward(self, x):
        score = x @ self.effective_weights().T + self.split.bias
        # The Warehouse forward routes in float64 and learns through a soft
        # left/right mixture. Include both subtree outputs in its derivative.
        hard = torch.as_tensor(np.einsum('...i,ji->...j', x.detach().cpu().numpy().astype(float),
            self.effective_weights().detach().cpu().numpy().astype(float), optimize=False)
            + self.split.bias.detach().cpu().numpy().astype(float), device=x.device)
        left = (hard <= 1e-10).to(x.dtype)
        soft = torch.sigmoid(-4. * score)
        gates = left + (soft - soft.detach())
        values = [self.leaf_logits[i].expand(*x.shape[:-1], -1) for i in range(16)]
        paths = [torch.full(x.shape[:-1], i, dtype=torch.long, device=x.device) for i in range(16)]
        for start, count in ((7, 8), (3, 4), (1, 2), (0, 1)):
            values = [gates[..., start+i, None] * values[2*i] + (1-gates[..., start+i, None]) * values[2*i+1] for i in range(count)]
            paths = [torch.where(left[..., start+i].bool(), paths[2*i], paths[2*i+1]) for i in range(count)]
        return values[0], paths[0]


class RoleActor(nn.Module):
    def __init__(self, role: RoleSpec):
        super().__init__(); self.spec = role
        d, a = len(role.features), len(role.actions)
        self.tree = SparseTree(role)
        self.gate = nn.Sequential(nn.Linear(d + 4, 64), nn.ReLU(), nn.Linear(64, 1))
        self.primary = nn.Sequential(nn.Linear(d + 6 + a + role.action_feature_dim, 256), nn.ReLU(),
                                     nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
        nn.init.zeros_(self.gate[-1].weight); nn.init.zeros_(self.gate[-1].bias)
        nn.init.normal_(self.primary[-1].weight, std=.01); nn.init.zeros_(self.primary[-1].bias)

    def components(self, x, action_features):
        a = len(self.spec.actions)
        if x.shape[-1] != len(self.spec.features) or action_features.shape != (*x.shape[:-1], a, self.spec.action_feature_dim):
            raise ValueError('public observation or per-action feature shape differs')
        own = x[..., list(self.spec.position_indices[:2])]
        partner = x[..., list(self.spec.position_indices[2:])]
        relative = own - partner
        gate = torch.tanh(self.gate(torch.cat((x, relative, relative.abs()), -1)))
        move = x.new_tensor(self.spec.action_deltas)
        gap = own.unsqueeze(-2) + move - partner.unsqueeze(-2)
        geometry = torch.cat((relative.unsqueeze(-2).expand_as(gap), gap, gap.abs()), -1)
        repeated = x.unsqueeze(-2).expand(*x.shape[:-1], a, x.shape[-1])
        eye = torch.eye(a, dtype=x.dtype, device=x.device).expand(*x.shape[:-1], a, a)
        primary = self.primary(torch.cat((repeated, geometry, eye, action_features), -1)).squeeze(-1)
        tree, _ = self.tree(x)
        full = primary + gate * (tree - tree.mean(-1, keepdim=True))
        return full, tree, primary

    def forward(self, x, action_features, masks):
        return mask_logits(self.components(x, action_features)[0], masks)


class DualActorCritic(nn.Module):
    def __init__(self, domain):
        super().__init__(); self.domain = domain; self.method_sha256 = MethodSpec().sha256
        self.actors = nn.ModuleDict({r.name: RoleActor(r) for r in domain.roles})
        d = sum(len(r.features) for r in domain.roles)
        self.reward_critic = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 2))
        self.cost_critic = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

    def role_hashes(self):
        return {name: tensor_hash(actor.state_dict()) for name, actor in self.actors.items()}


def tensor_hash(state):
    h = sha256()
    for name, tensor in sorted(state.items()):
        a = np.ascontiguousarray(tensor.detach().cpu().numpy())
        h.update(name.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


def _linear(x, layer):
    # Separate frozen implementation, never calls Torch or a teacher.
    return np.einsum('...i,ji->...j', x, layer[0], optimize=False) + layer[1]


class FrozenRole:
    def __init__(self, role, arrays):
        self.spec = role
        self.arrays = {k: np.array(v, copy=True) for k, v in arrays.items()}
        for value in self.arrays.values():
            value.flags.writeable = False

    @classmethod
    def from_actor(cls, actor):
        return cls(actor.spec, {k: v.detach().cpu().numpy() for k, v in actor.state_dict().items()})

    def logits(self, x, action_features, masks):
        x = np.asarray(x, np.float32); af = np.asarray(action_features, np.float32)
        masks = np.asarray(masks)
        s, w = self.spec, self.arrays
        if (x.shape[-1] != len(s.features) or af.shape != (*x.shape[:-1], len(s.actions), s.action_feature_dim)
                or masks.shape != (*x.shape[:-1], len(s.actions)) or not np.isin(masks, [0, 1]).all()
                or not masks.any(-1).all() or not np.isfinite(x).all() or not np.isfinite(af).all()):
            raise ValueError('frozen inference input differs')
        linear = lambda z, prefix: _linear(z, (w[prefix + '.weight'], w[prefix + '.bias']))
        own, partner = x[..., list(s.position_indices[:2])], x[..., list(s.position_indices[2:])]
        relative = own - partner
        g = np.concatenate((x, relative, np.abs(relative)), -1)
        gate = np.tanh(linear(np.maximum(linear(g, 'gate.0'), 0), 'gate.2'))
        gap = own[..., None, :] + np.asarray(s.action_deltas, np.float32) - partner[..., None, :]
        geometry = np.concatenate((np.broadcast_to(relative[..., None, :], gap.shape), gap, np.abs(gap)), -1)
        a = len(s.actions)
        relation = np.concatenate((np.broadcast_to(x[..., None, :], (*x.shape[:-1], a, x.shape[-1])), geometry,
                                   np.broadcast_to(np.eye(a, dtype=np.float32), (*x.shape[:-1], a, a)), af), -1)
        primary = linear(np.maximum(linear(np.maximum(linear(relation, 'primary.0'), 0), 'primary.2'), 0), 'primary.4')[..., 0]
        score = np.einsum('...i,ji->...j', x.astype(float),
            (w['tree.split.weight'] * w['tree.support_mask']).astype(float), optimize=False) + w['tree.split.bias'].astype(float)
        node = np.zeros(x.shape[:-1], dtype=np.int64)
        for _ in range(4):
            selected = np.take_along_axis(score, node[..., None], -1)[..., 0]
            node = node * 2 + 1 + (selected > 1e-10)
        tree = w['tree.leaf_logits'][node - 15]
        logits = primary + gate * (tree - tree.mean(-1, keepdims=True))
        if not np.isfinite(logits).all(): raise ValueError('frozen NN produced nonfinite logits')
        return np.where(masks, logits, -np.inf)
