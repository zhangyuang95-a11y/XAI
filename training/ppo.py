import numpy as np
import torch


def epoch_minibatches(length, batch_size, rng):
    order = rng.permutation(length)
    return [order[i:i+batch_size] for i in range(0,length,batch_size)]


def update(model, actor_optimizer, critic_optimizer, batch, cfg, rng, feedback=None):
    device = next(model.parameters()).device
    data = {k:torch.as_tensor(v,device=device) for k,v in batch.items() if k != 'features'}
    advantage = data['advantages'].clone()
    mask = data['mask'].bool()
    if mask.any():
        selected = advantage[mask]
        advantage[mask] = (selected-selected.mean()) / (selected.std(unbiased=False)+1e-8)
    metrics = dict(actor_loss=0.,critic_loss=0.,feedback_loss=0.,feedback_kl=0.,feedback_grad_norm=0.,actor_updates=0,critic_updates=0)
    for _ in range(cfg['epochs']):
        for ix in epoch_minibatches(len(mask),cfg['minibatch_size'],rng):
            ix = torch.as_tensor(ix,device=device)
            neural = ix[mask[ix]]
            if len(neural):
                logits = model.logits(data['obs'][neural],data['roles'][neural])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = (dist.log_prob(data['actions'][neural])-data['logp'][neural]).exp()
                a = advantage[neural]
                loss = -torch.minimum(ratio*a,ratio.clamp(1-cfg['clip'],1+cfg['clip'])*a).mean() - cfg['entropy']*dist.entropy().mean()
                penalty = logits.sum()*0
                if feedback is not None:
                    penalty = feedback.loss(logits,[batch['features'][i] for i in neural.cpu().tolist()])
                if penalty.requires_grad and feedback is not None and feedback.weight > 0:
                    grads = torch.autograd.grad(penalty,model.actor.parameters(),retain_graph=True,allow_unused=True)
                    metrics['feedback_grad_norm'] += float(sum(g.square().sum() for g in grads if g is not None).sqrt())
                actor_optimizer.zero_grad()
                (loss+penalty).backward()
                torch.nn.utils.clip_grad_norm_(model.actor.parameters(),cfg['max_grad_norm'])
                actor_optimizer.step()
                metrics['actor_loss'] += float(loss.detach())
                metrics['feedback_loss'] += float(penalty.detach())
                if feedback is not None and feedback.weight > 0:
                    metrics['feedback_kl'] += float(penalty.detach()) / feedback.weight
                metrics['actor_updates'] += 1
            value = model.values(data['states'][ix],data['roles'][ix])
            value_loss = .5*(value-data['returns'][ix]).square().mean()
            critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.critic.parameters(),cfg['max_grad_norm'])
            critic_optimizer.step()
            metrics['critic_loss'] += float(value_loss.detach())
            metrics['critic_updates'] += 1
    for name in ['actor_loss','feedback_loss','feedback_kl','feedback_grad_norm']:
        metrics[name] /= max(1,metrics['actor_updates'])
    metrics['critic_loss'] /= max(1,metrics['critic_updates'])
    return metrics
