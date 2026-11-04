"""Check the selected six-task L1 label/Student path independently of old E400."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from check_l1_six_task import model, assert_nested
from check_utility_e400 import make_batch
from policy.multitask_dp.l1 import FINAL_TASKS
from policy.multitask_dp.l1_uvip import L1Stage2Policy, evaluate_draws, selected_index, load_teacher
from policy.multitask_dp.stage2 import set_stage_trainability
from policy.multitask_dp.train_stage2 import Stage2Trainer
from policy.multitask_dp.utility import window_draws


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--teacher', type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ['UNIVTAC_STAGE2_STOP_FILE'] = str(args.output / 'STOP')
    torch.set_num_threads(2)
    cfg = yaml.safe_load((ROOT / 'configs/l1_six_task.yaml').read_text())
    cfg.update(down_dims=[16, 32], diffusion_step_embed_dim=16, num_inference_steps=2)
    teacher = model(cfg).eval().requires_grad_(False)
    value = make_batch()
    steps, noise = zip(*(window_draws(i, 11, 22) for i in range(2)))
    steps, noise = np.stack(steps), np.stack(noise)
    actual = evaluate_draws(teacher, value, steps, noise, 73)
    for draw in (0, 17, 63):
        expected = teacher.paired_denoising_losses(value, torch.from_numpy(noise[:, draw]), torch.from_numpy(steps[:, draw]))
        np.testing.assert_allclose(actual[:, draw], expected['losses'].numpy(), rtol=5e-5, atol=3e-6)
    alternate = evaluate_draws(teacher, value, steps, noise, 128)
    np.testing.assert_allclose(actual, alternate, rtol=5e-5, atol=3e-6)
    print('PASS cached L1 K64 scoring equals native paired loss and is microbatch independent', flush=True)
    def student():
        result = L1Stage2Policy(cfg, np.ones(8), copy.deepcopy(teacher.obs_encoder).requires_grad_(True))
        result.initialize_from_teacher(teacher)
        return result
    policy = student()
    encoded = policy.encode_stage2(value['obs'])
    tokens = teacher.obs_encoder.encode_tokens(value['obs'], teacher.normalizer)
    expected = teacher.obs_encoder.condition(tokens, value['obs']['language'][:, 0], teacher.condition_presence[:1].expand(2,-1))
    torch.testing.assert_close(encoded['z_t'], expected, rtol=0, atol=0)
    torch.testing.assert_close(encoded['tokens'], encoded['modulated_tokens'], rtol=0, atol=0)
    assert len(policy.film_hypernets) == 4 and encoded['u_hat'].shape == (2,8)
    assert any(not torch.equal(v, policy.model.state_dict()[k]) for k,v in teacher.model.state_dict().items())
    value['utility'] = torch.randn(2,8)
    for phase in ('A','B','C'):
        set_stage_trainability(policy, phase)
        policy.zero_grad(set_to_none=True)
        result = policy.training_losses(value, phase)
        result['utility_loss'].backward(retain_graph=True)
        assert any(p.grad is not None for p in policy.utility_predictor.parameters())
        assert all(p.grad is None for name,p in policy.named_parameters() if not name.startswith('utility_predictor.'))
        policy.zero_grad(set_to_none=True)
        if phase != 'A':
            result['action_loss'].backward()
            assert all(p.grad is None for p in policy.utility_predictor.parameters())
            assert any(p.grad is not None for p in policy.obs_encoder.parameters()) == (phase == 'C')
        assert all(p.requires_grad == (phase != 'A') for p in policy.language_token_projection.parameters())
    assert policy.eval().predict_action(value['obs'])['action'].shape == (2,8,8)
    print('PASS exact L1 context, four identity FiLMs, fresh actor, A/B/C gradients and native action sampling',flush=True)
    class Source:
        manifest = dict(tasks=['excluded_a','excluded_b',*FINAL_TASKS], total_steps=160,
            episodes=[dict(start=20*i,end=20*(i+1),task_index=i) for i in range(8)])
    index = selected_index(Source())
    assert len(index)==30 and set(index['task_index'])==set(range(2,8))
    np.testing.assert_array_equal(index['window_id'],np.arange(30))
    print('PASS six-task complete-window filtering excludes both omitted tasks',flush=True)
    class Labels(torch.utils.data.Dataset):
        contract_id = 'fixture'
        contract = {'teacher':{'epoch':300,'weights':'ema'}}
        def __len__(self):return 2
        def __getitem__(self,i):
            return {k:({s:t[i] for s,t in v.items()} if isinstance(v,dict) else v[i]) for k,v in value.items()}
        def epoch_indices(self,seed,epoch):return [0,1]
    initial = copy.deepcopy(student())
    config=dict(seed=42,batch_size=2,num_epochs=7,num_workers=0)
    def trainer(name):
        torch.manual_seed(42);np.random.seed(42);random.seed(42)
        return Stage2Trainer(copy.deepcopy(initial), Labels(), config, args.output / name, 'cpu')
    a=trainer('continuous');a.fit()
    b=trainer('resumed');b.fit(max_steps=6)
    c=trainer('resumed');c.restore(args.output / 'resumed/checkpoints/latest.ckpt');c.fit()
    assert_nested(a.policy.state_dict(),c.policy.state_dict())
    assert_nested(a.ema.state_dict(),c.ema.state_dict())
    assert_nested(a.optimizer.state_dict(),c.optimizer.state_dict())
    assert_nested(a.scheduler.state_dict(),c.scheduler.state_dict())
    print('PASS exact raw/EMA/Adam/scheduler continuation across A/B/C',flush=True)
    report=dict(passed=True,checks=['native_paired_loss','batch_independence','teacher_context','identity_film','fresh_actor',
        'phase_freeze','gradient_isolation','native_sampling','six_task_filtering','exact_resume'])
    if args.teacher:
        real, identity=load_teacher(args.teacher)
        report['teacher']=identity
        print('PASS actual selected E300 EMA strict restore and normalizer',flush=True)
    (args.output / 'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
