"""Synthetic CPU generation; real official GSM8K scoring, never a model engine."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time

from areal import workflow_context
from areal.api import ModelResponse
from areal.reward.gsm8k import gsm8k_reward_fn


def counted_gsm8k(prompt, completions, prompt_ids, completion_ids, **data):
    # Test evidence only: the method never reads this log as recovery authority.
    if data.get('_cpu_delay'):
        time.sleep(data['_cpu_delay'])
    if data.get('_cpu_inner_exception'):
        # Explicit test injection into the actual strict verifier boundary.
        from unittest.mock import patch
        from areal.reward import get_math_verify_worker
        with patch.object(get_math_verify_worker(), '_verify_impl', side_effect=RuntimeError('injected inner verifier exception')):
            value = gsm8k_reward_fn(prompt, completions, prompt_ids, completion_ids, **data)
    else:
        value = gsm8k_reward_fn(prompt, completions, prompt_ids, completion_ids, **data)
    entry = {'pid': os.getpid(), 'completion': completions, 'value': value}
    fd = os.open(data['_cpu_log'], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, (json.dumps(entry) + '\n').encode())
    finally:
        os.close(fd)
    return value


counted_gsm8k.strict_scoring = True


def asset_hash(root):
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(Path(root).iterdir()) if p.is_file()
              and (p.suffix == '.json' or p.name in ('merges.txt', 'vocab.txt', 'tokenizer.model'))}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


class FakeEngine:
    """Deliberately synthetic tokens/logprobs/versions; optional test-only gate."""
    def __init__(self, tokenizer, *, answer='4', mixed=False, gate=None, gated_index=1):
        self.tokenizer, self.answer, self.mixed = tokenizer, answer, mixed
        self.gate, self.gated_index = gate, gated_index
        self.calls = []
        self.entered = asyncio.Event()

    async def agenerate(self, request):
        index = workflow_context.get().sample_idx
        self.calls.append(index)
        if self.gate is not None and index == self.gated_index:
            self.entered.set()
            await self.gate.wait()
        text = ('Reasoning. ' * (index or 0)) + 'The answer is \\boxed{' + (self.answer[index] if isinstance(self.answer, dict) else self.answer) + '}.'
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        versions = [3] * len(tokens)
        if self.mixed:
            versions[-1] = 4
        return ModelResponse(input_tokens=list(request.input_ids), output_tokens=tokens,
                             output_logprobs=[-0.125] * len(tokens), output_versions=versions,
                             stop_reason='stop', tokenizer=self.tokenizer)


def prepare_response(root, tokenizer_path):
    """A real short-lived owner publishes response only; no ownership spoofing."""
    from transformers import AutoTokenizer
    from areal.api.cli_args import GenerationHyperparameters
    from areal.infra.workflow_context import WorkflowContext
    from scripts.ft import state
    from scripts.ft.rlvr_replay import CallReturnRLVR
    from replay_fixture import opened
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    verifier = hashlib.sha256(Path(gsm8k_reward_fn.__code__.co_filename).read_bytes()).hexdigest()
    owner = state.acquire_owner(root / 'state', -1, run_nonce='cpu-run',
                                config_sha256='b' * 64, verifier_version=verifier)
    with opened(root / 'draw') as loader:
        data = next(loader)[0]
    data.update(answer='4', _cpu_log=str(root / 'score.jsonl'))
    sample = data['_r_draw']['group_id'] + ':0'
    attempt = state.authorize_attempt(owner, sample, None, 'old', expected_policy_version=3)
    workflow = CallReturnRLVR(owner=owner, root=root / 'artifacts', attempts={sample: attempt},
                                 tokenizer_sha256=asset_hash(tokenizer_path), reward_fn=counted_gsm8k,
                                 gconfig=GenerationHyperparameters(n_samples=1), tokenizer=tokenizer)
    async def stop(*args): raise InterruptedError('after durable response')
    workflow._compute_rewards = stop
    workflow_context.set(WorkflowContext(task_id=1, sample_idx=0))
    try:
        try:
            asyncio.run(workflow.arun_episode(FakeEngine(tokenizer), data))
        except InterruptedError:
            pass
        (root / 'old.json').write_text(json.dumps({'data': data, 'attempt': attempt}))
    finally:
        workflow.close(); owner.close()


if __name__ == '__main__':
    import sys
    # Stable pickleable callable module identity, even when run as a script.
    from rlvr_replay_fixture import prepare_response
    prepare_response(sys.argv[1], sys.argv[2])
