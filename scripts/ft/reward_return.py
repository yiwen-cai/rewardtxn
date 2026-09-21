"""Bound reward evidence; strict scorers cannot return fallback failure zeros."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path


def verifier_fingerprint():
    from areal.reward import MathVerifyWorker
    from areal.reward.gsm8k import gsm8k_reward_fn
    from areal.api import AsyncRewardWrapper
    from areal.utils import strict_reward, reward_status
    from math_verify import parser, grader
    import inspect
    objects = (MathVerifyWorker, gsm8k_reward_fn, AsyncRewardWrapper,
               strict_reward, reward_status, parser, grader)
    hashes = {obj.__module__ if isinstance(obj, type) or callable(obj) else obj.__name__:
              hashlib.sha256(Path(inspect.getfile(obj)).read_bytes()).hexdigest() for obj in objects}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def input_digest(prompt, completion, prompt_ids, completion_ids, data):
    raw = json.dumps([prompt, completion, prompt_ids, completion_ids, data],
                     sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass(frozen=True)
class RewardReturn:
    schema: int
    invocation_nonce: str
    input_sha256: str
    verifier_sha256: str
    status: str
    score: float


@dataclass(frozen=True)
class ReturningReward:
    scorer: object

    @property
    def strict_scoring(self):
        return bool(getattr(self.scorer, 'strict_scoring', False))

    def __call__(self, prompt, completion, prompt_ids, completion_ids, *, _r_reward_request, **data):
        # Retry/timeout/worker cleanup belongs to the common AsyncRewardWrapper.
        actual = input_digest(prompt, completion, prompt_ids, completion_ids, data)
        if actual != _r_reward_request['input_sha256']:
            raise ValueError('scoring input binding mismatch')
        score = self.scorer(prompt, completion, prompt_ids, completion_ids, **data)
        if type(score) not in (int, float) or not math.isfinite(score) or score not in (0, 1):
            raise ValueError('unsupported official scoring return')
        return RewardReturn(2 if self.strict_scoring else 1, _r_reward_request['invocation_nonce'], actual,
                            _r_reward_request['verifier_sha256'],
                            'scored' if self.strict_scoring else 'official_call_returned', float(score))
