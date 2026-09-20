"""Method-owned call-return evidence; NOT proof of internal verifier success."""
from dataclasses import dataclass
import hashlib
import json
import math


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

    def __call__(self, prompt, completion, prompt_ids, completion_ids, *, _r_reward_request, **data):
        # Do not catch scorer exceptions or implement retries. The official
        # wrapper owns that policy; inner scorer fallback zeros remain zeros.
        actual = input_digest(prompt, completion, prompt_ids, completion_ids, data)
        if actual != _r_reward_request['input_sha256']:
            raise ValueError('scoring input binding mismatch')
        score = self.scorer(prompt, completion, prompt_ids, completion_ids, **data)
        if type(score) not in (int, float) or not math.isfinite(score) or score not in (0, 1):
            raise ValueError('unsupported official scoring return')
        return RewardReturn(1, _r_reward_request['invocation_nonce'], actual,
                            _r_reward_request['verifier_sha256'], 'official_call_returned', float(score))
