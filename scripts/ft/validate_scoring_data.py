"""Read-only reference-answer preflight through the bounded common scorer."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path


def check_answers(entries):
    from areal.reward import get_math_verify_worker
    from areal.utils.reward_status import ScoringFailure
    worker = get_math_verify_worker()
    for row, answer in entries:
        if not isinstance(answer, str):
            raise ScoringFailure('invalid_gold', f'row {row}: reference is not a string')
        try:
            score = worker.verify_strict(answer, answer)
        except ScoringFailure as exc:
            raise ScoringFailure(exc.status, f'row {row}: {exc.detail}') from exc
        if score != 1:
            raise ScoringFailure('invalid_gold', f'row {row}: reference failed self-comparison')
    return 1.0


check_answers.strict_scoring = True


async def validate(path):
    from areal.api import AsyncRewardWrapper
    from areal.utils.reward_status import RewardEvaluationError
    from scripts.ft.reward_return import verifier_fingerprint
    raw = path.read_bytes()
    entries = [(i + 1, json.loads(line)['label']) for i, line in enumerate(raw.splitlines())]
    wrapper = AsyncRewardWrapper(check_answers)
    result = {'dataset_sha256': hashlib.sha256(raw).hexdigest(), 'rows': len(entries),
              'verifier_sha256': verifier_fingerprint(), 'checked_rows': 0,
              'scope': 'reference extraction and self-comparison; no answer correctness claim'}
    for offset in range(0, len(entries), 64):
        try:
            await wrapper(entries[offset:offset + 64])
        except RewardEvaluationError as exc:
            return {**result, 'verified': False, 'failed_chunk_first_row': offset + 1,
                    'attempts': exc.attempts}
        result['checked_rows'] += len(entries[offset:offset + 64])
    return {**result, 'verified': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = asyncio.run(validate(args.dataset))
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    raise SystemExit(0 if result['verified'] else 1)
