#!/usr/bin/env python3
"""One-shot, read-only status check of the two Oracle repetitions."""
import datetime
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'runs/oracle-repeat-s11-20260911-083022'


def main():
    manifest = json.loads((DIRECTORY / 'manifest.json').read_text())
    lines = ['# Oracle 重复实验定时检查', '',
             '检查时间：' + datetime.datetime.now().astimezone().isoformat(), '']
    for name in manifest['runs']:
        run = ROOT / 'runs' / name
        lines.extend(['## ' + name, ''])
        state = subprocess.run(['docker', 'inspect', '--format', '{{json .State}}', 'rtx-p2-' + name],
                               capture_output=True, text=True, timeout=20)
        if state.returncode == 0:
            d = json.loads(state.stdout)
            lines.append('容器状态：{}；退出码：{}；OOM：{}'.format(d['Status'], d['ExitCode'], d['OOMKilled']))
        else:
            lines.append('容器状态未取得：' + state.stderr.strip())
        log = run / 'logs/train.log'
        if log.exists():
            text = log.read_text(errors='replace')
            steps = re.findall(r'model.py:\d+ - step (\d+):', text)
            lines.append('最新训练步：' + (str(int(steps[-1]) + 1) + '/500' if steps else '尚无训练步记录'))
            errors = [x for x in text.splitlines() if re.search(r'Traceback|OutOfMemoryError|ERROR', x)]
            if errors:
                lines.extend(['错误记录（需结合终态判断）：', '```', *errors[-5:], '```'])
        evaluation = run / 'restart_eval/validation.json'
        if evaluation.exists():
            d = json.loads(evaluation.read_text())
            lines.append('验证准确率：{}/{}；截断率：{:.1%}'.format(d['n_correct'], d['n_total'], d['truncated_fraction']))
        else:
            lines.append('验证评估结果尚未生成。')
        lines.append('')
    comparison = DIRECTORY / 'comparison.json'
    if comparison.exists():
        lines.extend(['## 两次结果对比', '', '```json', comparison.read_text().strip(), '```'])
    else:
        lines.append('最终比较尚未生成；本任务只检查一次，不自动重跑或继续轮询。')
    output = DIRECTORY / 'scheduled_status.md'
    output.write_text('\n'.join(lines) + '\n')
    print(output)


if __name__ == '__main__':
    main()
