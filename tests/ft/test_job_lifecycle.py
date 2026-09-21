"""CPU verification of the proposed, not-yet-installed common job supervisor."""
import importlib.util
import contextlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace


class JobLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def start(self, code):
        worker = self.root / ('worker-' + str(time.monotonic_ns()) + '.py')
        worker.write_text(code)
        log = self.root / (worker.stem + '.log')
        proc = subprocess.Popen([sys.executable, '-m', 'scripts.ft.job_lifecycle',
            '--log', str(log), '--command', 'exec ' + shlex.join([sys.executable, str(worker)])])
        self.addCleanup(lambda: (proc.kill(), proc.wait()) if proc.poll() is None else None)
        return proc, log

    def receipts(self):
        return [json.loads(p.read_text()) for p in (self.root / 'job-lifecycle').glob('*.json')]

    def wait_file(self, path):
        deadline = time.monotonic() + 5
        while not path.exists():
            if time.monotonic() > deadline:
                self.fail('missing child witness')
            time.sleep(.02)

    def test_actual_success_failure_and_sigkill_exit_codes(self):
        for code in [0, 7, -9]:
            proc, log = self.start('import os,signal\nprint("stdout witness",flush=True)\n'
                + ('os.kill(os.getpid(),signal.SIGKILL)\n' if code == -9 else f'raise SystemExit({code})\n'))
            self.assertEqual(proc.wait(timeout=5), 137 if code == -9 else code)
            record = next(r for r in self.receipts() if r['supervisor_pid'] == proc.pid)
            self.assertEqual(record['root_exit_code'], code)
            self.assertTrue(record['cleanup']['empty'])
            self.assertIn('stdout witness', log.read_text())

    def test_orphan_double_fork_stubborn_children_and_unrelated_sentinel(self):
        sentinel = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'])
        self.addCleanup(lambda: (sentinel.kill(), sentinel.wait()))
        pids = self.root / 'pids'
        proc, _ = self.start(f'''import os,signal,time
from pathlib import Path
p=Path({str(pids)!r})
if os.fork()==0:
 os.setsid()
 signal.signal(signal.SIGTERM,signal.SIG_IGN)
 if os.fork()==0:
  with p.open('a') as f: f.write(str(os.getpid())+'\\n'); f.flush(); os.fsync(f.fileno())
  while True: time.sleep(1)
 with p.open('a') as f: f.write(str(os.getpid())+'\\n'); f.flush(); os.fsync(f.fileno())
 while True: time.sleep(1)
while not p.exists() or len(p.read_text().splitlines())<2: time.sleep(.01)
raise SystemExit(7)
''')
        self.assertEqual(proc.wait(timeout=8), 7)
        record = self.receipts()[0]
        self.assertTrue(record['cleanup']['empty'])
        self.assertTrue(any(s['signal'] == 9 for s in record['cleanup']['signals']))
        for pid in map(int, pids.read_text().splitlines()):
            self.assertFalse(Path(f'/proc/{pid}').exists(), pid)
        self.assertIsNone(sentinel.poll())

    def test_sigterm_supervisor_waits_for_stubborn_root_cleanup(self):
        ready = self.root / 'ready'
        proc, _ = self.start(f'import signal,time\nfrom pathlib import Path\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\nPath({str(ready)!r}).touch()\nwhile True: time.sleep(1)\n')
        self.wait_file(ready)
        proc.terminate()
        self.assertEqual(proc.wait(timeout=5), 137)
        self.assertTrue(self.receipts()[0]['cleanup']['empty'])
        self.assertEqual(self.receipts()[0]['external_signals'], [15])

    def test_cleanup_failure_stays_noncompleted_to_block_native_retry(self):
        log = self.root / 'quarantine.log'
        code = ('from scripts.ft import job_lifecycle as j\n'
                'def fail(): raise RuntimeError("injected cleanup timeout")\n'
                'j.drain_children=fail\n' + f'j.supervise("exit 0",{str(log)!r})\n')
        proc = subprocess.Popen([sys.executable, '-c', code])
        try:
            deadline = time.monotonic() + 5
            while not self.receipts() and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertFalse(self.receipts()[0]['cleanup']['empty'])
            self.assertIsNone(proc.poll())
        finally:
            proc.kill(); proc.wait()


@unittest.skipUnless(os.environ.get('FT_LIFECYCLE_NATIVE_CPU') == '1', 'requires actual AReaL CPU image')
class NativeLauncher(unittest.TestCase):
    def local_copy(self, root):
        repo = Path(__file__).resolve().parents[2]
        original = (repo / 'third_party/areal/areal/infra/launcher/local.py').read_text()
        lines = (repo / 'docs/experiments/rewardtxn-ft-20260916/local-job-lifecycle.patch').read_text().splitlines(True)
        hunk = lines[next(i for i,l in enumerate(lines) if l.startswith('@@')) + 1:]
        old = ''.join(l[1:] for l in hunk if l[0] in ' -')
        new = ''.join(l[1:] for l in hunk if l[0] in ' +')
        if old in original:
            original = original.replace(old, new)
        else:
            self.assertIn(new, original)
        proposed = root / 'proposed_local.py'
        proposed.write_text(original)
        spec = importlib.util.spec_from_file_location('proposed_local', proposed)
        local = importlib.util.module_from_spec(spec); spec.loader.exec_module(local)
        return local

    def test_proposed_patch_preserves_native_submit_stop_and_log_environment(self):
        with contextlib.nullcontext(tempfile.mkdtemp(prefix='native-', dir='/output')) as tmp:
            root = Path(tmp); local = self.local_copy(root)
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}):
                launcher = local.LocalLauncher('cpu-lifecycle', 'proposal', str(root))
                try:
                    command = shlex.join([sys.executable, '-c', 'import os;print(os.environ["LIFECYCLE_VALUE"],flush=True)'])
                    launcher.submit('success', command, env_vars={'LIFECYCLE_VALUE': 'literal_value'})
                    with self.assertRaises(local.JobException):
                        launcher.wait(timeout=8, check_status=(local.JobState.COMPLETED,), remove_status=())
                    log = Path(launcher.log_path_of('success'))
                    self.assertIn('literal_value', log.read_text())
                    launcher.stop_all('SIGTERM')
                    launcher = local.LocalLauncher('cpu-lifecycle', 'proposal-stop', str(root))
                    launcher.submit('long', shlex.join([sys.executable, '-c', 'import time;time.sleep(30)']))
                    time.sleep(.5)
                    launcher.stop_all('SIGTERM')
                    self.assertFalse(launcher._jobs)
                finally:
                    if launcher._jobs: launcher.stop_all('SIGKILL')
                    launcher._jobs.clear()


    def test_actual_local_main_retries_killed_torchrun_with_orphan_stdout(self):
        from omegaconf import OmegaConf
        repo = Path(__file__).resolve().parents[2]
        with contextlib.nullcontext(tempfile.mkdtemp(prefix='native-', dir='/output')) as tmp:
            root = Path(tmp); local = self.local_copy(root)
            worker = root / 'retry_worker.py'
            witness = root / 'incarnations.jsonl'
            worker.write_text("import os,json,signal,time\nfrom pathlib import Path\n"
                + f"p=Path({str(witness)!r})\nfirst=not p.exists()\n"
                + "with p.open('a') as f: f.write(json.dumps({'pid':os.getpid(),'first':first})+'\\n'); f.flush(); os.fsync(f.fileno())\n"
                + "if first:\n if os.fork()==0:\n  signal.signal(signal.SIGTERM,signal.SIG_IGN)\n  while True: time.sleep(1)\n os.kill(os.getpid(),signal.SIGKILL)\n")
            config = OmegaConf.load(repo / 'docs/experiments/rewardtxn-ft-20260916/native-trainer.yaml')
            config.cluster.fileroot = str(root / 'areal')
            config.cluster.name_resolve.nfs_record_root = str(root / 'names')
            config.actor.scheduling_spec[0].cpu = 1
            # Only replace GPU allocation/validation; local_main retry, real
            # torchrun, Popen/wait/stop and the 10-second retry interval are real.
            allocation = SimpleNamespace(gen_backend='cpu-fixture', type_=None,
                allocations=[], train=SimpleNamespace(world_size=1))
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':'0'}), \
                 patch.object(local, 'current_platform', SimpleNamespace(device_control_env_var='CUDA_VISIBLE_DEVICES', device_count=lambda:1)), \
                 patch.object(local, 'validate_config_for_launcher'), \
                 patch.object(local._AllocationMode, 'from_str', return_value=allocation), \
                 patch.object(sys, 'argv', ['local', str(worker)]):
                with self.assertRaises(local.JobException):
                    local.local_main(config)
            incarnations = [json.loads(l) for l in witness.read_text().splitlines()]
            self.assertEqual([r['first'] for r in incarnations], [True, False])
            self.assertNotEqual(incarnations[0]['pid'], incarnations[1]['pid'])
            records = [json.loads(p.read_text()) for p in root.rglob('job-lifecycle/*.json')]
            records.sort(key=lambda r:r['started_ns'])
            self.assertEqual(len(records),2)
            self.assertNotEqual(records[0]['root_exit_code'],0)
            self.assertEqual(records[1]['root_exit_code'],0)
            self.assertTrue(all(r['cleanup']['empty'] for r in records))
            self.assertLess(records[0]['finished_ns'], records[1]['started_ns'])
            self.assertTrue(records[0]['cleanup']['signals'])

if __name__ == '__main__':
    unittest.main()
