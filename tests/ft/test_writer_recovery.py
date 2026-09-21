"""Real subreaper cleanup and adversarial pending-save receipt checks on CPU."""
import copy
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

from scripts.ft import state, writer_recovery


class PendingWriter(unittest.TestCase):
    def test_real_orphan_cleanup_required_before_owner_takeover(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worker = root / 'worker.py'
            worker.write_text('''import os,time,signal
from pathlib import Path
from scripts.ft import state,writer_recovery
root=Path(__file__).parent
owner=state.acquire_owner(root/'state',-1,run_nonce='cpu',config_sha256='a'*64,verifier_version='cpu')
gate=writer_recovery.pending_gate(owner,'unfinished-candidate')
state._write(root/'gate.json',gate)
if os.fork()==0:
 signal.signal(signal.SIGTERM,signal.SIG_IGN)
 with (root/'partial.distcp').open('wb') as f:
  f.write(b'partial');f.flush();os.fsync(f.fileno())
  state._write(root/'writer.json',state.process_identity(os.getpid()))
  while True:time.sleep(1)
while True:time.sleep(1)
''')
            proc = subprocess.Popen([sys.executable, '-m', 'scripts.ft.job_lifecycle',
                '--log', str(root/'trainer.log'), '--command', 'exec '+shlex.join([sys.executable,str(worker)])])
            trainer_fd = None
            try:
                deadline = time.monotonic()+10
                while not (root/'writer.json').exists():
                    if time.monotonic()>deadline: self.fail('CPU writer did not start')
                    time.sleep(.02)
                gate=state._read(root/'gate.json');control=state._read(root/'state/control.json')
                with self.assertRaisesRegex(RuntimeError,'still alive'):
                    writer_recovery.verify_cleanup(gate,control)
                with self.assertRaises((BlockingIOError,state.StateError)):
                    state.acquire_owner(root/'state',0,control['processes'])
                trainer_fd=os.pidfd_open(gate['owner']['pid'])
                signal.pidfd_send_signal(trainer_fd,signal.SIGKILL)
                self.assertEqual(proc.wait(timeout=10),137)
                proof=writer_recovery.verify_cleanup(gate,control)
                receipt=state._read(Path(proof['receipt']))
                writer=state._read(root/'writer.json')
                self.assertTrue(state._exited(writer))
                self.assertTrue(any(e['pid']==writer['pid'] for e in receipt['cleanup']['exits']))
                self.assertEqual((root/'partial.distcp').read_bytes(),b'partial')
                for field,value in [('pid_namespace',-1),('owner_nonce','wrong'),('epoch',7),('job',None)]:
                    bad=copy.deepcopy(gate);bad[field]=value
                    with self.subTest(field=field),self.assertRaises(RuntimeError):
                        writer_recovery.verify_cleanup(bad,control)
                for field,value in [('captured_ns',receipt['finished_ns']+1),('command','foreign job')]:
                    bad=copy.deepcopy(gate);bad['job'][field]=value
                    with self.subTest(field=field),self.assertRaises(RuntimeError):
                        writer_recovery.verify_cleanup(bad,control)
                path=Path(proof['receipt'])
                for cleanup in [{'empty':False},{'empty':True,'error':'quarantined'}]:
                    bad=copy.deepcopy(receipt);bad['cleanup']=cleanup
                    path.write_text(json.dumps(bad))
                    with self.subTest(cleanup=cleanup),self.assertRaises(RuntimeError):
                        writer_recovery.verify_cleanup(gate,control)
                path.unlink()
                with self.assertRaises(RuntimeError):writer_recovery.verify_cleanup(gate,control)
                path.write_text(json.dumps(receipt))
                owner=state.acquire_owner(root/'state',0,control['processes'])
                try:
                    self.assertEqual(owner.epoch,1)
                    self.assertIsNone(state.select_recovery(owner)['generation'])
                    with self.assertRaises(RuntimeError):
                        writer_recovery.verify_cleanup(gate,state._read(root/'state/control.json'))
                finally:owner.close()
            finally:
                if trainer_fd is not None:os.close(trainer_fd)
                if proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()


if __name__=='__main__':unittest.main()
