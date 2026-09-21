"""Exercise the actual installed MCore DCP tensor writer on CPU."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest


class NativeWriterProbe(unittest.TestCase):
    def run_writer(self, enabled):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            worker=root/'worker.py'
            worker.write_text('''from pathlib import Path
from types import SimpleNamespace
import torch
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync
from scripts.ft.areal_midwrite_fault import install_writer_probe
root=Path(__file__).parent
path=root/'actual.distcp'
runtime=SimpleNamespace(root=root,generation='cpu',ordinal=2,probe_target=str(path))
install_writer_probe(runtime,ENABLED)
planner=DefaultSavePlanner()
planner.set_up_planner({'a':torch.arange(1024),'b':torch.ones(1024)},is_coordinator=True)
plan=planner.create_local_plan()
writer=FileSystemWriterAsync(root,thread_count=1)
transforms=[writer.transforms] if hasattr(writer,'transforms') else []
bucket=(path,'actual.distcp',([] ,[(item,planner.resolve_data(item)) for item in plan.items]))
result=FileSystemWriterAsync.write_preloaded_data(transforms,0,bucket,None,None,True)
assert isinstance(result[1],list) and len(result[1])==2,result
'''.replace('ENABLED',repr(enabled)))
            with (root/'log').open('w+') as log:
                proc=subprocess.Popen([sys.executable,str(worker)],stdout=log,stderr=subprocess.STDOUT)
                try:
                    if enabled:
                        deadline=time.monotonic()+40
                        while not (root/'midwrite-witness.json').exists():
                            if proc.poll() is not None or time.monotonic()>deadline:
                                log.seek(0);self.fail(log.read())
                            time.sleep(.02)
                        witness=json.loads((root/'midwrite-witness.json').read_text())
                        self.assertEqual(witness['remaining_tensors'],1)
                        self.assertEqual(witness['writer']['pid'],proc.pid)
                        stat=Path(f"/proc/{proc.pid}/fd/{witness['fd']}").stat()
                        self.assertEqual((stat.st_size,stat.st_ino),(witness['size'],witness['inode']))
                        self.assertGreater(stat.st_size,0)
                        self.assertIsNone(proc.poll())
                        fd=os.pidfd_open(proc.pid)
                        try:signal.pidfd_send_signal(fd,signal.SIGKILL)
                        finally:os.close(fd)
                        self.assertEqual(proc.wait(timeout=10),-9)
                    else:
                        code=proc.wait(timeout=40);log.seek(0)
                        self.assertEqual(code,0,log.read())
                        self.assertFalse((root/'midwrite-witness.json').exists())
                        self.assertGreater((root/'actual.distcp').stat().st_size,0)
                finally:
                    if proc.poll() is None:proc.kill();proc.wait()

    def test_enabled_stops_after_actual_tensor_with_open_file(self):self.run_writer(True)
    def test_disabled_completes_native_bucket(self):self.run_writer(False)


if __name__=='__main__':unittest.main()
