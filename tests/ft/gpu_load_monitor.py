"""Record selected GPU load and stop our training container on foreign use."""
import csv
import json
from pathlib import Path
import subprocess
import threading
import time


class GpuLoadMonitor:
    def __init__(self, root, devices, remaining_gpu_hours):
        self.root = Path(root)
        self.devices = set(devices)
        self.remaining_seconds = remaining_gpu_hours * 900  # Four allocated GPUs.
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.violation = None
        self.samples = 0
        # PIDs ever listed in our container: a killed trainer vanishes from
        # docker top before nvidia-smi stops reporting its memory.
        self.seen_own = set()
        # Unowned PIDs from the previous sample; a violation needs two in a row
        # (a child born between docker top and nvidia-smi is transient).
        self.pending = set()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, *_):
        self.stop.set()
        self.thread.join()
        if self.samples == 0 and self.violation is None and (self.root / 'exitcode').exists():
            self.violation = {'reason': 'monitor_error', 'error': 'no GPU load samples'}
        if self.root.exists():
            (self.root / 'gpu-load-summary.json').write_text(json.dumps(
                {'samples': self.samples, 'violation': self.violation,
                 'scope': 'selected GPU load and compute PIDs during the training container'}, indent=2))
        if self.violation is not None and exc_type is None:
            raise RuntimeError(f'GPU monitor stopped training: {self.violation}')

    @staticmethod
    def _in_container(pid, cid):
        try:
            return cid in Path(f'/proc/{pid}/cgroup').read_text()
        except OSError:
            return False

    def _query(self, args):
        failure = None
        for _ in range(3):
            try:
                return subprocess.run(['nvidia-smi', args, '--format=csv,noheader,nounits'],
                                      check=True, capture_output=True, text=True, timeout=10).stdout
            except (OSError, subprocess.SubprocessError) as exc:
                failure = exc
                time.sleep(0.2)
        raise failure

    def _watch(self):
        started = None
        seen_running = False
        while not self.stop.is_set():
            cid_file = self.root / 'container.id'
            if (self.root / 'exitcode').exists():
                return
            if cid_file.exists():
                cid = cid_file.read_text().strip()
                try:
                    top = subprocess.run(['docker', 'top', cid, '-eo', 'pid'],
                                         capture_output=True, text=True, timeout=10)
                except (OSError, subprocess.SubprocessError) as exc:
                    self.violation = {'reason': 'monitor_error', 'error': repr(exc)}
                    top = None
                if top is not None and top.returncode == 0:
                    if started is None:
                        started = time.monotonic()
                    seen_running = True
                    try:
                        own = {int(line.strip()) for line in top.stdout.splitlines()[1:] if line.strip()}
                        self.seen_own |= own
                        gpu_raw = self._query('--query-gpu=uuid,memory.used,utilization.gpu')
                        compute_raw = self._query('--query-compute-apps=gpu_uuid,pid,used_gpu_memory')
                        gpu = [row for row in csv.reader(gpu_raw.splitlines()) if row and row[0].strip() in self.devices]
                        compute = [row for row in csv.reader(compute_raw.splitlines()) if row and row[0].strip() in self.devices]
                        if len(gpu) != 4:
                            raise RuntimeError('selected GPU load rows missing')
                        foreign = []
                        for row in compute:
                            pid = int(row[1].strip())
                            if pid in self.seen_own:
                                continue
                            if self._in_container(pid, cid):
                                self.seen_own.add(pid)
                                continue
                            foreign.append(row)
                        unowned = {int(row[1].strip()) for row in foreign}
                        confirmed = [row for row in foreign if int(row[1].strip()) in self.pending]
                        self.pending = unowned
                        sample = {'monotonic_ns': time.monotonic_ns(), 'gpu': gpu,
                                  'compute': compute, 'own_pids': sorted(own), 'foreign': foreign,
                                  'foreign_confirmed': confirmed}
                        with (self.root / 'gpu-load.jsonl').open('a') as stream:
                            stream.write(json.dumps(sample) + '\n')
                        self.samples += 1
                        if confirmed:
                            self.violation = {'reason': 'foreign_compute_process', 'sample': sample}
                        elif time.monotonic() - started >= self.remaining_seconds:
                            self.violation = {'reason': 'gpu_hour_cap', 'sample': sample}
                    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                        self.violation = {'reason': 'monitor_error', 'error': repr(exc)}
                elif top is not None and seen_running and not (self.root / 'exitcode').exists():
                    # A stopped container may precede docker wait writing exitcode.
                    try:
                        status = subprocess.run(['docker', 'inspect', '-f', '{{.State.Running}}', cid],
                                                capture_output=True, text=True, timeout=10)
                        if status.returncode == 0 and status.stdout.strip() == 'true':
                            self.violation = {'reason': 'monitor_error', 'error': top.stderr}
                    except (OSError, subprocess.SubprocessError) as exc:
                        self.violation = {'reason': 'monitor_error', 'error': repr(exc)}
                if self.violation is not None:
                    (self.root / 'gpu-load-violation.json').write_text(json.dumps(self.violation, indent=2))
                    subprocess.run(['docker', 'kill', cid], capture_output=True, timeout=15)
                    return
            self.stop.wait(5)
