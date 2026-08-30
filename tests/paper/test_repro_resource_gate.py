import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "resource_gate", ROOT / "scripts" / "resource_gate.py"
)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


class ResourceGateTests(unittest.TestCase):
    def _gpu(self, utilization=0.0):
        return [{
            "index": 3, "uuid": "GPU-test", "memory_total_gb": 80.0,
            "memory_used_gb": 1.0, "memory_free_gb": 79.0,
            "utilization_gpu": utilization,
        }]

    def _check(self, gpus, processes):
        env = {
            "RTX_GATE_MIN_PUB_GB": "1", "RTX_GATE_MIN_ROOT_GB": "1",
            "RTX_GATE_MIN_GPU_FREE_GB": "30", "RTX_GATE_MAX_GPU_UTIL": "10",
            "RTX_GATE_ALLOW_COMPUTE_PROCESSES": "0",
        }
        with mock.patch.dict(os.environ, env, clear=False), \
                mock.patch.object(gate, "_disk_avail_gb", return_value=999.0), \
                mock.patch.object(gate, "_nvidia_gpus", return_value=gpus), \
                mock.patch.object(gate, "_nvidia_compute_processes", return_value=processes), \
                mock.patch.object(gate, "_system_provenance", return_value={"cpu_numa": {}, "io": []}):
            return gate._gate_check("device=3")

    def test_idle_gpu_passes(self):
        report, ok = self._check(self._gpu(), [])
        self.assertTrue(ok)
        self.assertTrue(report["checks"]["gpu_utilization"]["pass"])
        self.assertTrue(report["checks"]["gpu_compute_processes"]["pass"])

    def test_high_utilization_is_rejected(self):
        report, ok = self._check(self._gpu(utilization=72.0), [])
        self.assertFalse(ok)
        self.assertFalse(report["checks"]["gpu_utilization"]["pass"])

    def test_existing_compute_process_is_rejected(self):
        process = [{
            "gpu_uuid": "GPU-test", "pid": 123, "process_name": "python",
            "used_gpu_memory_mib": 4096.0,
        }]
        report, ok = self._check(self._gpu(), process)
        self.assertFalse(ok)
        self.assertEqual(report["checks"]["gpu_compute_processes"]["processes"][0]["pid"], 123)


if __name__ == "__main__":
    unittest.main()
