"""CPU process tree and controller failure fixtures; never imported by production."""
import argparse
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def output():
    return Path(os.environ["FT_CONTROL_SOCKET"]).parent


def run(mode, count):
    from scripts.ft.descendants import Client
    if mode == "launcher":
        for event in range(count):
            child = subprocess.Popen([sys.executable, __file__, "middle", "--count", str(event)])
            if child.wait(timeout=20):
                raise RuntimeError("fixture middle failed")
    elif mode == "middle":
        children = [subprocess.Popen([sys.executable, __file__, role, "--count", str(count)], start_new_session=True)
                    for role in ("target", "waiter")]
        status = [child.wait(timeout=15) for child in children]
        (output() / f"parent_wait-{count}.json").write_text(json.dumps(status))
        if status != [-9, 0]:
            raise RuntimeError(f"unexpected parent wait statuses {status}")
    elif mode in ("target", "waiter"):
        client = Client(mode)
        try:
            key = f"event-{count}"
            client.ready(key, {"boundary": "cpu_fixture"})
            client.ready(key, {"boundary": "cpu_fixture"})
            client.wait_release(key)
        finally:
            client.close()
    elif mode == "dormant":
        child = subprocess.Popen([sys.executable, __file__, "sleeper"])
        (output() / "dormant.json").write_text(json.dumps({"pid": os.getpid(), "child": child.pid}))
        child.wait()
    elif mode == "sleeper":
        time.sleep(60)


def controller_fault(fault, args):
    from scripts.ft import namespace_run
    if fault == "pidfd_failure":
        original = os.pidfd_open
        def fail_child(pid, *rest):
            if pid != os.getpid():
                # Ensure a real unregistered subtree exists at the failure point.
                root = Path(args[args.index("--output") + 1])
                deadline = time.monotonic() + 3
                while not (root / "dormant.json").exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                raise OSError(errno.EMFILE, "fixture first child pidfd acquisition failure")
            return original(pid, *rest)
        os.pidfd_open = fail_child
    elif fault == "registration_failure":
        def fail(*args, **kwargs):
            raise RuntimeError("fixture registration failure")
        namespace_run.Descendant = fail
    sys.argv = [sys.argv[0], *args]
    namespace_run.main()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith("controller_"):
        controller_fault(sys.argv[1].removeprefix("controller_"), sys.argv[2:])
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("mode")
        parser.add_argument("--count", type=int, default=2)
        options = parser.parse_args()
        run(options.mode, options.count)
