"""A live fork child must not prolong a completed metadata critical section."""
import fcntl
import os
from pathlib import Path
import tempfile
import unittest

from scripts.ft import state


class ForkedMutationLock(unittest.TestCase):
    def test_parent_unlocks_while_fork_child_keeps_descriptor(self):
        with tempfile.TemporaryDirectory() as tmp:
            with state.acquire_owner(Path(tmp), -1, run_nonce='fork-lock',
                    config_sha256='a' * 64, verifier_version='fixture') as owner:
                read_fd, write_fd = os.pipe()
                child = None
                try:
                    with state._locked(owner):
                        child = os.fork()
                        if child == 0:
                            os.close(write_fd)
                            os.read(read_fd, 1)
                            os._exit(0)
                    # Child is still alive with the inherited locked description.
                    fd = os.open(owner.root / 'mutation.lock', os.O_RDWR)
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    finally:
                        os.close(fd)
                    state.authorize_attempt(owner, 'group:0', None, 'attempt',
                                            expected_policy_version=0)
                finally:
                    os.close(write_fd)
                    if child:
                        os.waitpid(child, 0)
                    os.close(read_fd)


if __name__ == '__main__':
    unittest.main()
