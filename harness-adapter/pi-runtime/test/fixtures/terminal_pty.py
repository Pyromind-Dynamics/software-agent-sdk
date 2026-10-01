"""Real canonical PTY used by the terminal transport regression tests."""

import os
import pty
import select
import signal
import sys


pid, master = pty.fork()
if pid == 0:
    os.execv("/bin/sh", ["sh", "-i"])

try:
    while True:
        readable, _, _ = select.select([master, sys.stdin.fileno()], [], [])
        if master in readable:
            try:
                chunk = os.read(master, 65536)
            except OSError:
                break
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        if sys.stdin.fileno() in readable:
            chunk = os.read(sys.stdin.fileno(), 65536)
            if not chunk:
                break
            while chunk:
                chunk = chunk[os.write(master, chunk) :]
finally:
    os.close(master)
    try:
        os.kill(pid, signal.SIGHUP)
    except ProcessLookupError:
        pass
    os.waitpid(pid, 0)
