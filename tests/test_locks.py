"""The file locks between real processes, as msvcrt and fcntl differ within one process."""

import itertools
import json
import subprocess
import sys

from nai_cli import auth

_TAKE_IN_TURN = """
import json, time
from nai_cli import ledger
with ledger.generation_lock(timeout=60):
    start = time.time()
    time.sleep(0.2)
    print(json.dumps([start, time.time()]))
"""

_HOLD = """
import time
from nai_cli import ledger
with ledger.generation_lock():
    print("held", flush=True)
    time.sleep(60)
"""

_TRY = """
import json, sys
from nai_cli import errors, ledger
try:
    with ledger.generation_lock(timeout=float(sys.argv[1])):
        print(json.dumps("got"))
except errors.NaiError as e:
    print(json.dumps(e.code))
"""

_EDIT = """
import sys
from nai_cli import auth
for j in range(10):
    with auth._editing() as store:
        store["logins"][f"{sys.argv[1]}_{j}"] = {"kind": "probe"}
"""


def _python(code: str, *args: str) -> subprocess.Popen[str]:
    return subprocess.Popen([sys.executable, "-c", code, *args], stdout=subprocess.PIPE, text=True)


def _result(process: subprocess.Popen[str]) -> object:
    out, _ = process.communicate(timeout=120)
    assert process.returncode == 0
    return json.loads(out)


def test_processes_take_the_generation_lock_in_turn():
    spans = sorted(_result(p) for p in [_python(_TAKE_IN_TURN) for _ in range(4)])
    assert all(later[0] >= earlier[1] for earlier, later in itertools.pairwise(spans)), spans


def test_a_held_lock_times_out_as_busy_and_a_killed_holder_frees_it():
    holder = _python(_HOLD)
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "held"
        assert _result(_python(_TRY, "1")) == "busy"
    finally:
        holder.kill()
        holder.wait(timeout=30)
    assert _result(_python(_TRY, "10")) == "got"


def test_processes_editing_the_login_store_lose_no_edit():
    for p in [_python(_EDIT, f"w{i}") for i in range(4)]:
        p.communicate(timeout=120)
        assert p.returncode == 0
    logins = json.loads(auth.credentials_path().read_text(encoding="utf-8"))["logins"]
    assert {f"w{i}_{j}" for i in range(4) for j in range(10)} <= set(logins)
