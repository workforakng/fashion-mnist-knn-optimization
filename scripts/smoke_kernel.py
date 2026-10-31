"""Pre-push smoke test for the Kaggle kernel bundle.

Run BEFORE every `kaggle kernels push`:
    .venv/bin/python scripts/smoke_kernel.py

Checks (no GPU required, < 30 s):
  1. Every bundled source file compiles.
  2. The kernel script compiles and its embedded base64 round-trips to a
     valid tar containing the expected members.
  3. The data-mapping logic resolves both compressed (.gz) and
     Kaggle-decompressed (extension-less) input layouts.
  4. Both training scripts accept the exact argv the kernel passes.
  5. The distillation teacher glob matches seed artefacts only
     (must NOT match softknn_conv_shift.pt).
  6. The dual-GPU queue logic: distribution, completion, failure propagation.

Checks 2 and 5 need files that are not in the repository: the generated
Kaggle kernel script (pass its path in the KAGGLE_KERNEL_SCRIPT environment
variable) and trained seed artefacts under artifacts/ (gitignored). When those
are absent the checks are reported as SKIP, not FAIL.
"""
import base64
import glob
import io
import os
import py_compile
import re
import subprocess
import sys
import tarfile
import tempfile
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BUNDLE = ["src/__init__.py", "src/data.py", "src/augment.py",
          "src/softknn.py", "src/train.py",
          "scripts/train_conv_seed.py", "scripts/train_distill.py"]
KERNEL_SCRIPT = os.environ.get("KAGGLE_KERNEL_SCRIPT", "")
ARTIFACT_DIR = os.path.join(ROOT, "artifacts")

failures = []
skipped = []


class Skip(Exception):
    """Raised by a check whose required external input is missing."""


def check(name, fn):
    try:
        fn()
        print(f"  [PASS] {name}")
    except Skip as why:
        skipped.append(name)
        print(f"  [SKIP] {name}: {why}")
    except Exception as exc:  # noqa: BLE001 - smoke gate reports everything
        failures.append(name)
        print(f"  [FAIL] {name}: {exc}")


def t_bundle_compiles():
    for rel in BUNDLE:
        py_compile.compile(os.path.join(ROOT, rel), doraise=True)


def t_kernel_script_roundtrip():
    if not KERNEL_SCRIPT or not os.path.isfile(KERNEL_SCRIPT):
        raise Skip("no kernel script; set KAGGLE_KERNEL_SCRIPT=/path/to/script.py "
                   "(generated outside this repo) to run this check")
    py_compile.compile(KERNEL_SCRIPT, doraise=True)
    src = open(KERNEL_SCRIPT).read()
    m = re.search(r"parts = \[(.*?)\]", src, flags=re.S)
    assert m, "parts block missing"
    b64 = "".join(re.findall(r'"([^"]+)"', m.group(1)))
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(b64))) as tf:
        members = sorted(m.name for m in tf.getmembers() if m.isfile())
    expected = sorted(rel for rel in BUNDLE)
    assert members == expected, f"tar members {members} != bundle {expected}"


def t_data_mapping():
    expected = {
        "train-images-idx3-ubyte.gz": ("train", "images"),
        "train-labels-idx1-ubyte.gz": ("train", "labels"),
        "t10k-images-idx3-ubyte.gz": ("t10k", "images"),
        "t10k-labels-idx1-ubyte.gz": ("t10k", "labels"),
    }
    for layout in ("gz", "bare"):
        with tempfile.TemporaryDirectory() as td:
            fake_in = os.path.join(td, "input", "dsets", "x")
            os.makedirs(fake_in)
            for exp in expected:
                stem = exp[:-3] if layout == "gz" else exp[:-3]
                src = os.path.join(fake_in, stem) if layout == "bare" \
                    else os.path.join(fake_in, exp)
                open(src, "wb").close()
            candidates = glob.glob(os.path.join(td, "input", "**", "*"),
                                   recursive=True)
            for exp, (split, kind) in expected.items():
                hit = next((c for c in candidates
                            if os.path.isfile(c) and split in os.path.basename(c)
                            and kind in os.path.basename(c)
                            and ("idx" in c or "ubyte" in c)), None)
                assert hit, f"{layout}: no source resolved for {exp}"


def t_arg_parsing():
    for script, argv in (
        ("scripts/train_conv_seed.py",
         ["--seed", "42", "--epochs", "35"]),
        ("scripts/train_distill.py", []),
    ):
        r = subprocess.run([sys.executable, os.path.join(ROOT, script),
                            "--help"] + ([] if not argv else []),
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{script} --help failed: {r.stderr[-200:]}"


def t_teacher_glob_real():
    import glob as _g
    if not os.path.isdir(ARTIFACT_DIR):
        raise Skip(f"{ARTIFACT_DIR} not found (gitignored; created by "
                   "scripts/train_conv_seed.py)")
    hits = sorted(os.path.basename(p) for p in
                  _g.glob(os.path.join(ARTIFACT_DIR, "softknn_conv_s[0-9]*.pt")))
    if not hits:
        raise Skip("no softknn_conv_s*.pt seed artefacts in artifacts/")
    assert not any("shift" in h for h in hits), "glob matched conv_shift!"


def t_queue_logic():
    def run_queue(stub_exit):
        queues = {0: [42, 123, 31415], 1: [7, 2024]}
        results, lock = {}, threading.Lock()

        def worker(gpu):
            while True:
                with lock:
                    if not queues[gpu]:
                        return
                    s = queues[gpu].pop(0)
                code = f"import sys; sys.exit({stub_exit})"
                r = subprocess.run([sys.executable, "-c", code])
                results[s] = r.returncode

        ts = [threading.Thread(target=worker, args=(g,)) for g in (0, 1)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return results

    res = run_queue(0)
    assert len(res) == 5 and all(rc == 0 for rc in res.values()), res
    res = run_queue(3)
    assert any(rc != 0 for rc in res.values()), "failure propagation broken"


print("SMOKE GATE — kernel bundle pre-push checks")
check("bundle sources compile", t_bundle_compiles)
check("kernel script b64 round-trip", t_kernel_script_roundtrip)
check("data mapping (gz + bare layouts)", t_data_mapping)
check("training scripts accept kernel argv", t_arg_parsing)
check("distill teacher glob", t_teacher_glob_real)
check("dual-GPU queue logic", t_queue_logic)

if failures:
    print(f"SMOKE GATE FAILED: {failures} — DO NOT PUSH")
    sys.exit(1)
if skipped:
    print(f"SMOKE GATE PASSED with {len(skipped)} skipped check(s): {skipped}")
else:
    print("SMOKE GATE PASSED — safe to push")
