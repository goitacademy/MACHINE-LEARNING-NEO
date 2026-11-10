"""Course-wide execution tests.

Discovers every script in scripts/, competition/ and homework/ and every notebook
in notebooks/, copies the course folders to a temporary directory and executes
each file there with the *student* interpreter (.venv from `uv sync`). Helper
modules are import-checked instead of run. Configuration: [tool.course_tests]
in pyproject.toml. Run: uv run --isolated --only-group maintainer pytest
"""
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
CFG = PYPROJECT["tool"]["course_tests"]
TIMEOUT = int(os.environ.get("COURSE_TEST_TIMEOUT", CFG["timeout"]))
EXCLUDE = CFG.get("exclude", {})            # {path: reason}
REQUIRES_URL = CFG.get("requires_url", {})  # {path: url}
MODULES = sorted(CFG.get("modules", []))    # helper modules: import-checked, not run
MLFLOW_URL = CFG.get("mlflow_url")
COURSE_DIRS = ["scripts", "notebooks", "competition", "homework", "datasets", "derived", "models"]
PINS = dict(m.groups() for m in (re.fullmatch(r"([A-Za-z0-9_.-]+)==(\S+)", d)
                                 for d in PYPROJECT["project"]["dependencies"]) if m)


def _discover(pattern):
    return sorted(p.relative_to(ROOT).as_posix() for p in ROOT.glob(pattern))


def _params(paths):
    """Excluded files stay visible in the report as explicit skips."""
    return [pytest.param(p, marks=pytest.mark.skip(reason=f"EXCLUDED: {EXCLUDE[p]}")) if p in EXCLUDE else p
            for p in paths]


SCRIPTS = [p for p in _discover("scripts/*.py") + _discover("competition/*.py") + _discover("homework/*.py")
           if p not in MODULES]
NOTEBOOKS = _discover("notebooks/*.ipynb")


def _answers(url, timeout=3):
    try:
        urllib.request.urlopen(url, timeout=timeout)
    except urllib.error.HTTPError:
        return True  # the server answered, even if the root path is not 200
    except Exception:  # connection refused, DNS failure, timeout
        return False
    return True


@pytest.fixture(scope="session")
def student_python():
    override = os.environ.get("COURSE_PYTHON")
    exe = Path(override) if override else ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not exe.exists():
        pytest.exit(f"Student environment not found at {exe}. Run `uv sync` first or set COURSE_PYTHON.", returncode=3)
    return exe


@pytest.fixture(scope="session")
def sandbox():
    """Temporary copy of the course folders (tempfile, not pytest's basetemp, which some Windows setups deny)."""
    dst = Path(tempfile.mkdtemp(prefix="course-tests-"))
    for name in COURSE_DIRS:
        if (ROOT / name).is_dir():
            shutil.copytree(ROOT / name, dst / name, ignore=shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints"))
    yield dst
    shutil.rmtree(dst, ignore_errors=True)


def _env(sandbox):
    env = os.environ.copy()
    env.update(
        # notebooks 13 and 14 import from scripts/; the README asks students to set this too
        PYTHONPATH=os.pathsep.join([str(sandbox), str(sandbox / "scripts")]),
        MPLBACKEND="Agg",  # plt.show() must not open windows
        PYTHONIOENCODING="utf-8",
        MLFLOW_DISABLE_AGENT_HINT="1",
    )
    return env


@pytest.fixture(scope="session")
def mlflow_server(sandbox, student_python):
    """Tracking server for topic 14: reuse one that already answers, otherwise start one whose
    store lives in the sandbox and kill its whole process tree afterwards."""
    if _answers(MLFLOW_URL):
        yield "external"
        return
    host, port = MLFLOW_URL.split("//", 1)[1].split(":")
    cmd = [str(student_python), "-m", "mlflow", "server", "--host", host, "--port", port,
           "--backend-store-uri", "sqlite:///mlflow.db", "--artifacts-destination", "./mlartifacts",
           "--workers", "1"]  # several workers on Windows sometimes leave a request unanswered (120 s timeout)
    group = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    proc = subprocess.Popen(cmd, cwd=sandbox, env=_env(sandbox),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **group)
    deadline = time.monotonic() + 120
    while proc.poll() is None and time.monotonic() < deadline and not _answers(MLFLOW_URL, timeout=1):
        time.sleep(1)
    try:
        yield "started"
    finally:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
        else:
            os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)


def _needs(rel, request):
    url = REQUIRES_URL.get(rel)
    if url is None:
        return
    if url == MLFLOW_URL:
        request.getfixturevalue("mlflow_server")
    if not _answers(url):
        pytest.skip(f"BLOCKED: {rel} needs {url}, which does not answer.")


def _run(cmd, cwd, env, rel):
    try:
        res = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        pytest.fail(f"{rel}: still running after {TIMEOUT}s (raise COURSE_TEST_TIMEOUT if legitimate)", pytrace=False)
    if res.returncode != 0:
        tail = "\n".join((res.stderr or res.stdout).splitlines()[-40:])
        pytest.fail(f"{rel}: exit code {res.returncode}\n{tail}", pytrace=False)


@pytest.mark.parametrize("rel", _params(SCRIPTS))
def test_script_runs(rel, request, sandbox, student_python):
    _needs(rel, request)
    path = sandbox / rel
    # A script that names files as '<its folder>/...' expects the repository root as working
    # directory; every other script uses paths relative to its own folder.
    root_relative = re.search(rf"""['"]{re.escape(path.parent.name)}/""", path.read_text(encoding="utf-8"))
    cwd = sandbox if root_relative else path.parent
    _run([str(student_python), str(path.relative_to(cwd))], cwd=cwd, env=_env(sandbox), rel=rel)


@pytest.mark.parametrize("rel", _params(NOTEBOOKS))
def test_notebook_executes(rel, request, sandbox, student_python):
    _needs(rel, request)
    path = sandbox / rel
    cmd = [str(student_python), "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
           f"--ExecutePreprocessor.timeout={TIMEOUT}", path.name]
    _run(cmd, cwd=path.parent, env=_env(sandbox), rel=rel)


@pytest.mark.parametrize("rel", MODULES)
def test_module_imports(rel, sandbox, student_python):
    path = sandbox / rel
    _run([str(student_python), "-c", f"import {path.stem}"], cwd=path.parent, env=_env(sandbox), rel=rel)


def test_every_topic_script_has_a_notebook_twin():
    """Course scripts (scripts/) are mirrored by notebooks; every notebook mirrors some script."""
    topic_scripts = {Path(s).stem for s in SCRIPTS if s.startswith("scripts/")}
    all_scripts = {Path(s).stem for s in SCRIPTS}
    notebooks = {Path(n).stem for n in NOTEBOOKS}
    assert topic_scripts <= notebooks, (
        f"script without notebook: {sorted(topic_scripts - notebooks)}. "
        "Add the twin, or list a helper module under [tool.course_tests].modules.")
    assert notebooks <= all_scripts, f"notebook without script: {sorted(notebooks - all_scripts)}."


def test_student_environment_has_no_maintainer_tooling(student_python):
    """Course code is validated in the student environment, which must not contain the test tooling."""
    tools = [re.match(r"[A-Za-z0-9_.-]+", req).group(0) for req in PYPROJECT["dependency-groups"]["maintainer"]]
    code = "import importlib.metadata as m\n" + "\n".join(
        f"try:\n    m.version({t!r}); print({t!r})\nexcept m.PackageNotFoundError:\n    pass" for t in tools)
    res = subprocess.run([str(student_python), "-c", code], capture_output=True, text=True, check=True)
    assert res.stdout.split() == [], (
        f"test tooling found in the student environment: {res.stdout.split()}. "
        "Recreate it with `uv sync` (no --group).")


def test_requirements_pip_mirrors_pyproject_pins():
    text = (ROOT / "requirements_pip.txt").read_text(encoding="utf-8")
    stale = {n: v for n, v in PINS.items() if not re.search(rf"(?im)^{re.escape(n)}=={re.escape(v)}\b", text)}
    assert not stale, f"regenerate requirements_pip.txt with `uv export` (see its header): {stale}"


def test_requirements_conda_mirrors_pyproject_pins():
    text = (ROOT / "requirements_conda.yml").read_text(encoding="utf-8").lower().replace("_", "-")
    stale = {n: v for n, v in PINS.items()
             if not re.search(rf"(?m)^\s*- {re.escape(n.lower())}==?{re.escape(v)}\s*$", text)}
    assert not stale, f"requirements_conda.yml does not mirror pyproject.toml: {stale}"
