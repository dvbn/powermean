"""Build from this checkout and test only the resulting installed alpha wheel in CI."""

from hashlib import file_digest
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig
from tempfile import TemporaryDirectory


SOURCE = Path(__file__).resolve().parents[2]
FLOORS = {"numpy": "2.3.1", "pandas": "2.2.3", "scipy": "1.15.0", "formulaic": "1.1.0"}
MODULES = ("test_after_estimation.py", "test_estimates.py", "test_interface.py")


def run(*args, cwd):
    subprocess.run(args, cwd=cwd, check=True)


def installed(profile, wheel_hash):
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    import powermean as pm
    import pytest

    assert SOURCE not in Path.cwd().parents and Path.cwd() != SOURCE
    assert sys.prefix != sys.base_prefix
    site = Path(sysconfig.get_path("purelib")).resolve()
    assert Path(pm.__file__).resolve() == site / "powermean/__init__.py"
    versions = {name: metadata.version(name) for name in (*FLOORS, "powermean", "pytest")}
    assert versions["powermean"] == "0.1.0a1"
    origin = json.loads(metadata.distribution("powermean").read_text("direct_url.json"))
    assert origin["archive_info"]["hashes"]["sha256"] == wheel_hash
    assert "dir_info" not in origin
    if profile == "MINIMUM":
        assert {name: versions[name] for name in FLOORS} == FLOORS

    expected = set()
    passed = set()

    class Outcomes:
        def pytest_collection_modifyitems(self, items):
            expected.update(item.nodeid for item in items)
            assert len(items) == len(expected) == 31

        def pytest_runtest_logreport(self, report):
            key = report.nodeid, report.when
            if report.passed and not hasattr(report, "wasxfail"):
                assert key not in passed
                passed.add(key)

    tests = SOURCE / "tests"
    flags = ["-c", str(SOURCE / "pyproject.toml"), "--confcutdir", str(tests),
             "-p", "no:cacheprovider",
             "--import-mode=prepend", "-ra"]
    rc = pytest.main([*flags, *(str(tests / name) for name in MODULES)], plugins=[Outcomes()])
    assert rc == 0 and len(expected) == 31
    assert passed == {(node, phase) for node in expected for phase in ("setup", "call", "teardown")}

    fixture = sys.modules["conftest"]
    assert Path(fixture.__file__).resolve() == tests / "conftest.py"
    result = pm.fit("y ~ 1 + x1 + x2 + d", fixture.counts.__wrapped__(),
                    omega=1, tau=1, solver="iols")
    provenance = result.provenance
    trace = provenance.trace
    assert trace["actual_backend"] == "numpy" and not trace["backend_explicit"]
    assert provenance.certified
    print(json.dumps({"profile": profile, "python": platform.python_version(),
                      "system": platform.system(), "architecture": platform.machine(),
                      "versions": versions, "pytest_items": 31, "passed_phases": 93,
                      "separate_iols_backend": trace["actual_backend"],
                      "iols_fallback": trace["fallback_used"],
                      "iols_certified": provenance.certified}))


def build_and_check(profile):
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip()
    assert sha == os.environ["GITHUB_SHA"]
    with TemporaryDirectory() as dirname:
        outside = Path(dirname).resolve()
        out = outside / "dist"
        run(sys.executable, "-I", "-B", "-m", "build", "--outdir", str(out), cwd=SOURCE)
        wheel, = out.glob("powermean-0.1.0a1-py3-none-any.whl")
        sdist, = out.glob("powermean-0.1.0a1.tar.gz")
        hashes = {}
        for artifact in (sdist, wheel):
            with artifact.open("rb") as stream:
                hashes[artifact] = file_digest(stream, "sha256").hexdigest()
            print(f"{sha} {artifact.name} sha256:{hashes[artifact]}", flush=True)
        venv = outside / "env"
        run(sys.executable, "-I", "-B", "-m", "venv", str(venv), cwd=outside)
        py = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        floors = [f"{name}=={ver}" for name, ver in FLOORS.items()] if profile == "MINIMUM" else []
        run(str(py), "-I", "-B", "-m", "pip", "--disable-pip-version-check", "install",
            "--only-binary=:all:", f"powermean[test] @ {wheel.as_uri()}", *floors, cwd=outside)
        run(str(py), "-I", "-B", str(Path(__file__).resolve()), "installed",
            profile, hashes[wheel], cwd=outside)


if __name__ == "__main__":
    assert __debug__ and sys.flags.isolated and sys.dont_write_bytecode
    if len(sys.argv) == 4 and sys.argv[1] == "installed":
        assert sys.argv[2] in {"NORMAL", "MINIMUM"}
        installed(sys.argv[2], sys.argv[3])
    else:
        assert len(sys.argv) == 1 and os.environ["PM_PROFILE"] in {"NORMAL", "MINIMUM"}
        build_and_check(os.environ["PM_PROFILE"])
