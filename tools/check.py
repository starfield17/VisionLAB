"""Run all gates in the active conda environment: python tools/check.py."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args, cwd=ROOT, env=None):
    subprocess.run([sys.executable, *map(str, args)], cwd=cwd, env=env, check=True)


def packaging_check():
    with tempfile.TemporaryDirectory(prefix='contractcheck-build-') as directory:
        work = Path(directory).resolve()
        dist = work / 'dist'
        run('-m', 'build', '--no-isolation', '--outdir', dist)
        # Rebuild from sdist to prove schemas survive both distribution formats.
        unpack = work / 'sdist'
        unpack.mkdir()
        with tarfile.open(next(dist.glob('*.tar.gz'))) as archive:
            for member in archive.getmembers():
                target = (unpack / member.name).resolve()
                if not target.is_relative_to(unpack) or member.issym() or member.islnk():
                    raise ValueError('unsafe sdist member')
            archive.extractall(unpack)
        rebuilt = work / 'rebuilt'
        run('-m', 'build', '--no-isolation', '--wheel', '--outdir', rebuilt, cwd=next(unpack.iterdir()))
        site = work / 'installed'
        run('-m', 'pip', 'install', '--no-deps', '--no-index', '--target', site, next(rebuilt.glob('*.whl')))
        env = dict(os.environ, PYTHONPATH=str(site))
        event = work / 'empty.json'
        event.write_bytes((ROOT / 'contracts/examples/empty-event.json').read_bytes())
        script = """
from pathlib import Path
import sys
import contractcheck, contracts
from contractcheck.loader import SchemaStore
assert Path(contractcheck.__file__).is_relative_to(Path.cwd() / 'installed')
assert Path(contracts.__file__).is_relative_to(Path.cwd() / 'installed')
assert len(SchemaStore().schemas) == 8
assert 'numpy' not in sys.modules
assert 'torch' not in sys.modules
from deploy import Pipeline
"""
        run('-c', script, cwd=work, env=env)
        run('-m', 'contractcheck', 'validate', event, cwd=work, env=env)
        run(site / 'bin' / 'contractcheck', 'validate', event, cwd=work, env=env)
        # Verify installed schema bytes exactly match the canonical sources.
        for path in (ROOT / 'contracts/schemas').glob('*.json'):
            assert (site / 'contracts/schemas' / path.name).read_bytes() == path.read_bytes()
        print('sdist/wheel, installed resources, module and console CLI passed', flush=True)


def main():
    run('tools/check_policy.py')
    run('-m', 'ruff', 'check', 'contractcheck', 'contracts', 'deploy', 'tests', 'tools')
    run('-m', 'mypy')
    run('-m', 'pytest', '-q')
    packaging_check()
    print(json.dumps({'status': 'passed', 'gates': ['boundaries', 'test-integrity', 'lint', 'types', 'tests', 'packaging']}))


if __name__ == '__main__':
    main()
