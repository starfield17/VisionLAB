"""Build a guarded YOLO training command from relative paths.

A caller-supplied custom relative project directory is resolved against the
repository root to an absolute path before being passed to `project=`, which
keeps Ultralytics from nesting output under its default `runs_dir` and never
hardcodes a directory name. The same resolution applies to cfg/model/data and
the guard output. This module only builds and validates the command; it never
imports a training framework.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple


class TrainCommand(NamedTuple):
    project: Path
    guard_dir: Path
    yolo_args: list


def build_train_command(*, root: Path, project: str, name: str, cfg: str, model: str,
                        data: str, device: str, guard_output: str) -> TrainCommand:
    """Return the resolved project/guard dirs and the absolute yolo argument list."""
    root = Path(root).resolve()
    if not name:
        raise ValueError('name must be a nonempty string')
    for label, value in (('project', project), ('cfg', cfg), ('model', model), ('data', data),
                         ('guard_output', guard_output)):
        if not isinstance(value, str) or not value:
            raise ValueError(f'{label} must be a nonempty string')

    def resolve(relative: str) -> Path:
        return (root / relative).resolve()

    project_abs = resolve(project)
    guard_dir = resolve(guard_output)
    yolo_args = ['yolo', 'detect', 'train',
                 f'cfg={resolve(cfg)}',
                 f'model={resolve(model)}',
                 f'data={resolve(data)}',
                 f'device={device}',
                 f'project={project_abs}',
                 f'name={name}']
    return TrainCommand(project_abs, guard_dir, yolo_args)
