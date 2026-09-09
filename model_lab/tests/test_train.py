from pathlib import Path

import pytest

from model_lab.train import build_train_command


def test_build_train_command_resolves_relative_to_absolute():
    command = build_train_command(root='/repo', project='workdir/training', name='smoke-1',
                                  cfg='model_lab/configs/smoke.yaml', model='workdir/models/yolo26n.pt',
                                  data='workdir/labeling/yolo/data.yaml', device='mps',
                                  guard_output='workdir/training/runs/smoke-guard')
    assert command.project == Path('/repo/workdir/training')
    assert command.guard_dir == Path('/repo/workdir/training/runs/smoke-guard')
    assert command.yolo_args[0] == 'yolo'
    assert 'project=/repo/workdir/training' in command.yolo_args
    assert 'name=smoke-1' in command.yolo_args
    assert 'cfg=/repo/model_lab/configs/smoke.yaml' in command.yolo_args
    assert 'device=mps' in command.yolo_args
    # every path argument is absolute; no bare relative 'workdir' or Ultralytics runs-dir prefix
    for argument in command.yolo_args:
        if '=' in argument and argument.startswith(('cfg=', 'model=', 'data=', 'project=')):
            assert argument.split('=', 1)[1].startswith('/repo/')
    assert not any('runs/detect/' in argument for argument in command.yolo_args)


def test_build_train_command_with_custom_project_root():
    command = build_train_command(root='/other-place', project='runs', name='exp',
                                  cfg='cfg.yaml', model='m.pt', data='data.yaml', device='cpu',
                                  guard_output='guard')
    assert command.project == Path('/other-place/runs')
    assert 'project=/other-place/runs' in command.yolo_args


@pytest.mark.parametrize('kwargs', [
    {'name': ''}, {'project': ''}, {'cfg': ''}, {'model': ''}, {'data': ''}, {'guard_output': ''},
])
def test_build_train_command_rejects_empty_required_fields(kwargs):
    base = dict(root='/repo', project='workdir/training', name='smoke-1',
                cfg='model_lab/configs/smoke.yaml', model='workdir/models/yolo26n.pt',
                data='workdir/labeling/yolo/data.yaml', device='mps',
                guard_output='workdir/training/runs/smoke-guard')
    base.update(kwargs)
    with pytest.raises(ValueError):
        build_train_command(**base)
