"""Guarded subprocess entrypoints that import a model runtime.

Only modules in this package may import a training or inference framework:
Model Lab orchestration stays offline and the guard owns process limits.
"""
