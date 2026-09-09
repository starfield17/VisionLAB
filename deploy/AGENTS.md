# Deploy map

- Public ports/values are in `types.py`, re-exported from `__init__.py`. Lifecycle belongs to `pipeline.py`, numerical transforms to `preprocessing.py`/`postprocessing.py`.
- `registry.py` binds trusted descriptions and factories. Core does not import concrete adapters; callers compose them.
- `sinks/` owns output implementations. All model adapters currently live only as test doubles in `tests/`.
- `README.md` defines partial-open cleanup and failure behavior. Run the root `python tools/check.py`; dependency rules are executable in `tools/check_policy.py`.
