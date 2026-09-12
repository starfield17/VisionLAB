# Deploy map

- Public ports/values are in `types.py`, re-exported from `__init__.py`. Lifecycle belongs to `pipeline.py`, numerical transforms to `preprocessing.py`/`postprocessing.py`.
- `registry.py` binds trusted descriptions and factories. Core does not import concrete adapters; callers compose them.
- `sinks/` owns output implementations. All model adapters currently live only as test doubles in `tests/`.
- `sources/image_file.py` decodes one image (EXIF orientation, alpha composited on white) into one contract frame; `adapters/onnx_detector.py` decodes an end-to-end ONNX graph and must not resize or letterbox.
- `compose.py` is the trusted static composition of the real slice; `cli.py` runs `python -m deploy detect --package <dir> --image <file>`.
- `README.md` defines partial-open cleanup and failure behavior. Run the root `python tools/check.py`; dependency rules are executable in `tools/check_policy.py`.
