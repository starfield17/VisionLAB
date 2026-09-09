# Validator map

- `loader.py` owns strict parsing, schema loading and structured schema errors; do not duplicate these in validators.
- `common.py` owns timestamps, path/hash resolution and shared primitives. `annotation.py` owns annotation semantics; dataset and package audit reuse them.
- `dataset.py` owns dataset semantics/revision checks; `package.py` owns adapter descriptions, preflight and artifact/run lineage. `detect_event.py` supports optional loaded-package context.
- `cli.py` composes validators and trusted adapter descriptions. No model code runs here.
- Run `python tools/check.py` from the root. The boundary check prohibits imports of Deploy and model frameworks.
