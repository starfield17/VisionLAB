# VisionLAB

Versioned vision contracts and a small, synchronous deployment core. Model Lab and Deploy communicate through a Model Package; training and VLM integrations are not part of the runtime.

```sh
conda activate <env>
python -m pip install -e '.[dev]'
python tools/check.py
```

For a validator-only installation, use `python -m pip install .`. To include deployment numerical processing, use `python -m pip install '.[deploy]'`.

- [Intent](INTENTION.md)
- [Contract baseline](contracts/README.md)
- [Validator usage](contractcheck/README.md)
- [Deployment ports and lifecycle](deploy/README.md)
- [Implementation status and next tasks](docs/IMPLEMENTATION_HANDOFF.md)
- [Contract coverage](docs/CONTRACT_COVERAGE.md)

The included model/image `.fixture` files are illustrative text, not executable assets. The deployment tests exercise the full pipeline with test-only adapters; no real model accuracy or runtime support is claimed.
