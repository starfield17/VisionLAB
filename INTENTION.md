# Project v2 — Intention

## Intent

Build a **configurable, traceable, replaceable traditional visual model production and deployment system**:

> Raw Data → Dataset → Train → Model Package → Deploy → Detection Event → Output

The current scenario may be garbage detection, but "garbage classification" must not become an architectural assumption. Datasets, models, deployment locations, input sources, and output protocols should be able to vary independently.

In the long term, the project maintains only two products:

- **Model Lab**: data, auto-labeling, training, evaluation, export.
- **Deploy**: runs traditional visual Model Packages and outputs standardized detection results.

VLM/LLM serve only as R&D and data production tools for Model Lab; they do not enter Deploy.

## Principles

1. **Model Lab and Deploy are decoupled**: they are connected only through a versioned Model Package.
2. **Ontology is configurable**: labels, IDs, mappings, and business semantics must not be hardcoded into Core.
3. **Machine annotations are only Candidates**: LocateAnything/VLM results must retain provenance and undergo verification.
4. **Avoid a self-labeling closed loop**: existing YOLO/Faster R-CNN models are not used as the default Auto Label source; if pseudo-labeling is needed, it belongs under an explicit Self-Training experiment.
5. **Small Core + Adapters**: models, cameras, transport protocols, Serial, etc., are all replaceable boundary implementations.
6. **Deploy does not use VLM**: no inference chains for Ollama, LM Studio, Qwen, Gemini, etc., are maintained.
7. **Remove RK3588/RKNN**: no longer supported on the v2 mainline; history is preserved via Git.

## Done when

- Changing the Dataset Ontology requires no changes to the Deploy Core.
- Replacing YOLO/Faster R-CNN/other traditional models requires no rewrite of the input and output layers.
- Serial can be replaced with a network protocol without modifying inference logic.
- Deploy can run independently in an environment with no Agent/VLM dependencies at all.
- Deployed models can be traced back to Dataset, Training, Evaluation, and Export records.

## Not building now

- Web Dashboard
- Online VLM inference
- LocateAnything/VLM fine-tuning
- RK3588/RKNN
- General plugin marketplace or complex dynamic plugin framework