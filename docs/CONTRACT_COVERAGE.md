# Contract coverage and verification limits

| Area | Implemented | Verification boundary |
| --- | --- | --- |
| JSON / Schema | Duplicate keys, UTF-8, finite numbers including overflow, version, closed structures, local refs, JSON Pointer errors | Producer field authenticity is not inferred |
| Time | UTC Z, calendar validity, arbitrary fractional-second ordering | Leap-second timestamps are rejected; acquisition-clock correctness remains a Source responsibility |
| Dataset | Ontology/item/hash/annotation uniqueness, group splits, references, approved annotations, image hashes, metadata dimensions and boxes | Does not decode images, discover duplicate scenes, or prove human review occurred |
| Annotation | Object uniqueness, geometry, review times, candidate retention, available revision identities/source continuity and cycle detection | Opaque VLM/model/prompt/producer-run refs require the producer's audit system; this validator does not authenticate those external systems |
| Model Package | Included hashes, containment, schema/trace IDs, ontology label agreement, supported adapter format/config | Metadata support does not certify model runtime compatibility or accuracy; caller trusts the adapter description |
| Audit | Dataset semantics with local retained roots, run artifact bytes, training/evaluation/export model associations, evaluation metadata, unavailable revision errors | Does not retrain, rerun metrics, authenticate actor identities or establish compliance of training methodology |
| Preprocessing | RGB/BGR, half-pixel stretch, float32 normalization, NCHW/NHWC, inverse geometry helper | No letterbox, dynamic batch, integer/quantized tensors |
| Event | Original-image boxes, class/label, package identity, finite scores, timestamps | Stateless validator cannot infer whether an empty result masks an inference failure |
| Pipeline | Success/empty/failure distinction, reverse cleanup, frame ordering and unique event IDs per run, NMS/filtering | No cross-run delivery guarantee, streaming retention policy or actual model adapter |
| Distribution | sdist → wheel → installed schemas and both CLI entrypoints outside source | CI matrix is configured; execution status depends on running the workflow |

The illustrative package under `contracts/examples` is a document fixture. Preflight can succeed with an explicit fixture description, while full audit rejects its incomplete evaluation/model lineage. Unit tests construct a separate consistent synthetic audit fixture; passing it proves metadata checking, not successful training.

The public contract JSON shapes remain version 1.0.0. Run `config` and I/O `role` fields are deliberately open in that schema; implementation conventions for audited evaluation/lineage are documented in the contract baseline and validator README. Any future change to supported preprocessing or task semantics requires explicit versioned contract work.
