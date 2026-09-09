"""contractcheck — portable validation of VisionLAB Core Contract 1.0.0 documents.

This library implements the validation layers described in contracts/README.md:

1. Strict JSON parsing (rejects duplicate keys and non-finite numbers), JSON
   Schema (Draft 2020-12) validation with format checking and local reference
   resolution.
2. Cross-file identities, uniqueness, coordinates, states, time ordering, path
   containment and hashes.
3. Provenance lineage, split leakage and adapter configuration/support.

It is independent of any model framework and never executes model code.
Model Lab and Deploy use it as a shared development/CI tool; the schemas live
in contracts/schemas and are the single source of truth.
"""

__version__ = "0.2.0"
