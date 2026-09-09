# Friction log

## Initial validator and deployment-core implementation

- Trigger: illustrative package records were being used as a full-audit success fixture.
- Constraint: the contract requires evaluation/model lineage that the illustrative records deliberately do not contain.
- Resolution: preserve the illustrative files; reject their full audit and build a separate consistent synthetic fixture for positive audit tests.
- Consequence: tests verify trace metadata without claiming actual training, accuracy or executable model support.
- Follow-up: supply a real reviewed dataset and model package when implementing the first runtime adapter.

## Audit metadata conventions

- Trigger: run-record config is an open object and I/O roles were strings without named lineage conventions.
- Constraint: checking whether evaluation/export consumed the trained model requires an unambiguous metadata edge.
- Resolution: use roles `dataset` and `model`; evaluation config has `split`, `metric_definitions`, `thresholds`, `sample_count`; metric keys must match definitions.
- Consequence: contract JSON shapes remain unchanged; prose and validator documentation now specify the convention.
- Follow-up: Model Lab producers must emit these fields and test against full audit.

## UTC precision

- Trigger: direct string comparison misordered timestamps with fractional seconds, while datetime alone loses sub-microsecond precision.
- Constraint: RFC 3339 fractions have no fixed digit count.
- Resolution: compare validated calendar seconds and exact Decimal fractions; reject leap seconds because the core has no leap-second time source.
- Consequence: portable deterministic ordering without truncation; timestamp producers must use ordinary UTC civil seconds.
- Follow-up: version a leap-second policy only if an actual Source requires it.
