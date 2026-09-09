"""Validation result types.

Every problem is reported with the three fields the handoff requires:
the document, the field (JSON-ish path), and the violated rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ValidationError:
    """A violated contract rule.

    - document: identifies the document/file the problem belongs to.
    - field: JSON pointer-ish path inside that document ("" for whole doc).
    - rule: short, stable rule name (e.g. "hash_mismatch", "split_leakage").
    - message: human-readable explanation.
    """

    document: str
    field: str
    rule: str
    message: str

    def __str__(self) -> str:
        location = f"{self.document}" + (f"#{self.field}" if self.field else "")
        return f"[{self.rule}] {location}: {self.message}"


@dataclass(frozen=True)
class ValidationWarning:
    """A non-fatal observation. Warnings never fail a validation."""

    document: str
    field: str
    rule: str
    message: str

    def __str__(self) -> str:
        location = f"{self.document}" + (f"#{self.field}" if self.field else "")
        return f"[{self.rule}] {location}: {self.message}"


@dataclass
class ValidationResult:
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationWarning] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, document: str, field: str, rule: str, message: str) -> None:
        self.errors.append(ValidationError(document, field, rule, message))

    def add_warning(self, document: str, field: str, rule: str, message: str) -> None:
        self.warnings.append(ValidationWarning(document, field, rule, message))

    def extend(self, other: "ValidationResult") -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
