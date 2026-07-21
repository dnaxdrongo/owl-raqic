"""Typed certificate/report payloads."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from .json import JSONValue


class ErrorPayload(TypedDict):
    type: str
    message: str


class IdentityPayload(TypedDict):
    source_sha256: str
    config_sha256: str
    plan_sha256: str
    environment_sha256: str
    scientific_contract_sha256: str


class CertificatePayload(TypedDict):
    schema_version: str
    certificate: str
    passed: bool
    classification: str
    reason_code: str | None
    identity: IdentityPayload
    error: ErrorPayload | None
    evidence: dict[str, JSONValue]
    artifact_sha256: NotRequired[str]
