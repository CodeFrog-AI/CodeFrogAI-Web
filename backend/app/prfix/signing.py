"""Authenticate review findings and fix plans without a database.

CodeFrog signs each finding it returns (bound to the repository, the pull request number,
and the head commit that was reviewed) and each fix plan it produces. A client can only send
back what CodeFrog actually produced, unchanged: it cannot invent or edit a finding, point
one at another pull request or repository, or widen an approved plan's file scope. The key
is the server's existing secret; signatures are HMAC-SHA256.
"""

import hashlib
import hmac
import json
import uuid

from app.core.config import get_settings
from app.schemas.plan import ImplementationPlan
from app.schemas.pull_request import ReviewFinding

FINDING_LABEL = b"codefrog.review-finding.v1|"
PLAN_LABEL = b"codefrog.fix-plan.v1|"


def _digest(label: bytes, payload: dict) -> str:
    key = get_settings().auth_secret_key.get_secret_value().encode()
    message = label + json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def _finding_payload(repository_id: uuid.UUID, number: int, head_sha: str, finding: ReviewFinding) -> dict:
    return {"repository_id": str(repository_id), "number": number, "head_sha": head_sha, "finding": finding.model_dump(mode="json")}


def sign_finding(repository_id: uuid.UUID, number: int, head_sha: str, finding: ReviewFinding) -> str:
    return _digest(FINDING_LABEL, _finding_payload(repository_id, number, head_sha, finding))


def finding_is_authentic(repository_id: uuid.UUID, number: int, head_sha: str, finding: ReviewFinding, signature: str) -> bool:
    return hmac.compare_digest(sign_finding(repository_id, number, head_sha, finding), signature)


def sign_plan(repository_id: uuid.UUID, number: int, head_sha: str, finding: ReviewFinding, plan: ImplementationPlan) -> str:
    payload = _finding_payload(repository_id, number, head_sha, finding)
    payload["plan"] = plan.model_dump(mode="json")
    return _digest(PLAN_LABEL, payload)


def plan_is_authentic(repository_id: uuid.UUID, number: int, head_sha: str, finding: ReviewFinding, plan: ImplementationPlan, signature: str) -> bool:
    return hmac.compare_digest(sign_plan(repository_id, number, head_sha, finding, plan), signature)
