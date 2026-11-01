"""Finite eager-prefill experiment. No torch import or runtime activation."""
from dataclasses import dataclass
import hashlib
import json
import re

REFERENCE = "REFERENCE"
MOE_TRY = "MOE_TRY"
FEATURES = {
    REFERENCE: (False, False, None),
    MOE_TRY: (True, False, None),
    'COMBINE': (False, True, None),
    'MOE_COMBINE': (True, True, None),
    'DENSE_CUBLAS': (False, False, 'cublas'),
    'MOE_DENSE_CUBLAS': (True, False, 'cublas'),
    'COMBINED_CUBLAS': (True, True, 'cublas'),
    'DENSE_TUNED': (False, False, 'tuned'),
    'MOE_DENSE_TUNED': (True, False, 'tuned'),
    'COMBINED': (True, True, 'tuned'),
}
FEATURES.update({'RAGGED':(False,False,None),'MOE_RAGGED':(True,False,None),
                 'RAGGED_CUBLAS':(True,False,'cublas'),'RAGGED_TUNED':(True,False,'tuned')})
RAGGED_PROFILES=frozenset(('RAGGED','MOE_RAGGED','RAGGED_CUBLAS','RAGGED_TUNED'))
PROFILES = tuple(FEATURES)
SUPPORTED = {(512, 4), (512, 6), (640, 4), (640, 6)}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256_string(value):
    return isinstance(value, str) and re.fullmatch("[a-f0-9]{64}", value) is not None


def eligible_geometry(rows, hidden, topk, intermediate, bits_x2, expert_count):
    return (type(rows) is int and 65 <= rows <= 2048 and hidden == 5120
            and topk == 6 and (intermediate, bits_x2) in SUPPORTED and expert_count == 384)


@dataclass
class Selection:
    identity: str
    candidate_ready: bool
    selected: str = REFERENCE
    session: str | None = None
    epoch: int = 0
    pending: tuple | None = None
    available: tuple | None = None

    def allowed(self):
        return self.available if self.available is not None else ((REFERENCE, MOE_TRY) if self.candidate_ready else (REFERENCE,))

    def prepare(self, request, active_abc_session):
        if not isinstance(request, dict) or set(request) != {"token", "profile", "session", "identity", "epoch"}:
            raise ValueError("Invalid prefill transaction envelope")
        if self.pending is not None:
            raise ValueError("Prefill transaction already pending")
        if request["identity"] != self.identity:
            raise ValueError("Wrong prepared prefill identity")
        if type(request["epoch"]) is not int or request["epoch"] != self.epoch + 1:
            raise ValueError("Stale prefill epoch")
        if not isinstance(request["token"], str) or not request["token"]:
            raise ValueError("Missing prefill transaction token")
        if not isinstance(active_abc_session, str) or not active_abc_session or request["session"] != active_abc_session:
            raise ValueError("Prefill selection requires the active exclusive ABC session")
        if request["profile"] not in self.allowed():
            raise ValueError("Prefill candidate was not qualified at startup")
        self.pending = (request["token"], request["profile"], request["session"], request["epoch"])

    def commit(self, token):
        if self.pending is None or self.pending[0] != token:
            raise ValueError("Unknown prefill commit")
        _, self.selected, self.session, self.epoch = self.pending
        self.pending = None

    def effective(self, active_abc_session):
        # Ending ABC or entering production cannot leave the prefill candidate on.
        if not active_abc_session:
            return REFERENCE
        if self.pending is not None:
            raise RuntimeError("Inference during a prefill transaction")
        if active_abc_session != self.session:
            return REFERENCE
        return self.selected

    def record(self, active_abc_session):
        return dict(identity=self.identity, available=list(self.allowed()),
                    selected=self.selected, effective=None if self.pending is not None else self.effective(active_abc_session),
                    session=self.session, epoch=self.epoch, pending=self.pending)
