"""CPU-only qualification and control schema for a FUTURE attention bank.

This is not an RPC implementation and is never imported by the retained owner.
Component numerical qualification permits controlled benchmarking, not promotion.
"""
from dataclasses import asdict, dataclass
import hashlib
import json

SCHEMA = 'w26.attention-bank.v1'
BASELINE = 'baseline'
CANDIDATE = 'attention-split-t6-v1'
DESCRIPTOR = (6, 1, True, False, 0)

@dataclass(frozen=True)
class Identity:
    owner_id: str
    baseline_capture_generation: int
    native_arm: str
    native_version: str
    native_hash: str
    collective_profile: str
    sampler_version: str
    attention_source_sha256: str
    contract_sha256: str
    descriptor: tuple = DESCRIPTOR

    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(',', ':')).encode()).hexdigest()

@dataclass(frozen=True)
class Qualification:
    source_layout_verified: bool = False
    resource_and_lifetime_verified: bool = False
    component_numerical_verified: bool = False
    model_state_restore_verified: bool = False
    native_collective_equivalent: bool = False
    all_rank_ready: bool = False
    numerical_receipt_sha256: str = ''
    state_receipt_sha256: str = ''
    # These are separate evidence fields. A dummy hidden-state check is NOT
    # model quality and is never allowed to populate these implicitly.
    end_to_end_quality_verified: bool = False
    end_to_end_performance_verified: bool = False
    end_to_end_receipt_sha256: str = ''

    def benchmark_ready(self):
        return all((self.source_layout_verified, self.resource_and_lifetime_verified,
                    self.component_numerical_verified, self.model_state_restore_verified,
                    self.native_collective_equivalent, self.all_rank_ready,
                    len(self.numerical_receipt_sha256) == 64,
                    len(self.state_receipt_sha256) == 64))

    def promotion_evidence_complete(self):
        return self.benchmark_ready() and all((self.end_to_end_quality_verified,
            self.end_to_end_performance_verified, len(self.end_to_end_receipt_sha256) == 64))

class AttentionControl:
    """Per-rank protocol; FUTURE owner must provide drain and CPU consensus.

    publish(), prepare(), commit() do not run a model or mutate W9 registries.
    Commit follows unanimous prepare while public admission remains closed.
    Private benchmark requests require a matching session token downstream.
    """
    def __init__(self, identity):
        self.identity = identity
        self.qualification = Qualification()
        self.entry = None
        self.next_entry = None
        self.next_qualification = None
        self.next_profile = None
        self.selected = BASELINE
        self.pending = None
        self.session_token = None
        self.reason = 'not_prepared'

    def publish(self, entry, qualification):
        if entry is None or not qualification.benchmark_ready():
            self.disable('candidate_preparation_incomplete')
            return False
        self.entry = entry
        self.qualification = qualification
        self.reason = None
        return True

    def disable(self, reason):
        # Retain strong graph/module refs even after rejection. Retirement needs
        # its own quiescent lifetime proof; a Python ref drop is not that proof.
        self.selected = BASELINE
        self.pending = None
        self.session_token = None
        self.reason = str(reason)

    def validate_next(self, entry, qualification, profile):
        from .attention_profiles import Profile,CURRENT
        if self.next_entry is not None:raise ValueError('Next bank already published')
        if type(profile) is not Profile or profile==CURRENT:raise ValueError('Unsupported next profile')
        if self.entry is None or not self.qualification.benchmark_ready() or not qualification.benchmark_ready():
            raise ValueError('Both attention banks need independent qualification')
        if (entry is self.entry or entry['graph'] is self.entry['graph'] or entry['pool']==self.entry['pool']
            or entry['attention_state'] is self.entry['attention_state']):
            raise ValueError('Current and next attention entries must own distinct graph/pool/state')
        return True

    def publish_next(self, entry, qualification, profile):
        self.validate_next(entry,qualification,profile)
        self.next_entry=entry;self.next_qualification=qualification;self.next_profile=profile

    def prepare(self, request, *, paused, drained, admission_closed, identity):
        if request.get('schema') != SCHEMA or request.get('mode') != 'benchmark_only':
            raise ValueError('Only the explicit benchmark-only protocol is supported')
        if not (paused and drained and admission_closed):
            raise ValueError('Selection requires paused/drained/closed owner')
        if identity != self.identity or request.get('identity_sha256') != identity.digest():
            raise ValueError('Owner, source, or cumulative native/profile identity changed')
        if self.pending is not None:
            raise ValueError('Transaction already pending')
        bank = request.get('bank_id')
        token = request.get('transaction_token')
        session = request.get('benchmark_session_token')
        if bank not in (BASELINE, CANDIDATE) or not isinstance(token, str) or not token or not isinstance(session, str) or not session:
            raise ValueError('Missing bank/transaction/session identity')
        if bank == CANDIDATE and (self.entry is None or not self.qualification.benchmark_ready()):
            raise ValueError('Candidate is not benchmark-qualified')
        self.pending = (token, bank, session)
        return dict(prepared=True, transaction_token=token, bank_id=bank, identity_sha256=identity.digest())

    def commit(self, token, *, all_rank_prepared):
        if not all_rank_prepared or self.pending is None or self.pending[0] != token:
            self.disable('all_rank_prepare_or_token_mismatch')
            return False
        _, self.selected, self.session_token = self.pending
        self.pending = None
        return True

    def lookup(self, *, runtime_mode, descriptor, benchmark_session_token, admission_closed, identity):
        if self.pending is not None:
            raise RuntimeError('Replay forbidden during bank transaction')
        if self.selected == BASELINE:
            return None
        if identity != self.identity or not admission_closed or benchmark_session_token != self.session_token:
            raise RuntimeError('Selected candidate requires unchanged identity and closed private benchmark session')
        if runtime_mode != 'FULL' or tuple(descriptor) != DESCRIPTOR:
            return None
        return self.entry

    def status(self):
        return dict(schema=SCHEMA, selected_bank=self.selected, default_bank=BASELINE,
                    candidate_bank=CANDIDATE, baseline_retained=True,
                    benchmark_ready=self.entry is not None and self.qualification.benchmark_ready(),
                    production_enabled=False, qualification=asdict(self.qualification),
                    identity=asdict(self.identity), identity_sha256=self.identity.digest(),
                    pending=self.pending is not None, reason=self.reason)
