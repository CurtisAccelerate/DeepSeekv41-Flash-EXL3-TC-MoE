"""One optional third capture at startup; no serving preparation or hot API."""
import hashlib,json,os
from pathlib import Path
from .native_mixed_private import requested,artifact,PrivateMixed,qualify_next_exact,PIN,PROOF_PIN,VERSION
from .attention_second_capture import second_runtime,capture_and_qualify
from .attention_contract import Qualification
from .attention_qualification import QualificationRejected

def prepare_mixed(worker,manifest,rt,journal,native,controller,roots,consensus,restore,agree,local_phase,health,current_identity,graph_capture,qualify,optional_type,unsafe_type):
    receipt=dict(enabled=requested(manifest),ready=False)
    worker._w35_mixed_bank=None
    if not receipt['enabled']:return receipt
    mixed=None
    def unsafe_phase(name,fn,allow_rejection=False):
        # These local CUDA driver operations have no collective inside. Finish
        # an all-rank decision even when a single rank's driver call fails.
        from .private_nodes import Rejected
        value=None;error=None;fatal=False
        try:value=fn()
        except BaseException as exc:
            error=repr(exc);fatal=not(allow_rejection and isinstance(exc,Rejected))
        votes=consensus(dict(phase=name,error=error,fatal=fatal))
        if len(votes)!=4 or any(x.get('phase')!=name or x.get('fatal') is not False for x in votes):raise unsafe_type(name+' failed; executable/context uncertain')
        if any(x.get('error') is not None for x in votes):raise optional_type(name+' rejected')
        return value
    try:
        agree(controller.next_entry is not None and controller.next_qualification is not None and controller.next_qualification.benchmark_ready(),'Mixed requires qualified existing next bank')
        from w9_exl_ab import params
        binary=local_phase('Pinned native mixed artifacts',lambda:artifact(os.environ['W9_VERSION_BUNDLE'],native.R.__file__,params.__file__))
        before=local_phase('Before mixed retained resource bound',rt.guard)
        mixed=PrivateMixed(native);roots.append(mixed) # module handle survives even failed load
        unsafe_phase('Private mixed module load',lambda:mixed.load(binary))
        local_phase('Mixed module resource bound',rt.guard)
        pool=local_phase('Mixed independent private pool',rt.torch.cuda.graph_pool_handle)
        with second_runtime(rt,pool,roots):
            from .w26_attention_split import AttentionState
            rt.attention_state=local_phase('Mixed next-profile workspace',lambda:AttentionState(rt.runner.device,controller.next_profile))
            probe=local_phase('Mixed next-profile compile probe',rt.attention_state.compile_probe)
            local_phase('Mixed workspace resource bound',rt.guard)
            with graph_capture(rt.runner.device,native.R.S.startup_capture_context):
                # First certify this independent boot capture with the exact
                # existing replay gates, before touching its executable.
                boot_nodes,boot_quality=capture_and_qualify(rt,journal,native,consensus,restore,agree,local_phase,qualify,unsafe_type)
                inventory=unsafe_phase('Mixed exact private node coverage',lambda:mixed.inventory(rt.entry),allow_rejection=True)
                update=unsafe_phase('Mixed private executable update',mixed.apply)
                quality=qualify_next_exact(rt,journal,native,consensus,controller.next_entry,qualify)
            restore();health()
            unsafe_phase('Mixed completed executable identity',mixed.assert_intact)
            local_phase('Mixed final baseline identity',current_identity)
            memory=local_phase('Mixed final global resource bound',rt.guard)
            agree(rt.attention_calls==38,'Mixed exact helper call coverage')
            bindings=local_phase('Mixed final journal verification',journal.verify)
            agree(bindings is True,'Mixed final journal bytes and bindings restored')
            state=dict(boot_capture_replay=boot_quality,replay=quality,before=before,memory=memory,
                bindings_verified=bindings,journal_bytes=journal.gpu.bytes,
                profile=controller.next_profile.record(),attention_calls=rt.attention_calls,
                native=inventory,update=update,descriptor=(6,1,True,False,0))
            pin=hashlib.sha256(json.dumps(state,sort_keys=True,allow_nan=False).encode()).hexdigest()
            q=Qualification(True,True,True,True,True,True,PROOF_PIN,pin)
            entry=rt.entry
            def separated():
                for prior in (controller.entry,controller.next_entry):
                    if entry is prior or entry['graph'] is prior['graph'] or entry['pool']==prior['pool'] or entry['attention_state'] is prior['attention_state']:raise ValueError('Mixed bank aliases retained bank')
                return True
            local_phase('Mixed publication separation',separated)
            agree(q.benchmark_ready(),'Mixed all-rank publication')
            worker._w35_mixed_bank=dict(entry=entry,qualification=q,profile=controller.next_profile,native=mixed,
                component=dict(version=VERSION,cubin_sha256=PIN,qualification_sha256=PROOF_PIN,descriptor=(6,1,True,False,0)))
            receipt.update(ready=True,state=state,probe=probe,boot_native=boot_nodes)
    except (optional_type,QualificationRejected) as exc:
        rt.mode='baseline';restore();health()
        if mixed is not None:unsafe_phase('Rejected mixed executable rollback',mixed.rollback)
        local_phase('Rejected mixed baseline identity',current_identity)
        try:retained=local_phase('Rejected mixed retained global resource bound',rt.guard)
        except optional_type as resource_error:raise unsafe_type('Retained mixed resources exceed global bound') from resource_error
        receipt.update(candidate_disabled=True,reason=str(exc),retained_memory=retained)
    return receipt
