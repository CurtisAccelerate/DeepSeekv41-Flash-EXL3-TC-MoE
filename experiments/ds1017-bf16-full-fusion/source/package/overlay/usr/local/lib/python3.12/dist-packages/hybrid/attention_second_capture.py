"""Startup-only second capture; no RPC, default profile or GPU import here."""
from contextlib import contextmanager

@contextmanager
def second_runtime(rt, pool, roots):
    """Restore dispatch references on every exit; never discard captured roots."""
    if rt.mode!='baseline':raise ValueError('Second bank requires baseline dispatch')
    if pool==rt.pool or pool==rt.wrapper.graph_pool:
        raise ValueError('Second attention bank needs a distinct private pool')
    saved=(rt.pool,rt.entry,rt.attention_state,rt.attention_calls)
    rt.pool=pool;rt.entry=None;rt.attention_state=None;rt.attention_calls=0
    try:yield
    finally:
        # Even rejected captures retain their output/pool/workspace bindings.
        roots.extend((rt.entry,rt.attention_state,pool))
        rt.mode='baseline'
        rt.pool,rt.entry,rt.attention_state,rt.attention_calls=saved

def capture_and_qualify(rt,journal,native,consensus,restore,agree,local_phase,qualify,unsafe_type):
    """Caller owns graph_capture context, inference_mode and second_runtime.

    Capture/communication errors stay globally fatal. Qualification's existing
    safe rejection/restoration logic is reused, including its unsafe path.
    """
    from .initial_decode_contract import CompletedMetadataRejection
    restore();rt.mode='capture'
    try:rt.dummy('FULL',target_only=True)
    except CompletedMetadataRejection:raise
    except BaseException as exc:raise unsafe_type('Second private FULL capture failed') from exc
    agree(True,'Second private FULL capture completed on every rank')
    rt.mode='baseline';restore()
    nodes=local_phase('Second private native/profile coverage',lambda:native.qualify_private(rt.entry))
    local_phase('Second post-capture global resource bound',rt.guard)
    quality=qualify(rt,journal,native,consensus)
    return nodes,quality
