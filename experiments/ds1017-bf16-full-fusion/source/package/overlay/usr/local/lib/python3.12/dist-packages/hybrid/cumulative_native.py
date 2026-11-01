"""Additional sampler-family binding check for the new cumulative bundle.

The original target boot/native/six inventory remains mandatory. This explicitly
extends its former incumbent-sampler-only assumption; it does not pretend that
an unrelated unrecognized sampler is part of attention-v2's qualified identity.
"""
def assert_sampler_family(runner):
    p=runner.drafter
    state=getattr(p,'_w26_sampler_bank_state',None)
    selector=getattr(p,'_w26_sampler_selector',None)
    if state is not None:
        from .w26_sampler_bank import BankSampler
        if type(state) is not BankSampler:raise ValueError('Unrecognized cumulative sampler state')
        state.assert_intact()
    if selector is not None:
        from .cumulative_worker import WorkerCumulative
        if type(selector) is not WorkerCumulative or selector.controller.sampler_state is not state:
            raise ValueError('Cumulative selector/prepared sampler binding changed')
