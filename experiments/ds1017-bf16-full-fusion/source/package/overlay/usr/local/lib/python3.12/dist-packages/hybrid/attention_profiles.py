"""Finite scratch-profile vocabulary. This module selects no next profile."""
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class Profile:
    splits: int = 16
    warps: int = 4  # core only; retained reducer always uses four warps
    direct: bool = False

    def __post_init__(self):
        if type(self.splits) is not int or type(self.warps) is not int or type(self.direct) is not bool:
            raise ValueError('Attention profile fields must have exact types')
        if (self.splits, self.warps, self.direct) not in (
            (16,4,False), (8,4,False), (16,8,False), (8,8,False), (16,4,True)
        ):
            raise ValueError('Profile is outside the frozen finite scratch matrix')

    def record(self):return asdict(self)

    @property
    def version(self):
        if self == CURRENT:return 'w26-compressed-t6-split16-v1'
        return f'w35-compressed-t6-split{self.splits}-corew{self.warps}-direct{int(self.direct)}-v1'

    @property
    def workspace_bytes(self):return (6*self.splits*16*2+6*self.splits*16*512)*4+6*16*512*2+4

CURRENT = Profile()

def next_profile(manifest):
    """Missing/null section means disabled. A future build must pin its proof."""
    value=manifest.get('next_attention')
    if value is None:return None
    if not isinstance(value,dict) or set(value)!={'profile','component_receipt_sha256'}:
        raise ValueError('Incomplete next attention manifest')
    record=value['profile']
    if not isinstance(record,dict) or set(record)!={'splits','warps','direct'}:
        raise ValueError('Unexpected profile fields')
    result=Profile(**record)
    if result == CURRENT:raise ValueError('Next bank must differ from current')
    pin=value['component_receipt_sha256']
    if not isinstance(pin,str) or len(pin)!=64 or any(c not in '0123456789abcdef' for c in pin):
        raise ValueError('Next profile needs a pinned component receipt')
    return result
