"""Drained all-rank selector inside the existing exclusive ABC benchmark session."""
import uuid
from .prefill_contract import PROFILES


def worker_invoke(worker,method,*args):
    bank=getattr(worker,"_prefill_bank",None)
    if bank is None:
        if method=="status":
            return dict(rank=int(worker.rank),ok=True,ready=False,selected="REFERENCE",effective="REFERENCE",
                        available=["REFERENCE"],identity=None,session=None,epoch=0,pending=None)
        raise ValueError("Prefill candidate was not prepared at startup")
    if method=="status":return bank.status()
    abc=worker._abc_bank.selection
    if abc.session is None or abc.pending is not None:
        raise ValueError("Prefill control requires an armed, settled ABC session")
    if method=="prepare":bank.selection.prepare(args[0],abc.session)
    elif method=="commit":
        bank.torch.cuda.synchronize()
        bank.selection.commit(args[0])
    else:raise ValueError("Unknown prefill worker method")
    return bank.status()


class Owner:
    def __init__(self,control):self.c=control

    @staticmethod
    def validate(q):
        shapes={"prefill_status":{"op"},"prefill_select":{"op","session","profile"}}
        if not isinstance(q,dict) or q.get("op") not in shapes or set(q)!=shapes[q["op"]]:
            raise ValueError("Invalid prefill control envelope")
        if q["op"]=="prefill_select":
            if not isinstance(q["session"],str) or not q["session"] or q["profile"] not in PROFILES:
                raise ValueError("Invalid prefill session/profile")

    async def rows(self,method,args=()):
        rows=await self.c.rpc("wpf_"+method,args,timeout=120)
        if len(rows)!=4 or sorted(r.get("rank") for r in rows)!=list(range(4)) or not all(r.get("ok") is True for r in rows):
            raise RuntimeError("Missing/failed prefill rank acknowledgement")
        return rows

    async def handle_locked(self,q):
        self.validate(q)
        if q["op"]=="prefill_status":return dict(prefill=await self.rows("status"))
        abc=self.c.abc_bank;g=self.c.admission
        if abc.session is None or q["session"]!=abc.session or g.production_active or g.production_transition:
            raise ValueError("Prefill selector is restricted to the current private ABC session")
        status=await self.rows("status")
        if (not all(r["ready"] and q["profile"] in r["available"] and r["pending"] is None for r in status)
                or len({r["identity"] for r in status})!=1 or len({r["epoch"] for r in status})!=1):
            raise ValueError("Prefill readiness/identity/epoch differs across ranks")
        token=uuid.uuid4().hex
        request=dict(token=token,profile=q["profile"],session=abc.session,
                     identity=status[0]["identity"],epoch=status[0]["epoch"]+1)
        try:
            await abc.close()  # closes admission, drains, pauses and qualifies existing workers
            prepared=await self.rows("prepare",(request,))
            pending=[token,q["profile"],abc.session,request["epoch"]]
            if not all(list(r["pending"])==pending for r in prepared):
                raise RuntimeError("Prefill prepare disagreement")
            committed=await self.rows("commit",(token,))
            if not all((r["selected"],r["effective"],r["session"],r["epoch"],r["identity"],r["pending"])
                       ==(q["profile"],q["profile"],abc.session,request["epoch"],request["identity"],None) for r in committed):
                raise RuntimeError("Prefill commit disagreement")
            # Old cache blocks reflect prior prefill math and must never contaminate a new arm.
            reset=await self.c.engine.reset_prefix_cache(reset_running_requests=False)
            if reset is not True:raise RuntimeError("Prefill prefix-cache reset failed")
            self.c.save("prefill-policy.json",dict(schema="star.prefill.policy.v1",**request))
            g.attention_session=abc.session
            await self.c.engine.resume_generation()
            if await self.c.engine.is_paused():raise RuntimeError("Prefill benchmark resume unconfirmed")
            self.c.phase="benchmark_serving";self.c.persist()
            return dict(prefill_profile=q["profile"],identity=request["identity"],epoch=request["epoch"],
                        session=abc.session,rows=committed,prefix_cache_reset=True,
                        ordinary_admission=False,model_quality_verified=False)
        except BaseException as exc:
            await self.c.quarantine("Uncertain prefill transaction: "+repr(exc),poison=True)
            raise
