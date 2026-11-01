"""Future admission overlay; original behavior until a bank transaction starts."""
import hmac,asyncio

class Gate:
    def __init__(self,token):
        self.token=token.encode('ascii');self.open=False
        self.attention_protocol_ready=True
        self.attention_guarded=False
        self.attention_session=None
        self.production_active=False;self.production_transition=False
        self.active_inference=0
        self.drained=asyncio.Event();self.drained.set()

    async def wait_drained(self):
        await asyncio.wait_for(self.drained.wait(),timeout=60)
        if self.active_inference:raise RuntimeError('Admission drain was not stable')

class AdmissionMiddleware:
    def __init__(self,app,state):self.app=app;self.state=state
    async def __call__(self,scope,receive,send):
        if scope['type']!='http':return await self.app(scope,receive,send)
        gate=self.state.w9_admission
        allowed=scope.get('method')=='GET' and scope.get('path','') in ('/health','/v1/models')
        headers=scope.get('headers',[])
        tokens=[v for k,v in headers if k.lower()==b'x-w9-benchmark-token']
        sessions=[v for k,v in headers if k.lower()==b'x-w26-benchmark-session']
        private=len(tokens)==1 and hmac.compare_digest(tokens[0],gate.token)
        if gate.production_transition:
            accepted=allowed
        elif gate.production_active:
            # Ordinary clients only; benchmark credentials never bypass a
            # production maintenance boundary or mix a private test into traffic.
            accepted=allowed or (gate.open and not tokens and not sessions)
        elif gate.attention_guarded:
            # A missing/revoked session during selection/drain blocks even old
            # benchmark-token holders. No wrong-session work may queue.
            accepted=allowed or (not gate.open and private and gate.attention_session is not None and len(sessions)==1 and hmac.compare_digest(sessions[0],gate.attention_session.encode('utf-8')))
        else:
            accepted=gate.open or allowed or private
        if accepted:
            if allowed:return await self.app(scope,receive,send)
            # Account the entire accepted ASGI handler, including the interval
            # before it enqueues an engine request and the complete SSE stream.
            gate.active_inference+=1;gate.drained.clear()
            try:
                from .abc_admission import receive_for  # STAR_ABC integration
                checked=await receive_for(gate,scope,receive,send)
                if checked is not None:return await self.app(scope,checked,send)
            finally:
                gate.active_inference-=1
                if gate.active_inference==0:gate.drained.set()
        body=b'{"error":{"message":"Exclusive maintenance window","type":"service_unavailable"}}'
        await send({'type':'http.response.start','status':503,'headers':[(b'content-type',b'application/json'),(b'content-length',str(len(body)).encode())]})
        await send({'type':'http.response.body','body':body})
