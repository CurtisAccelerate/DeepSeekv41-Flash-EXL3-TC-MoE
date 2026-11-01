"""Narrow/auto verification tests accept explicit greedy requests only."""
import json
async def receive_for(gate,scope,receive,send):
    if not getattr(gate,'abc_narrow',False) or scope.get('method')!='POST':return receive
    data=[]
    while True:
        item=await receive()
        if item.get('type')!='http.request':raise RuntimeError('Disconnected ABC request body')
        data.append(item.get('body',b''))
        if not item.get('more_body',False):break
    body=b''.join(data)
    try:
        obj=json.loads(body)
        valid=type(obj.get('temperature')) in (int,float) and obj['temperature']==0 and obj.get('n',1)==1
    except (ValueError,AttributeError):valid=False
    if not valid:
        error=b'{"error":{"message":"Narrow verification tests require explicit temperature=0 and n=1"}}'
        await send({'type':'http.response.start','status':400,'headers':[(b'content-type',b'application/json')]})
        await send({'type':'http.response.body','body':error})
        return None
    consumed=False
    async def replay():
        nonlocal consumed
        if not consumed:consumed=True;return {'type':'http.request','body':body,'more_body':False}
        return await receive()
    return replay
