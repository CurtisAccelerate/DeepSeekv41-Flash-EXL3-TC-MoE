"""Generated-only CUDA graph smoke of real sampler code; simulated pair all-gathers.

No checkpoint or model load. Actual four-rank communication and model-state
qualification remain mandatory in abc_startup. These are implementation tests.
"""
def main():
    import argparse,gc,json,torch
    import torch.nn.functional as F
    from types import SimpleNamespace as NS
    from hybrid import abc_sampler as S,abc_gate as G
    from hybrid.abc_contract import capture_policy
    ap=argparse.ArgumentParser();ap.add_argument('--scope',choices=('all','abc'),default='all');args=ap.parse_args()
    torch.cuda.set_device(0);device=torch.device('cuda',0)
    H,V,N,D=5120,129280,32320,256
    class Pairs:
        def all_gather(self,out,x):out.view((4,)+x.shape).copy_(x[None].expand((4,)+x.shape))
    with torch.inference_mode():
        torch.manual_seed(1017)
        norm=torch.nn.RMSNorm(H,device=device,dtype=torch.bfloat16)
        lm=torch.nn.Linear(H,N,bias=False,device=device,dtype=torch.bfloat16)
        w1=torch.nn.Embedding(V,D,device=device,dtype=torch.bfloat16)
        w2=torch.nn.Linear(D,V,bias=False,device=device,dtype=torch.bfloat16)
        model=NS(config=NS(hidden_size=H),model=NS(norm=norm,markov_head=NS(markov_w1=w1,markov_w2=w2)),lm_head=lm)
        model.logits_processor=lambda head,h,**kw:F.linear(h,head.weight)
        model.markov_embed=w1;model.map_draft_to_target=lambda ids:ids
        model.compute_confidence=lambda h,e:((h[:,:D].float()*e.float()).sum(-1)*.001).sigmoid()
        p=NS(model=model,device=device,input_ids=torch.arange(8,device=device,dtype=torch.int64),_anchor_idx=torch.arange(8,device=device,dtype=torch.int64))
        def incumbent(b,h):
            local=F.linear(norm(h),lm.weight).view(b,5,N)
            prev=p.input_ids[p._anchor_idx[:b]];ids=[]
            for i in range(5):
                prev=(local[:,i]+F.linear(w1(prev),w2.weight[:N])).argmax(-1)
                ids.append(prev)
            return torch.stack(ids,1)
        p._sample_sequential_incumbent=incumbent
        router=G.prepare(device);q8=S.quantize_head(p);states=[];rows=[]
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();before=torch.cuda.mem_get_info()[0]
        for key in capture_policy(args.scope,True)['samplers']:
            b,fp8,fused,gate=key
            state,receipt=S.prepare(p,Pairs(),0,N,b,fp8,fused,gate,q8,lambda x:[x.copy() for _ in range(4)])
            states.append(state)
            if gate:
                assert state.keep.dtype==torch.int32
                assert torch.equal(G.MASK[G.INDICES[b]],state.keep.flatten())
            rows.append(dict(key=key,**receipt))
            print('SAMPLER_SMOKE '+json.dumps(dict(key=key,passed=True)),flush=True)
        torch.cuda.synchronize()
        streams={r['capture_stream'] for r in rows};pools={tuple(s.graph.pool()) for s in states}
        if len(streams)!=1 or len(pools)!=len(states):raise ValueError('Expected one capture stream and distinct private graph pools')
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();growth=max(0,before-torch.cuda.mem_get_info()[0])
        if args.scope=='abc' and growth>256*1024**2:raise ValueError('Eight new C sampler banks exceed standalone256MiB retained budget: '+str(growth))
        print('STAR_GENERATED_SAMPLER_GPU '+json.dumps(dict(passed=True,scope=args.scope,capture_stream_count=len(streams),private_graph_pool_count=len(pools),retained_sampler_growth_bytes=growth,retained_sampler_budget_bytes=256*1024**2 if args.scope=='abc' else None,generated_only=True,model_loaded=False,model_quality_verified=False,simulated_collectives=True,actual_tp4_qualification_still_required=True,hidden=H,local_vocab=N,global_vocab=V,markov_rank=D,router=router,variants=rows)),flush=True)

if __name__=='__main__':main()
