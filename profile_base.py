import sys,os,json
from pathlib import Path
import torch,torch_npu
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "base"))
import base_executor as e
mode=sys.argv[1]
count=20 if mode=='normal' else 3
device=int(os.environ.get("NPU_DEVICE_ID","0"))
torch_npu.npu.set_device(device)
torch.set_num_threads(1)
records=[]
index=0
for t in [1,4]:
 e.DECODE_STEP=t
 for b in [1,4,16,64]:
  print('BEGIN_CASE',t,b,'TARGET_START',index+20,'COUNT',count,flush=True)
  x=e._move_inputs(e._build_cpu_inputs(b,'base',e.FIXED_SEED),torch.device(f'npu:{device}'))
  kw=e.RecurrentKdaAtkApi._common_kwargs(x)
  for i in range(20+count):
   e.npu_recurrent_kda(x['q'],x['k'],x['v'],x['g'],x['beta'],x['initial_state'],**kw)
   torch_npu.npu.synchronize()
   records.append(dict(index=index,T=t,B=b,warmup=i<20,local=i,q_shape=list(x['q'].shape),v_shape=list(x['v'].shape),q_stride=list(x['q'].stride()),state_dtype=str(x['initial_state'].dtype),flags={k:v for k,v in kw.items() if not isinstance(v,torch.Tensor)}))
   index+=1
  print('END_CASE',t,b,flush=True)
Path(sys.argv[2]).write_text(json.dumps(records,indent=2))
print('BATCH_PROFILE_DONE',mode,index,flush=True)
