"""Independent effective-batch oracle: gradients, clipping, Adam state and G/D cadence."""
import os,json,torch
import torch.nn.functional as F
from types import SimpleNamespace as NS
from unittest.mock import patch
from main.accumulation import accumulated_step
from torch.nn.parallel import DistributedDataParallel as DDP
rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
if world>1:torch.distributed.init_process_group('gloo')
class Counter:
 def __init__(self):self.n=0
 def step(self):self.n+=1
class Accelerator:
 device=torch.device('cpu')
 def __init__(self):self.gradients=[]
 def backward(self,x):x.backward()
 def clip_grad_norm_(self,parameters,max_norm):
  parameters=list(parameters);self.gradients.append([p.grad.detach().clone() for p in parameters]);return torch.nn.utils.clip_grad_norm_(parameters,max_norm)
class Model(torch.nn.Module):
 def __init__(self):
  super().__init__();self.feedforward_model=torch.nn.Linear(2,2).double();self.guidance_model=torch.nn.Linear(2,1).double()
 def forward(self,noise,tokens,uncond,**kw):
  if kw['generator_turn']:
   if kw['compute_generator_gradient']:image=self.feedforward_model(noise.flatten(1))
   else:
    with torch.no_grad():image=self.feedforward_model(noise.flatten(1))
   losses={}
   if kw['compute_generator_gradient']:
    self.guidance_model.requires_grad_(False);pred=self.guidance_model(image);self.guidance_model.requires_grad_(True)
    losses={'loss_dm':(image-kw['real_train_dict']['images']).square().mean(),'gen_cls_loss':F.softplus(-pred).mean()}
   return losses,{'guidance_data_dict':{'image':image.detach(),'real':kw['real_train_dict']['images']}}
  d=kw['guidance_data_dict'];pred=self.guidance_model(d['image']);real=self.guidance_model(d['real'])
  return {'loss_fake_mean':pred.square().mean(),'guidance_cls_loss':F.softplus(pred).mean()+F.softplus(-real).mean()},{}
def close(a,b):torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
checks=[]
for count in [1,2,8]:
 torch.manual_seed(123);model=Model();reference=Model();reference.load_state_dict(model.state_dict())
 if world>1:
  model.feedforward_model=DDP(model.feedforward_model);model.guidance_model=DDP(model.guidance_model)
 opt=lambda module:torch.optim.AdamW(module.parameters(),lr=.01,betas=(.9,.999),weight_decay=.01)
 a=Accelerator();g=opt(model.feedforward_model);d=opt(model.guidance_model);rg=opt(reference.feedforward_model);rd=opt(reference.guidance_model)
 updates={'g':0,'d':0}
 g.register_step_post_hook(lambda *_:updates.__setitem__('g',updates['g']+1));d.register_step_post_hook(lambda *_:updates.__setitem__('d',updates['d']+1))
 micro=8//count
 t=NS(accelerator=a,model=model,args=NS(gradient_accumulation_steps=count,gan_alone=False,dm_loss_weight=1.,schedule_world_size=64),step=0,dfake_gen_update_ratio=10,wandb_iters=50,batch_size=micro,latent_channel=2,latent_resolution=1,uncond_embedding=torch.zeros(1,1,2),cls_on_clean_image=True,gen_cls_loss=True,gen_cls_loss_weight=.1,guidance_cls_loss_weight=.2,max_grad_norm=.5,optimizer_generator=g,optimizer_guidance=d,scheduler_generator=Counter(),scheduler_guidance=Counter())
 for step in [0,1,9,10]:
  t.step=step;a.gradients=[];ref_grads=[]
  x=torch.arange(world*16,dtype=torch.float64).reshape(world*8,2)/10+step/20;y=x.flip(1)*.3
  localx=x[rank*8:(rank+1)*8];localy=y[rank*8:(rank+1)*8]
  tokens=[{'text_input_ids_one':torch.zeros(micro,1,1,dtype=torch.long)} for _ in range(count)]
  t.dataloader=iter(tokens);t.guidance_dataloader=iter(tokens);t.real_dataloader=iter([{'images':z} for z in localy.split(micro)])
  xs=iter(z.reshape(micro,2,1,1) for z in localx.split(micro))
  with patch('torch.randn',side_effect=lambda *args,**kwargs:next(xs)):accumulated_step(t)
  # Independent full-batch equations; D reuses the PRE-update image.
  if step%10==0:
   image=reference.feedforward_model(x);pred=F.linear(image,reference.guidance_model.weight.detach(),reference.guidance_model.bias.detach())
   loss=(image-y).square().mean()+.1*F.softplus(-pred).mean();loss.backward()
   ref_grads.append([p.grad.clone() for p in reference.feedforward_model.parameters()]);torch.nn.utils.clip_grad_norm_(reference.feedforward_model.parameters(),.5);rg.step();rg.zero_grad(set_to_none=True)
  else:
   with torch.no_grad():image=reference.feedforward_model(x)
  pred=reference.guidance_model(image.detach());real=reference.guidance_model(y)
  loss=pred.square().mean()+.2*(F.softplus(pred).mean()+F.softplus(-real).mean());loss.backward()
  ref_grads.append([p.grad.clone() for p in reference.guidance_model.parameters()]);torch.nn.utils.clip_grad_norm_(reference.guidance_model.parameters(),.5);rd.step();rd.zero_grad(set_to_none=True)
  assert len(ref_grads)==len(a.gradients)
  for actual,expected in zip(a.gradients,ref_grads):
   for u,v in zip(actual,expected):close(u,v)
  for u,v in zip(model.parameters(),reference.parameters()):close(u,v)
  for actual,expected in [(g,rg),(d,rd)]:
   for u,v in zip(actual.state.values(),expected.state.values()):
    for key in ['step','exp_avg','exp_avg_sq']:close(u[key],v[key])
 assert updates=={'g':2,'d':4};assert t.scheduler_generator.n==t.scheduler_guidance.n==256
 checks.append({'accumulation':count,'local_effective_batch':8,'preclip_gradients':True,'clipped_parameters':True,'adam_moments_and_steps':True,'pre_generator_update_samples':True,'optimizer_updates':updates,'scheduler_ticks':256})
print(json.dumps({'status':'PASS','rank':rank,'world':world,'dtype':'float64','rtol':1e-10,'atol':1e-10,'checks':checks}),flush=True)
if world>1:torch.distributed.destroy_process_group()
