#!/usr/bin/env python3
"""Render crimson README figures (the Faster-LLaVA style) from archived measurements. No GPU work."""
from pathlib import Path
import json
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','font.size':10})
RED='#8C0000'; MUTED='#79696B'; INK='#292122'
def save(fig,path):
 for ext in ['svg','png']:
  p=path.with_suffix('.'+ext);fig.savefig(p,dpi=170,facecolor='white')
  if ext=='svg':p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines())+'\n')
 plt.close(fig)
def hero():
 d=json.loads((ROOT.parent.parent/'evaluation/20261017/completed_comparison_summary.json').read_text())
 fig=plt.figure(figsize=(9,4.5));fig.text(.06,.90,'Faster-DMD, optimized by AI agents',color=RED,weight='bold',size=19)
 fig.text(.06,.80,'SD1.5 DMD2 · 8 H20 GPUs · global batch 2,048 · 39k → 41k continuation',color=MUTED,size=10)
 ax=fig.add_axes([.23,.24,.40,.46]);vals=[d['timing'][x]['mean_step_seconds'] for x in ['original','optimized']]
 ax.barh([1,0],vals,color=['#D7BDBF',RED],height=.24,zorder=3);ax.set_yticks([1,0],['Original precision\n+ accumulation','Faster-DMD\n+ accumulation'],color=MUTED)
 for y,v in zip([1,0],vals):ax.text(v+.8,y,f'{v:.3f} s',va='center',color=INK,size=10,weight='bold')
 ax.set_xlim(0,39);ax.set_ylim(-.6,1.6);ax.set_xticks([0,10,20,30]);ax.tick_params(length=0,labelsize=9,colors=MUTED,pad=10)
 for k in ['top','right','left']:ax.spines[k].set_visible(False)
 ax.spines['bottom'].set_color(GRID);ax.grid(axis='x',color=GRID,lw=.7,zorder=0);ax.set_xlabel('Mean effective iteration time (s) ↓',color=MUTED,labelpad=13)
 fig.text(.76,.55,f"{d['optimized_vs_original_step_speedup']:.2f}×",color=RED,size=33,weight='bold');fig.text(.76,.45,'step speedup',color=MUTED,size=11)
 fig.text(.06,.055,'All 2,000 effective iterations per arm. Checkpoint writes excluded; one training seed per arm.',color=MUTED,size=9)
 save(fig,ROOT/'01-h20-benchmark')
CARD='#FAF5F5'; GRID='#EADFE0'
def history():
 rows=json.loads((ROOT/'history.json').read_text())['rows'];height=8.4;top=lambda inches:1-inches/height
 fig=plt.figure(figsize=(9,height),facecolor='white')
 fig.text(.09,top(.36),'Faster-DMD',size=11,weight='bold',color=RED)
 fig.text(.09,top(.79),'Agent Optimization History',size=21.5,weight='bold',color=INK)
 fig.text(.94,top(.65),f"{rows[-1]['speedup']:.2f}×",size=29,ha='right',weight='bold',color=RED)
 fig.text(.94,top(.88),'step speedup',size=10.5,ha='right',color=MUTED)
 fig.text(.09,top(1.31),'Cumulative speedup · 4 H20 GPUs · global batch 128',size=11,color=MUTED)
 ax=fig.add_axes([.09,top(4.19),.85,2.61/height]);x=list(range(len(rows)));y=[v['speedup'] for v in rows]
 tags=[('Final' if v['round']=='final' else f"r{v['round']}") for v in rows]
 ax.plot(x,y,color=RED,lw=2.3,zorder=2)
 for xi,v,row in zip(x,y,rows):ax.plot([xi],[v],marker='o',ms=6.5,zorder=3,color=RED,mfc=RED if row['kind']=='full_check' else 'white',mew=1.8)
 for i,v in enumerate(y):ax.annotate(f'{v:.2f}×',(i,v),xytext=(0,11 if i%2==0 else -23),textcoords='offset points',ha='center',size=12,color=RED if i==len(rows)-1 else INK,weight='bold')
 ax.set_xticks(x,tags);ax.set_yticks([1.0,1.2,1.4,1.6,1.8]);ax.yaxis.set_major_formatter(lambda v,_:f'{v:.2f}×')
 ax.set_ylim(.9,1.86);ax.set_xlim(-.5,len(rows)-.5)
 ax.spines[['top','right','left']].set_visible(False);ax.spines['bottom'].set_color(GRID)
 ax.tick_params(length=0,labelsize=11.5,colors=MUTED,pad=9);ax.grid(axis='y',color=GRID,lw=.7,zorder=0);ax.set_axisbelow(True)
 for i,(row,tag) in enumerate(zip(rows,tags)):
  bx,by=.09+(i%2)*.44,top(5.18+(i//2)*.74)
  fig.add_artist(FancyBboxPatch((bx,by-.378/height),.41,.585/height,boxstyle='round,pad=0.004,rounding_size=0.007',transform=fig.transFigure,facecolor=CARD,edgecolor='none'))
  fig.text(bx+.014,by-.08/height,tag,size=12,weight='bold',color=RED)
  fig.text(bx+.079,by-.08/height,textwrap.fill(row['label'],width=24,break_long_words=False,break_on_hyphens=False),size=11.5,color=INK,weight='bold',va='center',linespacing=1.35)
 fig.text(.09,top(8.15),'Open marker: timing screen against a frozen reference. Filled: final full check, reference remeasured.\nCumulative configurations, not isolated ablations. Separate workload from the 8-GPU continuation headline.',size=9.5,color=MUTED,linespacing=1.4)
 save(fig,ROOT/'02-history')
def modulemap():
 fig=plt.figure(figsize=(9,6.6),facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
 ax.text(.055,.92,'What changed in Faster-DMD',size=23,color=INK,weight='bold')
 cards=[('Generator','Compiled UNet forward\nSelected BF16 no-grad computation','main/sd_unified_model.py','1 file'),
        ('Guidance','Compiled fake UNet · BF16 no-grad score\nLess checkpoint recomputation','main/sd_guidance.py','1 file'),
        ('Training loop','Autotuning · DDP buckets · batched gathers\nAccumulation dispatch · update / save audits','main/train_sd.py','1 file'),
        ('GPU memory adaptation','Explicit G / D accumulation phases\nGlobal batch 2,048 · scheduler clock 64','main/accumulation.py','1 new file')]
 for i,(title,desc,path,count) in enumerate(cards):
  x,y=.055+(i%2)*.455,.49-(i//2)*.39
  ax.add_patch(FancyBboxPatch((x,y),.425,.34,boxstyle='round,pad=0.009,rounding_size=0.012',facecolor=CARD,edgecolor=GRID))
  ax.plot([x,x+.425],[y+.34,y+.34],color=RED,linewidth=2,solid_capstyle='round')
  ax.text(x+.014,y+.30,f'{i+1:02d}',size=11,color=RED,weight='bold');ax.text(x+.41,y+.30,count,size=10,color=MUTED,ha='right')
  ax.text(x+.014,y+.254,title,size=13,color=INK,weight='bold');ax.text(x+.014,y+.198,desc,size=10.3,color=INK,va='top',linespacing=1.6)
  ax.text(x+.014,y+.035,path,size=9.5,color=MUTED,linespacing=1.5)
 save(fig,ROOT.parent/'change-views/module-map')
if __name__=='__main__':hero();history();modulemap()
