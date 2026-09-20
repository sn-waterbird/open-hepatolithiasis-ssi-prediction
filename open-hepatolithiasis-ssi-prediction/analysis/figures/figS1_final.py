"""Quantitative two-panel figure from the final-model paired predictions.
Conclusion: estimates are imprecise and do not establish equivalence.
Panel A shows paired probabilities; panel B shows every probability change.
Python-only exports: 183 mm wide; editable PDF/SVG and 600 dpi PNG/TIFF.
"""
import sys,os,json,csv
from pathlib import Path
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input-dir',type=Path,default=Path(__file__).resolve().parents[1]/'NRI_IDI分析')
args=parser.parse_args()
O=args.input_dir.resolve()
os.environ.setdefault('MPLCONFIGDIR',str(O/'mplconfig'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=json.loads((O/'final_model_NRI_IDI_results.json').read_text())
with (O/'final_model_reclassification_predictions.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
y=np.array([int(x['y_true']) for x in rows]);a=np.array([float(x['p_full']) for x in rows]);b=np.array([float(x['p_reduced']) for x in rows]);delta=b-a
plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7,'axes.titlesize':8,'xtick.labelsize':6.5,'ytick.labelsize':7,'legend.fontsize':6.5,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(7.2,3.15),facecolor='white')
ax=fig.add_axes([.085,.19,.365,.70]);bx=fig.add_axes([.59,.19,.385,.70])
colors={0:'#6F879B',1:'#B65C3A'}
ax.plot([0,1],[0,1],ls='--',color='#9AA3A8',lw=.7,label='No change',zorder=1)
for g in [0,1]:
 m=y==g;ax.scatter(a[m],b[m],s=16 if g==0 else 23,c=colors[g],marker='o' if g==0 else '^',alpha=.8,edgecolors='white',linewidths=.35,label=f'{"No SSI" if g==0 else "SSI"} (n = {m.sum()})',zorder=3)
ax.set(xlim=(-.025,1.025),ylim=(-.025,1.025),xlabel='Full model probability (64 predictors)',ylabel='Final model probability (4 predictors)')
ax.set_title('Paired predicted probabilities',loc='left',pad=8)
ax.legend(loc='upper left',frameon=False,handletextpad=.4,labelspacing=.45)
stats='\n'.join(f"{k} {r[k]['estimate']:+.3f} ({r[k]['CI'][0]:.3f} to {r[k]['CI'][1]:.3f})\np = {r[k]['p']:.3f}" for k in ['NRI','IDI'])
ax.text(.975,.03,stats,transform=ax.transAxes,ha='right',va='bottom',fontsize=6.3,linespacing=1.2,bbox={'facecolor':'white','edgecolor':'none','pad':2})
rng=np.random.default_rng(0)
for g in [0,1]:
 m=y==g;vals=delta[m];bx.scatter(vals,g+rng.uniform(-.08,.08,len(vals)),s=17 if g==0 else 24,c=colors[g],marker='o' if g==0 else '^',alpha=.8,edgecolors='white',linewidths=.35,zorder=3)
 bx.scatter([vals.mean()],[g],s=38,marker='D',color=colors[g],edgecolors='black',linewidths=.6,zorder=5)
 bx.text(0,g+.26,f'Mean change {vals.mean():+.3f}',ha='center',va='bottom',fontsize=7,color=colors[g])
 c=r['counts']['SSI' if g else 'nonSSI'];bx.text(.98,g-.26,f"Up {c['up']}   Down {c['down']}   Ties {c['ties']}",transform=bx.get_yaxis_transform(),ha='right',va='top',fontsize=6.5,color=colors[g])
lim=max(.4,np.ceil(np.max(np.abs(delta))/.1)*.1)
bx.axvline(0,ls='--',color='#9AA3A8',lw=.7,zorder=1)
bx.set(xlim=(-lim,lim),ylim=(-.5,1.5),xlabel='Probability change (final − full)',yticks=[0,1],yticklabels=['No SSI\n(n = 59)','SSI\n(n = 11)'])
bx.set_title('Probability changes by SSI status',loc='left',pad=8)
for axis,label in [(ax,'A'),(bx,'B')]:
 axis.text(-.17,1.10,label,transform=axis.transAxes,fontweight='bold',fontsize=10)
 axis.grid(axis='x' if axis is bx else 'both',ls=':',lw=.5,color='#D9E1E5');axis.set_axisbelow(True)
fig.text(.5,.025,'Uncalibrated CatBoost + S0 models; paired held-out validation patients (n = 70)',ha='center',fontsize=6.5,color='#444444')
for ext in ['png','tiff','pdf','svg']:
 kwargs={'pil_kwargs':{'compression':'tiff_lzw'}} if ext=='tiff' else {}
 fig.savefig(O/f'Figure_S1_final.{ext}',dpi=600,facecolor='white',**kwargs)
plt.close(fig)
print('Figure S1 exported in PNG, TIFF, PDF and SVG.')
