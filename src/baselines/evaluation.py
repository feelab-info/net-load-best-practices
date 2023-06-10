import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from utils.metrics import  get_pointwise_metrics, get_prediction_interval_scores, get_cwi_score
from utils.visual_functions import plot_prediction_with_scale
from utils.visual_functions import  plot_prediction_with_upper_lower
from utils.visual_functions import  plot_prediction_with_pi
import matplotlib_inline.backend_inline
import matplotlib.pyplot as plt
import arviz as az
import matplotlib.dates as mdates
matplotlib_inline.backend_inline.set_matplotlib_formats("svg", "pdf")
az.style.use(["science", "grid", "arviz-doc", 'tableau-colorblind10'])

def evaluate_point_forecast(outputs, target_range, hparams, exp_name, file_name, ghi_dx=1, show_fig=False):
    
    pd_metrics, spilit_metrics = {}, {}
    logs = {}
    for j in range(outputs['true'].shape[-1]):
        metrics=[]
        for i in range(0, len(outputs['true'])):
         
            true = outputs['true'][i,:, j]
            pred = outputs['pred'][i,:, j]
            
            R = target_range[j]
            t_nmpic = true.std()/R
            
            df = pd.DataFrame(outputs['index'][i])
            df.columns=['Date']
            index=df.Date.dt.round("D").unique()[-1]
            point_scores = get_pointwise_metrics(pred, true, R, 1)
            point_scores =pd.DataFrame.from_dict(point_scores, orient='index').T
            point_scores['timestamp']=index
            metrics.append(point_scores)

        metrics = pd.concat(metrics)
        outputs[f"{hparams['targets'][j]}_metrics"]=metrics
        print(f"Results for {hparams['targets'][j]}")
        print(pd.DataFrame(metrics.median()).T[[  'mae', 'nrmse',  'corr',  'nbias']].round(3))

        bad=np.where(metrics['mae']==metrics['mae'].max())[0][0]
        good=np.where(metrics['mae']==metrics['mae'].min())[0][0]
        outputs[f"{hparams['targets'][j]}_bad"]=bad
        outputs[f"{hparams['targets'][j]}_good"]=good
        
        colors = ['C0', 'C3', 'C5']
        fig, ax = plt.subplots(1,2, figsize=(9,2))
        ax = ax.ravel()
        ax[0], lines, label=plot_prediction_with_upper_lower(ax[0],outputs['true'][good][:, j].flatten(),
                                     outputs['pred'][good][:, j].flatten(),
                                    None,
                                    None)
        
        

        met=metrics[['nrmse', 'mae',  'corr']].iloc[good].values
        ax[0].set_title("Good Day NMRSE: {:.2g}, \n MAE: {:.3g}, CORR: {:.3g}%".format(met[0], met[1], met[2]), fontsize=15)
        leg = ax[0].legend(lines, label, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

        ax[1],lines, label=plot_prediction_with_upper_lower(ax[1], outputs['true'][bad][:, j].flatten(),
                                     outputs['pred'][bad][:, j].flatten(),
                                    None,
                                    None)
        
        met=metrics[['nrmse', 'mae',  'corr']].iloc[bad].values
        ax[1].set_title("Bad Day NMRSE: {:.2g}, \n MAE: {:.3g}, CORR: {:.3g}%".format(met[0], met[1], met[2]), fontsize=15)
        leg = ax[1].legend(lines, label, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        
        
        fig.tight_layout(pad=1.08, h_pad=0.5, w_pad=0.5)
        fig.savefig(f"../figures/{exp_name}/{hparams['encoder_type']}/{file_name}_{hparams['targets'][j]}_results.pdf", dpi=480)
        if not show_fig:
            plt.close()


   
    outputs["targets_range"]=target_range
    return outputs 

def evaluate_prob_forecast(outputs, target_range, hparams, exp_name, file_name, show_fig=False):
    
    pd_metrics, spilit_metrics = {}, {}
    logs = {}
    for j in range(outputs['loc'].shape[-1]):
        metrics=[]
        for i in range(0, len(outputs['loc'])):
            p=outputs['loc'][i,: ,j]
            t=outputs['true'][i,:, j]
            R = target_range[j]#t.max()-t.min()
            t_nmpic = t.std()/R
            sample=outputs['samples'][:, i, :, j]
            low = outputs['lower'][i,:, j] if not hparams['conformalize'] else outputs['lower-calib'][i,:, j]
            upp = outputs['upper'][i,:, j] if not hparams['conformalize'] else outputs['upper-calib'][i,:, j]
            
            df = pd.DataFrame(outputs['index'][i][-len(p):])
            df.columns=['Date']
            index=df.Date.dt.round("D").unique()[-1]

            point_scores = get_pointwise_metrics(p, t, R, 1)
            pic_scores = get_prediction_interval_scores(pred=p, 
                                               true=t,  
                                               target_range=R, 
                                                samples=sample,
                                              lower=low,
                                              upper=upp)

            ciwe = get_cwi_score( pic_scores['nmpi'], 
                                  pic_scores['pic'], 
                                  point_scores['nrmse'], 
                                  t_nmpic)
            
            cirps = get_cwi_score( pic_scores['nmpi'], 
                                  pic_scores['pic'], 
                                  pic_scores['ncrps'], 
                                  t_nmpic)

            point_scores.update(pic_scores)
            point_scores['ciwe'] = ciwe 
            point_scores['cirps'] = cirps
            point_scores =pd.DataFrame.from_dict(point_scores, orient='index').T
            point_scores['timestamp']=index
            metrics.append(point_scores)

        metrics = pd.concat(metrics)
        outputs[f"{hparams['targets'][j]}_metrics"]=metrics



        print(f"Results for {hparams['targets'][j]}")
        print(pd.DataFrame(metrics.median()).T[[  'mae', 'nrmse',  'ncrps',  'pic',  'nmpi', 'ciwe', 'cwc', 'corr', 'bias', 'nbias']].round(3))

        bad=np.where(metrics['nrmse']==metrics['nrmse'].max())[0][0]
        good=np.where(metrics['nrmse']==metrics['nrmse'].min())[0][0]
        
        colors = ['C0', 'C3', 'C5']
        fig, ax = plt.subplots(2,2, figsize=(8,3))
        ax = ax.ravel()
        ax[0], lines, label=plot_prediction_with_scale(ax[0],outputs['true'][good][:, j].flatten(),
                                     outputs['loc'][good][:, j].flatten(),
                                      outputs['scale'][good][:, j])
        met=metrics[['nrmse', 'ciwe', 'ncrps',  'corr']].iloc[good].values
        ax[0].set_title("Good Day NMRSE: {:.2g}, \n CWE: {:.3g}, CRPS: {:.3g}%".format(met[0], met[1], met[2]), fontsize=15)
        leg = ax[0].legend(lines, label, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)


        ax[1],lines, label=plot_prediction_with_scale(ax[1],outputs['true'][bad][:, j].flatten(),
                                     outputs['loc'][bad][:, j].flatten(),
                                      outputs['scale'][bad][:, j])
        
        met=metrics[['nrmse', 'ciwe', 'ncrps',  'corr']].iloc[bad].values
        ax[1].set_title("Bad Day NMRSE: {:.2g}, \n CWE: {:.3g}, CRPS: {:.3g}%".format(met[0], met[1], met[2]), fontsize=15)
        leg = ax[1].legend(lines, label, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        
        
        N=len((outputs['loc'][good]))
        ax[2].plot(outputs['index'][good][-N:], outputs['inputs'][good][-N:,1], '.', mec="#ff7f0e", mfc="None", label="Ghi")
        hfmt = mdates.DateFormatter('%d %H')
        ax[2].xaxis.set_major_formatter(hfmt)
        ax[2].set_ylim(0, 1)
        plt.setp( ax[2].xaxis.get_majorticklabels(), rotation=90 );
        
        ax[3].plot(outputs['index'][bad][-N:], outputs['inputs'][bad][-N:,1], '.', mec="#ff7f0e", mfc="None", label="Ghi")
        ax[3].set_ylim(0, 1)
        hfmt = mdates.DateFormatter('%d %H')
        ax[3].xaxis.set_major_formatter(hfmt)
        plt.setp( ax[3].xaxis.get_majorticklabels(), rotation=90 );
        fig.tight_layout(pad=1.08, h_pad=0.5, w_pad=0.5)
        fig.savefig(f"../figures/{exp_name}/{hparams['encoder_type']}/{file_name}_{hparams['targets'][j]}_results.pdf", dpi=480)
        if not show_fig:
            plt.close()

   
    return outputs 


def evaluate_quantile_forecast(outputs, target_range, hparams, exp_name, file_name, show_fig=False):
   

    pd_metrics, spilit_metrics = {}, {}
    logs = {}
    for j in range(outputs['pred'].shape[-1]):
        metrics=[]
        for i in range(0, len(outputs['pred'])):

            p=outputs['pred'][i,: ,j]
            t=outputs['true'][i,:, j]
            R = target_range[j]
            t_nmpic = t.std()/R
            q_p=outputs['quantile_hats'][i, :, :, j].T
            tau = outputs['tau_hats'][i].T
            sample=outputs['quantile_hats'][i, :, :, j]
            low = outputs['quantile_hats'][i, 0, :, j] if not hparams['conformalize'] else outputs['lower-calib'][i,:, j]
            upp = outputs['quantile_hats'][i, -1, :, j] if not hparams['conformalize'] else outputs['upper-calib'][i,:, j]
            
            df = pd.DataFrame(outputs['index'][i][-len(p):])
            df.columns=['Date']
            index=df.Date.dt.round("D").unique()[-1]

            point_scores = get_pointwise_metrics(p, t, R, 1)
            pic_scores = get_prediction_interval_scores(pred=p, 
                                               true=t,  
                                               target_range=R, 
                                                samples=sample,
                                              lower=low,
                                              upper=upp)
            cirps = get_cwi_score( pic_scores['nmpi'], 
                                  pic_scores['pic'], 
                                  pic_scores['ncrps'], 
                                  t_nmpic)
            ciwe = get_cwi_score(pic_scores['nmpi'], pic_scores['pic'], point_scores['nrmse'], t_nmpic)

            point_scores.update(pic_scores)
            point_scores['ciwe'] = ciwe 
            point_scores['cirps'] = cirps
            
           
            point_scores =pd.DataFrame.from_dict(point_scores, orient='index').T
            point_scores['timestamp']=index
            metrics.append(point_scores)

        metrics = pd.concat(metrics)
        outputs[f"{hparams['targets'][j]}_metrics"]=metrics



        print(f"Results for {hparams['targets'][j]}")
        print(pd.DataFrame(metrics.median()).T[[  'mae', 'nrmse',  'ncrps',  'pic',  'nmpi', 'ciwe', 'cwc', 'corr', 'bias', 'nbias']].round(3))

        bad=np.where(metrics['nrmse']==metrics['mae'].max())[0][0]
        good=np.where(metrics['nrmse']==metrics['min'].min())[0][0]
        
        colors = ['C0', 'C3', 'C5']
        fig, ax = plt.subplots(2,2, figsize=(8,3))
        ax = ax.ravel()
        ax[0], lines, label=plot_prediction_with_pi(ax[0],outputs['true'][good][:, j].flatten(),
                                     outputs['pred'][good][:, j].flatten(),
                                      outputs['quantile_hats'][good][:, :, j])

        met=metrics[['nrmse', 'ciwe', 'ncrps',  'corr']].iloc[good].values
        ax[0].set_title("Good Day NMRSE: {:.2g}, \n CWE: {:.3g}, CRPS: {:.3g}%".format(met[0], met[1], met[2]), fontsize=15)
        leg = ax[0].legend(lines, label, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

        ax[1],lines, label=plot_prediction_with_pi(ax[1],outputs['true'][bad][:, j].flatten(),
                                     outputs['pred'][bad][:, j].flatten(),
                                      outputs['quantile_hats'][bad][:,:, j])
        
        met=metrics[['nrmse', 'ciwe', 'ncrps',  'corr']].iloc[bad].values
        ax[1].set_title("Bad Day NMRSE: {:.2g}, \n CWE: {:.3g}, CRPS: {:.3g}%".format(met[0], met[1], met[2]), fontsize=15)
        leg = ax[1].legend(lines, label, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        
        
        N=len((outputs['pred'][good]))
        ax[2].plot(outputs['index'][good][-N:], outputs['inputs'][good][-N:,1], '.', mec="#ff7f0e", mfc="None", label="Ghi")
        hfmt = mdates.DateFormatter('%d %H')
        ax[2].xaxis.set_major_formatter(hfmt)
        ax[2].set_ylim(0, 1)
        plt.setp( ax[2].xaxis.get_majorticklabels(), rotation=90 );
        
        ax[3].plot(outputs['index'][bad][-N:], outputs['inputs'][bad][-N:,1], '.', mec="#ff7f0e", mfc="None", label="Ghi")
        ax[3].set_ylim(0, 1)
        hfmt = mdates.DateFormatter('%d %H')
        ax[3].xaxis.set_major_formatter(hfmt)
        plt.setp( ax[3].xaxis.get_majorticklabels(), rotation=90 );
        fig.tight_layout(pad=1.08, h_pad=0.5, w_pad=0.5)
        fig.savefig(f"../figures/{exp_name}/{hparams['encoder_type']}/{file_name}_{hparams['targets'][j]}_results.pdf", dpi=480)
        if not show_fig:
            plt.close()

    return outputs 


           
            




