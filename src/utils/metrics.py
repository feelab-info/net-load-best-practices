import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error,  r2_score
from pyro.contrib.forecast import  eval_crps
import torch
import pandas as pd
from scipy import stats
import properscoring as ps

def numpy_normalised_quantile_loss(y, y_pred, quantile):
    prediction_underflow = y - y_pred
    weighted_errors = quantile * np.maximum(prediction_underflow, 0.0) + (1.0 - quantile) * np.maximum(
        -prediction_underflow, 0.0
    )
    loss = weighted_errors.mean()
    normaliser = abs(y).mean()
    return 2 * loss / normaliser


def get_forecast_bias(true, pred):
    """
    Tracking Signal quantifies “Bias” in a forecast. No product can be planned from a severely biased forecast. Tracking Signal is the gateway test for evaluating forecast accuracy.
    Once this is calculated, for each period, the numbers are added to calculate the overall tracking signal. A forecast history entirely void of bias will return a value of zero
    """
    return (true - pred).flatten()/np.abs(true - pred).flatten()

def get_normalizedForecastMetric(true, pred):
    """
    this metric will stay between -1 and 1, with 0 indicating the absence of bias. Consistent negative values indicate a tendency to under-forecast whereas constant positive values indicate a tendency to over-forecast
    if the added values are more than 2, we consider the forecast to be biased towards over-forecast. Likewise, if the added values are less than -2, we find the forecast to be biased towards under-forecast.
    """
    return (pred - true).flatten()/(true + pred).flatten()

def get_pointwise_metrics(pred:np.array, true:np.array, target_range:float, scale=1):
    """calculate pointwise metrics
    Args:   pred: predicted values
            true: true values
            target_range: target range          
    Returns:    rmse: root mean square error                


    """
    assert pred.ndim == 1, "pred must be 1-dimensional"
    assert true.ndim == 1, "pred must be 1-dimensional"
    assert pred.shape == true.shape, "pred and true must have the same shape"
    #target_range = true.max() - true.min()
    
    rmse = np.sqrt(mean_squared_error(true, pred))
    nrmse =min( rmse/target_range, 1)
    mae = mean_absolute_error(true, pred)/scale
    nd=(np.abs(true - pred) / np.abs(true)).mean()
    bias=get_forecast_bias(true, pred).sum()/target_range
    nbias=get_normalizedForecastMetric(true, pred).sum()/target_range
    smape = 200. * np.mean(np.abs(true - pred) / (np.abs(true) + np.abs(pred)))
    wmsmape=100.0 / len(true) * np.sum(2 * np.abs(pred - true) / (np.maximum(true, 1) + np.abs(pred)))
    corr = np.corrcoef(true, pred)[0, 1]
    r2=r2_score(pred, true)
    return dict(nrmse=nrmse, mae=mae, smape=smape, r2=r2,
                corr=corr, rmse=rmse, nd=nd, wmsmape=wmsmape,  bias=bias, nbias=nbias)



def quantile_loss(target: np.ndarray, quantile_hats: np.ndarray, tau_hats: float) -> float:
    
    error = quantile_hats - target[None, :]
    return 2 * np.sum(np.abs(error * (( target[None, :] <= quantile_hats) - tau_hats[:, None])))

def get_realibility_scores(true:np.array, q_pred:np.array, tau:np.array):
    
    #https://github.com/tony-psq/QRMGM_KDE/blob/master/QRMGM_KDE/evaluation/Evaluation.py
    assert true.ndim == 1, "pred must be 1-dimensional"
    assert q_pred.ndim == 2, "pred must be 1-dimensional"
    assert tau.ndim == 2, "pred must be 1-dimensional"
    assert tau.shape == q_pred.shape, "pred and true must have the same shape"
    assert len(true) == q_pred.shape[0], "pred and true must have the same shape"
    
    y_cdf = np.zeros((q_pred.shape[0], q_pred.shape[1] + 2))
    y_cdf[:, 1:-1] = q_pred
    y_cdf[:, 0] = 2.0 * q_pred[:, 1] - q_pred[:, 2]
    y_cdf[:, -1] = 2.0 * q_pred[:, -2] - q_pred[:, -3]
    
    
    qs = np.zeros((q_pred.shape[0], q_pred.shape[1] + 2))
    qs[:, 1:-1] = tau
    qs[:, 0] = 0.0
    qs[:, -1] = 1.0
    
    PIT = np.zeros(true.shape)
    for i in range(true.shape[0]):
        PIT[i] = np.interp(np.squeeze(true[i]), np.squeeze(y_cdf[i, :]), np.squeeze(qs[i, :]))
        
    return PIT
        
    



def get_quantile_crps_scores(true:np.array, 
                    q_pred:np.array, 
                    tau:np.array,      
                    target_range:float=None):
    """calculate prediction interval scores

    Args:
        pred (np.array): predicted values
        true (np.array): true values
        q_pred (float): prediction interval
        target_range (float): target range

    Returns:
        [pic(float), nmpi(float), nrmsq(float), ncrsp(float)]: prediction interval scores
    """

    
    assert true.ndim == 1, "pred must be 1-dimensional"
    assert q_pred.ndim == 2, "pred must be 1-dimensional"
    assert tau.ndim == 2, "pred must be 1-dimensional"
    #assert tau.shape == q_pred.shape, "pred and true must have the same shape"
    assert len(true) == q_pred.shape[0], "pred and true must have the same shape"
    
    y_cdf = np.zeros((q_pred.shape[0], q_pred.shape[1] + 2))
    y_cdf[:, 1:-1] = q_pred
    y_cdf[:, 0] = 2.0 * q_pred[:, 1] - q_pred[:, 2]
    y_cdf[:, -1] = 2.0 * q_pred[:, -2] - q_pred[:, -3]
    
    
    qs = np.zeros((q_pred.shape[0], q_pred.shape[1] + 2))
    qs[:, 1:-1] = tau
    qs[:, 0] = 0.0
    qs[:, -1] = 1.0
    
    
    ind = np.zeros(y_cdf.shape)
    ind[y_cdf > true.reshape(-1, 1)] = 1.0
    CRPS = np.trapz((qs - ind) ** 2.0, y_cdf)
    CRPS = np.mean(CRPS)
    if target_range is not None:
        CRPS = CRPS/target_range
    return CRPS

  

def standardized_nrmscore(nrms_score):
    score=np.exp(-nrms_score)
    score=(score-np.exp(-1))/(np.exp(0)-np.exp(-1))
    return score

def standardize_picscore(pic=1, alpha=0.99):
    pic_diff = abs(alpha-pic)
    score=np.exp(-pic_diff)
    score=(score-np.exp(-alpha))/(np.exp(-abs(alpha-1))-np.exp(-alpha))
    return score

def get_cwi_score(nmpi:float, pic:float, nrmse:float,  true_nmpic:float, alpha:float=0.95, eps=1e-8):
    """calculate combined CIPWRMSE score

    Args:
        nmpi (float): nmpi score
        pic (float): p
        nrmse (float): nrmse score
        true_nmpic (float): true nmpic score

    Returns:
        [score(float)]: combined CIPWRMSE score
    """
    #get nmpi difference
   
    nmpic_diff = np.abs(true_nmpic-nmpi)
    pic_diff = np.abs(alpha-pic)
    #get pic score
   
    #pic_score=(np.exp(-nrmse*pic_diff))*pic
    
    
    #get nmpi score
    error  = standardized_nrmscore(nrmse)
    pic_score=error*standardize_picscore(pic, alpha)
    
    nmpic_score = error*np.exp(-nmpic_diff)/(1+np.abs(nmpic_diff))
    #get nmpi score
    num = 2*nmpic_score* (pic_score + eps)
    denom = (pic_score + eps + nmpic_score) 
    score=np.true_divide(num, denom)
    #if denom<=0.0:
    #    denom=1.0
    #score = np.divide(num, denom)
    return score


def get_prediction_interval_scores(pred:np.array, 
                                    true:np.array,  
                                    target_range:float, 
                                    q_pred:np.array=None,
                                    samples:np.array=None,
                                    lower:np.array=None,
                                    upper:np.array=None,
                                    tau:np.array=None,
                                    eta=30, mu=0.9
                                    ):
    """calculate prediction interval scores

    Args:
        pred (np.array): predicted values
        true (np.array): true values
        q_pred (float): prediction interval
        target_range (float): target range

    Returns:
        [pic(float), nmpi(float), nrmsq(float), ncrsp(float)]: prediction interval scores
    """

    assert pred.ndim == 1, "pred must be 1-dimensional"
    assert true.ndim == 1, "pred must be 1-dimensional"
    assert pred.shape == true.shape, "pred and true must have the same shape"

    
    
    if q_pred is not None:
        upper = q_pred[:, -1]
        lower = q_pred[:, 0]

        var = (q_pred/target_range).var(1)

        ncrps = get_quantile_crps_scores(true, q_pred, tau)
        crps=ps.crps_ensemble(true,q_pred).mean()/target_range
    if samples is not None:
        ncrps = eval_crps(torch.tensor(samples), torch.tensor(true))/target_range
    
        lower = samples.mean(0) - samples.std(0) if lower is None else lower
        upper = samples.mean(0) + samples.std(0) if upper is None else upper

        var = (samples/target_range).var(0)
    

    #get prediction interval probability
    samples = samples.reshape(len(true), -1)
    #ncrps=ps.crps_ensemble(true,samples.T)/target_range
    pic = np.intersect1d(np.where(true > lower)[0], np.where(true < upper)[0])
    pic = len(pic)/len(true)

    #get nmpi
    interval =  np.mean(abs(upper - lower))/target_range
    nmpic = interval.mean()

    
    cwc = (1-nmpic)*np.exp(-eta*(pic-mu)**2)
   

    return dict(pic=pic, nmpi=nmpic, ncrps=ncrps, cwc=cwc)

def get_daily_metrics(pred:np.array, true:np.array,  
                      q_pred:np.array, target_range:float, 
                      true_nmpic:float, 
                      samples=None, 
                      tau=None,
                     alpha=0.95, 
                     scale=1):
    """calculate daily metrics

    Args:
        pred (np.array): predicted values
        true (np.array): true values
        target_range (float): target range
        q_pred (np.array): prediction interval

    Returns:
        [daily_metrics(dict)]: daily metrics
    """
    assert pred.ndim == 1, "pred must be 1-dimensional"
    assert true.ndim == 1, "pred must be 1-dimensional"
    assert q_pred.ndim == 2, "pred must be 1-dimensional"
    assert pred.shape == true.shape, "pred and true must have the same shape"

    #get pointwise metrics
    metrics = get_pointwise_metrics(pred, true, target_range, scale)
    
    #get prediction interval scores
    prediction_interval_scores = get_prediction_interval_scores(pred, 
                                                                true,  
                                                                target_range, 
                                                                q_pred, 
                                                                samples,
                                                                tau)
    #get combined CWE
    ciwe = get_cwi_score(prediction_interval_scores['nmpi'], 
                                                    prediction_interval_scores['pic'], 
                                                    metrics['nrmse'], 
                                                    true_nmpic, alpha)

    metrics.update(prediction_interval_scores)
    metrics['ciwe'] = ciwe 
    metrics =pd.DataFrame.from_dict(metrics, orient='index').T

    return metrics



def get_daily_pointwise_metrics(pred:np.array, true:np.array, target_range:float):
    assert pred.ndim == 1, "pred must be 1-dimensional"
    assert true.ndim == 1, "pred must be 1-dimensional"
    assert pred.shape == true.shape, "pred and true must have the same shape"

    #get pointwise metrics
    metrics = get_pointwise_metrics(pred, true, target_range)
    metrics =pd.DataFrame.from_dict(metrics, orient='index').T
    return metrics
    





