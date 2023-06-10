
from darts.models import   NHiTSModel, NBEATSModel, RNNModel, TCNModel, TFTModel, DLinearModel
from darts.models import TransformerModel, ExponentialSmoothing
from darts.models import  CatBoostModel,   LinearRegressionModel, RandomForest
from statsforecast.models import AutoARIMA, SeasonalNaive, MSTL
from timeit import default_timer
from pytorch_lightning.callbacks import Callback, EarlyStopping
import numpy as np
import torch
from net.evaluation import evaluate_point_forecast
from statsforecast.models import AutoARIMA, SeasonalNaive, MSTL
from statsforecast import StatsForecast



class BaselineDNNModel(object):
    
    def get_model(self, hparams, path, callbacks):
        if hparams['encoder_type']=='NHiTS':
            model = self.get_nhits_model(hparams, path, callbacks)
            
        if hparams['encoder_type']=='NBEATS':
            model = self.get_nbeats_model(hparams, path, callbacks)
            
        if hparams['encoder_type']=='RNN':
            model = self.get_rnn_model(hparams, path, callbacks)
            
        if hparams['encoder_type']=='TCN':
            model = self.get_tcn_model(hparams, path, callbacks)
            
        if hparams['encoder_type']=='TFT':
            model = self.get_tft_model(hparams, path, callbacks)
            
        if hparams['encoder_type']=='TRANSFORMER':
            model = self.get_transformer_model(hparams, path, callbacks)
            
        if hparams['encoder_type']=='D-LINEAR':
            model = self.get_dlinear_model(hparams, path, callbacks)
            
        if hparams['encoder_type'] in ['MSTL',  'SeasonalNaive', 'AutoARIMA']:
            model = self.get_statistical_baselines(hparams)
            
        if hparams['encoder_type'] in ['CATBOOST', 'RF', 'LREGRESS']:
            model = self.get_conventional_baseline_model(hparams)
        return model
            
    def get_hyparams(self, trial, hparams):
        if hparams['encoder_type']=='NHiTS':
            params = self.get_nhits_search_params(trial, hparams)
            
        if hparams['encoder_type']=='NBEATS':
            params = self.get_nbeats_search_params(trial, hparams)
            
        if hparams['encoder_type']=='RNN':
            params = self.get_rnn_search_params(trial, hparams)
            
        if hparams['encoder_type']=='TCN':
            params = self.get_tcn_search_params(trial, hparams)
            
        if hparams['encoder_type']=='TFT':
            params = self.get_tft_search_params(trial, hparams)
        
        if hparams['encoder_type']=='D-LINEAR':
            params = self.get_dlinear_search_params(trial, hparams)
            
        if hparams['encoder_type']=='TRANSFORMER':
            params=self.get_transformer_search_params(trial, hparams)
        return params
    
    
    def get_conventional_baseline_model(self, hparams):
        encoders = {"cyclic": {"past": ["dayofweek", 'hour', 'day'], 'future': ["dayofweek", 'hour', 'day']}} 
        if hparams['encoder_type']=='CATBOOST': 
            model=CatBoostModel(lags=2,
                                lags_past_covariates=None,
                                lags_future_covariates=[1,2],
                                add_encoders =encoders,
                                output_chunk_length=hparams['horizon'])
                
        elif hparams['encoder_type']=='RF':
            model=RandomForest(lags=2,
                                lags_past_covariates=None,
                                lags_future_covariates=[1,2],
                                add_encoders =encoders,
                                output_chunk_length=hparams['horizon'])
        elif hparams['encoder_type']=='LREGRESS':
            model=LinearRegressionModel(lags=2,
                                lags_past_covariates=None,
                                lags_future_covariates=[1,2],
                                add_encoders =encoders,
                                output_chunk_length=hparams['horizon'])
        return model
    
    

    
    def get_statistical_baselines(self, hparams):
        period=int(24*60/hparams['horizon'])
        if hparams['encoder_type']=='MSTL':
            mstl = MSTL(
                        season_length=[hparams['SAMPLES_PER_DAY'], hparams['SAMPLES_PER_DAY'] * 7], # seasonalities of the time series 
                        trend_forecaster=AutoARIMA() # model used to forecast trend
                    )
            model = StatsForecast(
                        models=[mstl], # model used to fit each time series 
                        freq=f'{period}T', # frequency of the data
                    )
        elif hparams['encoder_type']== 'SeasonalNaive':
            model = StatsForecast(models=[SeasonalNaive(season_length= hparams['SAMPLES_PER_DAY'])], # model used to fit each time series 
                        freq=f'{period}T')
            
        elif hparams['encoder_type']=='AutoARIMA':
            model = StatsForecast(models=[AutoARIMA(season_length= hparams['SAMPLES_PER_DAY'])], # model used to fit each time series 
                        freq=f'{period}T')
        
        return model
            
    def set_callback(self, callbacks):
        # throughout training we'll monitor the validation loss for early stopping
        early_stopper = EarlyStopping("val_loss", min_delta=0.001, patience=3, verbose=True)
        if callbacks is None:
            callbacks = [early_stopper]
        else:
            callbacks = [early_stopper] + callbacks


        # detect if a GPU is available
        if torch.cuda.is_available():
            pl_trainer_kwargs = {
                "accelerator": "gpu",
                 "devices": [0],
                "callbacks": callbacks,
            }
            num_workers = 4
        else:
            pl_trainer_kwargs = {"callbacks": callbacks}
            num_workers = 0
        return callbacks, pl_trainer_kwargs, num_workers
    
    
    def get_transformer_model(self, hparams, path, callback=None):
        encoders = {"cyclic": {"past": ["dayofweek", 'hour', 'day'], 'future': ["dayofweek", 'hour', 'day']}} 
        callback, pl_trainer_kwargs, num_workers=self.set_callback(callback)
    

        model = TransformerModel(
                        input_chunk_length=hparams['window_size'],
                        output_chunk_length=hparams['horizon'],
                        model_name=hparams['encoder_type'],
                        n_epochs=hparams['max_epochs'],
                        batch_size=hparams['batch_size'], 
                        d_model=hparams['latent_size'],
                        nhead=4,
                        norm_type=hparams['norm_type'],
                        num_encoder_layers=hparams['depth'],
                        num_decoder_layers=hparams['depth'],
                        dim_feedforward=hparams['latent_size']*2,
                        dropout=hparams['dropout'],
                        activation=hparams['activation'],
                        add_encoders=encoders if hparams['include_dayofweek'] else None,
                        force_reset=True,
                        work_dir=path, 
                        optimizer_kwargs={"lr": hparams['lr']},
                        pl_trainer_kwargs=pl_trainer_kwargs)
        return model
    
    
   
        
    def get_nhits_model(self, hparams, path, callback=None):
        encoders = {"cyclic": {"past": ["dayofweek", 'hour', 'day'], 'future': ["dayofweek", 'hour', 'day']}} 
        callback, pl_trainer_kwargs, num_workers=self.set_callback(callback)
    

        model = NHiTSModel(
                        input_chunk_length=hparams['window_size'],
                        output_chunk_length=hparams['horizon'],
                        model_name=hparams['encoder_type'],
                        num_stacks=hparams['num_stacks'],
                        num_blocks=hparams['num_blocks'],
                        #pooling_kernel_sizes=hparams['pooling_kernel_sizes'],
                        #n_freq_downsample=hparams['n_freq_downsample'],
                        dropout=hparams['dropout'],
                        num_layers=hparams['depth'],
                        MaxPool1d = hparams['MaxPool1d'],
                        activation=hparams['activation'],
                        layer_widths=hparams['latent_size'],
                        n_epochs=hparams['max_epochs'],
                        batch_size=hparams['batch_size'], 
                        save_checkpoints=True, 
                        add_encoders=encoders if hparams['include_dayofweek'] else None,
                        force_reset=True,
                        work_dir=path, 
                        optimizer_kwargs={"lr": hparams['lr']},
                        pl_trainer_kwargs=pl_trainer_kwargs)
        return model
   
    def get_nbeats_model(self, hparams, path, callback=None):
        encoders = {"cyclic": {"past": ["dayofweek", 'hour', 'day'], 'future': ["dayofweek", 'hour', 'day']}} 
        callback, pl_trainer_kwargs, num_workers=self.set_callback(callback)

       
        model = NBEATSModel(
                        input_chunk_length=hparams['window_size'],
                        output_chunk_length=hparams['horizon'],
                        generic_architecture=hparams['generic_architecture'],
                        model_name=hparams['encoder_type'],
                        num_stacks=hparams['num_stacks'],
                        num_blocks=hparams['num_blocks'],
                        num_layers=hparams['depth'],
                        activation=hparams['activation'],
                        layer_widths=hparams['latent_size'],
                        n_epochs=hparams['max_epochs'],
                        batch_size=hparams['batch_size'], 
                        dropout=hparams['dropout'],
                        save_checkpoints=True, 
                        add_encoders=encoders if hparams['include_dayofweek'] else None,
                        force_reset=True,
                        optimizer_kwargs={"lr": hparams['lr']},
                        work_dir=path, 
                        pl_trainer_kwargs=pl_trainer_kwargs)
        return model
    
    
    
    def get_rnn_model(self, hparams, path, callback=None):
        
        encoders = {"cyclic": {"past": ["dayofweek", 'hour', 'day'], 'future': ["dayofweek", 'hour', 'day']}} 
        callback, pl_trainer_kwargs, num_workers=self.set_callback(callback)
    
        model = RNNModel(
                        input_chunk_length=hparams['window_size'],
                        model=hparams['rnn_type'],
                        model_name=hparams['encoder_type'],
                        n_rnn_layers=hparams['depth'],
                        hidden_dim =hparams['latent_size'],
                        n_epochs=hparams['max_epochs'],
                        batch_size=hparams['batch_size'], 
                        dropout=hparams['dropout'],
                        save_checkpoints=True, 
                        training_length=hparams['window_size']+hparams['horizon'],
                        add_encoders=encoders if hparams['include_dayofweek'] else None,
                        force_reset=True,
                        optimizer_kwargs={"lr": hparams['lr']},
                        work_dir=path, 
                        pl_trainer_kwargs=pl_trainer_kwargs)
        return model
    
    
    def get_tcn_model(self, hparams, path, callback=None):
        
        encoders = {"cyclic": {"past": ["dayofweek", 'hour', 'day'], 'future': ["dayofweek", 'hour', 'day']}} 
        callback, pl_trainer_kwargs, num_workers=self.set_callback(callback)
    
        model = TCNModel(input_chunk_length=hparams['window_size'],
                        output_chunk_length=hparams['horizon'],
                        kernel_size=hparams['kernel_size'],
                        num_filters=hparams['num_filters'],
                        model_name=hparams['encoder_type'],
                        weight_norm=hparams['weight_norm'],
                        dilation_base=hparams['dilation_base'],
                        n_epochs=hparams['max_epochs'],
                        batch_size=hparams['batch_size'], 
                        save_checkpoints=True, 
                        dropout=hparams['dropout'],
                        add_encoders=encoders if hparams['include_dayofweek'] else None,
                        force_reset=True,
                        optimizer_kwargs={"lr": hparams['lr']},
                        work_dir=path, 
                        pl_trainer_kwargs=pl_trainer_kwargs)
        return model
    
    
    def get_tft_model(self, hparams, path, callback=None):
        
        encoders = {"cyclic": {"past": ["dayofweek", 'hour', 'day'], 'future': ["dayofweek", 'hour', 'day']}} 
        callback, pl_trainer_kwargs, num_workers=self.set_callback(callback)
    
        model = TFTModel(
                        input_chunk_length=hparams['window_size'],
                        output_chunk_length=hparams['horizon'],
                        model_name=hparams['encoder_type'],
                        hidden_size=hparams['latent_size'],
                        lstm_layers=hparams['depth'],
                        num_attention_heads=hparams['num_attn_head'],
                        n_epochs=hparams['max_epochs'],
                        batch_size=hparams['batch_size'], 
                        add_relative_index=hparams['add_relative_index'],
                        save_checkpoints=True, 
                        dropout=hparams['dropout'],
                        optimizer_kwargs={"lr": hparams['lr']},
                        add_encoders=encoders if hparams['include_dayofweek'] else None,
                        force_reset=True,
                        work_dir=path, 
                        pl_trainer_kwargs=pl_trainer_kwargs)
        return model
    
    
    def get_dlinear_model(self, hparams, path, callback=None):
        
        encoders = {"cyclic": {"past": ["dayofweek", 'hour', 'day'], 'future': ["dayofweek", 'hour', 'day']}} 
        callback, pl_trainer_kwargs, num_workers=self.set_callback(callback)
    
        model = DLinearModel(
                        input_chunk_length=hparams['window_size'],
                        output_chunk_length=hparams['horizon'],
                        model_name=hparams['encoder_type'],
                        #shared_weights=hparams['shared_weights'],
                        #kernel_size=hparams['kernel_size'],
                        const_init=hparams['const_init'],
                        n_epochs=hparams['max_epochs'],
                        batch_size=hparams['batch_size'], 
                        save_checkpoints=True, 
                        optimizer_kwargs={"lr": hparams['lr']},
                        add_encoders=encoders if hparams['include_dayofweek'] else None,
                        force_reset=True,
                        work_dir=path, 
                        pl_trainer_kwargs=pl_trainer_kwargs)
        return model
    
    def get_dlinear_search_params(self, trial, params):
    
        
    
        const_init = trial.suggest_int("const_init", 2, 4)
        params.update({'const_init': const_init})

    
        lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
        params.update({'lr': lr})

        include_dayofweek = trial.suggest_categorical("dayofweek", [False, True])
        params.update({'include_dayofweek': include_dayofweek})
        return params
    
    
    def get_transformer_search_params(self, trial, params):
        latent_size = {'latent_size': trial.suggest_categorical("latent_size", [16, 32, 64, 128, 256, 512] )}
        params.update(latent_size)

        depth = {'depth':trial.suggest_categorical("depth", [1, 2, 3, 4, 5])}
        params.update(depth)

        nhead = {'nhead':trial.suggest_categorical("nhead", [4,8,16])}
        params.update(nhead)
        

        norm_type = trial.suggest_categorical("norm_type", ["LayerNorm", "RMSNorm", "LayerNormNoBias", None])
        params.update({'norm_type': norm_type})
        
        activation = trial.suggest_categorical("activation", ["GLU", "Bilinear", "ReGLU", "GEGLU", "SwiGLU", "ReLU", "GELU"])
        params.update({'activation': activation})

        
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        params.update({'dropout': dropout})


        include_dayofweek = trial.suggest_categorical("dayofweek", [False, True])
        params.update({'include_dayofweek': include_dayofweek})
        return params
    
    def get_tft_search_params(self, trial, params):
        latent_size = {'latent_size': trial.suggest_categorical("latent_size", [16, 32, 64, 128, 256, 512] )}
        params.update(latent_size)

        depth = {'depth':trial.suggest_categorical("depth", [1, 2, 3, 4, 5])}
        params.update(depth)

        num_attn_head = {'num_attn_head':trial.suggest_categorical("num_attn_head", [4,8,16])}
        params.update(num_attn_head)
      
    
        add_relative_index = trial.suggest_categorical("add_relative_index", [False, True])
        params.update({'add_relative_index': add_relative_index})
        
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        params.update({'dropout': dropout})

        lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
        params.update({'lr': lr})

        include_dayofweek = trial.suggest_categorical("dayofweek", [False, True])
        params.update({'include_dayofweek': include_dayofweek})
        return params
    
    
    def get_tcn_search_params(self, trial, params):
        latent_size = {'latent_size': trial.suggest_categorical("latent_size", [16, 32, 64, 128, 256, 512] )}
        params.update(latent_size)

        depth = {'depth':trial.suggest_categorical("depth", [1, 2, 3, 4, 5])}
        params.update(depth)


        
        kernel_size = trial.suggest_int("kernel_size", 5, 25)
        params.update({'kernel_size': kernel_size})
        
        num_filters = trial.suggest_int("num_filters", 5, 25)
        
        params.update({'num_filters': num_filters})
        weight_norm = trial.suggest_categorical("weight_norm", [False, True])
        params.update({'weight_norm': weight_norm})
        
        dilation_base = trial.suggest_int("dilation_base", 2, 4)
        params.update({'dilation_base': dilation_base})

        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        params.update({'dropout': dropout})

        lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
        params.update({'lr': lr})

        include_dayofweek = trial.suggest_categorical("dayofweek", [False, True])
        params.update({'include_dayofweek': include_dayofweek})
        return params
    
    def get_rnn_search_params(self, trial, params):
        
        rnn_type =trial.suggest_categorical("rnn_type", ['GRU','LSTM'])
        params.update({'rnn_type': rnn_type})
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        params.update({'dropout': dropout})

        lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
        params.update({'lr': lr})

        include_dayofweek = trial.suggest_categorical("dayofweek", [False, True])
        params.update({'include_dayofweek': include_dayofweek})
        
        latent_size = {'latent_size': trial.suggest_categorical("latent_size", [16, 32, 64, 128, 256, 512] )}
        params.update(latent_size)

        depth = {'depth':trial.suggest_categorical("depth", [1, 2, 3, 4, 5])}
        params.update(depth)

       
        return params
    
    def get_nhits_search_params(self, trial, params):
        
        num_stacks=trial.suggest_int('num_stacks', 2, 10)
        params.update({'num_stacks': num_stacks})
        num_blocks=trial.suggest_int('num_blocks', 1, 10)
        params.update({'num_blocks': num_blocks})
        pooling_kernel_sizes=trial.suggest_categorical('pooling_kernel_sizes', [(2, 2, 2), (16, 8, 1)])
        params.update({'pooling_kernel_sizes': pooling_kernel_sizes})
        n_freq_downsample=trial.suggest_categorical('n_freq_downsample', [[168, 24, 1], [24, 12, 1], [1, 1, 1]]) 
        params.update({'n_freq_downsample': n_freq_downsample})

        MaxPool1d =trial.suggest_categorical("MaxPool1d", [False, True])
        params.update({'MaxPool1d': MaxPool1d})

        activation=trial.suggest_categorical("activation", ['ReLU','RReLU', 'PReLU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'Sigmoid'])
        params.update({'activation': activation})

        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        params.update({'dropout': dropout})

        lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
        params.update({'lr': lr})

        include_dayofweek = trial.suggest_categorical("dayofweek", [False, True])
        params.update({'include_dayofweek': include_dayofweek})
        
        latent_size = {'latent_size': trial.suggest_categorical("latent_size", [16, 32, 64, 128, 256, 512] )}
        params.update(latent_size)

        depth = {'depth':trial.suggest_categorical("depth", [1, 2, 3, 4, 5])}
        params.update(depth)

        
        return params
    
    
    
    
    
    
    def get_nbeats_search_params(self, trial, params):
        num_stacks=trial.suggest_int('num_stacks', 2, 10)
        params.update({'num_stacks': num_stacks})
        num_blocks=trial.suggest_int('num_blocks', 1, 10)
        params.update({'num_blocks': num_blocks})
        

        generic_architecture =trial.suggest_categorical("generic_architecture", [False, True])
        params.update({'generic_architecture': generic_architecture})

        activation=trial.suggest_categorical("activation", ['ReLU','RReLU', 'PReLU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'Sigmoid'])
        params.update({'activation': activation})

        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        params.update({'dropout': dropout})

        lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
        params.update({'lr': lr})

        include_dayofweek = trial.suggest_categorical("dayofweek", [False, True])
        params.update({'include_dayofweek': include_dayofweek})
        
        latent_size = {'latent_size': trial.suggest_categorical("latent_size", [16, 32, 64, 128, 256, 512] )}
        params.update(latent_size)

        depth = {'depth':trial.suggest_categorical("depth", [1, 2, 3, 4, 5])}
        params.update(depth)

        
        return params
    
    
    
    
    