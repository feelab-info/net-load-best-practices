import os
from symbol import test
import numpy as np
import optuna
from optuna.study import StudyDirection
from optuna.samplers import TPESampler
from optuna.pruners import PatientPruner
from .model import MLPForecast



class MLPForecastParamOptimizer(object):
    def __init__(self,hparams,  experiment,  exp_name, train_df, val_df, test_df=None):

        
        self.hparams=hparams
        self.exp_name=exp_name
        self.experiment = experiment
        self.train_df = train_df
        self.validation_df = val_df
        self.test_df = test_df if test_df is not None else val_df
        
    def get_search_params(self, trial, params):
        # We optimize the number of layers, hidden units and dropout ratio in each layer.

        latent_size = {'latent_size': trial.suggest_categorical("latent_size", [16, 32, 64, 128, 256, 512] )}
        params.update(latent_size)

        depth = {'depth':trial.suggest_categorical("depth", [1, 2, 3, 4, 5])}
        params.update(depth)

        dropout = {'dropout':trial.suggest_float("dropout", 0.1, 0.9)}
        params.update(dropout)

        activation  = {'activation':trial.suggest_categorical("activation", [0, 1, 2, 3, 4])}
        params.update(activation)
    
        emb_size = {'emb_size':trial.suggest_categorical("emb_size",[8,  16, 32, 64])}
        params.update(emb_size)
        

        
        alpha = {'alpha':trial.suggest_float("alpha", 0.01, 0.9)}
        params.update(alpha)
       
        return params
    


    def objective(self, trial=None):
        hparams =  self.get_search_params(trial, self.hparams)
        model=MLPForecast(hparams, exp_name=f"{self.exp_name}", seed=42, trial=trial, rich_progress_bar=True)
        val_cost = model.fit(self.train_df, self.validation_df, self.experiment)
       
        return  val_cost


    def optimise(self,  num_trials=2, patience=5):
        
        def print_callback(study, trial):
            print(f"Current value: {trial.value}, Current params: {trial.params}")
            print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

        study_name=f"{self.exp_name}_{self.hparams['encoder_type']}"
        storage_name = "sqlite:///{}.db".format(study_name)
        
       
     
        storage_name = "sqlite:///{}.db".format(self.exp_name)
        base_pruner = pruner=optuna.pruners.SuccessiveHalvingPruner()
        pruner=optuna.pruners.PatientPruner(base_pruner, patience=patience)
        study = optuna.create_study( direction="minimize", pruner=pruner,  study_name=self.exp_name, 
                                    storage=storage_name,
                                    load_if_exists=True)
        study.optimize(self.objective, n_trials=num_trials)
        print("Number of finished trials: {}".format(len(study.trials)))
        print(f"Best trial: {study.best_trial.number} ")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        return study

