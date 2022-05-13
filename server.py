import flwr as fl
import utils
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from typing import Dict
import sys
import numpy as np
import pickle
import numpy as np
from pytest import importorskip
import os 
import datetime as dt
#from sklearn.model_selection import fit_grid_point

from sympy import arg
#from fastapi import FastAPI
import argparse

#parse command line argument 
parser = argparse.ArgumentParser(description="Loan Default")
parser.add_argument("--num_rounds", type=int , required=True)
#parser.add_argument("--ipadress", type=str, required=True)
#parser.add_argument("--port", type=int, required=True)
#parser.add_argument("--resume",default=False, action="store_true")
args = parser.parse_args()


#define date and time to save weghts in directorles
today = dt.datetime.today()
session = today.strftime("%d-%n-%H-%M-%S")

def fit_round(rnd: int) -> Dict:
    """Send round number to client."""
    return {"rnd": rnd}


def get_eval_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    _, (X_test, y_test) = utils.load_data()

    # The `evaluate` function will be called after every round
    def evaluate(parameters: fl.common.Weights):
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate

###############################################################

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd,
        results,
        failures
    ):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
             # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            
            if not os.path.exists(f"Session.(session)"):
                os.makedirs(f"Session.(session)")
                if rnd < args.num_rounds:
                   np.save(f"Session.(session)/round-(rnd).weights.npy", aggregated_weights)
                elif rnd == args.num_rounds:
                   np.save(f"Session.(session)/global_session_model.npy", aggregated_weights)
            else:
                 if rnd < args.num_rounds:
                     np.save(f"Session.(session)/round-(rnd).weights.npy", aggregated_weights)
                 elif rnd == args.num_rounds:
                   np.save(f"Session.(session)/global_session_model.npy", aggregated_weights)
           
            
        return aggregated_weights
################################################################
import os
from typing import Callable
#define hyper_parameter for evaluation
def evaluate_config(rnd: int):
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps, "verbose": 0 }

#create strategy and run server 
#load last session weights if they exists 
#define batch.size , nb of epochs and verbose for fitting 
def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
    """Return a function which returns training configurations."""

    def fit_config(rnd: int) -> Dict[str, str]:
        config = {
              "batch_size": 32,
              "epochs": 10,
              "verbose": 0,
        }
        return config
    return fit_config
       
sessions = ['no session']
for root , dirs , files in os.walk(".", topdown=False):
    for name in dirs:
        if name.find('Session') != -1:
             sessions.append(name)
if os.path.exists(f'sessions[-1])/global_session_model.npy'):
     initial_parameters = np.load(f"(sessions[-1])/global_session_model.npy", allow_pickle=True)
     initial_weights = initial_parameters[0]
else:
     initial_weights=None
# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression()
    utils.set_initial_params(model)
    strategy =SaveModelStrategy(

        fraction_fit=1.0,
        fraction_eval=1.0,
        min_fit_clients=2,
        min_eval_clients=2,
        min_available_clients=2,
        initial_parameters=initial_weights,
        on_fit_config_fn=get_on_fit_config_fn(),
        on_evaluate_config_fn=evaluate_config,
     )
    
    fl.server.start_server(
        server_address = 'localhost:8080',
        strategy=strategy,
        config={"num_rounds": 20},
    )
    utils.save_data(model)