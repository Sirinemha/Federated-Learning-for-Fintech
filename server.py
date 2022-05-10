import flwr as fl
import utils
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from typing import Dict
import sys
import numpy as np
import pickle

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
        roc_auc_score = model.score(X_test, y_test)
        return loss, {"roc_auc_score": roc_auc_score}

    return evaluate

###############################################################

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd,
        results,
        failures
    ):
        aggregated_weights = self.aggregate_fit(rnd, results, failures)
        #if aggregated_weights is not None:
            # Save aggregated_weights
            #print(f"Saving round {rnd} aggregated_weights...")
            #np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

strategy1= SaveModelStrategy()

################################################################

# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression()
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        eval_fn=get_eval_fn(model), #for validation
        on_fit_config_fn=fit_round, #configure training
    )
    
    fl.server.start_server(
        server_address = 'localhost:'+str(sys.argv[1]),
        strategy=strategy,
        config={"num_rounds": 100},
    )
    utils.save_data(model)