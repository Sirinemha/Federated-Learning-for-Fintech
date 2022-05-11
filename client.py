import warnings
import flwr as fl
import numpy as np
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import utils

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = utils.load_data()

    # Split train set into 10 partitions and randomly use one for training.
    partition_id = np.random.choice(5)
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!de X train')
    print(partition_id)
    (X_train, y_train) = utils.partition(X_train, y_train, 5)[partition_id]
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!de X test') 
    print(partition_id)
    (x_test, y_test) = utils.partition(X_test, y_test, 5)[partition_id]


    # Create LogisticRegression Model
    model = LogisticRegression(
        penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )
 
    # Define Flower client
    class FlowerClient(fl.client.NumPyClient):
        #return the model weight as a list of NumPy ndarrays
        def get_parameters(self):  # type: ignore
            return utils.get_model_parameters(model)
            

        #update the local model weights with the parameters received from the server
      #  def set_parameters(self,parameters):
       #     return utils.set_model_params(model,parameters)

        #Set the local model weights, train the local model and receive the updated local model weights
        def fit(self, parameters, config):  # type: ignore
            
            utils.set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['rnd']}")
            return utils.get_model_parameters(model), len(X_train), {}
            
        #test the local model
        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(x_test))
            accuracy = model.score(x_test, y_test)
            return loss, len(x_test), {"accuracy": accuracy}

    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model)

    # Start Flower client
    fl.client.start_numpy_client(server_address="localhost:"+str(sys.argv[1]), client=FlowerClient())