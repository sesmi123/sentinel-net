# server.py
import flwr as fl
import numpy as np

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        # Call the default aggregation logic
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            # Convert Flower Parameters to NumPy arrays
            ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            
            # Save the weights to a file
            print(f"Saving round {server_round} weights...")
            np.savez(f"final_model_weights.npz", *ndarrays)
            
        return aggregated_parameters, aggregated_metrics
    
if __name__ == "__main__":
    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=SaveModelStrategy(),
    )
