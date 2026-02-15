from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from pytorchexample.task import TriageNet, get_parameters

# 1. (Optional) Custom Metrics Aggregator
# This calculates global accuracy from individual hospital results
def weighted_average(metrics):
    # Multiply accuracy by number of examples to get weighted sum
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

# 2. Define the Global Strategy
# This is the "brain" that averages the math from hospitals
strategy = FedAvg(
    fraction_fit=1.0,             # Use 100% of available hospitals for training
    fraction_evaluate=1.0,        # Use 100% for validation
    min_fit_clients=3,            # Wait for at least 3 hospitals before starting
    min_available_clients=3,      # Min threshold to launch the round
    evaluate_metrics_aggregation_fn=weighted_average, # How to merge accuracy reports
)

# 3. Initialize the Global Model
# We start with a "blank slate" model or a pre-trained baseline
global_model = TriageNet()
initial_parameters = fl.common.ndarrays_to_parameters(get_parameters(global_model))

# 4. Configure the Server App
config = ServerConfig(num_rounds=5) # Run 5 cycles of learning

app = ServerApp(
    config=config,
    strategy=strategy,
)