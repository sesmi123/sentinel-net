# sentinel-net

https://account.flower.ai/realms/flower/device/status

flwr --version
flwr config list
flwr login supergrid
    Username: sesmi
    Password: Northeastern123

pip install -U "flwr[simulation]"
flwr new @flwrlabs/quickstart-pytorch
pip install -e .
pip install scikit-learn numpy pandas torch
flwr run . --num-supernodes 3
flwr run . --run-config "num-supernodes=3"

python server_app.py
python client_app.py
python client_app.py
streamlit run app.py