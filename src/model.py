import torch
from torch import nn
from torch.nn import functional as F
import pennylane as qml


def VariationalAnsatz(weights):
    """
    Subroutine for the custom variatioanl ansatz.
    """
    if isinstance(weights, torch.Tensor):
        n_layers = weights.size()[0]
        n_wires = weights.size()[1]
    elif isinstance(weights, np.ndarray):
        n_layers = weights.shape[0]
        n_wires = weights.shape[1]

    for layer in range(n_layers):
        # Rotation layer
        for r in range(n_wires):
            for c in range(2):
                if c == 0:
                    qml.RX(weights[layer, r, c], wires=r)
                else:
                    qml.RZ(weights[layer, r, c], wires=r)
    
        # CNOTs
        for _ in range(2):
            for w in range(1, (n_wires + 1) // 2):
                qml.CNOT(wires=[2*w-1, 2*w])
            for w in range(n_wires // 2):
                qml.CNOT(wires=[2*w, 2*w+1])
    
        # Rotation layer
        for r in range(n_wires):
            for c in range(2):
                if c == 0:
                    qml.RX(weights[layer, r, c], wires=r)
                else:
                    qml.RZ(weights[layer, r, c], wires=r)


def get_qnn(n_layers: int, n_wires: int):
    def custom_vqc(inputs, weights):
        if isinstance(weights, torch.Tensor):
            n_wires = weights.size()[1]
        elif isinstance(weights, np.ndarray):
            n_wires = weights.shape[1]
        
        qml.AmplitudeEmbedding(inputs, wires=range(n_wires), normalize=True)
        VariationalAnsatz(weights)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]
    
    qnn_weight_shape = {"weights": (n_layers, n_wires, 4)}
    return custom_vqc, qnn_weight_shape


class TorchHybridModel(nn.Module):
    def __init__(
        self,
        out_shape: int,
        n_qnn_layers: int,
        n_wires: int,
        n_hidden_neurons: int,
        qpu_device: str,
        torch_device: torch.device,
    ):
        super(TorchHybridModel, self).__init__()

        dev = qml.device(qpu_device, wires=8, torch_device=torch_device)
        qnn, qnn_weight_shape = get_qnn(n_qnn_layers, n_wires)
        qnode = qml.QNode(qnn, dev, interface="torch")
        self.q_layer = qml.qnn.TorchLayer(qnode, qnn_weight_shape)
        self.fc_layer = nn.Sequential(
            nn.Linear(n_wires, n_hidden_neurons),
            nn.ReLU(),
            nn.Linear(n_hidden_neurons, out_shape),
        )

    def forward(self, x):
        x = self.q_layer(x)
        x = self.fc_layer(x)
        return x
