import torch
import torch.nn as nn
import torch.nn.functional as F
from .mvfem import MVFEM
from .vmgcm import VMGCM
from .arfem import ARFEM


class WeightedBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, pos_weight=10.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, predictions, targets):
        bce_loss = F.binary_cross_entropy(
            predictions,
            targets,
            reduction='none'
        )

        weights = torch.where(
            targets == 1,
            torch.ones_like(targets) * self.pos_weight,
            torch.ones_like(targets)
        )

        weighted_loss = weights * bce_loss

        return weighted_loss.mean()


class VorRelNet(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config if config else {}

        self.mvfem = MVFEM(self.config)
        self.vmgcm = VMGCM(self.config)
        self.arfem = ARFEM(self.config)

        model_config = self.config.get('model', {}) if self.config else {}
        self.pos_weight = model_config.get('pos_weight', 10.0)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, batch, return_features=False):
        sequence = batch.get('sequence', None)
        pdb_path = batch.get('pdb_path', None)
        residue_coords = batch.get('residue_coords', None)
        residue_types = batch.get('residue_types', None)

        h_initial = self.mvfem(sequence=sequence, pdb_path=pdb_path)

        edge_features = None
        if residue_coords is not None:
            raw_edges, processed_edges = self.vmgcm(
                residue_coords=residue_coords,
                residue_types=residue_types,
                sequence=sequence
            )
            edge_features = (raw_edges, processed_edges)

        predictions = self.arfem(
            h_initial=h_initial,
            edge_features_dict=edge_features,
            return_features=return_features
        )

        return predictions

    def compute_loss(self, predictions, targets):
        criterion = WeightedBinaryCrossEntropyLoss(pos_weight=self.pos_weight)
        return criterion(predictions, targets)


def create_vorrel_net(config=None):
    model = VorRelNet(config)
    return model
