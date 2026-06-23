import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GCNIILayer(nn.Module):
    def __init__(self, hidden_dim, alpha=0.2, beta=1.3, lambda_val=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.beta = beta
        self.lambda_val = lambda_val

        self.weight = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, h, adj):
        batch_size, num_nodes, _ = h.shape

        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)

        adj_with_self = adj + torch.eye(num_nodes, device=adj.device).unsqueeze(0)

        degree = adj_with_self.sum(dim=-1, keepdim=True)
        degree = torch.where(degree > 0, degree, torch.ones_like(degree))
        d_inv_sqrt = degree.pow(-0.5)
        d_inv_sqrt = torch.where(torch.isinf(d_inv_sqrt), torch.zeros_like(d_inv_sqrt), d_inv_sqrt)
        adj_norm = d_inv_sqrt * adj_with_self * d_inv_sqrt.transpose(-2, -1)

        beta_l = math.log(self.lambda_val * (1 + 1) + 1)

        h_prop = torch.matmul(adj_norm, h)

        h_conv = (1 - self.alpha) * h_prop + self.alpha * h

        identity_part = (1 - beta_l) * h

        weight_part = self.weight(self.dropout(h_conv))

        h_new = identity_part + weight_part

        return F.gelu(h_new)


class ChannelAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, h):
        z = h.mean(dim=1)
        w = self.channel_attention(z)
        return h * w.unsqueeze(1)


class HeterogeneousEdgeAttention(nn.Module):
    EDGE_TYPES = ['voronoi', 'hydrogen_bond', 'hydrophobic', 'salt_bridge']

    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_weights = nn.Parameter(torch.ones(len(self.EDGE_TYPES)) / len(self.EDGE_TYPES))

    def forward(self, edge_features_dict):
        weights = F.softmax(self.edge_weights, dim=0)

        weighted_features = []
        for i, edge_type in enumerate(self.EDGE_TYPES):
            if edge_type in edge_features_dict:
                weighted_features.append(weights[i] * edge_features_dict[edge_type])

        if weighted_features:
            combined = torch.stack(weighted_features, dim=0).sum(dim=0)
        else:
            combined = torch.zeros_like(list(edge_features_dict.values())[0])

        return combined


class FeedbackMechanism(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.feedback_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h_initial, predictions):
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(-1)

        feedback_signal = self.feedback_mlp(predictions.float())

        return h_initial + feedback_signal.unsqueeze(1)


class Classifier(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h):
        return self.classifier(h)


class ARFEM(nn.Module):
    def __init__(self, config=None):
        super().__init__()

        model_config = config.get('model', {}) if config else {}
        self.hidden_dim = model_config.get('hidden_dim', 256)
        self.num_layers = model_config.get('gcnii_layers', 6)
        self.alpha = model_config.get('alpha', 0.5)
        self.beta = model_config.get('beta', 1.3)
        self.dropout = model_config.get('dropout', 0.1)

        self.gcnii_layers = nn.ModuleList([
            GCNIILayer(
                self.hidden_dim,
                alpha=self.alpha,
                beta=self.beta,
                lambda_val=math.exp(self.beta - 1)
            )
            for _ in range(self.num_layers)
        ])

        self.channel_attention = ChannelAttention(self.hidden_dim)

        self.heterogeneous_edge_attention = HeterogeneousEdgeAttention(self.hidden_dim)

        self.feedback_mechanism = FeedbackMechanism(self.hidden_dim)

        self.classifier = Classifier(self.hidden_dim, self.dropout)

        self.edge_transforms = nn.ModuleDict({
            edge_type: nn.Linear(64, self.hidden_dim)
            for edge_type in ['voronoi', 'hydrogen_bond', 'hydrophobic', 'salt_bridge']
        })

        self.edge_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, h_initial, edge_features_dict, return_features=False,
                return_edge_attn=False, return_stage_features=False):
        h = h_initial

        raw_edges = None
        processed_edges = None
        if edge_features_dict is not None:
            raw_edges, processed_edges = edge_features_dict

        stage_feats = {}
        if return_stage_features:
            stage_feats['input_embedding'] = h.detach()

        for layer in self.gcnii_layers:
            if raw_edges is not None and 'voronoi' in raw_edges:
                adj = raw_edges['voronoi']
            else:
                batch_size, num_nodes = h.shape[0], h.shape[1]
                adj = torch.ones(batch_size, num_nodes, num_nodes, device=h.device)

            h = layer(h, adj)

        if return_stage_features:
            stage_feats['after_gcnii'] = h.detach()

        h = self.channel_attention(h)

        if return_stage_features:
            stage_feats['after_channel_attn'] = h.detach()

        edge_aggregated = None
        if processed_edges and raw_edges:
            hea_weights = F.softmax(
                self.heterogeneous_edge_attention.edge_weights, dim=0
            )

            combined = None
            edge_aggregated = {} if return_edge_attn else None

            for i, edge_type in enumerate(self.heterogeneous_edge_attention.EDGE_TYPES):
                if edge_type not in processed_edges or edge_type not in raw_edges:
                    continue
                if edge_type not in self.edge_transforms:
                    continue

                edge_feat = processed_edges[edge_type]
                adj = raw_edges[edge_type]

                adj_w = adj.unsqueeze(-1)
                weighted_64 = edge_feat * adj_w
                del edge_feat, adj_w

                degree = adj.sum(dim=-1, keepdim=True).clamp(min=1)
                agg_64 = weighted_64.sum(dim=2) / degree
                del weighted_64, degree

                agg = self.edge_transforms[edge_type](agg_64)
                del agg_64

                contribution = hea_weights[i] * agg

                if combined is None:
                    combined = contribution
                else:
                    combined = combined + contribution

                if return_edge_attn:
                    edge_aggregated[edge_type] = agg.detach()
                else:
                    del agg

            if combined is not None:
                gate = torch.sigmoid(self.edge_gate)
                h = h + gate * combined
                del combined

        logits = self.classifier(h).squeeze(-1)
        predictions = torch.sigmoid(logits)

        outputs = [predictions]
        if return_features:
            outputs.append(h)
        if return_edge_attn:
            hea_weights = F.softmax(
                self.heterogeneous_edge_attention.edge_weights, dim=0
            )
            outputs.append(edge_aggregated)
            outputs.append(hea_weights)

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)
