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

    def forward(self, h, h_initial, adj, layer_idx):
        batch_size, num_nodes, _ = h.shape

        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)

        adj_with_self = adj + torch.eye(num_nodes, device=adj.device).unsqueeze(0)

        degree = adj_with_self.sum(dim=-1, keepdim=True)
        degree = torch.where(degree > 0, degree, torch.ones_like(degree))
        d_inv_sqrt = degree.pow(-0.5)
        d_inv_sqrt = torch.where(torch.isinf(d_inv_sqrt), torch.zeros_like(d_inv_sqrt), d_inv_sqrt)
        adj_norm = d_inv_sqrt * adj_with_self * d_inv_sqrt.transpose(-2, -1)

        beta_l = math.log(self.lambda_val / (layer_idx + 1) + 1)

        h_prop = torch.matmul(adj_norm, h)

        h_conv = (1 - self.alpha) * h_prop + self.alpha * h_initial

        identity_part = (1 - beta_l) * h_initial

        weight_part = self.weight(self.dropout(h_conv))

        h_new = identity_part + beta_l * weight_part

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
        
        self.communication_proj = nn.ModuleDict({
            edge_type: nn.Linear(hidden_dim, hidden_dim)
            for edge_type in self.EDGE_TYPES
        })

    def forward(self, h, raw_edges):
        weights = F.softmax(self.edge_weights, dim=0)

        combined = None
        for i, edge_type in enumerate(self.EDGE_TYPES):
            if edge_type not in raw_edges:
                continue

            adj = raw_edges[edge_type]
            batch_size, num_nodes, _ = adj.shape

            adj_with_self = adj + torch.eye(num_nodes, device=adj.device).unsqueeze(0)
            degree = adj_with_self.sum(dim=-1, keepdim=True).clamp(min=1)
            d_inv_sqrt = degree.pow(-0.5)
            adj_norm = d_inv_sqrt * adj_with_self * d_inv_sqrt.transpose(-2, -1)

            h_comm = torch.matmul(adj_norm, h)

            h_comm = self.communication_proj[edge_type](h_comm)

            contribution = weights[i] * h_comm

            if combined is None:
                combined = contribution
            else:
                combined = combined + contribution

        return combined


class SemanticStructuralFeedback(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.feedback_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU()
        )

        self.feedback_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, predictions, h_initial, adj=None):
        if adj is None:
            neighbor_feats = torch.zeros(
                *predictions.shape, 3, device=predictions.device
            )
            context = torch.cat([
                predictions.unsqueeze(-1), neighbor_feats
            ], dim=-1)
        else:
            B, N = predictions.shape[0], predictions.shape[1]
            pred_expanded = predictions.unsqueeze(-1)
            adj_ = adj.float()

            eye = torch.eye(N, device=adj.device).unsqueeze(0)
            adj_no_self = adj_ * (1.0 - eye)

            neighbor_count = adj_no_self.sum(dim=-1, keepdim=True).clamp(min=1)

            neighbor_sum = torch.bmm(
                adj_no_self, pred_expanded
            )
            neighbor_mean = neighbor_sum / neighbor_count

            neg_inf = torch.full((B, N, N), -1e9, device=predictions.device)
            pos_inf = torch.full((B, N, N), 1e9, device=predictions.device)
            pred_rep = pred_expanded.squeeze(-1).unsqueeze(1).expand(-1, N, -1)

            neighbor_max = torch.where(
                adj_no_self.bool(), pred_rep, neg_inf
            ).max(dim=-1).values.unsqueeze(-1)
            neighbor_min = torch.where(
                adj_no_self.bool(), pred_rep, pos_inf
            ).min(dim=-1).values.unsqueeze(-1)

            has_neighbors = (adj_no_self.sum(dim=-1, keepdim=True) > 0).float()
            neighbor_mean = neighbor_mean * has_neighbors + pred_expanded * (1 - has_neighbors)
            neighbor_max = neighbor_max * has_neighbors + pred_expanded * (1 - has_neighbors)
            neighbor_min = neighbor_min * has_neighbors + pred_expanded * (1 - has_neighbors)

            context = torch.cat([
                pred_expanded, neighbor_mean, neighbor_max, neighbor_min
            ], dim=-1)

        feedback_signal = self.feedback_mlp(context)
        gate = torch.sigmoid(self.feedback_gate)
        h_updated = h_initial + gate * feedback_signal

        return h_updated


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
        self.num_feedback_iterations = model_config.get('num_feedback_iterations', 2)

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

        self.semantic_structural_feedback = SemanticStructuralFeedback(self.hidden_dim)

        self.classifier = Classifier(self.hidden_dim, self.dropout)

        self.edge_gate = nn.Parameter(torch.tensor(-2.0))

    def _forward_pass(self, h, h_initial, raw_edges, stage_feats=None, iteration=0):
        for layer_idx, layer in enumerate(self.gcnii_layers):
            if raw_edges is not None and 'voronoi' in raw_edges:
                adj = raw_edges['voronoi']
            else:
                batch_size, num_nodes = h.shape[0], h.shape[1]
                adj = torch.ones(batch_size, num_nodes, num_nodes, device=h.device)

            h = layer(h, h_initial, adj, layer_idx)

            edge_aggregated = None
            if raw_edges is not None:
                edge_aggregated = self.heterogeneous_edge_attention(h, raw_edges)

                if edge_aggregated is not None:
                    gate = torch.sigmoid(self.edge_gate)
                    h = h + gate * edge_aggregated

            h = self.channel_attention(h)

            if stage_feats is not None:
                stage_feats[f'after_iter_{iteration}_layer_{layer_idx}'] = h.detach()

        return h

    def forward(self, h_initial, edge_features_dict, return_features=False,
                return_edge_attn=False, return_stage_features=False):
        h = h_initial

        raw_edges = None
        if edge_features_dict is not None:
            if isinstance(edge_features_dict, tuple):
                raw_edges, _ = edge_features_dict
            else:
                raw_edges = edge_features_dict

        feedback_adj = None
        if raw_edges is not None and 'voronoi' in raw_edges:
            feedback_adj = (raw_edges['voronoi'] > 0).float()

        stage_feats = {}
        if return_stage_features:
            stage_feats['input_embedding'] = h.detach()

        h = self._forward_pass(h_initial, h_initial, raw_edges, stage_feats, iteration=0)
        if return_stage_features:
            stage_feats['after_iter_0_gcnii'] = h.detach()

        logits = self.classifier(h).squeeze(-1)
        predictions = torch.sigmoid(logits)

        h_updated = self.semantic_structural_feedback(predictions, h_initial, adj=feedback_adj)
        h = self._forward_pass(h_updated, h_updated, raw_edges, stage_feats, iteration=1)
        if return_stage_features:
            stage_feats['after_iter_1_gcnii'] = h.detach()

        logits = self.classifier(h).squeeze(-1)
        predictions = torch.sigmoid(logits)

        h_updated = self.semantic_structural_feedback(predictions, h_initial, adj=feedback_adj)
        h = self._forward_pass(h_updated, h_updated, raw_edges, stage_feats, iteration=2)
        if return_stage_features:
            stage_feats['after_iter_2_gcnii'] = h.detach()

        logits = self.classifier(h).squeeze(-1)
        predictions = torch.sigmoid(logits)

        outputs = [predictions, logits]
        if return_features:
            outputs.append(h)
        if return_edge_attn:
            hea_weights = F.softmax(self.heterogeneous_edge_attention.edge_weights, dim=0)
            outputs.append(hea_weights)

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)
