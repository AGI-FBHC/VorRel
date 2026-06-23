import os
import sys
import csv
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.vmgcm import VMGCM
from models.arfem import ARFEM
from models.vorrel_net import WeightedBinaryCrossEntropyLoss


class FocalLoss(nn.Module):
    """
    Focal Loss 用于应对样本不平衡问题
    FL = -α_t (1 - p_t)^γ log(p_t)
    兼容原始 WeightedBinaryCrossEntropyLoss 接口
    """
    def __init__(self, pos_weight=10.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma

    def forward(self, predictions, targets):
        """
        predictions: [batch_size, num_residues], sigmoid 后的概率
        targets: [batch_size, num_residues], 0 或 1
        """
        # 计算 BCE Loss
        bce_loss = F.binary_cross_entropy(
            predictions,
            targets,
            reduction='none'
        )
        
        # 计算 pt
        pt = torch.where(targets == 1, predictions, 1 - predictions)
        
        # Focal term
        focal_weight = (1 - pt) ** self.gamma
        
        # 应用 pos_weight
        weights = torch.where(
            targets == 1,
            torch.ones_like(targets) * self.pos_weight,
            torch.ones_like(targets)
        )
        
        # 组合 Loss
        loss = weights * focal_weight * bce_loss
        
        return loss.mean()

FEATURE_DIR = '/home/shihj/shj/VorRel/feature'
INFO_PATH = '/home/shihj/shj/GraphPRNet/data/scPDB/info.txt'


class PrecomputedDataset(Dataset):
    def __init__(self, keys, feature_dir, info_map):
        self.keys = keys
        self.feature_dir = feature_dir
        self.info_map = info_map

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        info = self.info_map[key]

        pssm = np.load(os.path.join(self.feature_dir, 'PSSM', f'{key}.npy'))
        esm2 = np.load(os.path.join(self.feature_dir, 'ESM-2', f'{key}.npy'))
        dssp = np.load(os.path.join(self.feature_dir, 'DSSP', f'{key}.npy'))
        atomic = np.load(os.path.join(self.feature_dir, 'Atomic', f'{key}.npy'))
        graph = np.load(os.path.join(self.feature_dir, 'Graph', f'{key}.npy'), allow_pickle=True).item()

        n = pssm.shape[0]
        esm2 = esm2[:n]
        if esm2.shape[0] < n:
            return None

        features = np.concatenate([esm2, pssm, dssp, atomic], axis=1).astype(np.float32)

        voronoi = graph['voronoi_edges'].astype(np.float32)
        hb = graph['hb_edges'].astype(np.float32)
        hp = graph['hp_edges'].astype(np.float32)
        sb = graph['sb_edges'].astype(np.float32)

        binding_str = info['binding']
        n = features.shape[0]
        labels = np.zeros(n, dtype=np.float32)
        for i, ch in enumerate(binding_str[:n]):
            if ch == '1':
                labels[i] = 1.0

        residue_types = list(info['sequence'][:n])

        return {
            'key': key,
            'features': torch.tensor(features),
            'voronoi': torch.tensor(voronoi),
            'hb': torch.tensor(hb),
            'hp': torch.tensor(hp),
            'sb': torch.tensor(sb),
            'labels': torch.tensor(labels),
            'residue_types': residue_types,
            'num_residues': n,
        }


def collate_fn(batch):
    max_n = max(item['num_residues'] for item in batch)
    feat_dim = batch[0]['features'].shape[1]

    padded_features = []
    padded_voronoi = []
    padded_hb = []
    padded_hp = []
    padded_sb = []
    padded_labels = []
    masks = []
    residue_types_list = []

    for item in batch:
        n = item['num_residues']
        pad_n = max_n - n

        feat = item['features']
        if pad_n > 0:
            feat = torch.cat([feat, torch.zeros(pad_n, feat_dim)], dim=0)
        padded_features.append(feat)

        for name, padded_list in [('voronoi', padded_voronoi), ('hb', padded_hb),
                                   ('hp', padded_hp), ('sb', padded_sb)]:
            adj = item[name]
            if pad_n > 0:
                adj = torch.cat([adj, torch.zeros(n, pad_n)], dim=1)
                adj = torch.cat([adj, torch.zeros(pad_n, max_n)], dim=0)
            padded_list.append(adj)

        label = item['labels']
        if pad_n > 0:
            label = torch.cat([label, torch.zeros(pad_n)], dim=0)
        padded_labels.append(label)

        mask = torch.zeros(max_n, dtype=torch.bool)
        mask[:n] = True
        masks.append(mask)

        residue_types_list.append(item['residue_types'])

    return {
        'features': torch.stack(padded_features),
        'voronoi': torch.stack(padded_voronoi),
        'hb': torch.stack(padded_hb),
        'hp': torch.stack(padded_hp),
        'sb': torch.stack(padded_sb),
        'labels': torch.stack(padded_labels),
        'masks': torch.stack(masks),
        'residue_types': residue_types_list,
    }


class VorRelTrainer(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.get('model', {}).get('hidden_dim', 256)
        use_focal = config.get('model', {}).get('use_focal_loss', False)
        gamma = config.get('model', {}).get('focal_gamma', 2.0)

        self.embedding_layer = nn.Sequential(
            nn.LayerNorm(1320),
            nn.Linear(1320, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.vmgcm = VMGCM(config)

        self.arfem = ARFEM(config)

        pos_weight = config.get('model', {}).get('pos_weight', 10.0)
        if use_focal:
            self.loss_fn = FocalLoss(pos_weight=pos_weight, gamma=gamma)
        else:
            self.loss_fn = WeightedBinaryCrossEntropyLoss(pos_weight=pos_weight)

    def forward(self, batch, return_edge_attn=False):
        features = batch['features']
        voronoi = batch['voronoi']
        hb = batch['hb']
        hp = batch['hp']
        sb = batch['sb']

        h = self.embedding_layer(features)

        device = h.device
        raw_edges = {
            'voronoi': voronoi.to(device),
            'hydrogen_bond': hb.to(device),
            'hydrophobic': hp.to(device),
            'salt_bridge': sb.to(device),
        }

        processed_edges = {}
        for edge_type, adj in raw_edges.items():
            adj_unsqueezed = adj.unsqueeze(-1)
            processed_edges[edge_type] = self.vmgcm.edge_type_embeddings[edge_type](adj_unsqueezed)

        return self.arfem(
            h_initial=h,
            edge_features_dict=(raw_edges, processed_edges),
            return_features=False,
            return_edge_attn=return_edge_attn,
        )

    def compute_loss(self, predictions, targets):
        return self.loss_fn(predictions, targets)


def load_info(info_path):
    info_map = {}
    with open(info_path) as f:
        for line in f:
            if line.startswith('pdb_id'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                pdb_id = parts[0].strip()
                struct_num = parts[1].strip()
                chain_id = parts[2].strip()
                sequence = parts[3].strip()
                binding = parts[4].strip()
                key = f'{pdb_id}_{chain_id}_{struct_num}'
                info_map[key] = {'sequence': sequence, 'binding': binding}
    return info_map


def find_valid_keys(feature_dir, info_map):
    dirs = ['PSSM', 'ESM-2', 'DSSP', 'Atomic', 'Graph']
    file_sets = {}
    for d in dirs:
        path = os.path.join(feature_dir, d)
        files = set(f[:-4] for f in os.listdir(path) if f.endswith('.npy'))
        file_sets[d] = files

    common = file_sets['PSSM']
    for d in dirs:
        common = common & file_sets[d]

    valid = []
    for key in sorted(common):
        if key not in info_map:
            continue

        pssm = np.load(os.path.join(feature_dir, 'PSSM', f'{key}.npy'))
        esm2 = np.load(os.path.join(feature_dir, 'ESM-2', f'{key}.npy'))
        graph_arr = np.load(os.path.join(feature_dir, 'Graph', f'{key}.npy'), allow_pickle=True)
        graph = graph_arr.item() if graph_arr.ndim == 0 else graph_arr
        n_feat = pssm.shape[0]
        n_graph = graph['voronoi_edges'].shape[0]
        n_esm2 = esm2.shape[0]
        seq_len = len(info_map[key]['sequence'])

        if n_feat == n_graph == seq_len and n_esm2 >= seq_len + 2:
            valid.append(key)

    return valid


def get_gpu_memory(device):
    if not torch.cuda.is_available() or device.type != 'cuda':
        return {}
    return {
        'alloc_mb': round(torch.cuda.memory_allocated(device) / 1024**2, 1),
        'reserved_mb': round(torch.cuda.memory_reserved(device) / 1024**2, 1),
        'peak_alloc_mb': round(torch.cuda.max_memory_allocated(device) / 1024**2, 1),
        'peak_reserved_mb': round(torch.cuda.max_memory_reserved(device) / 1024**2, 1),
    }


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    total_fwd_time = 0.0
    total_bwd_time = 0.0
    total_data_time = 0.0
    gpu_mem_samples = []

    data_start = time.time()
    pbar = tqdm(dataloader, desc="训练")
    for batch in pbar:
        total_data_time += time.time() - data_start

        features = batch['features'].to(device)
        voronoi = batch['voronoi'].to(device)
        hb = batch['hb'].to(device)
        hp = batch['hp'].to(device)
        sb = batch['sb'].to(device)
        labels = batch['labels'].to(device)
        masks = batch['masks'].to(device)

        optimizer.zero_grad()

        torch.cuda.synchronize(device) if device.type == 'cuda' else None
        fwd_start = time.time()
        predictions = model({
            'features': features,
            'voronoi': voronoi,
            'hb': hb,
            'hp': hp,
            'sb': sb,
        })
        pred_masked = predictions[masks]
        label_masked = labels[masks]
        loss = model.compute_loss(pred_masked.unsqueeze(0), label_masked.unsqueeze(0))
        torch.cuda.synchronize(device) if device.type == 'cuda' else None
        total_fwd_time += time.time() - fwd_start

        torch.cuda.synchronize(device) if device.type == 'cuda' else None
        bwd_start = time.time()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        torch.cuda.synchronize(device) if device.type == 'cuda' else None
        total_bwd_time += time.time() - bwd_start

        total_loss += loss.item()
        num_batches += 1

        mem = get_gpu_memory(device)
        if mem:
            gpu_mem_samples.append(mem['alloc_mb'])

        pbar.set_postfix(loss=f'{loss.item():.4f}',
                         fwd=f'{total_fwd_time/num_batches*1000:.0f}ms',
                         mem=f"{mem.get('alloc_mb', 0):.0f}MB")

        data_start = time.time()

    result = {
        'loss': total_loss / num_batches,
        'fwd_time': total_fwd_time,
        'bwd_time': total_bwd_time,
        'data_time': total_data_time,
        'avg_fwd_ms': total_fwd_time / num_batches * 1000 if num_batches else 0,
        'avg_bwd_ms': total_bwd_time / num_batches * 1000 if num_batches else 0,
        'avg_data_ms': total_data_time / num_batches * 1000 if num_batches else 0,
        'peak_gpu_alloc_mb': max(gpu_mem_samples) if gpu_mem_samples else 0,
        'avg_gpu_alloc_mb': sum(gpu_mem_samples) / len(gpu_mem_samples) if gpu_mem_samples else 0,
        'num_batches': num_batches,
    }
    return result


def evaluate(model, dataloader, device, threshold=0.5, return_all_thresholds=False):
    model.eval()
    all_predictions = []
    all_targets = []

    infer_times = []
    gpu_mem_samples = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估"):
            features = batch['features'].to(device)
            voronoi = batch['voronoi'].to(device)
            hb = batch['hb'].to(device)
            hp = batch['hp'].to(device)
            sb = batch['sb'].to(device)
            labels = batch['labels']
            masks = batch['masks']

            torch.cuda.synchronize(device) if device.type == 'cuda' else None
            t0 = time.time()
            predictions = model({
                'features': features,
                'voronoi': voronoi,
                'hb': hb,
                'hp': hp,
                'sb': sb,
            })
            torch.cuda.synchronize(device) if device.type == 'cuda' else None
            infer_times.append(time.time() - t0)

            mem = get_gpu_memory(device)
            if mem:
                gpu_mem_samples.append(mem['alloc_mb'])

            pred_cpu = predictions.cpu()
            for i in range(len(masks)):
                m = masks[i]
                all_predictions.extend(pred_cpu[i][m].numpy())
                all_targets.extend(labels[i][m].numpy())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    results = {}
    try:
        results['auroc'] = roc_auc_score(all_targets, all_predictions)
    except ValueError:
        results['auroc'] = 0.0

    try:
        results['auprc'] = average_precision_score(all_targets, all_predictions)
    except ValueError:
        results['auprc'] = 0.0

    if return_all_thresholds:
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        results['thresholds'] = {}
        for t in thresholds:
            pred_binary = (all_predictions > t).astype(int)
            results['thresholds'][t] = {
                'accuracy': accuracy_score(all_targets, pred_binary),
                'precision': precision_score(all_targets, pred_binary, zero_division=0),
                'recall': recall_score(all_targets, pred_binary, zero_division=0),
                'f1': f1_score(all_targets, pred_binary, zero_division=0),
                'mcc': matthews_corrcoef(all_targets, pred_binary),
            }
    
    pred_binary = (all_predictions > threshold).astype(int)
    results['threshold'] = threshold
    results['accuracy'] = accuracy_score(all_targets, pred_binary)
    results['precision'] = precision_score(all_targets, pred_binary, zero_division=0)
    results['recall'] = recall_score(all_targets, pred_binary, zero_division=0)
    results['f1'] = f1_score(all_targets, pred_binary, zero_division=0)
    results['mcc'] = matthews_corrcoef(all_targets, pred_binary)

    n_pos = int(all_targets.sum())
    n_neg = len(all_targets) - n_pos
    results['n_samples'] = len(all_targets)
    results['n_positive'] = n_pos
    results['n_negative'] = n_neg

    results['infer_time'] = sum(infer_times)
    results['avg_infer_ms'] = sum(infer_times) / len(infer_times) * 1000 if infer_times else 0
    results['peak_gpu_alloc_mb'] = max(gpu_mem_samples) if gpu_mem_samples else 0

    return results


def main():
    parser = argparse.ArgumentParser(description="VorRel 预计算特征训练")
    parser.add_argument("--feature_dir", type=str, default=FEATURE_DIR)
    parser.add_argument("--info_path", type=str, default=INFO_PATH)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--gcnii_layers", type=int, default=6)
    parser.add_argument("--pos_weight", type=float, default=10.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数（调试用）")
    parser.add_argument("--gpu", type=int, default=3, help="GPU 设备编号")
    parser.add_argument("--threshold", type=float, default=0.5, help="分类阈值（提升Precision可以提高这个值）")
    parser.add_argument("--use_focal_loss", action="store_true", help="使用 Focal Loss 代替加权 BCE Loss")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal Loss 的 gamma 参数（默认 2.0）")

    args = parser.parse_args()

    config = {
        'model': {
            'hidden_dim': args.hidden_dim,
            'gcnii_layers': args.gcnii_layers,
            'alpha': 0.5,
            'beta': 1.3,
            'dropout': 0.1,
            'pos_weight': args.pos_weight,
            'use_focal_loss': args.use_focal_loss,
            'focal_gamma': args.focal_gamma,
        },
        'graph': {
            'hydrogen_bond_threshold': 6.0,
            'hydrophobic_threshold': 8.0,
            'salt_bridge_threshold': 12.0,
        }
    }

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    print("加载 info.txt...")
    info_map = load_info(args.info_path)
    print(f"info.txt 条目数: {len(info_map)}")

    print("查找有效样本...")
    valid_keys = find_valid_keys(args.feature_dir, info_map)
    print(f"有效样本数: {len(valid_keys)}")

    if args.max_samples and args.max_samples < len(valid_keys):
        np.random.seed(args.seed)
        valid_keys = list(np.random.choice(valid_keys, args.max_samples, replace=False))
        print(f"限制样本数: {len(valid_keys)}")

    train_keys, val_keys = train_test_split(
        valid_keys, test_size=args.val_ratio, random_state=args.seed
    )
    print(f"训练集: {len(train_keys)}, 验证集: {len(val_keys)}")

    train_dataset = PrecomputedDataset(train_keys, args.feature_dir, info_map)
    val_dataset = PrecomputedDataset(val_keys, args.feature_dir, info_map)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )

    print("创建模型...")
    model = VorRelTrainer(config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    buffer_size_mb = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024**2
    print(f"总参数: {total_params:,}, 可训练: {trainable_params:,}")
    print(f"模型大小: {model_size_mb:.2f} MB (参数) + {buffer_size_mb:.2f} MB (缓冲)")

    gpu_mem_before = get_gpu_memory(device)
    print(f"模型加载后 GPU 显存: {gpu_mem_before}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_time = time.strftime('%Y%m%d_%H%M%S')
    log_dir = output_path / "logs" / run_time
    log_dir.mkdir(parents=True, exist_ok=True)

    csv_path = log_dir / "training_log.csv"
    csv_fields = ['epoch', 'train_loss', 'lr', 'epoch_time',
                  'train_fwd_ms', 'train_bwd_ms', 'train_data_ms',
                  'train_peak_gpu_mb', 'train_avg_gpu_mb',
                  'val_auroc', 'val_auprc', 'val_accuracy', 'val_precision',
                  'val_recall', 'val_f1', 'val_mcc',
                  'val_n_samples', 'val_n_positive', 'val_n_negative',
                  'val_infer_ms', 'val_peak_gpu_mb',
                  'best_auroc']
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()
    csv_file.flush()

    # tb_dir = log_dir / "tb"
    # tb_writer = SummaryWriter(log_dir=str(tb_dir))
    # print(f"TensorBoard 日志: {tb_dir}")
    # print(f"启动 TensorBoard: tensorboard --logdir {tb_dir} --port 6006")

    realtime_log_path = log_dir / "realtime.log"
    realtime_log = open(realtime_log_path, 'w')
    realtime_log.write(f"{'Epoch':>6} | {'Loss':>8} | {'LR':>10} | {'Time':>6} | "
                       f"{'AUROC':>6} {'AUPRC':>6} {'Acc':>6} {'Prec':>6} "
                       f"{'Recall':>6} {'F1':>6} {'MCC':>6} | {'Best':>6}\n"
                       f"        | {'Fwd':>7} {'Bwd':>7} {'Data':>7} | "
                       f"{'GPU峰值':>10} {'GPU均值':>10} | {'Val推理':>8}\n")
    realtime_log.write("-" * 110 + "\n")
    realtime_log.flush()

    train_config = {
        'args': vars(args),
        'model_config': config,
        'model_complexity': {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': total_params - trainable_params,
            'model_size_mb': round(model_size_mb, 2),
            'buffer_size_mb': round(buffer_size_mb, 2),
            'input_feature_dim': 1320,
            'hidden_dim': args.hidden_dim,
            'gcnii_layers': args.gcnii_layers,
            'batch_size': args.batch_size,
            'gpu_mem_before_load': gpu_mem_before,
        },
        'train_samples': len(train_keys),
        'val_samples': len(val_keys),
        'device': str(device),
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'log_dir': str(log_dir),
    }
    config_path = log_dir / "train_config.json"
    with open(config_path, 'w') as f:
        json.dump(train_config, f, indent=2, ensure_ascii=False)
    print(f"训练配置保存至: {config_path}")

    best_auroc = 0.0
    best_epoch = 0
    training_start = time.time()

    for epoch in range(args.epochs):
        print(f"\n{'='*60}", flush=True)
        print(f"Epoch {epoch + 1}/{args.epochs}", flush=True)
        print(f"{'='*60}", flush=True)

        epoch_start = time.time()
        train_stats = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        epoch_time = time.time() - epoch_start
        train_loss = train_stats['loss']

        current_lr = scheduler.get_last_lr()[0]
        print(f"训练 Loss: {train_loss:.4f}, LR: {current_lr:.6f}, 耗时: {epoch_time:.1f}s", flush=True)
        print(f"  前向: {train_stats['avg_fwd_ms']:.1f}ms/batch, "
              f"反向: {train_stats['avg_bwd_ms']:.1f}ms/batch, "
              f"数据加载: {train_stats['avg_data_ms']:.1f}ms/batch", flush=True)
        print(f"  GPU 显存: 峰值 {train_stats['peak_gpu_alloc_mb']:.0f}MB, "
              f"平均 {train_stats['avg_gpu_alloc_mb']:.0f}MB", flush=True)

        val_metrics = evaluate(model, val_loader, device, threshold=args.threshold)
        print(f"验证 AUROC: {val_metrics['auroc']:.4f}, "
              f"AUPRC: {val_metrics['auprc']:.4f}", flush=True)
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}, "
              f"Precision: {val_metrics['precision']:.4f}, "
              f"Recall: {val_metrics['recall']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, "
              f"MCC: {val_metrics['mcc']:.4f}", flush=True)
        print(f"  推理: {val_metrics['avg_infer_ms']:.1f}ms/batch, "
              f"GPU峰值: {val_metrics['peak_gpu_alloc_mb']:.0f}MB", flush=True)

        is_best = val_metrics['auroc'] > best_auroc
        if is_best:
            best_auroc = val_metrics['auroc']
            best_epoch = epoch + 1

        log_row = {
            'epoch': epoch + 1,
            'train_loss': round(train_loss, 6),
            'lr': round(current_lr, 8),
            'epoch_time': round(epoch_time, 2),
            'train_fwd_ms': round(train_stats['avg_fwd_ms'], 2),
            'train_bwd_ms': round(train_stats['avg_bwd_ms'], 2),
            'train_data_ms': round(train_stats['avg_data_ms'], 2),
            'train_peak_gpu_mb': round(train_stats['peak_gpu_alloc_mb'], 1),
            'train_avg_gpu_mb': round(train_stats['avg_gpu_alloc_mb'], 1),
            'val_auroc': round(val_metrics['auroc'], 6),
            'val_auprc': round(val_metrics['auprc'], 6),
            'val_accuracy': round(val_metrics['accuracy'], 6),
            'val_precision': round(val_metrics['precision'], 6),
            'val_recall': round(val_metrics['recall'], 6),
            'val_f1': round(val_metrics['f1'], 6),
            'val_mcc': round(val_metrics['mcc'], 6),
            'val_n_samples': val_metrics['n_samples'],
            'val_n_positive': val_metrics['n_positive'],
            'val_n_negative': val_metrics['n_negative'],
            'val_infer_ms': round(val_metrics['avg_infer_ms'], 2),
            'val_peak_gpu_mb': round(val_metrics['peak_gpu_alloc_mb'], 1),
            'best_auroc': round(best_auroc, 6),
        }
        csv_writer.writerow(log_row)
        csv_file.flush()

        global_step = epoch + 1
        # tb_writer.add_scalar('Loss/train', train_loss, global_step)
        # tb_writer.add_scalar('LR', current_lr, global_step)
        # tb_writer.add_scalar('Time/epoch', epoch_time, global_step)
        # tb_writer.add_scalar('Time/fwd_ms', train_stats['avg_fwd_ms'], global_step)
        # tb_writer.add_scalar('Time/bwd_ms', train_stats['avg_bwd_ms'], global_step)
        # tb_writer.add_scalar('Time/data_ms', train_stats['avg_data_ms'], global_step)
        # tb_writer.add_scalar('Time/val_infer_ms', val_metrics['avg_infer_ms'], global_step)
        # tb_writer.add_scalar('GPU/train_peak_mb', train_stats['peak_gpu_alloc_mb'], global_step)
        # tb_writer.add_scalar('GPU/train_avg_mb', train_stats['avg_gpu_alloc_mb'], global_step)
        # tb_writer.add_scalar('GPU/val_peak_mb', val_metrics['peak_gpu_alloc_mb'], global_step)
        # tb_writer.add_scalar('Val/AUROC', val_metrics['auroc'], global_step)
        # tb_writer.add_scalar('Val/AUPRC', val_metrics['auprc'], global_step)
        # tb_writer.add_scalar('Val/Accuracy', val_metrics['accuracy'], global_step)
        # tb_writer.add_scalar('Val/Precision', val_metrics['precision'], global_step)
        # tb_writer.add_scalar('Val/Recall', val_metrics['recall'], global_step)
        # tb_writer.add_scalar('Val/F1', val_metrics['f1'], global_step)
        # tb_writer.add_scalar('Val/MCC', val_metrics['mcc'], global_step)
        # tb_writer.add_scalar('Best/AUROC', best_auroc, global_step)
        # tb_writer.flush()

        realtime_log.write(
            f"{epoch + 1:>6} | {train_loss:>8.4f} | {current_lr:>10.6f} | {epoch_time:>5.1f}s | "
            f"{val_metrics['auroc']:>6.4f} {val_metrics['auprc']:>6.4f} "
            f"{val_metrics['accuracy']:>6.4f} {val_metrics['precision']:>6.4f} "
            f"{val_metrics['recall']:>6.4f} {val_metrics['f1']:>6.4f} "
            f"{val_metrics['mcc']:>6.4f} | {best_auroc:>6.4f}\n"
            f"        | fwd:{train_stats['avg_fwd_ms']:>5.0f}ms bwd:{train_stats['avg_bwd_ms']:>5.0f}ms "
            f"data:{train_stats['avg_data_ms']:>5.0f}ms | "
            f"GPU: peak {train_stats['peak_gpu_alloc_mb']:>6.0f}MB avg {train_stats['avg_gpu_alloc_mb']:>6.0f}MB "
            f"| val_infer:{val_metrics['avg_infer_ms']:>5.0f}ms\n"
        )
        realtime_log.flush()

        if is_best:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': val_metrics,
                'config': config,
            }, output_path / "best_model.pt")
            print(f"  ★ 保存最佳模型 (AUROC: {best_auroc:.4f}, Epoch: {best_epoch})", flush=True)

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': val_metrics,
                'config': config,
            }, output_path / f"checkpoint_epoch_{epoch + 1}.pt")

    print(f"\n{'='*60}")
    print(f"评估多个分类阈值对 Precision 的影响:")
    print(f"{'='*60}")
    final_val_metrics = evaluate(model, val_loader, device, threshold=0.5, return_all_thresholds=True)
    if 'thresholds' in final_val_metrics:
        print(f"\n{'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'MCC':>10}")
        print(f"-"*70)
        for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
            m = final_val_metrics['thresholds'][t]
            print(f"{t:>10.2f} {m['accuracy']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['mcc']:>10.4f}")
        print()

    total_time = time.time() - training_start
    csv_file.close()
    # tb_writer.close()
    realtime_log.close()

    torch.save(model.state_dict(), output_path / "final_model.pt")

    gpu_mem_final = get_gpu_memory(device)

    summary = {
        'best_auroc': round(best_auroc, 6),
        'best_epoch': best_epoch,
        'total_epochs': args.epochs,
        'total_time_seconds': round(total_time, 2),
        'total_time_human': f"{total_time / 3600:.1f}h" if total_time >= 3600 else f"{total_time / 60:.1f}min",
        'end_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model_complexity': {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': round(model_size_mb, 2),
            'buffer_size_mb': round(buffer_size_mb, 2),
            'input_feature_dim': 1320,
            'hidden_dim': args.hidden_dim,
            'gcnii_layers': args.gcnii_layers,
            'batch_size': args.batch_size,
            'gpu_mem_final': gpu_mem_final,
        },
        'time_per_epoch_avg': round(total_time / args.epochs, 2) if args.epochs else 0,
        'log_csv': str(csv_path),
        'realtime_log': str(realtime_log_path),
        'tb_logs': str(tb_dir),
        'config_json': str(config_path),
        'best_model': str(output_path / "best_model.pt"),
        'final_model': str(output_path / "final_model.pt"),
    }
    summary_path = log_dir / "train_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"  最佳 AUROC: {best_auroc:.4f} (Epoch {best_epoch})")
    print(f"  总耗时: {summary['total_time_human']}")
    print(f"  日志 CSV: {csv_path}")
    print(f"  训练摘要: {summary_path}")
    print(f"  模型保存在: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
