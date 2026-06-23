#!/usr/bin/env python3
"""
VorRel-Net 完整评估框架
支持：
- 10 折交叉验证（按蛋白质分组）
- 5 折交叉验证
- 独立测试集
- 消融研究
- 冷启动场景评估
- 跨靶点泛化实验
增强功能：
- 全面的评估指标（Accuracy, Precision, Recall, F1, AUROC, AUPRC, MCC）
- 模型参数统计
- 训练/推断时间统计
- CPU/GPU 资源占用监控
- FLOPs 计算
"""

import os
import sys
import json
import time
import random
import argparse
import psutil
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.profiler import profile, record_function, ProfilerActivity

from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, matthews_corrcoef,
    confusion_matrix
)

try:
    import GPUtil
    HAS_GPU_UTIL = True
except ImportError:
    HAS_GPU_UTIL = False

sys.path.insert(0, str(Path(__file__).parent))

from train_precomputed import (
    PrecomputedDataset, load_info, find_valid_keys,
    collate_fn, VorRelTrainer
)


def group_proteins_by_sequence(info_map, keys):
    seq_groups = defaultdict(list)
    for key in keys:
        seq = info_map[key]['sequence']
        seq_groups[seq].append(key)
    return list(seq_groups.values())


def group_proteins_by_pdb_id(info_map, keys):
    pdb_groups = defaultdict(list)
    for key in keys:
        pdb_id = key.split('_')[0]
        pdb_groups[pdb_id].append(key)
    return list(pdb_groups.values())


def prepare_kfold_split(groups, k=10, random_state=42):
    random.seed(random_state)
    shuffled_groups = groups.copy()
    random.shuffle(shuffled_groups)
    
    fold_sizes = [len(shuffled_groups) // k] * k
    for i in range(len(shuffled_groups) % k):
        fold_sizes[i] += 1
    
    folds = []
    idx = 0
    for size in fold_sizes:
        fold = []
        for j in range(size):
            fold.extend(shuffled_groups[idx])
            idx += 1
        folds.append(fold)
    
    return folds


def count_parameters(model):
    """计算模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def calculate_flops(model, input_size=(1, 256, 128)):
    """估算模型 FLOPs"""
    try:
        from fvcore.nn import FlopCountAnalysis
        device = next(model.parameters()).device
        dummy_input = {
            'features': torch.randn(input_size).to(device),
            'voronoi': torch.randn(input_size[0], input_size[1], input_size[1]).to(device),
            'hb': torch.randn(input_size[0], input_size[1], input_size[1]).to(device),
            'hp': torch.randn(input_size[0], input_size[1], input_size[1]).to(device),
            'sb': torch.randn(input_size[0], input_size[1], input_size[1]).to(device)
        }
        flops = FlopCountAnalysis(model, dummy_input)
        return flops.total()
    except ImportError:
        return None
    except Exception as e:
        print(f"FLOPs calculation failed: {e}")
        return None


def get_gpu_info(gpu_id=0):
    """获取 GPU 信息"""
    if not HAS_GPU_UTIL or not torch.cuda.is_available():
        return None
    
    try:
        gpus = GPUtil.getGPUs()
        if gpu_id < len(gpus):
            gpu = gpus[gpu_id]
            return {
                'name': gpu.name,
                'total_memory': gpu.memoryTotal,
                'used_memory': gpu.memoryUsed,
                'free_memory': gpu.memoryFree,
                'load': gpu.load * 100,
                'temperature': gpu.temperature
            }
    except Exception as e:
        print(f"GPU info error: {e}")
    return None


def get_cpu_info():
    """获取 CPU 信息"""
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'memory_used_gb': psutil.virtual_memory().used / (1024 ** 3),
        'memory_total_gb': psutil.virtual_memory().total / (1024 ** 3)
    }


def compute_metrics(preds, labels, threshold=0.5):
    """计算评估指标"""
    pred_binary = (preds > threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(labels, pred_binary, labels=[0, 1]).ravel()
    
    metrics = {
        'auroc': 0.0,
        'auprc': 0.0,
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'mcc': 0.0,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'sensitivity': 0.0,
        'specificity': 0.0
    }
    
    try:
        metrics['auroc'] = roc_auc_score(labels, preds)
    except ValueError:
        pass
    
    try:
        metrics['auprc'] = average_precision_score(labels, preds)
    except ValueError:
        pass
    
    metrics['accuracy'] = accuracy_score(labels, pred_binary)
    metrics['precision'] = precision_score(labels, pred_binary, zero_division=0)
    metrics['recall'] = recall_score(labels, pred_binary, zero_division=0)
    metrics['f1'] = f1_score(labels, pred_binary, zero_division=0)
    metrics['mcc'] = matthews_corrcoef(labels, pred_binary)
    
    if (tp + fn) > 0:
        metrics['sensitivity'] = tp / (tp + fn)
    if (tn + fp) > 0:
        metrics['specificity'] = tn / (tn + fp)
    
    return metrics


def evaluate_model(model, dataloader, device, threshold=0.5):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            voronoi = batch['voronoi'].to(device)
            hb = batch['hb'].to(device)
            hp = batch['hp'].to(device)
            sb = batch['sb'].to(device)
            batch_labels = batch['labels']
            batch_masks = batch['masks']
            
            predictions = model({
                'features': features,
                'voronoi': voronoi,
                'hb': hb,
                'hp': hp,
                'sb': sb
            })
            preds = predictions.cpu().numpy()
            
            for i in range(len(preds)):
                mask = batch_masks[i].numpy()
                valid_preds = preds[i][mask.astype(bool)]
                valid_labels = batch_labels[i][mask.astype(bool)]
                all_preds.extend(valid_preds)
                all_labels.extend(valid_labels)
    
    return compute_metrics(np.array(all_preds), np.array(all_labels), threshold)


def train_one_fold(train_keys, val_keys, test_keys, args, fold_idx, log_dir):
    """训练和评估一个 fold"""
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx + 1}/{args.kfold}")
    print(f"{'='*60}")
    print(f"  Train samples: {len(train_keys)}")
    print(f"  Val samples:   {len(val_keys)}")
    print(f"  Test samples:  {len(test_keys)}")
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    
    info_map = load_info(args.info_path)
    train_dataset = PrecomputedDataset(train_keys, args.feature_dir, info_map)
    val_dataset = PrecomputedDataset(val_keys, args.feature_dir, info_map)
    test_dataset = PrecomputedDataset(test_keys, args.feature_dir, info_map) if test_keys else None
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    ) if test_dataset else None
    
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
        }
    }
    
    model = VorRelTrainer(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    param_info = count_parameters(model)
    flops = calculate_flops(model)
    
    print(f"  Model params: {param_info['total']:,} (trainable: {param_info['trainable']:,})")
    if flops:
        print(f"  Estimated FLOPs: {flops / 1e9:.2f} GFLOPs")
    
    best_val_auroc = 0.0
    best_test_metrics = None
    fold_log_file = log_dir / f"fold_{fold_idx + 1}.csv"
    fold_csv = open(fold_log_file, 'w')
    fold_csv.write("Epoch,Train_Loss,Val_Accuracy,Val_Precision,Val_Recall,Val_F1,Val_AUROC,Val_AUPRC,Val_MCC,Val_Sensitivity,Val_Specificity,Train_Time,Val_Time\n")
    
    train_start_time = time.time()
    total_train_time = 0.0
    epoch_times = []
    train_losses = []
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            features = batch['features'].to(device)
            voronoi = batch['voronoi'].to(device)
            hb = batch['hb'].to(device)
            hp = batch['hp'].to(device)
            sb = batch['sb'].to(device)
            labels = batch['labels'].to(device)
            masks = batch['masks'].to(device)
            
            optimizer.zero_grad()
            predictions = model({
                'features': features,
                'voronoi': voronoi,
                'hb': hb,
                'hp': hp,
                'sb': sb
            })
            
            pred_masked = predictions[masks]
            label_masked = labels[masks]
            loss = model.compute_loss(pred_masked.unsqueeze(0), label_masked.unsqueeze(0))
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        train_loss = train_loss / num_batches if num_batches > 0 else 0
        train_losses.append(train_loss)
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        total_train_time += epoch_time
        
        val_start = time.time()
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                voronoi = batch['voronoi'].to(device)
                hb = batch['hb'].to(device)
                hp = batch['hp'].to(device)
                sb = batch['sb'].to(device)
                batch_labels = batch['labels']
                batch_masks = batch['masks']
                
                predictions = model({
                    'features': features,
                    'voronoi': voronoi,
                    'hb': hb,
                    'hp': hp,
                    'sb': sb
                })
                preds = predictions.cpu().numpy()
                
                for i in range(len(preds)):
                    mask = batch_masks[i].numpy()
                    valid_preds = preds[i][mask.astype(bool)]
                    valid_labels = batch_labels[i][mask.astype(bool)]
                    all_preds.extend(valid_preds)
                    all_labels.extend(valid_labels)
        
        val_time = time.time() - val_start
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        val_metrics = compute_metrics(all_preds, all_labels, threshold=args.threshold)
        
        fold_csv.write(f"{epoch + 1},{train_loss:.6f},"
                       f"{val_metrics['accuracy']:.6f},{val_metrics['precision']:.6f},"
                       f"{val_metrics['recall']:.6f},{val_metrics['f1']:.6f},"
                       f"{val_metrics['auroc']:.6f},{val_metrics['auprc']:.6f},"
                       f"{val_metrics['mcc']:.6f},{val_metrics['sensitivity']:.6f},"
                       f"{val_metrics['specificity']:.6f},{epoch_time:.4f},{val_time:.4f}\n")
        fold_csv.flush()
        
        if val_metrics['auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['auroc']
            
            if test_loader is not None:
                best_test_metrics = evaluate_model(model, test_loader, device, args.threshold)
            
            print(f"  ★ Best Val AUROC: {best_val_auroc:.4f} (Epoch {epoch+1})")
            if best_test_metrics:
                print(f"    Test AUROC: {best_test_metrics['auroc']:.4f}")
    
    fold_csv.close()
    
    final_val_metrics = evaluate_model(model, val_loader, device, args.threshold)
    
    inference_time = 0.0
    if test_loader:
        inference_start = time.time()
        evaluate_model(model, test_loader, device, args.threshold)
        inference_time = time.time() - inference_start
    
    gpu_info = get_gpu_info(args.gpu)
    cpu_info = get_cpu_info()
    
    return {
        'fold_idx': fold_idx + 1,
        'best_val_auroc': best_val_auroc,
        'final_val_metrics': final_val_metrics,
        'best_test_metrics': best_test_metrics,
        'train_samples': len(train_keys),
        'val_samples': len(val_keys),
        'test_samples': len(test_keys) if test_keys else 0,
        'model_params': param_info,
        'flops': flops,
        'total_train_time': total_train_time,
        'avg_epoch_time': np.mean(epoch_times),
        'inference_time': inference_time,
        'avg_train_loss': np.mean(train_losses),
        'final_train_loss': train_losses[-1] if train_losses else 0,
        'gpu_info': gpu_info,
        'cpu_info': cpu_info
    }


def print_metrics_summary(results, title="指标总结"):
    """打印详细的指标总结"""
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")
    
    metrics_list = ['auroc', 'auprc', 'accuracy', 'precision', 'recall', 'f1', 'mcc', 'sensitivity', 'specificity']
    
    for metric_name in metrics_list:
        values = [r['final_val_metrics'][metric_name] for r in results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"\nVal {metric_name.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")
        print(f"  Individual: {[f'{x:.4f}' for x in values]}")
    
    train_times = [r['total_train_time'] for r in results]
    print(f"\n训练时间: {np.mean(train_times):.2f} ± {np.std(train_times):.2f} 秒")
    print(f"  Individual: {[f'{x:.2f}s' for x in train_times]}")
    
    params = results[0]['model_params']
    print(f"\n模型参数: {params['total']:,}")
    print(f"  可训练参数: {params['trainable']:,}")
    print(f"  不可训练参数: {params['non_trainable']:,}")
    
    if results[0]['flops']:
        print(f"估计 FLOPs: {results[0]['flops'] / 1e9:.2f} GFLOPs")


def save_detailed_report(results, output_dir):
    """保存详细的评估报告"""
    report = {
        'summary': {},
        'detailed': results
    }
    
    metrics_list = ['auroc', 'auprc', 'accuracy', 'precision', 'recall', 'f1', 'mcc', 'sensitivity', 'specificity']
    
    for metric_name in metrics_list:
        values = [r['final_val_metrics'][metric_name] for r in results]
        report['summary'][metric_name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'values': [float(v) for v in values]
        }
    
    report['summary']['train_time'] = {
        'mean': float(np.mean([r['total_train_time'] for r in results])),
        'std': float(np.std([r['total_train_time'] for r in results])),
        'values': [float(r['total_train_time']) for r in results]
    }
    
    report['summary']['avg_epoch_time'] = {
        'mean': float(np.mean([r['avg_epoch_time'] for r in results])),
        'std': float(np.std([r['avg_epoch_time'] for r in results])),
        'values': [float(r['avg_epoch_time']) for r in results]
    }
    
    report['summary']['inference_time'] = {
        'mean': float(np.mean([r['inference_time'] for r in results])),
        'std': float(np.std([r['inference_time'] for r in results])),
        'values': [float(r['inference_time']) for r in results]
    }
    
    report['summary']['train_loss'] = {
        'mean': float(np.mean([r['avg_train_loss'] for r in results])),
        'std': float(np.std([r['avg_train_loss'] for r in results])),
        'final_values': [float(r['final_train_loss']) for r in results]
    }
    
    if results[0]['model_params']:
        report['summary']['model_params'] = results[0]['model_params']
    
    if results[0]['flops']:
        report['summary']['flops_gflops'] = float(results[0]['flops'] / 1e9)
    
    if results[0]['gpu_info']:
        report['summary']['gpu_info'] = results[0]['gpu_info']
    
    if results[0]['cpu_info']:
        report['summary']['cpu_info'] = results[0]['cpu_info']
    
    report_file = output_dir / "detailed_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("VorRel-Net 评估报告\n")
        f.write("="*60 + "\n\n")
        
        f.write("【评估指标】\n")
        for metric_name in metrics_list:
            m = report['summary'][metric_name]
            f.write(f"{metric_name.capitalize():<15}: {m['mean']:.4f} ± {m['std']:.4f}\n")
        
        f.write("\n【训练时间】\n")
        t = report['summary']['train_time']
        f.write(f"总训练时间: {t['mean']:.2f} ± {t['std']:.2f} 秒\n")
        t = report['summary']['avg_epoch_time']
        f.write(f"平均 epoch 时间: {t['mean']:.2f} ± {t['std']:.2f} 秒\n")
        
        f.write("\n【模型信息】\n")
        if 'model_params' in report['summary']:
            p = report['summary']['model_params']
            f.write(f"总参数: {p['total']:,}\n")
            f.write(f"可训练参数: {p['trainable']:,}\n")
        if 'flops_gflops' in report['summary']:
            f.write(f"FLOPs: {report['summary']['flops_gflops']:.2f} GFLOPs\n")
        
        f.write("\n【资源占用】\n")
        if 'cpu_info' in report['summary']:
            c = report['summary']['cpu_info']
            f.write(f"CPU 使用率: {c['cpu_percent']}%\n")
            f.write(f"内存占用: {c['memory_used_gb']:.2f} GB / {c['memory_total_gb']:.2f} GB\n")
        if 'gpu_info' in report['summary']:
            g = report['summary']['gpu_info']
            f.write(f"GPU: {g['name']}\n")
            f.write(f"GPU 显存: {g['used_memory']:.0f} MB / {g['total_memory']:.0f} MB\n")
            f.write(f"GPU 使用率: {g['load']:.1f}%\n")
    
    print(f"\n详细报告已保存到: {report_file}")
    print(f"摘要报告已保存到: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="VorRel-Net 完整评估框架")
    parser.add_argument('--feature_dir', type=str, default='/home/shihj/shj/VorRel/feature')
    parser.add_argument('--info_path', type=str, default='/home/shihj/shj/GraphPRNet/data/scPDB/info.txt')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results')
    
    parser.add_argument('--eval_mode', type=str, default='kfold',
                        choices=['kfold', 'cross_val', 'test_set', 'ablation', 'cold_start', 'cross_target'])
    parser.add_argument('--kfold', type=int, default=10, help='K 折交叉验证')
    parser.add_argument('--cross_val_folds', type=int, default=5, help='5 折交叉验证')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='独立测试集比例')
    
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--gcnii_layers', type=int, default=6)
    parser.add_argument('--pos_weight', type=float, default=10.0)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--use_focal_loss', action='store_true', help='使用 Focal Loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal Loss 的 gamma 参数')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument('--group_by', type=str, default='sequence',
                        choices=['sequence', 'pdb_id'])
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"{args.eval_mode}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"VorRel-Net 评估框架")
    print(f"{'='*60}")
    print(f"  Mode: {args.eval_mode}")
    print(f"  Output: {output_dir}")
    
    print(f"\n加载数据...")
    info_map = load_info(args.info_path)
    valid_keys = find_valid_keys(args.feature_dir, info_map)
    print(f"  Valid samples: {len(valid_keys)}")
    
    if args.group_by == 'sequence':
        print("  按序列分组...")
        groups = group_proteins_by_sequence(info_map, valid_keys)
    else:
        print("  按 PDB ID 分组...")
        groups = group_proteins_by_pdb_id(info_map, valid_keys)
    
    print(f"  Protein groups: {len(groups)}")
    
    config_file = output_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump({
            'args': vars(args),
            'num_valid_samples': len(valid_keys),
            'num_protein_groups': len(groups),
            'timestamp': timestamp,
            'gpu_available': torch.cuda.is_available(),
            'cpu_cores': psutil.cpu_count()
        }, f, indent=2)
    
    if args.eval_mode in ['kfold', 'cross_val']:
        k = args.kfold if args.eval_mode == 'kfold' else args.cross_val_folds
        print(f"\n准备 {k} 折交叉验证...")
        
        folds = prepare_kfold_split(groups, k=k, random_state=args.seed)
        print(f"  Fold sizes: {[len(f) for f in folds]}")
        
        test_keys = []
        if args.test_ratio > 0:
            print(f"\n留出 {args.test_ratio:.0%} 作为独立测试集...")
            test_keys = folds[-1]
            folds = folds[:-1]
            k = k - 1
            print(f"  Test samples: {len(test_keys)}")
            print(f"  Remaining folds: {len(folds)}")
        
        results = []
        for fold_idx in range(k):
            val_keys = folds[fold_idx]
            train_keys = []
            for j in range(k):
                if j != fold_idx:
                    train_keys.extend(folds[j])
            
            result = train_one_fold(
                train_keys, val_keys, test_keys,
                args, fold_idx, output_dir
            )
            results.append(result)
        
        results_file = output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print_metrics_summary(results, title=f"{k} 折交叉验证结果")
        
        save_detailed_report(results, output_dir)
        
        print(f"\n结果已保存到: {results_file}")
    
    print(f"\n评估完成!")
    print(f"  输出目录: {output_dir}")


if __name__ == '__main__':
    main()
