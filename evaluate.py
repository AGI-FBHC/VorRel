import os
import argparse
import yaml
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import create_vorrel_net


class ProteinDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        protein = self.data[idx]
        return {
            'pdb_id': protein.get('pdb_id', f'protein_{idx}'),
            'sequence': protein.get('sequence', ''),
            'pdb_path': protein.get('pdb_path'),
            'residue_coords': protein.get('residue_coords'),
            'residue_types': protein.get('residue_types', []),
            'binding_labels': protein.get('binding_labels'),
        }


def collate_fn(batch):
    max_residues = max(item['residue_coords'].shape[0] for item in batch if item['residue_coords'] is not None)

    padded_coords = []
    padded_labels = []
    sequences = []
    pdb_paths = []
    residue_types_list = []

    for item in batch:
        num_residues = item['residue_coords'].shape[0] if item['residue_coords'] is not None else 0

        if num_residues > 0:
            coords = item['residue_coords']
            labels = item['binding_labels']

            padding_size = max_residues - num_residues
            if padding_size > 0:
                coords = torch.cat([coords, torch.zeros(padding_size, 3)], dim=0)
                labels = torch.cat([labels, torch.zeros(padding_size)], dim=0)

            padded_coords.append(coords)
            padded_labels.append(labels)
        else:
            padded_coords.append(torch.zeros(max_residues, 3))
            padded_labels.append(torch.zeros(max_residues))

        sequences.append(item.get('sequence', ''))
        pdb_paths.append(item.get('pdb_path'))
        residue_types_list.append(item.get('residue_types', []))

    return {
        'sequence': sequences,
        'pdb_path': pdb_paths,
        'residue_coords': torch.stack(padded_coords),
        'residue_types': residue_types_list,
        'binding_labels': torch.stack(padded_labels),
    }


def compute_metrics(all_predictions, all_targets, threshold=0.5):
    pred_binary = (all_predictions > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(all_targets, pred_binary, labels=[0, 1]).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mcc = matthews_corrcoef(all_targets, pred_binary)

    auroc = roc_auc_score(all_targets, all_predictions)
    auprc = average_precision_score(all_targets, all_predictions)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'auroc': auroc,
        'auprc': auprc,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
    }


def evaluate_model(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            predictions = model(batch)
            targets = batch['binding_labels']

            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    metrics = compute_metrics(all_predictions, all_targets)

    return metrics, all_predictions, all_targets


def main():
    parser = argparse.ArgumentParser(description="Evaluate VorRel-Net model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--output", type=str, default=None, help="Output file for results")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_dataset = ProteinDataset(args.test_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn
    )

    model = create_vorrel_net(config)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)

    metrics, predictions, targets = evaluate_model(model, test_loader, device)

    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"MCC:       {metrics['mcc']:.4f}")
    print(f"AUROC:     {metrics['auroc']:.4f}")
    print(f"AUPRC:     {metrics['auprc']:.4f}")
    print("-"*50)
    print(f"TP: {metrics['tp']}, TN: {metrics['tn']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    print("="*50)

    if args.output:
        with open(args.output, 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
