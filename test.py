import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import MVFEM, VMGCM, ARFEM, create_vorrel_net

FEATURE_DIR = '/home/shihj/shj/VorRel/feature'
PDB_DIR = '/home/shihj/shj/GraphPRNet/data/scPDB/pdb_files'
TEST_PROTEINS = ['10mh_A_1', '11bg_A_2', '11bg_B_2']


def load_precomputed_features(key):
    pssm = np.load(os.path.join(FEATURE_DIR, 'PSSM', f'{key}.npy'))
    esm2 = np.load(os.path.join(FEATURE_DIR, 'ESM-2', f'{key}.npy'))
    dssp = np.load(os.path.join(FEATURE_DIR, 'DSSP', f'{key}.npy'))
    atomic = np.load(os.path.join(FEATURE_DIR, 'Atomic', f'{key}.npy'))
    graph = np.load(os.path.join(FEATURE_DIR, 'Graph', f'{key}.npy'), allow_pickle=True).item()
    return pssm, esm2, dssp, atomic, graph


def get_sequence_from_info(key):
    pdb_id, chain_id, struct_num = key.split('_')
    info_path = '/home/shihj/shj/GraphPRNet/data/scPDB/info.txt'
    with open(info_path) as f:
        for line in f:
            if line.startswith(f'{pdb_id}\t{struct_num}\t{chain_id}\t'):
                return line.strip().split('\t')[3]
    return None


def test_step1_load_features():
    print("=" * 60)
    print("Step 1: 加载预计算特征")
    print("=" * 60)

    for key in TEST_PROTEINS:
        pssm, esm2, dssp, atomic, graph = load_precomputed_features(key)
        n_feat = pssm.shape[0]
        n_graph = graph['voronoi_edges'].shape[0]
        match = "✓" if n_feat == n_graph else "✗ (维度不匹配)"

        print(f"\n  {key}:")
        print(f"    PSSM:   {pssm.shape}")
        print(f"    ESM-2:  {esm2.shape}")
        print(f"    DSSP:   {dssp.shape}")
        print(f"    Atomic: {atomic.shape}")
        print(f"    Graph:  voronoi={graph['voronoi_edges'].shape}")
        print(f"    特征-图维度一致: {match}")

        if n_feat != n_graph:
            return None

    print(f"\n  Step 1 通过 ✓")
    return TEST_PROTEINS[0]


def test_step2_embedding(key):
    print("\n" + "=" * 60)
    print("Step 2: 特征拼接 + Embedding 层")
    print("=" * 60)

    pssm, esm2, dssp, atomic, graph = load_precomputed_features(key)
    n = pssm.shape[0]

    config = {
        'model': {'hidden_dim': 256, 'gcnii_layers': 6, 'alpha': 0.5, 'beta': 1.3, 'dropout': 0.1}
    }

    mvfem = MVFEM(config)
    mvfem.eval()

    device = next(mvfem.parameters()).device

    concatenated = np.concatenate([esm2, pssm, dssp, atomic], axis=1)
    print(f"  拼接后特征: {concatenated.shape} (1280+20+13+7={1280+20+13+7})")

    concat_tensor = torch.tensor(concatenated, dtype=torch.float32).to(device)

    with torch.no_grad():
        h_initial = mvfem.embedding_layer(concat_tensor)
        if h_initial.dim() == 2:
            h_initial = h_initial.unsqueeze(0)

    print(f"  Embedding 输出: {h_initial.shape}")
    print(f"  Step 2 通过 ✓")

    return h_initial, graph


def test_step3_graph_construction(key):
    print("\n" + "=" * 60)
    print("Step 3: 图结构构建 (VMGCM)")
    print("=" * 60)

    pssm, esm2, dssp, atomic, graph = load_precomputed_features(key)

    pdb_id = key.split('_')[0]
    chain_id = key.split('_')[1]
    pdb_path = os.path.join(PDB_DIR, f'{pdb_id}.pdb')
    sequence = get_sequence_from_info(key)

    config = {
        'graph': {
            'hydrogen_bond_threshold': 6.0,
            'hydrophobic_threshold': 8.0,
            'salt_bridge_threshold': 12.0
        }
    }

    vmgcm = VMGCM(config)
    vmgcm.eval()

    residue_types = list(sequence) if sequence else None

    coords_tensor = torch.tensor(pssm[:, :3], dtype=torch.float32)
    if coords_tensor.shape[1] != 3:
        from scripts.extract_edge_matrices import extract_residue_coords_and_types
        coords_np, _, _ = extract_residue_coords_and_types(pdb_path, chain_id)
        coords_tensor = torch.tensor(coords_np, dtype=torch.float32)

    with torch.no_grad():
        raw_edges, processed_edges = vmgcm(
            residue_coords=coords_tensor,
            residue_types=residue_types,
            sequence=sequence
        )

    print(f"  raw_edges keys: {list(raw_edges.keys())}")
    for k, v in raw_edges.items():
        print(f"    {k}: {v.shape}")
    print(f"  processed_edges keys: {list(processed_edges.keys())}")
    for k, v in processed_edges.items():
        print(f"    {k}: {v.shape}")

    print(f"\n  预计算图 vs VMGCM 实时构建:")
    for edge_type in ['voronoi', 'hydrogen_bond', 'hydrophobic', 'salt_bridge']:
        precomputed_key = {'voronoi': 'voronoi_edges', 'hydrogen_bond': 'hb_edges',
                          'hydrophobic': 'hp_edges', 'salt_bridge': 'sb_edges'}[edge_type]
        precomputed = graph[precomputed_key]
        live = raw_edges[edge_type].squeeze(0).cpu().numpy()
        n_pre = (precomputed > 0).sum() // 2
        n_live = (live > 0).sum() // 2
        print(f"    {edge_type}: 预计算={n_pre}边, 实时={n_live}边")

    print(f"  Step 3 通过 ✓")
    return raw_edges, processed_edges


def test_step4_arfem(h_initial, raw_edges, processed_edges):
    print("\n" + "=" * 60)
    print("Step 4: 自适应关系特征增强 (ARFEM)")
    print("=" * 60)

    config = {
        'model': {
            'hidden_dim': 256,
            'gcnii_layers': 6,
            'alpha': 0.5,
            'beta': 1.3,
            'dropout': 0.1,
        }
    }

    arfem = ARFEM(config)
    arfem.eval()

    device = next(arfem.parameters()).device
    h_initial = h_initial.to(device)
    edge_features = (raw_edges, processed_edges)

    with torch.no_grad():
        predictions = arfem(
            h_initial=h_initial,
            edge_features_dict=edge_features,
            return_features=False
        )

    print(f"  输入: h_initial={h_initial.shape}")
    print(f"  输出: predictions={predictions.shape}")
    print(f"  预测范围: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
    print(f"  Step 4 通过 ✓")

    return predictions


def test_step5_loss(predictions, binding_labels):
    print("\n" + "=" * 60)
    print("Step 5: 损失计算")
    print("=" * 60)

    config = {'model': {'pos_weight': 10.0}}
    model = create_vorrel_net(config)

    targets = torch.tensor(binding_labels, dtype=torch.float32).unsqueeze(0)

    loss = model.compute_loss(predictions, targets)

    print(f"  predictions: {predictions.shape}")
    print(f"  targets: {targets.shape}")
    print(f"  正样本数: {int(binding_labels.sum())}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Step 5 通过 ✓")

    return loss


def test_step6_full_pipeline(key):
    print("\n" + "=" * 60)
    print("Step 6: 完整端到端前向传播 (预计算特征 + 模型)")
    print("=" * 60)

    pssm, esm2, dssp, atomic, graph = load_precomputed_features(key)
    n = pssm.shape[0]

    pdb_id = key.split('_')[0]
    chain_id = key.split('_')[1]
    pdb_path = os.path.join(PDB_DIR, f'{pdb_id}.pdb')
    sequence = get_sequence_from_info(key)

    from scripts.extract_edge_matrices import extract_residue_coords_and_types
    coords_np, _, _ = extract_residue_coords_and_types(pdb_path, chain_id)
    if coords_np is None:
        print(f"  无法提取坐标，跳过")
        return
    coords_tensor = torch.tensor(coords_np, dtype=torch.float32).unsqueeze(0)

    config = {
        'model': {
            'hidden_dim': 256,
            'gcnii_layers': 6,
            'alpha': 0.5,
            'beta': 1.3,
            'dropout': 0.1,
            'pos_weight': 10.0,
        },
        'graph': {
            'hydrogen_bond_threshold': 6.0,
            'hydrophobic_threshold': 8.0,
            'salt_bridge_threshold': 12.0,
        }
    }

    print(f"  蛋白质: {key}")
    print(f"  残基数: {n}")

    model = create_vorrel_net(config)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  设备: {device}")

    concatenated = np.concatenate([esm2, pssm, dssp, atomic], axis=1)
    concat_tensor = torch.tensor(concatenated, dtype=torch.float32).to(device)

    with torch.no_grad():
        h_initial = model.mvfem.embedding_layer(concat_tensor)
        if h_initial.dim() == 2:
            h_initial = h_initial.unsqueeze(0)

        residue_types = list(sequence)[:n] if sequence else ['ALA'] * n
        raw_edges, processed_edges = model.vmgcm(
            residue_coords=coords_tensor.to(device),
            residue_types=residue_types,
            sequence=sequence[:n] if sequence else None
        )

        predictions = model.arfem(
            h_initial=h_initial,
            edge_features_dict=(raw_edges, processed_edges),
            return_features=False
        )

    print(f"  输出形状: {predictions.shape}")
    print(f"  预测范围: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")

    info_path = '/home/shihj/shj/GraphPRNet/data/scPDB/info.txt'
    binding_str = ''
    with open(info_path) as f:
        for line in f:
            if line.startswith(f'{pdb_id}\t{key.split("_")[2]}\t{chain_id}\t'):
                binding_str = line.strip().split('\t')[4]
                break

    binding_labels = np.zeros(n, dtype=np.float32)
    for i, ch in enumerate(binding_str[:n]):
        if ch == '1':
            binding_labels[i] = 1.0

    targets = torch.tensor(binding_labels, dtype=torch.float32).unsqueeze(0).to(device)
    loss = model.compute_loss(predictions, targets)

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Step 6 通过 ✓")


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        VorRel 端到端 Pipeline 测试                       ║")
    print("╚══════════════════════════════════════════════════════════╝")

    key = test_step1_load_features()
    if key is None:
        print("\nStep 1 失败，终止测试")
        return

    h_initial, graph = test_step2_embedding(key)

    raw_edges, processed_edges = test_step3_graph_construction(key)

    predictions = test_step4_arfem(h_initial, raw_edges, processed_edges)

    pssm, esm2, dssp, atomic, _ = load_precomputed_features(key)
    info_path = '/home/shihj/shj/GraphPRNet/data/scPDB/info.txt'
    pdb_id, chain_id, struct_num = key.split('_')
    binding_str = ''
    with open(info_path) as f:
        for line in f:
            if line.startswith(f'{pdb_id}\t{struct_num}\t{chain_id}\t'):
                binding_str = line.strip().split('\t')[4]
                break

    n = pssm.shape[0]
    binding_labels = np.zeros(n, dtype=np.float32)
    for i, ch in enumerate(binding_str[:n]):
        if ch == '1':
            binding_labels[i] = 1.0

    test_step5_loss(predictions, binding_labels)

    test_step6_full_pipeline(key)

    print("\n" + "=" * 60)
    print("  所有测试通过! ✓")
    print("=" * 60)


if __name__ == '__main__':
    main()
