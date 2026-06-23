import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial.distance import cdist

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vmgcm import VMGCM, VoronoiGraphBuilder


THREE_TO_ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}


def parse_info_txt(info_path):
    """解析 scPDB 的 info.txt 文件，返回蛋白质信息列表"""
    proteins = []
    with open(info_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('pdb_id'):
                continue
            parts = line.split('\t')
            if len(parts) >= 5:
                pdb_id = parts[0].strip()
                structure_num = int(parts[1].strip())
                chain_id = parts[2].strip()
                sequence = parts[3].strip()
                binding_labels = parts[4].strip()
                proteins.append({
                    'pdb_id': pdb_id,
                    'structure_num': structure_num,
                    'chain_id': chain_id,
                    'sequence': sequence,
                    'binding_labels': binding_labels
                })
    return proteins


def get_site_num_from_residues(binding_residues_str, chain_id):
    """从结合残基字符串中提取 binding site 数量"""
    if not binding_residues_str or binding_residues_str == '-':
        return 0
    residues = [r.strip() for r in binding_residues_str.split(',') if r.strip()]
    sites = set()
    for res in residues:
        parts = res.split('.')
        if len(parts) == 2 and parts[0].strip() == chain_id:
            sites.add(parts[0].strip())
    return max(len(sites), 1) if residues else 0


def extract_residue_coords_and_types(pdb_path, chain_id):
    """
    从 PDB 文件中提取指定链的 CA 坐标、残基类型（三字母代号）和残基序列号
    直接解析 PDB 文件，不依赖 BioPython
    """
    coords = []
    residue_types = []
    residue_numbers = []

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and len(line) >= 54 and line[12:16].strip() == 'CA':
                chain = line[21].strip()
                if chain == chain_id:
                    resname = line[17:20].strip().upper()
                    resseq = int(line[22:26].strip())
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    if resseq not in residue_numbers:
                        coords.append([x, y, z])
                        residue_types.append(resname)
                        residue_numbers.append(resseq)

    if not coords:
        return None, None, None

    return np.array(coords, dtype=np.float32), residue_types, residue_numbers


def build_multirelational_graph(residue_coords, residue_types, config=None):
    """
    构建 Voronoi + 多关系边图

    返回:
        graph_data: dict，包含以下键:
            - voronoi_adj: Voronoi 拓扑邻接矩阵 (n, n)
            - hydrogen_bond: 氢键边矩阵 (n, n)
            - hydrophobic: 疏水作用边矩阵 (n, n)
            - salt_bridge: 盐桥边矩阵 (n, n)
            - combined_adj: 合并的邻接矩阵 (n, n)
    """
    if config is None:
        config = {}
    graph_builder = VoronoiGraphBuilder(
        hydrogen_bond_threshold=config.get('hydrogen_bond_threshold', 3.5),
        hydrophobic_threshold=config.get('hydrophobic_threshold', 5.5),
        salt_bridge_threshold=config.get('salt_bridge_threshold', 7.5)
    )

    coords_tensor = torch.tensor(residue_coords, dtype=torch.float32)

    voronoi_adj = graph_builder.build_voronoi_topology(coords_tensor, cutoff=12.0)
    hydrogen_bond = graph_builder.build_hydrogen_bond_edges(coords_tensor, residue_types)
    hydrophobic = graph_builder.build_hydrophobic_edges(coords_tensor, residue_types)
    salt_bridge = graph_builder.build_salt_bridge_edges(coords_tensor, residue_types)

    combined_adj = voronoi_adj + hydrogen_bond + hydrophobic + salt_bridge
    combined_adj = torch.clamp(combined_adj, 0, 1)

    n = voronoi_adj.shape[0]
    degree = combined_adj.sum(dim=1, keepdim=True)
    degree = torch.where(degree > 0, degree, torch.ones_like(degree))
    normalized_adj = combined_adj / degree

    return {
        'voronoi_adj': voronoi_adj.numpy(),
        'hydrogen_bond': hydrogen_bond.numpy(),
        'hydrophobic': hydrophobic.numpy(),
        'salt_bridge': salt_bridge.numpy(),
        'combined_adj': combined_adj.numpy(),
        'normalized_adj': normalized_adj.numpy(),
        'residue_coords': residue_coords,
        'residue_types': residue_types,
        'num_residues': len(residue_types)
    }


def extract_binding_labels(binding_labels_str, residue_numbers):
    """根据二进制字符串和PDB残基序列号生成 binding 标签（正确对齐）"""
    labels = np.zeros(len(residue_numbers), dtype=np.float32)
    if not binding_labels_str:
        return labels

    for i, resseq in enumerate(residue_numbers):
        idx = resseq - 1
        if 0 <= idx < len(binding_labels_str) and binding_labels_str[idx] == '1':
            labels[i] = 1.0
    return labels


def process_single_protein(pdb_id, structure_num, chain_id, pdb_dir, binding_labels_str='', sequence='', config=None):
    """处理单个蛋白质，返回构图结果，会补齐到序列长度"""
    pdb_path = os.path.join(pdb_dir, f'{pdb_id}.pdb')
    if not os.path.exists(pdb_path):
        return None

    coords, residue_types, residue_numbers = extract_residue_coords_and_types(pdb_path, chain_id)
    if coords is None:
        return None

    if sequence:
        seq_len = len(sequence)
        pdb_len = len(residue_numbers)
        
        # 如果 PDB CA 数量不等于序列长度，补齐到序列长度
        if seq_len != pdb_len:
            # 创建完整的坐标、残基类型、残基编号数组
            full_coords = np.zeros((seq_len, 3), dtype=np.float32)
            full_res_types = []
            full_res_nums = []
            
            # 创建映射：残基编号到索引的映射
            res_to_idx = {resseq: i for i, resseq in enumerate(residue_numbers)}
            
            # 先找到第一个有效残基的坐标，用于前面的填充
            first_valid_idx = None
            for i in range(seq_len):
                expected_resseq = i + 1
                if expected_resseq in res_to_idx:
                    first_valid_idx = i
                    break
            
            # 填充
            for i in range(seq_len):
                # 序列编号 i + 1（从1-based）
                expected_resseq = i + 1
                full_res_nums.append(expected_resseq)
                
                if expected_resseq in res_to_idx:
                    # PDB 里有这个残基，用真实数据
                    idx = res_to_idx[expected_resseq]
                    full_coords[i] = coords[idx]
                    full_res_types.append(residue_types[idx])
                else:
                    # PDB 里没有这个残基，用虚拟数据
                    # 优先用前一个残基的坐标；如果是第一个残基缺失，用第一个有效残基
                    if i > 0:
                        full_coords[i] = full_coords[i-1]
                    elif first_valid_idx is not None:
                        full_coords[i] = full_coords[first_valid_idx]
                    else:
                        full_coords[i] = [0, 0, 0]
                    full_res_types.append('ALA')  # 默认用丙氨酸
            
            coords = full_coords
            residue_types = full_res_types
            residue_numbers = full_res_nums
    
    graph_data = build_multirelational_graph(coords, residue_types, config)
    
    binding_labels = extract_binding_labels(binding_labels_str, residue_numbers)
    graph_data['binding_labels'] = binding_labels
    graph_data['pdb_id'] = pdb_id
    graph_data['structure_num'] = structure_num
    graph_data['chain_id'] = chain_id

    return graph_data


def batch_extract(info_path, pdb_dir, output_dir, config=None, force=False):
    """批量处理 scPDB 数据集，提取 Voronoi + 多关系边图"""
    os.makedirs(output_dir, exist_ok=True)

    proteins = parse_info_txt(info_path)
    print(f"从 info.txt 解析到 {len(proteins)} 个蛋白质条目")

    success_count = 0
    fail_count = 0
    skip_count = 0

    for entry in tqdm(proteins, desc="提取 Voronoi + 多关系边图"):
        pdb_id = entry['pdb_id']
        structure_num = entry['structure_num']
        chain_id = entry['chain_id']
        binding_labels_str = entry['binding_labels']
        sequence = entry['sequence']

        key = f"{pdb_id}_{chain_id}_{structure_num}"
        save_path = os.path.join(output_dir, f"{key}.npy")

        if not force and os.path.exists(save_path):
            skip_count += 1
            continue

        try:
            result = process_single_protein(
                pdb_id, structure_num, chain_id, pdb_dir,
                binding_labels_str=binding_labels_str,
                sequence=sequence,
                config=config
            )
            if result is not None:
                np.save(save_path, {
                    'voronoi_edges': result['voronoi_adj'],
                    'hb_edges': result['hydrogen_bond'],
                    'hp_edges': result['hydrophobic'],
                    'sb_edges': result['salt_bridge'],
                })
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"处理 {key} 时出错: {e}")
            fail_count += 1

    print(f"\n处理完成! 成功: {success_count}, 失败: {fail_count}, 跳过: {skip_count}")
    print(f"结果保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='从 scPDB 数据集提取 Voronoi + 多关系边图')
    parser.add_argument('--info_path', type=str,
                        default='/home/shihj/shj/GraphPRNet/data/scPDB/info.txt',
                        help='info.txt 文件路径')
    parser.add_argument('--pdb_dir', type=str,
                        default='/home/shihj/shj/GraphPRNet/data/scPDB/pdb_files',
                        help='PDB 文件目录')
    parser.add_argument('--output_dir', type=str,
                        default='/home/shihj/shj/VorRel/feature/Graph',
                        help='输出目录')
    parser.add_argument('--hydrogen_bond_threshold', type=float, default=6.0)
    parser.add_argument('--hydrophobic_threshold', type=float, default=8.0)
    parser.add_argument('--salt_bridge_threshold', type=float, default=12.0)
    parser.add_argument('--force', action='store_true', help='强制重新处理已存在的文件')

    args = parser.parse_args()

    config = {
        'hydrogen_bond_threshold': args.hydrogen_bond_threshold,
        'hydrophobic_threshold': args.hydrophobic_threshold,
        'salt_bridge_threshold': args.salt_bridge_threshold,
    }

    batch_extract(args.info_path, args.pdb_dir, args.output_dir, config, force=args.force)


if __name__ == '__main__':
    main()
