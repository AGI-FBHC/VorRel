import pickle
import random
import os


def load_info_as_dict(info_path):
    data = {}
    with open(info_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 5:
                continue
            pdb_id, structure, chain, sequence, binding_str = parts
            key = f"{pdb_id}_{chain}"

            # 检查 binding_str 是否只包含 0 和 1
            if not all(c in '01' for c in binding_str):
                print(f"[Warning] Invalid binding string for {key}, skip: {binding_str[:20]}...")
                continue

            # 长度对不上也跳过
            if len(sequence) != len(binding_str):
                print(f"[Warning] Sequence/binding length mismatch for {key}, skip.")
                continue

            binding = list(map(int, list(binding_str)))
            data[key] = [sequence, binding]
    return data


def split_data(data, val_ratio=0.2, seed=42):
    keys = list(data.keys())
    random.seed(seed)
    random.shuffle(keys)

    val_size = int(len(keys) * val_ratio)
    val_keys = keys[:val_size]
    train_keys = keys[val_size:]

    train_data = {k: data[k] for k in train_keys}
    test_data = {k: data[k] for k in val_keys}
    return train_data, test_data


def save_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    info_path = './data/scPDB/info.txt'  # ← 请根据你的路径确认
    output_dir = './data/scPDB/split'

    os.makedirs(output_dir, exist_ok=True)

    all_data = load_info_as_dict(info_path)
    train_data, test_data = split_data(all_data, val_ratio=0.2)

    save_pkl(train_data, os.path.join(output_dir, 'train_data.pkl'))
    save_pkl(test_data, os.path.join(output_dir, 'test_data.pkl'))

    print(f"✅ 数据划分完成：train_data.pkl({len(train_data)}) / test_data.pkl({len(test_data)})")
