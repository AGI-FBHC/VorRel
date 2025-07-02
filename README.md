# 一、数据处理
&emsp;&emsp;原有数据集保存在data文件夹下，scPDB文件夹下。分别有五个文件，分别是info.txt、no_one_msa_info.txt、no_one_msa_unique、sequences、unique五个文件。  
&emsp;&emsp;info.txt是数据集中每个蛋白质样本的基本信息汇总表，包含以下字段（以制表符分隔）：
| 字段名                | 含义                                                      |
| ------------------ | ------------------------------------------------------- |
| `pdb_id`           | 蛋白质的 PDB 编号（含结构编号与链 ID），例如：`10mh_A`                     |
| `structure`        | 三维结构文件名，通常对应于 PDB 条目                                    |
| `chain`            | 蛋白质所在的链 ID，例如 `A`                                       |
| `sequence`         | 氨基酸序列，由单字母表示法组成                                         |
| `binding_residues` | 结合位点标签，为与 `sequence` 等长的 0/1 字符串，1 表示该残基为结合位点，0 表示非结合位点 |  

&emsp;&emsp;no_one_msa_info.txt是与info.txt 类似的文件，只不过该版本的数据不依赖MSA信息，用于评估模型在无进化信息下的泛化能力，其字段信息与info.txt一样。  
&emsp;&emsp;no_one_msa_unique 是从 no_one_msa_info.txt中筛选出的去重后、序列不冗余的蛋白质样本列表，用于消除同源序列带来的偏差影响，确保训练和测试的公平性。  
&emsp;&emsp;sequences 文件中保存了所有蛋白质的氨基酸序列数据，一行一个pdb_id + sequence 组合，形式为：10mh_A MIEIKDKQLTGLRFIDLFAGLGGFR...  
&emsp;&emsp;unique文件是从no_one_msa_info.txt中提取的去重、不冗余的蛋白质列表，主要用于交叉验证或测试集划分。  
&emsp;&emsp;在数据处理阶段，首先需要从.pdb结构文件中提取信息，根据info.txt中的信息，生成PCA_residue_feas_PHSA.pkl和PCA_psepos_SC.pkl两个文件。  
**PCA_residue_feas_PHSA.pkl：** 包含每个样本的残基级别节点特征，维度为 len × 71  
**PCA_psepos_SC.pkl：** 存储每个残基的空间坐标（经过 PCA 降维后为 len × 3）  
&emsp;&emsp;在生成这两个文件后，根据以下代码生成模型输入的.npy文件，用于模型的训练和推理：
<pre> def create_adjacency_matrix(data,  dist_threshold=14, output_dir='./data/adjacency_matrix'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    sigma = 10
    for name, array in tqdm(data.items()):
        # 将array转换为NumPy数组以便于计算
        points = np.array(array)
        # 初始化邻接矩阵
        n = len(points)
        adjacency_matrix = np.zeros((n, n), dtype=int)
        # 计算每对点之间的距离
        for i in range(n):
            for j in range(n):
                # 计算欧氏距离
                distance = np.linalg.norm(points[i] - points[j])
                # 设置邻接矩阵的值
                if distance <= dist_threshold:
                    adjacency_matrix[i, j] = 1
        # 保存邻接矩阵为.npy文件
        file_path = os.path.join(output_dir, f'{name}.npy')
        np.save(file_path, adjacency_matrix)  
                    
def create_feature_matrix(data, output_dir='./data/node_features'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for name, matrix in data.items():
        np_matrix = np.array(matrix)
        file_path = os.path.join(output_dir, f'{name}.npy')
        np.save(file_path, np_matrix)
    print("success! ---create_feature_matrix()---")</pre>  

&emsp;&emsp;node_features/{pdb_id}.npy是从PCA_residue_feas_PHSA.pkl 中提取，维度为len × 71,表示每个残基的节点特征向量；adjacency_matrix14/{pdb_id}.npy使用psepos_SC.pkl中的坐标构建残基间的邻接关系，若两个残基之间的欧式距离小于14Å则认为存在边
