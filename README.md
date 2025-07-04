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

&emsp;&emsp;node_features/{pdb_id}.npy是从PCA_residue_feas_PHSA.pkl 中提取，维度为len × 71,表示每个残基的节点特征向量，adjacency_matrix14/{pdb_id}.npy使用psepos_SC.pkl中的坐标构建残基间的邻接关系，若两个残基之间的欧式距离小于14Å则认为存在边。再利用create_label_matrix(data,key)、create_label_matrix_more(data, key)两个函数生成训练和测试所需要的pkl文件。

# 二、训练模型  
&emsp;&emsp;此项目使用GraphPLBR模型，这是一种是一个基于PyTorch的深度图卷积神经网络 (deepGCN) 模型，模型输入蛋白质的结构信息（邻接矩阵）、节点特征和残基空间位置。以下是模型输入数据和输出数据：
| 输入项                | 形状        | 描述                                  |
| ------------------ | --------- | ----------------------------------- |
| `sequence_name`    | -         | 蛋白质序列的唯一 ID，对应数据文件名                 |
| `sequence`         | `L`       | 蛋白质氨基酸序列，长度为残基数 L                   |
| `label`            | `[L]`     | 每个残基是否为结合位点（0: 非结合, 1: 结合）          |
| `node_features`    | `[L, 71]` | 每个残基的 71 维生化特征（如氨基酸类型、理化性质等）        |
| `adjacency_matrix` | `[L, L]`  | 蛋白质残基邻接矩阵，表示残基之间是否有空间接触（1: 有, 0: 无） |
| `residue_pos`      | `[L, 3]`  | 每个残基的 3D 坐标 (PCA 降维后)               |  

| 输出项           | 形状       | 描述                          |
| ------------- | -------- | --------------------------- |
| `output`      | `[L, 2]` | 每个残基属于“非结合(0)”或“结合(1)”的概率分布 |
| `pred_labels` | `[L]`    | 概率最大类别作为预测标签（0 或 1）         |  

&emsp;&emsp;训练过程的超参数如下：
| 参数                | 默认值  | 说明                                 |
| ----------------- | ---- | ---------------------------------- |
| `LAYER`           | 6    | GCN 层数（更多层可捕获更远邻域信息，但可能过拟合）        |
| `INPUT_DIM`       | 71   | 输入节点特征维度                           |
| `HIDDEN_DIM`      | 256  | 隐藏层维度（可以尝试 512 提升模型容量）             |
| `DROPOUT`         | 0.2  | Dropout 比例，防止过拟合                   |
| `LEARNING_RATE`   | 1e-3 | Adam 优化器学习率                        |
| `WEIGHT_DECAY`    | 1e-5 | 权重衰减 (L2 正则化)，有助于模型泛化              |
| `POSITIVE_WEIGHT` | 3.0  | 在 CrossEntropyLoss 中对正样本加权，缓解类别不平衡 |
| `BATCH_SIZE`      | 1    | 批量大小（受限于图数据大小，建议尝试 4、8、16）         |
| `NUMBER_EPOCHS`   | 20   | 训练轮数                               |  

&emsp;&emsp;在训练过程中会生成如下评估指标：
| 指标          | 描述                              |
| ----------- | ------------------------------- |
| `Loss`      | 加权交叉熵损失                         |
| `Precision` | 预测为正的残基中有多少比例是真的正样本             |
| `Recall`    | 真正的正样本中有多少被正确预测为正样本             |
| `MCC`       | Matthews 相关系数（更适合类别不平衡时评估二分类任务） |
| `Run Time`  | 每轮训练用时                          |  






