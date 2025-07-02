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
&emsp;&emsp;sequences 文件中保存了所有蛋白质的氨基酸序列数据，一行一个pdb_id + sequence 组合，形式为：  
10mh_A MIEIKDKQLTGLRFIDLFAGLGGFR...
&emsp;&emsp;unique文件是从no_one_msa_info.txt中提取的去重、不冗余的蛋白质列表，主要用于交叉验证或测试集划分。  
