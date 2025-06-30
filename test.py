"""
# -*- coding: utf-8 -*-
# @Author : Sun JJ
# @File : test.py
# @Time : 2022/5/9 18:25
# code is far away from bugs with the god animal protecting
#         ┌─┐       ┌─┐
#      ┌──┘ ┴───────┘ ┴──┐
#      │                 │
#      │       ───       │
#      │  ─┬┘       └┬─  │
#      │                 │
#      │       ─┴─       │
#      │                 │
#      └───┐         ┌───┘
#          │         │
#          │         │
#          │         │
#          │         └──────────────┐
#          │                        │
#          │                        ├─┐
#          │                        ┌─┘
#          │                        │
#          └─┐  ┐  ┌───────┬──┐  ┌──┘
#            │ ─┤ ─┤       │ ─┤ ─┤
#            └──┴──┘       └──┴──┘
"""


import os
import pickle
import pandas as pd
from torch.utils.data import DataLoader

from Model1 import *
from sklearn import metrics
from torch.autograd import Variable
import warnings
warnings.filterwarnings('ignore')


Model_path = './Model/'
Dataset_path = './data/'


def evaluate(model, data_loader):

    model.eval()

    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []
    pred_dict = {}

    for data in data_loader:

        with torch.no_grad():
            sequence_names, seq, labels, node_features, adj = data

            if torch.cuda.is_available():
                node_features = Variable(node_features.cuda())
                adj = Variable(adj.cuda())
                y_true = Variable(labels.cuda())
            else:
                node_features = Variable(node_features)
                adj = Variable(adj)
                y_true = Variable(labels)

            node_features = torch.squeeze(node_features)
            adj = torch.squeeze(adj)
            y_true = torch.squeeze(y_true)

            y_pred = model(node_features, adj)
            y_true = torch.tensor(y_true, dtype=torch.long)
            loss = model.criterion(y_pred, y_true)
            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred)
            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
            valid_pred += [pred[1] for pred in y_pred]
            valid_true += list(y_true)
            pred_dict[sequence_names[0]] = [pred[1] for pred in y_pred]

            epoch_loss += loss.item()
            n += 1

    epoch_loss_avg = epoch_loss / n

    return epoch_loss_avg, valid_true, valid_pred, pred_dict

def analysis(y_true, y_pred, best_threshold = None):

    if best_threshold == None:
        best_f1 = 0
        best_threshold = 0
        for threshold in range(0, 20):

            threshold = threshold / 20
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            f1 = metrics.f1_score(binary_true, binary_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]

    binary_true = y_true

    # binary evaluate
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    AUC = metrics.roc_auc_score(binary_true, y_pred)
    precisions, recalls, thresholds = metrics.precision_recall_curve(binary_true, y_pred)
    AUPRC = metrics.auc(recalls, precisions)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)

    results = {
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'AUC': AUC,
        'AUPRC': AUPRC,
        'mcc': mcc,
        'threshold': best_threshold
    }

    return results

def filter_missing_samples(df):
    valid_rows = []
    for _, row in df.iterrows():
        sid = row["ID"]
        adj_path = f"./data/scPDB/adjacency_matrix14/{sid}.npy"
        feat_path = f"./data/scPDB/node_features/{sid}.npy"
        if os.path.exists(adj_path) and os.path.exists(feat_path):
            valid_rows.append(row)
        else:
            print(f"跳过缺失样本: {sid}")
    return pd.DataFrame(valid_rows)

def test(test_dataframe):
    test_dataframe = filter_missing_samples(test_dataframe)
    test_loader = DataLoader(dataset=ProDataset(test_dataframe), batch_size=1, shuffle=True, num_workers=4)

    for model_name in sorted(os.listdir(Model_path)):
        print(model_name)
        model = GraphPLBR(LAYER, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT, LAMBDA, ALPHA, VARIANT)
        if torch.cuda.is_available():
            model.cuda()
        model.load_state_dict(torch.load(Model_path + model_name, map_location='cuda:0'),strict=False)

        epoch_loss_test_avg, test_true, test_pred, pred_dict = evaluate(model, test_loader)

        result_test = analysis(test_true, test_pred)

        print("========== Evaluate Test set ==========")
        print("Test loss: ", epoch_loss_test_avg)
        print("Test binary acc: ", result_test['binary_acc'])
        print("Test precision:", result_test['precision'])
        print("Test recall: ", result_test['recall'])
        print("Test f1: ", result_test['f1'])
        print("Test AUC: ", result_test['AUC'])
        print("Test AUPRC: ", result_test['AUPRC'])
        print("Test mcc: ", result_test['mcc'])
        print("Threshold: ", result_test['threshold'])
        print()

        with open('./records/test.txt', 'a') as f:
            f.write("========== {} Evaluate Test set ========== \n".format(model_name[:5].lower()))
            f.write("Test loss: {} \n".format(epoch_loss_test_avg))
            f.write("Test binary acc: {} \n".format(result_test['binary_acc']))
            f.write("Test precision: {} \n".format(result_test['precision']))
            f.write("Test recall: {} \n".format(result_test['recall']))
            f.write("Test f1: {} \n".format(result_test['f1']))
            f.write("Test AUC: {} \n".format(result_test['AUC']))
            f.write("Test AUPRC: {} \n".format(result_test['AUPRC']))
            f.write("Test mcc: {} \n".format(result_test['mcc']))
            f.write("Threshold: {} \n".format(result_test['threshold']))
            f.write('\n')

def test_one_dataset(dataset):
    IDs, sequences, labels = [], [], []
    for ID in dataset:
        IDs.append(ID)
        item = dataset[ID]
        sequences.append(item[0])
        labels.append(item[1])
    test_dic = {"ID": IDs, "sequence": sequences, "label": labels}
    test_dataframe = pd.DataFrame(test_dic)
    print("test_dataframe:",test_dataframe)
    test(test_dataframe)

def main():

    with open(Dataset_path + 'test_data_more.pkl','rb') as f:
        Test = pickle.load(f)

    print('Evaluate GraphPLBS on testset')
    test_one_dataset(Test)

if __name__ == "__main__":
    main()

'''
{
    '1izh_1': 
    [
        'PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNFPQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF', 
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
}
'''