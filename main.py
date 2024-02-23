import torch
from dataloader import KfoldDataset, dataloader
import argparse
from tqdm import tqdm
from model import AntibodyFormer
from transformers import BertTokenizer
from torch import optim
from tool import Accuracy_score, F1_score, AUROC_score, Recall_score, Precision_score
import pickle
def prepare():
    parser = argparse.ArgumentParser(description='AI4Bio')

    # training parameters
    parser.add_argument('--ep_num', type=int, default=1, help='epoch number of training')
    parser.add_argument('--train_batch_size', type=int, default=64, help='')
    parser.add_argument('--test_batch_size', type=int, default=128, help='')
    parser.add_argument('--f_path', type=str, default='./subdt', help='')
    parser.add_argument('--lr', type=int, default=0.0001, help='')
    parser.add_argument('--folds', type=int, default=5, help='')
    parser.add_argument('--folds_list', type=list, default=[1, 0, 2, 3, 4], help='must match with folds')
    parser.add_argument('--subset_size', type=int, default=200, help='')


    # model parameters
    parser.add_argument('--hidden_size', type=int, default=256, help='')
    parser.add_argument('--num_hidden_layers', type=int, default=12, help='')
    parser.add_argument('--num_attention_heads', type=int, default=8, help='')
    parser.add_argument('--intermediate_size', type=int, default=2048, help='')
    parser.add_argument('--max_position_embeddings', type=int, default=512, help='')
    parser.add_argument('--hidden_dropout_prob', type=int, default=0.1, help='')

    args = parser.parse_args()


    trdt_list, tedt_list = KfoldDataset(args)


    return args, trdt_list, tedt_list



def run():
    seed = 24
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args, trdt_list, tedt_list = prepare()

    for f in args.folds_list:

        model = AntibodyFormer(args)

        loss_function = torch.nn.BCEWithLogitsLoss()
        model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = optim.AdamW(model_parameters, lr=args.lr)

        model = model.to(device)

        train_data_loader, test_data_loader = dataloader(current_fold=f, train_list=trdt_list, test_list=tedt_list,
                                                         tr_bs=args.train_batch_size, te_bs=args.test_batch_size)

        for epoch in range(args.ep_num):
            loss_sum = 0
            Acc = []
            F1 = []
            AUROC = []
            Precision = []
            Recall = []


            with tqdm(train_data_loader, ncols=80, position=0, leave=True) as batches:
                for b in batches:  # sample
                    input_ids, attention_mask, token_type_ids, labels, node_map, adjs = b  # batch_size*seq_len
                    if torch.all(torch.logical_or(torch.all(labels == torch.tensor([1, 0]).to(device)),
                                                  torch.all(labels == torch.tensor([0, 1]).to(device)))) == True:
                        continue
                    pred = model(input_ids, attention_mask, token_type_ids, node_map, adjs)
                    loss = loss_function(pred, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_sum = loss_sum + loss
                    acc = Accuracy_score(pred, labels)
                    f1 = F1_score(pred, labels)
                    aur = AUROC_score(pred, labels)
                    pre = Precision_score(pred, labels)
                    rcl = Recall_score(pred, labels)

                    Acc.append(acc)
                    F1.append(f1)
                    AUROC.append(aur)
                    Precision.append(pre)
                    Recall.append(rcl)
            print('Training epoch:', epoch, 'Current_fold:', f, 'loss:', loss_sum, 'Accuracy:', sum(Acc) / len(Acc), 'AUROC:',sum(AUROC) / len(AUROC),
                  'Precision:', sum(Precision) / len(Precision), 'Recall:', sum(Recall) / len(Recall), 'F1:',sum(F1) / len(F1))


            loss_sum = 0
            Acc = []
            F1 = []
            AUROC = []
            Precision = []
            Recall = []

            with torch.no_grad():
                with tqdm(test_data_loader, ncols=80, position=0, leave=True) as batches:
                    for b in batches:  # sample
                        input_ids, attention_mask, token_type_ids, labels, node_map, adjs = b  # batch_size*seq_len
                        if torch.all(torch.logical_or(torch.all(labels == torch.tensor([1, 0]).to(device)),
                                                      torch.all(labels == torch.tensor([0, 1]).to(device)))) == True:
                            continue
                        pred = model(input_ids, attention_mask, token_type_ids, node_map, adjs)
                        loss = loss_function(pred, labels)
                        loss_sum = loss_sum + loss

                        acc = Accuracy_score(pred, labels)
                        f1 = F1_score(pred, labels)
                        aur = AUROC_score(pred, labels)
                        pre = Precision_score(pred, labels)
                        rcl = Recall_score(pred, labels)

                        Acc.append(acc)
                        F1.append(f1)
                        AUROC.append(aur)
                        Precision.append(pre)
                        Recall.append(rcl)
                    print('Testing epoch:', epoch, 'Current_fold:', f, 'loss:', loss_sum, 'Accuracy:', sum(Acc) / len(Acc), 'AUROC:',sum(AUROC) / len(AUROC),
                          'Precision:', sum(Precision) / len(Precision), 'Recall:', sum(Recall) / len(Recall),'F1:', sum(F1) / len(F1))

            #torch.save(model.state_dict(), './model_save/model.ckpt')


if __name__ == '__main__':
    run()


