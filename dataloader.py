from sklearn.model_selection import KFold

from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import re
import torch
from tqdm import tqdm
from datasets import load_from_disk
import random
class BioDataset(Dataset):
    def __init__(self, args):
        super(BioDataset, self).__init__()
        print('loading dataset...')
        self.dataset = load_from_disk(args.f_path)

        # for demonstration, just sample a subset of samples
        self.subset_size = args.subset_size
        random_subset_indices = random.sample(range(len(self.dataset)), self.subset_size)
        self.subdataset = self.dataset.select(random_subset_indices)


        self.tokens = self.subdataset['input_ids']
        self.labels = self.subdataset['label']

        self.length = len(self.tokens)
        print('number of sequence:',self.length)
    def __getitem__(self, item):
        return self.tokens[item], self.labels[item]

    def __len__(self):
        return self.length


def bio_collate_fn(batch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    input_ids = []
    attention_mask = []
    token_type_ids = []
    labels = []

    for ist in batch:
        id = ist[0]['input_ids']
        mk = ist[0]['attention_mask']
        ty = ist[0]['token_type_ids']
        while len(id)<512:
            id.append(0)
        while len(mk)<512:
            mk.append(0)
        while len(ty)<512:
            ty.append(0)
        input_ids.append(torch.Tensor(id[:512]))
        attention_mask.append(torch.Tensor(mk[:512]))
        token_type_ids.append(torch.Tensor(ty[:512]))

        labels.append(int(ist[1]))

    input_ids = torch.stack(input_ids).long()
    attention_mask = torch.stack(attention_mask).long()
    token_type_ids = torch.stack(token_type_ids).long()

    new_list = [[0, 0] for _ in range(len(labels))]
    for i in range(len(labels)):
        if labels[i] == 1:
            new_list[i][1] = 1
        else:
            new_list[i][0] = 1
    labels = torch.Tensor(new_list)


    # knowledge graph
    num_channels = 1
    node_map = [input_ids.float() for _ in range(num_channels)]  # 输入特征
    node_map = torch.stack(node_map)
    adjs = [torch.ones(len(batch), len(batch)) for _ in range(num_channels)]
    adjs = torch.stack(adjs)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    token_type_ids = token_type_ids.to(device)
    labels = labels.to(device)
    node_map = node_map.to(device)
    adjs =adjs.to(device)


    return input_ids, attention_mask, token_type_ids, labels, node_map, adjs

def KfoldDataset(args):
    biodataset = BioDataset(args)
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    trdt_list = []
    tedt_list = []

    for train_indices, val_indices in kf.split(biodataset):
        train_dataset = torch.utils.data.Subset(biodataset, train_indices)
        test_dataset = torch.utils.data.Subset(biodataset, val_indices)

        trdt_list.append(train_dataset)
        tedt_list.append(test_dataset)


    return trdt_list, tedt_list




def dataloader(current_fold,train_list,test_list,tr_bs,te_bs):
    train_data_loader = DataLoader(dataset=train_list[current_fold], batch_size=tr_bs, shuffle=True,
                                   collate_fn=bio_collate_fn)
    test_data_loader = DataLoader(dataset=test_list[current_fold], batch_size=te_bs, shuffle=True,
                                  collate_fn=bio_collate_fn)

    return train_data_loader,test_data_loader





