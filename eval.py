import torch
import ast
from model import AntibodyFormer
import os
from torch.utils.data import Dataset, DataLoader


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AntibodyFormer()

model.load_state_dict(torch.load('./model_save/model.pickle'))
model.eval()
model = model.to(device)
root_path = './data_vdj'
type_path = ['/AS2', '/HC2', '/SM(MD)2', '/SM(SDR)2']



class BioDataset(Dataset):
    def __init__(self,samples_input_ids,samples_token_type_ids,samples_attention_mask):
        super(BioDataset, self).__init__()
        self.input_ids = samples_input_ids
        self.token_type_ids = samples_token_type_ids
        self.attention_mask = samples_attention_mask


        self.length = len(self.input_ids)

    def __getitem__(self, item):
        return self.input_ids[item], self.token_type_ids[item], self.attention_mask[item]

    def __len__(self):
        return self.length


def bio_collate_fn(batch):
    '''
    Data processing includes three essential inputs of Bert
    :param bag:
    :return: input_ids, attention_mask, token_type_ids
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    input_ids = []
    attention_mask = []
    token_type_ids = []

    for ist in batch:
        id = ist[0]
        mk = ist[2]
        ty = ist[1]
        while len(id)<512:
            id.append(0)
        while len(mk)<512:
            mk.append(0)
        while len(ty)<512:
            ty.append(0)
        input_ids.append(torch.Tensor(id))
        attention_mask.append(torch.Tensor(mk))
        token_type_ids.append(torch.Tensor(ty))

    input_ids = torch.stack(input_ids).long()
    attention_mask = torch.stack(attention_mask).long()
    token_type_ids = torch.stack(token_type_ids).long()

    # knowledge graph
    num_channels = 1
    node_map = [input_ids.float() for _ in range(num_channels)]
    node_map = torch.stack(node_map)
    adjs = [torch.ones(len(batch), len(batch)) for _ in range(num_channels)]
    adjs = torch.stack(adjs)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    token_type_ids = token_type_ids.to(device)

    node_map = node_map.to(device)
    adjs =adjs.to(device)


    return input_ids, attention_mask, token_type_ids, node_map, adjs




def eval(root_path, type_path):
    dir_list = []

    with torch.no_grad():
        for t in type_path:
            dir_list.append(root_path + t)

        for d in dir_list:  # type level
            f_list = os.listdir(d)
            for f in f_list:  # patient level
                print(f)
                l = d+'/'+f
                with open(l, 'r') as file:
                    samples_input_ids = []
                    samples_token_type_ids = []
                    samples_attention_mask = []
                    for line in file:
                        data_dict = ast.literal_eval(line)
                        samples_input_ids.append(data_dict['input_ids'])
                        samples_token_type_ids.append(data_dict['token_type_ids'])
                        samples_attention_mask.append(data_dict['attention_mask'])

                    data_set = BioDataset(samples_input_ids, samples_token_type_ids, samples_attention_mask)
                    data_loader = DataLoader(dataset=data_set, batch_size=128, shuffle=False,
                                                   collate_fn=bio_collate_fn)

                    sv_pth = d+'_results/'+f[:-4] + '_output_pobs.txt'
                    with open(sv_pth, 'w') as file:
                        for b in data_loader:  # sample
                            input_ids, attention_mask, token_type_ids, node_map, adjs = b  # batch_size*seq_len

                            output_pobs = model(input_ids, attention_mask, token_type_ids, node_map, adjs)
                            sigmoid_output = torch.sigmoid(output_pobs)
                            sigmoid_output = sigmoid_output.tolist()
                            rounded_output = [[round(num, 5) for num in inner_list] for inner_list in sigmoid_output]
                            for line in rounded_output:
                                file.write(str(line) + '\n')





eval(root_path,type_path)


