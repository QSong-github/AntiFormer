from sklearn.model_selection import KFold
import os

from transformers import BertTokenizer
from datasets import Dataset
from datasets import load_from_disk


root_path = 'dataset_aa.txt'


def Freader(root_path):
    tokenizer = BertTokenizer.from_pretrained("./protbert", do_lower_case=False)
    dt = {}
    f_list = os.listdir(root_path)
    input_ids = []
    labels = []
    for f in f_list:
        print(f)
        l = root_path+'/'+f
        with open(l, 'r') as file:
            for line in f.readlines():
                input_ids_temp = line[0]
                label_temp = line[1]

                strings = input_ids_temp
                spaced_string = ' '.join([char for char in strings])

                token = tokenizer(spaced_string, truncation=True, max_length=512, padding=True)
                input_ids.append(token)
                labels.append(int(label_temp))

    # save
    dt['input_ids'] = input_ids
    dt['label'] = labels

    my_dataset = Dataset.from_dict(dt)
    my_dataset.save_to_disk('./dt')

Freader(root_path)
