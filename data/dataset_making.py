import multiprocessing

import pandas as pd
import os
from tqdm import tqdm






directory = '/home/exouser/ai4bio/data/data1/'




def dataset_making_stage2(data):
    list = ['cdr1H.txt', 'cdr2H.txt', 'cdr3H.txt', 'cdr1L.txt', 'cdr2L.txt', 'cdr3L.txt']
    mix_seq = data[0]
    cdr1_heavy = data[1]
    cdr2_heavy = data[2]
    cdr3_heavy = data[3]
    cdr1_light = data[4]
    cdr2_light = data[5]
    cdr3_light = data[6]

    freq_sum = []
    with open(list[0], 'r') as f:
        for line in f:
            try:
                temp_tuple = eval(line)
                sub_seq, freq = temp_tuple
                if cdr1_heavy == sub_seq:
                    freq_sum.append(int(freq))
            except:
                pass
                # print(cdr,line)
    with open(list[1], 'r') as f:
        for line in f:
            try:
                temp_tuple = eval(line)
                sub_seq, freq = temp_tuple
                if cdr2_heavy == sub_seq:
                    freq_sum.append(int(freq))
            except:
                pass
                # print(cdr,line)
    with open(list[2], 'r') as f:
        for line in f:
            try:
                temp_tuple = eval(line)
                sub_seq, freq = temp_tuple
                if cdr3_heavy == sub_seq:
                    freq_sum.append(int(freq))
            except:
                pass
                # print(cdr,line)
    with open(list[3], 'r') as f:
        for line in f:  # every line
            try:
                temp_tuple = eval(line)
                sub_seq, freq = temp_tuple
                if cdr1_light == sub_seq:
                    freq_sum.append(int(freq))
            except:
                pass
                # print(cdr,line)
    with open(list[4], 'r') as f:
        for line in f:
            try:
                temp_tuple = eval(line)
                sub_seq, freq = temp_tuple
                if cdr2_light == sub_seq:
                    freq_sum.append(int(freq))
            except:
                pass
                # print(cdr,line)
    with open(list[5], 'r') as f:
        for line in f:
            try:
                temp_tuple = eval(line)
                sub_seq, freq = temp_tuple
                if cdr3_light == sub_seq:
                    freq_sum.append(int(freq))
            except:
                pass
                # print(cdr,line)
    total = sum(freq_sum)
    info = []

    # set threshold
    if total > 288888:
        info.append(mix_seq + ' ')
        info.append('1')
        return info
    if total < 28888:
        info.append(mix_seq + ' ')
        info.append('0')
        return info




def wrapper(data):
    return dataset_making_stage2(data)



def dataset_making_stage1(directory):
    file_list = os.listdir(directory)
    data = []
    #file_list = ['_Eccles_2020_SRR10358524_paired.csv']
    for file_name in file_list:
        mix_seq = []
        cdr1_heavy = []
        cdr2_heavy = []
        cdr3_heavy = []
        cdr1_light = []
        cdr2_light = []
        cdr3_light = []
        h = []
        l = []

        with open(directory + file_name, 'r') as file:
            df = pd.read_csv(file, header=1)
            temp_h = df['sequence_alignment_aa_heavy']
            temp_l = df['sequence_alignment_aa_light']
            h.extend(temp_h)
            l.extend(temp_l)
            temp_mix_seq = [str1 + str2 for str1, str2 in zip(h, l)]
            mix_seq.extend(temp_mix_seq)

            temp_cdr1_h = df['cdr1_aa_heavy']
            temp_cdr2_h = df['cdr2_aa_heavy']
            temp_cdr3_h = df['cdr3_aa_heavy']
            temp_cdr1_l = df['cdr1_aa_light']
            temp_cdr2_l = df['cdr2_aa_light']
            temp_cdr3_l = df['cdr3_aa_light']
            cdr1_heavy.extend(temp_cdr1_h)
            cdr2_heavy.extend(temp_cdr2_h)
            cdr3_heavy.extend(temp_cdr3_h)
            cdr1_light.extend(temp_cdr1_l)
            cdr2_light.extend(temp_cdr2_l)
            cdr3_light.extend(temp_cdr3_l)

            for j in range(len(temp_cdr1_l)):
                temp = []
                temp.append(mix_seq[j])
                temp.append(temp_cdr1_h[j])
                temp.append(temp_cdr2_h[j])
                temp.append(temp_cdr3_h[j])
                temp.append(temp_cdr1_l[j])
                temp.append(temp_cdr2_l[j])
                temp.append(temp_cdr3_l[j])
                data.append(temp)

    file_name = 'dataset_aa.txt'
    with open(file_name, 'w') as file:
        with multiprocessing.Pool(processes=32) as pool:
            L = list(tqdm(pool.imap(wrapper, data), total=len(data)))
        for l in L:
            l = str(l)
            if l !='None':
                file.write(l)







if __name__ == "__main__":
    dataset_making_stage1(directory)

