from collections import Counter
import csv
import pandas as pd
import os


directory = '/home/exouser/ai4bio/data/train_data/'


def cdr1_light_process(directory,file_name):
    with open(file_name, 'w') as file:
        file_list = os.listdir(directory)
        cdr1_light = []


        for filename in file_list:
            df = pd.read_csv(directory + filename, header=1)
            temp_subseq = df['cdr1_aa_light']
            cdr1_light.extend(temp_subseq)

        counts = Counter(cdr1_light)
        most_common_item = counts.most_common()

        for i in most_common_item:
            file.write(str(i))
            file.write('\n')





def cdr2_light_process(directory, file_name):
    with open(file_name, 'w') as file:
        file_list = os.listdir(directory)
        cdr2_light = []

        for filename in file_list:
            df = pd.read_csv(directory + filename, header=1)
            temp_subseq = df['cdr2_aa_light']
            cdr2_light.extend(temp_subseq)

        counts = Counter(cdr2_light)
        most_common_item = counts.most_common()

        for i in most_common_item:
            file.write(str(i))
            file.write('\n')

def cdr3_light_process(directory, file_name):
    with open(file_name, 'w') as file:
        file_list = os.listdir(directory)
        cdr3_light = []

        for filename in file_list:
            df = pd.read_csv(directory + filename, header=1)
            temp_subseq = df['cdr3_aa_light']
            cdr3_light.extend(temp_subseq)

        counts = Counter(cdr3_light)
        most_common_item = counts.most_common()

        for i in most_common_item:
            file.write(str(i))
            file.write('\n')

def cdr1_heavy_process(directory, file_name):
    with open(file_name, 'w') as file:
        file_list = os.listdir(directory)
        cdr1_heavy = []

        for filename in file_list:
            df = pd.read_csv(directory + filename, header=1)
            temp_subseq = df['cdr1_aa_heavy']
            cdr1_heavy.extend(temp_subseq)

        counts = Counter(cdr1_heavy)
        most_common_item = counts.most_common()

        for i in most_common_item:
            file.write(str(i))
            file.write('\n')

def cdr2_heavy_process(directory, file_name):
    with open(file_name, 'w') as file:
        file_list = os.listdir(directory)
        cdr2_heavy = []

        for filename in file_list:
            df = pd.read_csv(directory + filename, header=1)
            temp_subseq = df['cdr2_aa_heavy']
            cdr2_heavy.extend(temp_subseq)

        counts = Counter(cdr2_heavy)
        most_common_item = counts.most_common()

        for i in most_common_item:
            file.write(str(i))
            file.write('\n')

def cdr3_heavy_process(directory, file_name):
    with open(file_name, 'w') as file:
        file_list = os.listdir(directory)
        cdr3_heavy = []


        for filename in file_list:
            df = pd.read_csv(directory + filename, header=1)
            temp_subseq = df['cdr3_aa_heavy']
            cdr3_heavy.extend(temp_subseq)

        counts = Counter(cdr3_heavy)
        most_common_item = counts.most_common()

        for i in most_common_item:
            file.write(str(i))
            file.write('\n')



cdr1_light_process(directory,'cdr1L.txt')
cdr2_light_process(directory,'cdr2L.txt')
cdr3_light_process(directory,'cdr3L.txt')
cdr1_heavy_process(directory,'cdr1H.txt')
cdr2_heavy_process(directory,'cdr2H.txt')
cdr3_heavy_process(directory,'cdr3H.txt')




