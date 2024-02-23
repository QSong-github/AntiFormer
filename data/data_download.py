import pandas as pd
import urllib.request
import gzip
import os
import urllib.request

manifest = pd.read_csv('manifest_230324.csv')

# Randomly sample 3% of the dataset.
# manifest = manifest.sample(frac=0.001, random_state=42)

f2study = {r.filename:r.study for r in manifest.to_records()}

filenames = manifest.filename.values




def download_file(url, save_path, file):
    try:
        urllib.request.urlretrieve(url, save_path)
        input_gz_file = save_path
        output_file = input_gz_file[:-3]
        with gzip.open(input_gz_file, 'rb') as f_in, open(output_file, 'wb') as f_out:
            f_out.write(f_in.read())
        os.remove(save_path)
    except Exception as e:
        print("cannot download:", e, url)
        file.write(url)
        file.write('\n')




sample_nums = len(filenames)
cnt = 0

file_name = 'error_log.txt'

with open(file_name, 'w') as file:
    for i in filenames:
        study = f2study[i]
        filename = i
        file_url = 'http://opig.stats.ox.ac.uk/webapps/ngsdb/paired/' + study + '/csv/' + filename + '.csv.gz'
        save_file_path = 'dataset_train/' + '_' + study + '_' + filename + '.csv.gz'
        download_file(file_url, save_file_path, file)
        cnt = cnt + 1
        print(cnt,'/',sample_nums)






