# AntiFormer

This code is prepared for "AntiFormer: graph enhanced large language model for binding affinity prediction".

## Overview

<embed src="./flowchart.pdf" type="application/pdf" width="600" height="400">

## Installation
Download Antiformer:
```git clone https://github.com/wq2581/Antiformer```

Install Environment:
```pip install -r requirements.txt```

Our code is all based on python. In order to install dependencies please enter the project path ```./AntiFormer``` and activate your python environment.


## Running

   The codes for dataset creating are stored in the ```./AntiFormer/data``` directory.
   
   (1) Run ```python data_download.py``` to download the required data.
   
   (2) Run ```python data_process.py``` for frequency statistics. Of course, we also provide the results of frequency statistics saved as txt files in this directory.
   
   (3) Run ```python dataset_making.py``` to build the dataset.
   
   (4) Run ```python dt_rebuild.py``` to tokenize the input sequence and save it as a file of type Dataset saved in the ```./AntiFormer/dt``` path.
  
   We also provide partially processed data (2000 sequences) as demo, located under the ```./AntiFormer/subdt``` path.
   
   (5) Run ```python main.py``` to get the results.

   However，if you have processed all the data, you can replace the ```./subdt``` path with your data path for training. And be careful to change the hyperparameters in the ```main.py``` to suit your hardware and target.

   
