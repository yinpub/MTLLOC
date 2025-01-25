# MTLLOC 
## Image-to-Image Prediction

### Setup environment

Create the conda environment and install torch
```bash
conda create -n mtl python=3.9.7
conda activate mtl
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

Install the repo:
```bash
git clone xxxxxxxxxxxx
cd LOC
pip install -e .
```

### Download dataset

We follow the [MTAN](https://github.com/lorenmt/mtan) paper. The datasets could be downloaded from [NYU-v2](https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa?dl=0) and [CityScapes](https://www.dropbox.com/sh/gaw6vh6qusoyms6/AADwWi0Tp3E3M4B2xzeGlsEna?dl=0). To download the CelebA dataset, please refer to this [link](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg). The dataset should be put under ```experiments/EXP_NAME/dataset/``` folder where ```EXP_NAME``` is chosen from ```nyuv2, cityscapes, celeba```. Note that ```quantum_chemistry``` will download the data automatically.

The file hierarchy should look like
```
LOC
 └─ experiments
     └─ utils.py                     (for argument parsing)
     └─ nyuv2
         └─ dataset                  (the dataset folder containing the MTL data)
         └─ trainer.py               (the main file to run the training)
         └─ run.sh                   (the command to reproduce LOC's results)
     └─ cityscapes
         └─ dataset                  (the dataset folder containing the MTL data)
         └─ trainer.py               (the main file to run the training)
         └─ run.sh                   (the command to reproduce LOC's results)
     └─ quantum_chemistry
         └─ dataset                  (the dataset folder containing the MTL data)
         └─ trainer.py               (the main file to run the training)
         └─ run.sh                   (the command to reproduce LOC's results)
     └─ celeba
         └─ dataset                  (the dataset folder containing the MTL data)
         └─ trainer.py               (the main file to run the training)
         └─ run.sh                   (the command to reproduce LOC's results)
 └─ methods
     └─ weight_methods.py            (the different MTL optimizers)
     └─ attention.py                 (the attention network)
```

### Run experiment

To run experiments, go to the relevant folder with name ```EXP_NAME```
```bash
cd experiment/EXP_NAME
bash run.sh
```
You can check the ```run.sh``` for details about training with **LOC**.


