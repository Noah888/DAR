# Enhancing Recipe Retrieval with Foundation Models


Official implementation of our ECCV2024 paper:

**[Enhancing Recipe Retrieval with Foundation Models: A Data Augmentation Perspective ](https://arxiv.org/abs/2312.04763)**

This paper proposes a new perspective on data augmentation using the Foundation Model (i.e., llama2 and SAM) to better learn multimodal representations in the common embedding space for the task of cross-modal recipe retrieval.

![Project Banner](figs.jpg)

---
## The main code has been uploaded first, the data as well as guidance will be updated soon.

## Installation

To install the required packages, please follow these steps:

```bash
# Clone the repository
git clone https://github.com/Noah888/DAR.git

# Create a virtual environment (Python 3.8 or above)
conda create --name your_env_name python=3.9

# Activate the conda environment
conda activate your_env_nam

# Install dependencies
pip install -r requirements.txt
```




## Usage
To reproduce the results,  Download Recipe1M [dataset](http://wednesday.csail.mit.edu/temporal/release/) and Generate enhanced data (to be uploaded in the future). Place the data in the ```DATASET_PATH``` directory with the following structure:
```bash
DATASET_PATH/
│
├── train/
│   ├── ...
├── val/
│   ├── ...
└── test/
│   ├── ...
├── segment/
│    ├── train/...
│    ├── val/...
│    ├── test/...
└── layer1.json
└── layer2.json





