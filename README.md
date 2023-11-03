# EASY

Source code for "EASY Framework Revisited: Achieving Faster and More Accurate Entity Alignment".

## Installation

To run our code, first install required packages. Then run preprocess

    pip install -r requirements.txt
    sh preprocess.sh

## Run 

Run on all dataset with default settings

First get MNEAP results.

    python neap.py --pair zh_en --device cpu --init_type "MNEAP-L" --do_sinkhorn

Then get ASRS results.

    python main.py --pair zh_en --device cpu --strategy "ASRS-TFIDF"


## Acknowledgement

We use the code and datasets of 
[EASY](https://github.com/gawkAtCode/Revisited-EASY),
