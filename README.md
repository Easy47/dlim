# DLIM: Image indexing
## Authors
- Mathieu Zimmermann
- Nicolas Romano
- Gauthier Lombard
- Ancelin Bouchet

## Installation

Please install python packages with pip:
```sh
pip install -r requirements.txt
```
This project uses tensorflow 2.9.2.

## Usage

We have two main scripts: main.py and main_flask.
*main.py* is used to train, save and measure performance of baseline and Proxy-Anchor models on the Paris6k dataset.
```
options:
  -h, --help            show this help message and exit
  -m {baseline,proxy_anchor}, --model {baseline,proxy_anchor}
                        CBIR models
  --dist {cosine,euclidean}
                        Distance metrics used to measure similarities

# Example of usage:
python src/main.py --dist "euclidean" -m "baseline"
python src/main.py --dist "cosine" -m "baseline"
```
*main_flask.py* is used to run Flask web application.
```
# Usage
flask --app src/main_flask run
```

## File Structure

The __model__ directory contains the saved models so we don't have to retrain them at each execution.

The __src__ directory contains the python scripts of our project. They are as follows :  

&nbsp;&nbsp;&nbsp;&nbsp;. __main.py__ : script to start the program on the shell. Compute the entire pipeline and print the mAp at the end.

&nbsp;&nbsp;&nbsp;&nbsp;. __main_flask.py__ : script to start the program using flask.

&nbsp;&nbsp;&nbsp;&nbsp;. __resnet50.py__ : the baseline implmentation.

&nbsp;&nbsp;&nbsp;&nbsp;. __resnet_50_proxy_anchor.py__ : the proxy anchor implmentation.

&nbsp;&nbsp;&nbsp;&nbsp;. __utils.py__ : various utility functions to get paths of jpgs in datasets, extracting a test set, etc.

The __static__ directory contains our lebellisation for the datasets.

The __template__ directory contains our template for the flask app.

The file __notebook_tripletloss.ipynb__ contains the resnet baseline, implementations and various tests about the triplet loss (batch all and batch hard strategies, data augmentation...).

## Datasets

### INRIA Holidays

https://lear.inrialpes.fr/~jegou/data.php

The INIRIA Holidays dataset consists of 1491 images in total: 500 queries and 991 corresponding relevant images.
 
### The Paris Dataset

https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/

The Paris Dataset consists of 6412 images collected from Flickr by searching for particular Paris landmarks.

## Tasks division

__Dataset Pesearch__ : Gauthier, Mathieu, Nicolas, Ancelin

__Data Preprocessing__ : Mathieu, Nicolas

__Benchmark__ : Gauthier, Mathieu, Nicolas

__Refactor__ : Gauthier, Mathieu

__Resnet__ : Gauthier, Nicolas

__Triplets__ : Gauthier, Mathieu, Nicolas

__Online Mining__ : Nicolas

__Proxy Anchor__ : Mathieu

__Flask__ : Mathieu
