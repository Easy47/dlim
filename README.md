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

Describe files of this repos:
Jupyter, py, directories

## Datasets

Add links and some explanations