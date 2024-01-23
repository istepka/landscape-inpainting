# In/Outpainting landscapes
## Introduction
---
## Installation
1. Clone the repository
```bash
git clone git@github.com:istepka/im-outpainting.git
```
2. Install the requirements
```bash
pip install -r requirements.txt
```
3. Download the data from the following url: https://1drv.ms/u/c/35ddce87939617c8/EWyelnH8qlJEgFNjAX_z3EABZiNsaDIGrOS8du5V52DSXA?e=mkizPd and uzip it into the `data/raw` folder.

## Usage
1. Prepare the data
```bash
python src/preprocess.py [DEFAULT OPTIONS: --data_dir data/raw, --out_dir data/processed, --num_workers CPU_COUNT, --size 256]
```
2. Train the model
```bash
python src/train.py [OPTIONS OPTIONS: --run_eval TRUE, --device GPU, --experiment_name JUST_GAN]
```
3. Evaluation
```bash
python src/eval.py [OPTIONS OPTIONS: --device GPU, --experiment_name JUST_GAN]
```
4. Experiment review
```bash
mlflow ui
```

## Data sources 
The dataset is comprised of 16k images of landscapes. They were collected from the following sources:
1. Landscape Pictures (~4k images)
https://www.kaggle.com/datasets/arnaud58/landscape-pictures?rvi=1

1. Landscape Recognition Image Dataset (~12k images)
https://www.kaggle.com/datasets/utkarshsaxenadn/landscape-recognition-image-dataset-12k-images


## Architectures
*JUST_GAN*: A simplest possible GAN architecture with a generator and a discriminator. The generator is a downsampling FCN 

## License
MIT License

## References
---