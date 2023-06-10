# Setting the Standard: Best Practices for Benchmarking Net-Load Forecasting Approaches in Low-Voltage Distribution Substations

This repository is the official implementation of [Setting the Standard: Best Practices for Benchmarking Net-Load Forecasting Approaches in Low-Voltage Distribution Substationse](). 



## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training
- To begin training the model(s), start by optimizing the hyperparameters of the deep-learning-based forecasting model. This can be achieved by executing the code provided in the [SmartGridcomHyper](https://github.com/feelab-info/net-load-best-practices/blob/main/notebook/SmartGridcomHyper-params.ipynb).

- Once the hyperparameters have been optimized, proceed to train the model for all experiments. This can be done by running the code available in the [SmartGridcomExperiment](https://github.com/feelab-info/net-load-best-practices/blob/main/notebook/SmartGridcomExperiment.ipynb) notebook.


## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
