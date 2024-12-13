# Large-Small-Language-Model

This repository contains a small text-generation project based on a Transformer model, designed as a practical exercise for the **Deep Learning course (14x050)** at the **University of Geneva**. The project demonstrates the implementation, training, and utilization of a Transformer-based language model for generating text based on the [Tiny Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).

## Contents

### 1. `model_training.ipynb`
- A Jupyter Notebook showcasing the full training pipeline.
- Includes training logs, loss values, and generated text 
- Requires `input.txt` (Tiny Shakespeare dataset)

### 2. `training.py`
- The standalone Python script for training the model.
- Contains the same logic as `model_training.ipynb` but designed for running directly from the command line.

### 3. `trained.py`
- A script for loading the pre-trained model (`model.pt`) and generating text.
- Ideal for users who want to test the model without retraining.
- Requires `input.txt` (Tiny Shakespeare dataset) and `model.pt`, which can be downlaoded from [Github Release](https://github.com/StepanStepanovic/Large-Small-Language-Model/releases/tag/v1.0).


## Dataset
The model is trained on the **Tiny Shakespeare** dataset, which is a simple and compact text dataset used frequently in NLP experiments. It is included in the repository.

## How to Use

### 1. Clone the Repository
```bash
git clone https://github.com/your_username/text-generation-transformer.git
cd text-generation-transformer
```
### 2. Train the Model
To train the model from scratch, run:

bash
Copy code
python training.py
Alternatively, if you don't have pytorch installed, open model_training.ipynb in google Colab  to view and experiment with the training process.

### 3. Generate Text Using the Pre-trained Model
To generate text using the pre-trained model, run:

```bash
python trained.py
```
The output will display generated text based on the learned patterns.

### 4. Modify Hyperparameters
You can modify the hyperparameters in training.py or model_training.ipynb to experiment with different configurations. Key hyperparameters include:

`batch_size`
`learning_rate`
`dropout`
`num_layers`
`num_heads`

### Requirements
Python 3.8+
PyTorch
Jupyter Notebook (optional, for using the notebook)
Additional packages: numpy, torch, tqdm (installable via requirements.txt)
Install all dependencies using:

```bash
pip install -r requirements.txt
```

### Key Features
Fully implemented Transformer-based model.
Configurable training pipeline.
Pre-trained model available for text generation.
Designed for educational purposes, following best practices in NLP and deep learning.

### Credits
This project is inspired by the work of [Andrej Karpathy](https://github.com/karpathy) and his [YouTube lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY), which served as excellent resources for understanding and implementing this transformer model.


