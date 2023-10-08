# Advancing Semantic Textual Similarity Modeling: A Regression Framework with Shift ReLU and Smooth K2 Loss

## Package

```python
python              3.9.16
datasets            2.13.0
huggingface-hub     0.15.1
numpy               1.25.0
pandas              2.0.2
scikit-learn        1.2.2
scipy               1.10.1
semantic-version    2.10.0
SentEval            0.1.0
torch               1.13.1
torchaudio          0.13.1
torchvision         0.14.1
tqdm                4.65.0
transformers        4.30.2
```

## Data

https://drive.google.com/file/d/1oWGNLR0gPAForAKu0AGVUrsSKS0mG1T4/view?usp=sharing

## Train

```bash
nohup torchrun --nproc_per_node=4 train.py > nohup.out &
```
