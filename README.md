# Redundant product titles compression
## Introduction
Code for redundant product titles compression, it includes GRU_SATT model and some baselines.
## Requirements
- Python 3.5+
- Tensorflow 1.10+
## Train
GRU_SATT means self_attention based GRU model for redundant product titles compression. GRU_SATT_SMD is an optimization on GRU_SATT. The simple training instructions are shown as follows. You can modify other parameters to train some baselines. "Compress" tasks represents the human label compression titles and "Phrase" means the labeled tokens are less than three.
### Train on GRU_SATT
- The default includs represents the "Phrase" tasks and SELF_ATT model
```
python -m main
```
- For "Compress" tasks
```
python -m main -task=Compress
```
### Train on GRU_SATT_SMD
- For Phrash tasks
```
python -m main -model=GRU_SATT_SMD
```
- For "Compress" tasks
```
python -m main -model=GRU_SATT_SMD -task=Compress
```
## Load pretrained train model
Several pretrained models have been provided in the model folder. You can use the following command to load them. 
- Load default pretrained model GRU_SATT for "Phrase" tasks.
```
python -m load
```
- Load pretrained model GRU_SATT for "Compress" tasks.
```
python -m load -task=Compress
```
- Load pretrained model GRU_SATT_SMD for "Phrase" tasks.
```
python -m load -model=GRU_SATT_SMD
```
- Load pretrained model GRU_SATT_SMD for "Compress" tasks.
```
python -m load -model=GRU_SATT_SMD -task=Compress
```
## Citation
If you use the code in your paper, please cite it as:
```
@article{fu2019ersnet,
  title={Self-attention based neural networks for product titles compression},
  author={FU Yu and LI You and LIN Yu-ming and ZHOU Ya and others},
  journal={Journal of East China Normal University (Natural Sciences)},
  volume={5},
  pages={113--122},
  year={2019}
}
```
