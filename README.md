# Semantify-NN: Towards Verifying Robustness of Neural Networks Against A Family of Semantic Perturbations

## About Semantify-NN:
We propose **Semantify-NN**, a model-agnostic and generic robustness verification toolkit for semantic perturbations. **Semantify-NN** can be viewed as a powerful extension module consisting of novel semantic perturbation layers (SP-layers) and is compatible to existing Lp-norm based verification tools. Semantify-NN can support robustness verification against a wide range of semantic perturbations: translation, occlusion, hue, saturation, lightness, brightness and contrast, rotations.

Cite our work:

Jeet Mohapatra, Tsui-Wei Weng, Pin-Yu Chen, Sijia Liu and Luca Daniel, "[**Towards Verifying Robustness of Neural Networks Against A Family of Semantic Perturbations**"](https://arxiv.org/abs/1912.09533), CVPR 2020.

```
@inproceedings{mohapatra2020SemantifyNN,
  author = "Jeet Mohapatra AND Tsui-Wei Weng AND Pin-Yu Chen AND Sijia Liu AND Luca Daniel",
  title = "Towards Verifying Robustness of Neural Networks Against A Family of Semantic Perturbations",
  booktitle = "CVPR",
  year = "2020",
  month = "June"
}
```

## Setup and Requirements:

The code is tested with python3 and TensorFlow v1.10 and v1.12 (TensorFlow v1.8 is not compatible). The following packages are required.
```
conda create --name cnncert python=3.6
source activate cnncert
conda install pillow numpy scipy pandas h5py tensorflow numba posix_ipc matplotlib
```

Clone this repository:
```
git clone https://github.com/JeetMo/Semantify-NN.git
cd Semantify-NN
```
## Pre-trained Models

Download the pre-trained CNN models used in the paper.
```
wget https://www.dropbox.com/sh/aap4zh8aclgw4s4/AADhQE4J54GicW9ueiQlV4Xna?dl=1 -O model.zip
unzip models.zip
```

## How to run (* Current code only supports fully connected models, CNN verification code to be added soon)

### Verification

In order to run the color-space experiments (without implicit splitting) we can use the following command:
```
python main_semantic_hsl.py --model=cifar --numimage=2 --numlayer=5 --hidden=2048 --eps=3 --delta=0.2 --hsl=hue
```
with the parameters 

-- model (gtsrb, cifar, mnist) : the dataset to use

-- hidden (int) : number of hidden units in the fully connected layers of the model

-- numlayer (int) : number of layers in the model

-- modelfile (string) : specify name of model file (optional)

-- numimage (int) : number of examples to run the attacks for

-- hsl (lighten, saturate, hue) : the threat model of the attack

-- eps (float) : maximum range of attack parameter to test over

-- delta (float) : size of each explicit split to be used


Similarly for the rotation experiments use :

```
python main_semantic_ rotate.py --model=mnist --numimage=2 --numlayer=3 --hidden=1024 --eps=100 --delta=0.5 --subdiv=100
```
where the extra parameter
-- subdiv (int) : number of implicit splits to use

### Attack

We have one interface to run the simple grid attacks for the threat models specified in the paper:

```
python main_attack.py --model=gtsrb --hidden=256 --numlayer=4 --attack='rotate' --eps=100 --delta=0.5 --numimage=200 
```
with the same parameters.
