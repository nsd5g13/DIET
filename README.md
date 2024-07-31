# DIET: Dependence based lIteral Excluding for compressed Tsetlin machine

<!-- ABOUT THE PROJECT -->
## About The Project

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

Install pyTsetlinMachine (original source code available at [https://github.com/cair/pyTsetlinMachine](https://github.com/cair/pyTsetlinMachine)):
   ```sh
   cd pyTsetlinMachine
   python3 setup.py install
   ```

## Usage

### Booleanization

A Tsetlin machine (TM) requires all raw features of a dataset represented in the form of Bool. We provide source code to booleanize multiple open source datasets:
- MNIST
- Fashion-MNIST (FMNIST)
- Kuzushiji-MNIST (KMNIST)
- CIFAR2
- Keyword Spotting (KWS): a mini speech commands dataset is availabe at [http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip](http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip). Please put the zip file in the path: DIET/Booleanization/
- Iris

   ```sh
   cd DIET/Booleanization
   python3 booleanization.py [dataset_name]
   ```
Options for [dataset_name] are mnist, fmnist, kmnist, cifar2, kws, iris and all, where all suggests to produce Booleanized datasets for all above.

### Vanilla TM Training

Source code to train a vanilla TM-based model is located in DIET/VanillaTM_training.

   ```sh
   usage: StandardTraining.py clauses T s epochs budget dataset_name

 positional arguments:
     clauses         Provide the number of clauses per class
     T               Provide the value of "Threshold"
     s               Provide the value of "Strength" for literal include
     epochs          Proivde the number of training epochs
     budget          Provide the constrain for the maximal number of literals included in each clause
     dataset_name    Provide the name of the dataset. Options include mnist, kmnist, fmnist, cifar2, kws and iris
   ```

Example:
   ```sh
   cd DIET/VanillaTM_training
   python3 StandardTraining.py 300 15 10 100 1568 mnist
   ```

### Iterative Training-Excluding for Compresed TM

Source code to use DIET to train a compressed TM is located in DIET/DIET_training.

   ```sh
   usage: diet.py clauses T s epochs epochs_every_exclude budget dataset_name

 positional arguments:
     clauses               Provide the number of clauses per class
     T                     Provide the value of "Threshold"
     s                     Provide the value of "Strength" for literal include
     epochs                Proivde the total number of training epochs
     epochs_every_exclude  Specify the number of epochs after each excluding process  
     budget                Provide the constrain for the maximal number of literals included in each clause
     dataset_name          Provide the name of the dataset. Options include mnist, kmnist, fmnist, cifar2, kws and iris
   ```

Example:
   ```sh
   cd DIET/DIET_training
   python3 diet.py 300 15 10 100 5 1568 mnist
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>
