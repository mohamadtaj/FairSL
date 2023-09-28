# FairSL

This is a framework for fair swarm learning which is a collaborative machine learning system that takes into account participant's contributions.

This repository provides simulations for the fair SL, typical SL, and local training (without cooperation). 
Simulations consist of neural networks and random forest training on different datasets.

## Options:

### Dataset:
| Dataset  | Option |
| ------------- | ------------- |
| CIFAR-10  | --dataset cifar10  |
| Reuters Newswire  | --dataset reuters  |
|  NATICUSdroid | --dataset android  |
|  Breast Cancer | --dataset breast_cancer  |
| Heart Failure | --dataset heart_failure  |
| Maternal Health | --dataset maternal_health  |
| Auction Verification  | --dataset auction  |
| Students' Dropout  | --dataset student  |

### Framework:
| framework  | Option |
| ------------- | ------------- |
| Neural Networks  | --mode nn  |
| Random Forest  | --mode rf  |

### Mode:
| Mode  | Option |
| ------------- | ------------- |
| Fair Swarm Learning  | --mode fair  |
| Typical Swarm Learning  | --mode unfair  |
|  Local learning | --mode local  |

## Dataset Preparation:
The CIFAR10 and Reuters newswire datasets will be automatically downloaded by the program. For the other six datasets, the user should download the datasets prior to running the code and put them in a folder named **datasets** in the same directory as the app files.
Here are the links to download each dataset and the directories that should be created for each:

| Dataset  | Download Link | Path |
| ------------- | ------------- | ------------- |
| NATICUSdroid | http://archive.ics.uci.edu/dataset/722/naticusdroid+android+permissions+dataset | ./datasets/android |
| Breast Cancer | http://archive.ics.uci.edu/dataset/451/breast+cancer+coimbra | ./datasets/breast_cancer |
| Heart Failure | http://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records | ./datasets/heart_failure |
| Maternal Health | http://archive.ics.uci.edu/dataset/863/maternal+health+risk | ./datasets/maternal_health_risk |
| Auction Verification | http://archive.ics.uci.edu/dataset/713/auction+verification | ./datasets/auction |
| Students' Dropout | http://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success | ./datasets/student_dropout |

## Requirements:
imbalanced_learn==0.7.0<br>
Keras==2.4.3<br>
matplotlib==3.3.1<br>
numpy==1.22.4<br>
opencv_python==4.5.5.64<br>
pandas==1.1.4<br>
scikit_learn==1.0.2<br>
scipy==1.10.0<br>
seaborn==0.12.2<br>
tensorflow==2.2.0<br>


## Run:
To run the simulation, simply execute the run.py file with the **dataset**, **framework**, and **mode** arguments.
For example, to run the simulation for the CIFAR10 dataset using the Fair Swarm Learning approach, type in:

```
python3 run.py --dataset cifar10 --framework nn --mode fair

```
