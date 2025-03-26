# Causality analysis of electricitymarket liberalization on electricit price using novel Machine Learning methods
Type: Master's Thesis

Author: Orr Shahar

1st Examiner: Prof. Doc. Stefan Lessmann

2nd Examiner: Doc. Sigbert Klinke

Electricity price data

![Screenshot 2025-03-26 at 16 04 07](https://github.com/user-attachments/assets/f524cf66-3c85-44e8-9986-0469c07b43ce)

![Screenshot 2025-03-26 at 16 09 36](https://github.com/user-attachments/assets/6675ed7f-0a43-4bd1-8cb5-18095b42982c)

Synthetic dataset results:

![Screenshot 2025-03-26 at 16 06 49](https://github.com/user-attachments/assets/d13d7c58-7d84-4b24-8f07-3055ba2d75ad)

Electricity price results:

![Screenshot 2025-03-26 at 16 07 22](https://github.com/user-attachments/assets/8fa236ee-fd6e-45cd-9a6a-5fea7f528609)

ATT estimation:

![Screenshot 2025-03-26 at 16 08 06](https://github.com/user-attachments/assets/40805db8-cb6f-4304-9a80-f4d062412f2e)

## Table of Content

- [Summary](#summary)
- [Working with the repo](#Working-with-the-repo)
    - [Dependencies](#Dependencies)
    - [Setup](#Setup)
- [Reproducing results](#Reproducing-results)
    - [Training code](#Training-code)
    - [Evaluation code](#Evaluation-code)
    - [Pretrained models](#Pretrained-models)
- [Results](#Results)
- [Project structure](-Project-structure)

## Summary

Relationships between the energy and the finance markets are increasingly important. Understanding these relationships is vital for policymakers and other stakeholders. One of the most complex yet important relationships is the identification and estimation of causality impact. The development of machine learning in recent years opened the door for a new branch of machine learning models for causality impact. In this research, we wish to evaluate novel machine learning methods for causality analysis and assess their adequacy for the single intervention case. We empirically compare the performance of different frameworks on synthetic data, and then utilize these models to estimate the causal effect of electricity market liberalization on the electricity price in the US. By performing this analysis, we aim to shed more light on the applicability of causal frameworks to energy policy intervention cases. We also aim to provide new insights into the ongoing debate about the benefits of electricity market liberalization

## Working with the repo

### Dependencies

Python = 3.9

### Setup

1. Clone this repository

2. Create an virtual environment and activate it
```bash
python -m venv thesis-env
source thesis-env/bin/activate
```

3. Install requirements
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
4. Path Variables
Set the PYTHONPATH env variable of the system. Append absolute paths of both the project root directory and the directory of the src/models/DeepProbCP/cocob_optimizer into the PYTHONPATH for the model DeepProbCP
Set the project root directory as the working directory.

For R scripts, make sure to set the working directory to the project root folder.


packages:

python  3.9
tensorflow 2.5.0
smac 1.0.0
scipy 1.10.0
pynisher 0.4.1
typeguard 2.10.0
pandas 1.4.2
ml-dtypes 0.0.2
numpy 1.19.5
statsmodels 0.13.0


