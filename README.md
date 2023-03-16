# STRAFE - Survival analysis TRAnsFormer-based architecture for Electronic heath records

STRAFE is a novel architecture for modeling time-series clinical data and predicting time-to-event. It is designed to provide better accuracy and interpretability than other models for this type of data, and can train on censored data. The STRAFE algorithm was applied to real-world claims data in the OMOP common data model (CDM) format.

The STRAFE architecture is an expansion of SARD, a Transformer-based architecture developed by Kodiolam et al. [1], which takes as input time-series OMOP CDM data built from claims data to predict clinical outcomes. The expansion of the SARD model to time-to-event prediction was inspired by Hu et al. [4].

This repository includes the implementation of the STRAFE algorithm from the pre-process phase until the prediction phase. We also included the implementation of SARD and logistic regression for risk prediction and the survival baselines: random survival forest (RSF) [2] and DeepHit [3].

## Purpose and Benefits of STRAFE
STRAFE is designed to provide better accuracy and interpretability than other models for time-series clinical data. It is particularly well-suited for predicting time-to-event in real-world claims data, and has been shown to outperform other models in this area. We described the details of the STRAFE architecture in [5]. If you use STRAFE in your work, please cite our paper: 

"""

## Documentation

Most of the scripts and code in this repository were taken from the omop-learn package: https://github.com/clinicalml/omop-learn. Every function that differs from the original is marked with the comment "# Changed" before the function definition. More detailed documentation and summaries for most of the code can be found in the omop-learn package.

The following files are included in this repository:

- prediction_preprocess.py: reads information from a cohort, creates a feature set, and saves datasets for further analysis (risk/surv).
- risk.py: runs SARD and logistic regression models for risk prediction.
- surv.py: runs STRAFE, DeepHit, and RSF for time-to-event prediction and risk prediction as described in the paper.
- survival.py: transformer-based Kaplan-Meier estimator (Hu et al.)
- sql/Cohorts/cohort_survival.py: implementation of the cohort defined in the paper.

## Running example

TODO

## Dependencies

The following libraries are required to run the code:

- copy
- gensim.models
- matplotlib
- numpy
- pandas
- pickle
- pycox
- random
- scipy.sparse
- sklearn
- sqlalchemy
- sparse
- torch
- torchtuples

## Contributors and Acknowledgements
TODO

[1] Rohan S. Kodialam, Rebecca Boiarsky, Justin Lim, Neil Dixit, Aditya
Sai, and David Sontag. Deep contextual clinical prediction with reverse
distillation, 2020.

[2] Hemant Ishwaran, Udaya B. Kogalur, Eugene H. Blackstone, and
Michael S. Lauer. Random survival forests. The Annals of Applied
Statistics, 2(3), sep 2008.

[3]  Changhee Lee, William Zame, Jinsung Yoon, and Mihaela van der
Schaar. Deephit: A deep learning approach to survival analysis with
competing risks. Proceedings of the AAAI Conference on Artificial
Intelligence, 32(1), Apr. 2018.

[4] Shi Hu, Egill Fridgeirsson, Guido van Wingen, and Max Welling.
Transformer-based deep survival analysis. In Survival PredictionAlgorithms, Challenges and Applications, pages 132–148. PMLR, 2021.
