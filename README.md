# strafe

STRAFE is a novel architecture we developed for modeling time-series clinical data and predicting time-to-event. The STRAFE algorithm was applied to real-world claims data using the OMOP common data model format.
The STRAFE architecture design is an expansion of SARD, a Trasnformer-based architecture developed by Kodiolam et al. [1], which takes as input time-series OMOP CDM data that was built from claims data to predict clinical outcomes.

This repository includes the implemenation of the STRAFE algorithm from the pre-process phase until the prediction phase. We also included the implementation of SARD and logistic regression for risk prediction and the survival baselines: random survival forest (RSF) [2] and DeepHit [3].

The expansion of the SARD model to time-to-event prediction was inspired by Hu at al. [4].

"""
We described the details of this method and the STRAFE architecture in our paper. If you use STRAFE in your work, please cite our paper:
...
"""

## Documentation
Most of the scripts and code in this repository was taken from omop-learn package: github.com/clinicalml/omop-learn.
This include some exceptions. Every function which is differ from the original is marked with the comment: "# Changed" before function definition.
Thus, for most of the code a a more detailed summary and documentation can be found in omop-learn package.

The following files are code scripts to run the models:
- prediction_preprocess.py : reading information from cohort, creating feature set and save datasets for further analysis (risk/surv)
- risk.py: running SARD and logistic regression model for risk tasks.
- surv.py: running STRAFE, DeepHit and RSF for time-to-event prediction and risk prediction as described in the paper.
- survival.py: transformer-based Kaplan-Meier estimator (Hu. at al)
- sql/Cohorts/cohort_survival.py: implementation the cohort defined in the paper.


## Running example
TODO

## Dependencies
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
