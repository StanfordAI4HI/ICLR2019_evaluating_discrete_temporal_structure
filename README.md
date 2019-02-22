# Evaluating Discrete Latent Temporal Structure

This repository provides code to run experiments from,
 
**Learning Procedural Abstractions and Evaluating Discrete Latent Temporal Structure**  
Karan Goel and Emma Brunskill  
_ICLR 2019_

Specifically, this repository reproduces experiments with the new evaluation criteria proposed in the paper. 
The results are reproduced from logged runs of the methods being compared (stored in ``logs/``).

Install the requirements into your virtual environment using ``pip install -r requirements.txt``.

Then, reproduce by running   

``> python evaluate.py --dataset bees_0``
  
will reproduce all the plots and visualizations from the first sequence of the Bees dataset in ``plots/``. 
Run ``python evaluate.py`` to see a list of the available datasets.


Please contact Karan Goel ``kgoel <at> cs <dot> stanford <dot> edu`` for questions!