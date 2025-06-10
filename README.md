# TCR-analysis

## Problem statement:
T-cell receptors (TCRs) are crucial components of the adaptive immune system responsible for recognizing and binding to specific antigens. The TCR is formed by recombination of the V, (D), and J segments of the genes that build its structure. The random process of recombination and further rearrangements provide a large sequence diversity and low overlap of immune repertoires.
Modern machine learning methods are able to analyze many sequences at once, detect complex patterns, and are also more flexible than traditional approaches to analysis. In this work, we developed algorithms for V and J gene classification, for TCR-epitope interaction modeling, and for generation of receptor sequences. As a result, we developed highly efficient models for dealing with TCR sequences. The classification of V and J genes and the generation of sequences were most successfully solved. In the process of evaluating the results, we found that the models effectively detect significant patterns in biological sequences, which contributes to the high quality of the analysis and the reliability of the results. 


## Navigation
| Name | Description |
|-----------------|-----------------|
| data    | contains all the data that used in the project    |
| models_mdf.py    | file with TCR-BERT and Prottrans based models, also containes required functions and methond for the next usage     |
|  Alpha and beta chain classification.ipynb   | alpha / beta subunit prediction     |
|  V_J_genes.ipynb   |  V and J genes prediction with transformers  |
|  KNN   |  KNN setup and amlication to simillatity sequences analysis   |
|  FCNN   |  V and J genes prediction with FCNN based model; implementation, training and application  |
|  epitopes   |  epitopes analysis; fine-tuning, usage, evaluation and visualisation  |
|  GAN   |  sequences generation; implementation, training, evaluation and application  |
