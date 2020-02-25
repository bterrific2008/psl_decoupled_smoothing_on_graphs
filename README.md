# Decoupled Smooting in Probabilistic Soft Logic

This project implements the concept of Decoupled Smoothing into Probabilistic Soft Logic (PSL). This code provides a set of PSL rules for one hop (homophily), two hop (monophily), and decoupled smoothing methods.

## Replication code: "Decoupled smoothings on graphs"

The original paper and code for "Decoupled smoothings on graphs" can be found here:

* [Decoupled smoothing on graphs](https://dl.acm.org/citation.cfm?doid=3308558.3313748) (WWW 2019) - [Alex Chin](https://ajchin.github.io/), [Yatong Chen](https://github.com/YatongChen/), [Kristen M. Altenburger](http://kaltenburger.github.io/), [Johan Ugander](https://web.stanford.edu/~jugander/).
* [Github Repository](https://github.com/YatongChen/decoupled_smoothing_on_graphs)

## Machine Learning Framework: "Probabilistic Soft Logic"

Probabilistic Soft Logic is a machine learning framework for developing probabilistic models. You can find more information about PSL available at the [PSL homepage](https://psl.linqs.org/). 

### Documentation

This repository contains code to run PSL rules for one hop (homophily) and two hop (monophily) methods to predict genders in a social network. 
We provide links to the datasets (Facebook100) in the data sub-folder.


### Reproducing results and figures

This repository set-up assumes that the FB100 (raw `.mat` files) have been acquired and are saved the data folder. Follow these steps:
1. The Facebook100 (FB100) dataset is publicly available from the Internet Archive at https://archive.org/details/oxford-2005-facebook-matrix and other public repositories. Download the datasets.
2. Save raw datasets in placeholder folder data. They should be in the following form: `Amherst41.mat`.
