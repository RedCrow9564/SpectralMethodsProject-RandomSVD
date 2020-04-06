# Spectral Methods in Data Processing (0372-4001) - Final Project 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RedCrow9564/SpectralMethodsProject-RandomSVD/blob/master/Spectral_Methods_Project_Random_SVD.ipynb) [![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)![Run Unit-Tests](https://github.com/RedCrow9564/SpectralMethodsProject-RandomSVD/workflows/Run%20Unit-Tests/badge.svg)![Compute Code Metrics](https://github.com/RedCrow9564/SpectralMethodsProject-RandomSVD/workflows/Compute%20Code%20Metrics/badge.svg)![GitHub last commit](https://img.shields.io/github/last-commit/RedCrow9564/SpectralMethodsProject-RandomSVD)

This is a project submitted as a requirement for this course. [The course](https://www30.tau.ac.il/yedion/syllabus.asp?course=0372400101) was administered in Fall 2019-2020 (before the Coronavirus outbreak...) in [Tel-Aviv University - School of Mathematical Sciences](https://en-exact-sciences.tau.ac.il/math), and taught by [Prof. Yoel Shkolnisky](https://english.tau.ac.il/profile/yoelsh). 
This project is a reconstruction of experiments of [[1]](#1) for algorithms for Randomized SVD and Randomized Interpolative Decompositions. A complete documentation of the code is available [here](docs/main_doc.html).

## Getting Started

The code can be fetched from [this repo](https://github.com/RedCrow9564/SpectralMethodsProject-RandomSVD.git). The Jupyter Notebook version does the same work, and can be deployed to [Google Colab](https://colab.research.google.com/github/RedCrow9564/SpectralMethodsProject-RandomSVD/blob/master/Spectral_Methods_Project_Random_SVD.ipynb). While the the notebook version can be used immediately, this code has some prerequisites.
Any questions about this project may be sent by mail to 'eladeatah' at mail.tau.ac.il (replace 'at' by @).

### Prerequisites

This code was developed for Windows10 OS and tested using the following Python 3.7.6 dependencies. These dependencies are listed in [requirements.txt](requirements.txt).
All these packages can be installed using the 'pip' package manager (when the command window is in the main directory where requirements.txt is located):
```
pip install -r requirements.txt
```
All the packages, except for Sacred, are available as well using 'conda' package manager.

## Running the Unit-Tests

The Unit-Test files are:

* [test_data_creation.py](UnitTests/test_data_creation.py) - Tests the data created for all experiments is the data described in the paper [[1]](#1) .
* [test_random_svd.py](UnitTests/test_random_svd.py) - Tests the implementation of the Random SVD algorithm.
* [test_random_id.py](UnitTests/test_random_id.py) - Tests the implementation of the Random Interpolative Decompositino algorithm.

Running any of these tests can be performed by:
```
<python_path> -m unittest <test_file_path>
```
## Acknowledgments
Credits for the original algorithms, paper and results of [[1]](#1) belong to its respectful authors: P.-G. Martinsson, V. Rokhlin, and M. Tygert.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## References
<a id="1">[1]</a> [P.-G. Martinsson, V. Rokhlin, and M. Tygert. A randomized al-
gorithm for the decomposition of matrices. Applied and Computa-
tional Harmonic Analysis, 30(1):47—68, 2011](https://www.sciencedirect.com/science/article/pii/S1063520310000242).
