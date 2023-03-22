# DipDECK - Dip-based Deep Embedded Clustering with k-Estimation

In this folder you will find the sourcecode to the paper "Dip-based Deep Embedded Clustering with k-Estimation".

## Preparation

An installation of python3 and a working C compiler are required!

Furthermore, following python packages must be installed:
- numpy
- scipy
- scikit-learn
- torch
- torchvision

We need to compile the dip.c file using a C compiler. For this following command needs to be executed (here we chose the gcc compiler):

On Windows: `gcc -fPIC -shared -std=c99 -o dip.dll dip.c`

On Linux: `gcc -fPIC -shared -o dip.so dip.c`

After this our code is ready to be executed.

## Data sets

The data sets USPS, MNIST, F-MNIST, K-MNIST, Optdigits, Pendigits and Letterregocnition will be downloadad automatically if the respective lines are uncommented.

All data sets will be downloaded to: [HOME]/Downloads/dipDECK_data/

GTSRB can be downloaded at: https://benchmark.ini.rub.de/gtsrb_news.html

If the automatic downloads do not work, the data sets can be downloaded here:
- USPS: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps
- MNIST: http://yann.lecun.com/exdb/mnist/
- F-MNIST: https://github.com/zalandoresearch/fashion-mnist
- K-MNIST: https://github.com/rois-codh/kmnist
- Optdigits: https://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits
- Pendigits: https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits
- Letterrecognition: https://archive.ics.uci.edu/ml/datasets/Letter+Recognition

## Execute DipDECK

Within the main block of dipdeck.py (marked by `if __name__ == "__main__:`) everything is prepared.
Just choose a data set, uncomment the respective line and run the code.

You can run the code by executing:

`python dipdeck.py`

or

`python3 dipdeck.py`

Alternatively, a python3 console can be opened and the following commands executed:

```python 
import dipdeck

# === Load dataset (can be exchanged eg by load_mnist()) ===
data, labels = load_usps()

# === Create DipDECK object ===
dipdeck = DipDECK()
dipdeck.fit(data)

# === Print results ===
print("K:", dipdeck.n_clusters_)
print("NMI:", nmi(labels, dipdeck.labels_))
```

## Extra: Examplary run

The directory contains the SYN4_k35.gif file that shows an exemplary run of the data set illustrated in Fig. 3.
It is used to better understand the process of our algorithm.