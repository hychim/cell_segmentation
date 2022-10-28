# cell_segmentation
## Introduction
Drug assay has been frequently used to test drug activity in different biological
mechanisms and there are already large databases with different drug activity
results publicly available, e.g. TOX21 database. However, preparing these
types of drug assays is time-consuming and expensive. In this research training
project, another approach is used to classify drug activity and cell painting im-
age profiles are used for the classification of drug activity instead. The cells are
cultured in wells that are seeded with chosen chemical or potential drug candi-
dates. The cell painting images are then taken using microscope with different
channels of light and features of the images are extracted using CellProfiler.
The profiles will then be used for the prediction of activity using different Ma-
chine learning methods.
Some of the most popular machine learning methods in drug discovery[4]
for classification are used to build the prediction model, for example, logis-
tic regression, support vector machines (SVMs), decision tree, random
forest, K nearest neighbor (KNN), XGBoost and also multilayer per-
ceptron (MLP). All the classifiers are implemented using Python with the
default settings in sklearn and XGBoost, no further optimization is per-
formed to improve the accuracy.
