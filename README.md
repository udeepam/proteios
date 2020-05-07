# Proteios:  Predicting Subcellular Location of Eukaryotic Proteins

This repository contains python code for predicting the subcellular location of eukaryotic proteins.

## Motivation
Within the last few years the complete sequence has been determined for over 3000 genomes. This has created the need for fully automated methods to analyse the vast amount of sequence data now available. Investigations into protein function as well as therapeutic interventions both require an understanding of protein subcellular localisation. This topic has been well-studied in bioinformatics and has led to the development of several machine learning algorithms. More recently deep learning methods such as recurrent neural networks have been employed to handle these sequences. However, deep learning architectures continue to be treated as black-box function approximators. Thus it is desirable to have methods for predicting protein locations whilst maintaining a degree of interpretability. This work presents an approach for predicting the subcellular location (cytosolic, secreted, nuclear, mitochondrial) of non-homologous proteins by using classical machine learning algorithms,
* Random Forest,
* Support Vector Machine (SVM),
* Multilayer Perceptron (MLP),
* Gaussian Näive Bayes (GNB),
* Linear Discriminant Analysis (LDA),
* K-Nearest Neighbours (KNN).

## Code
* The data is provided in the FASTA format. We use the `biopython` package to process the data. 
* For feature extraction we use the `ProtoParam` module from the `biopython` package.

## Resources
Useful resources
* Motivation and dataset: http://www0.cs.ucl.ac.uk/staff/D.Jones/coursework/
* Biopython: http://biopython.org/DIST/docs/tutorial/Tutorial.pdf
* Feature extraction: https://biopython.org/DIST/docs/api/Bio.SeqUtils.ProtParam.ProteinAnalysis-class.html
