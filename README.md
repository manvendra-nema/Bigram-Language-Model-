
# Bigram Language Model Project

## Overview

This project involves creating a Bigram Language Model from scratch using standard Python methods and the NumPy library. The primary objectives are:

- Learning the bigram model from a dataset.
- Implementing and comparing two smoothing algorithms: Laplace and Kneser-Ney.
- Modifying the bigram probabilities using emotion scores to generate emotion-oriented samples.
- Evaluating the model using generated samples and extrinsic evaluation methods.

## Features

### 1. Learning the Bigram Model

Develop a `BigramLM` class that learns the bigram model from a given dataset. The class should store the learned language model and support other methods required for processing and evaluation.

### 2. Smoothing Algorithms

Implement the following smoothing algorithms in the `BigramLM` class:

- **Laplace Smoothing**: This technique adds a small constant to all counts to ensure no probability is zero.
- **Kneser-Ney Smoothing**: This is a more advanced smoothing technique that adjusts probabilities based on the diversity of contexts a word appears in.

Compare the probabilities obtained from both smoothing algorithms and provide an argument for which one is better.

### 3. Emotion-Oriented Modifications

Utilize the `emotion_scores()` function from the provided `utils.py` file to get emotion scores for a sample sentence. Modify the standard probability of the bigram model using these emotion scores with the following formula:

\[ P(w_i | w_{i-1}) = \left(\frac{\text{count}(w_i)}{\text{count}(w_{i-1})}\right) + \beta \]

where \(\beta\) is the emotion component. This modification can be applied at various levels: unigram, bigram, or sample level. Use this modified model to generate samples oriented towards specific emotions.

### 4. Extrinsic Evaluation

#### a. Generate Emotion-Oriented Samples

Generate 50 samples for each of the 6 emotions, using the file name format `gen_<emotion>.txt`. These samples will be used for extrinsic evaluation of the language model.

#### b. Train and Evaluate an SVC Model

Use the original corpus as the training data and the generated samples as the testing data. Train a Support Vector Classifier (SVC) from the Scikit-Learn library and use the TF-IDF vectorizer for text samples. Conduct a Grid Search to find the best parameters and evaluate the performance of the emotion-based modifications.
