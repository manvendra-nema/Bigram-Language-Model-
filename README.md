# 📚 Bigram Language Model Project

## 🔍 Overview

Dive into the world of language models with this project, where we build a Bigram Language Model from scratch using Python and NumPy. Our goals include:

- Developing a bigram model from a dataset.
- Implementing and contrasting Laplace and Kneser-Ney smoothing techniques.
- Enhancing bigram probabilities with emotion scores for emotion-oriented text generation.
- Evaluating the model through generated samples and extrinsic methods.

## 🌟 Features

### 1. 📖 Learning the Bigram Model

Create a `BigramLM` class that learns from a given dataset, storing the language model and supporting essential processing and evaluation methods.

### 2. 🔧 Smoothing Algorithms

Implement and compare two smoothing algorithms within the `BigramLM` class:

- **📈 Laplace Smoothing**: Adds a constant to all counts to ensure no probability is zero.
- **🔍 Kneser-Ney Smoothing**: Adjusts probabilities based on the diversity of contexts where a word appears.

Evaluate and discuss the efficacy of each smoothing method.

### 3. 🎭 Emotion-Oriented Modifications

Utilize the `emotion_scores()` function from `utils.py` to integrate emotion scores into the bigram model. Adjust probabilities using:

\[ P(w_i | w_{i-1}) = \left( \frac{\text{count}(w_i)}{\text{count}(w_{i-1})} \right) + \beta \]

where \(\beta\) is the emotion component. Apply this at various levels (unigram, bigram, or sample) to generate emotion-specific text samples.

### 4. 🧪 Extrinsic Evaluation

#### a. ✍️ Generate Emotion-Oriented Samples

Create 50 samples for each of the six emotions, saving them with the format `gen_<emotion>.txt`. Use these samples to assess the language model externally.

#### b. 🧠 Train and Evaluate an SVC Model

Leverage the original corpus for training and the emotion-based samples for testing. Train a Support Vector Classifier (SVC) using Scikit-Learn and a TF-IDF vectorizer. Perform a Grid Search to optimize parameters and evaluate the emotion-enhanced modifications.

## 🎯 Conclusion

This project aims to provide a deep understanding of bigram language models, smoothing techniques, and the impact of emotion scores on text generation. By exploring these elements, we create a robust model capable of generating emotion-specific samples and evaluating its performance comprehensively.

---

Happy coding! 🚀🔍
