#!/usr/bin/env python
# coding: utf-8

# In[6]:


from transformers import pipeline
from tqdm import tqdm
import numpy as np
from collections import Counter
import math



from itertools import product
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True,)

def emotion_scores_function(sample): 
    emotion=classifier(sample)
    return emotion[0]


class BigramLM:
    def __init__(self):
        self.vocab = set()
        self.bigram_counts = {}
        self.unigram_counts = {}
        self.bigram_probs = {}
        self.emotions = ['joy','sadness','love','anger','fear','surprise']
        self.all_bigram_probs = {i:{} for i in self.emotions }
        self.total_prob = {i:0 for i in self.emotions   }

    def scale_bigram_probabilities(self, bigram_probabilities):

      # Find the score corresponding to the given emotion label
      
      total_probability = {}
      for bigram, prob in tqdm( bigram_probabilities.items(), desc="SCALING EMOTIONS"): #IMPLEMNET TQDM
          # Obtain emotional scores for the current bigram
          emotion_scores = emotion_scores_function(" ".join(bigram))

          for score in emotion_scores:
              scaled_probabilities = self.all_bigram_probs[score["label"]]
              scaled_probabilities[bigram] = prob * score["score"]
              self.total_prob[score["label"]]+=(prob * score["score"])
      
      for i in self.emotions:
        scaled_probabilities = self.all_bigram_probs[i]
        for bigram, prob in scaled_probabilities.items():
            scaled_probabilities[bigram] = prob / self.total_prob[score["label"]]


      

    def learn_model(self, corpus,smooth ="laplace"):
        # Implement code to learn bigram model from the dataset
        for sentence in corpus:
            tokens = sentence.split()
            # Add a start token at the beginning of each sentence
            tokens = ['<start>'] + tokens
            self.vocab.update(tokens)
        print(len(self.vocab))
        # Generate all possible bigrams using product
        all_bigrams = list(product(self.vocab, repeat=2))

        # Initialize bigram counts and unigram counts
        self.bigram_counts = {bigram: 0 for bigram in all_bigrams}
        self.unigram_counts = {word: 0 for word in self.vocab}

        # Update bigram and unigram counts
        for sentence in corpus:
            tokens = sentence.split()
            # Add a start token at the beginning of each sentence
            tokens = ['<start>'] + tokens
            for i in range(len(tokens) - 1):
                bigram = (tokens[i], tokens[i + 1])
                self.bigram_counts[bigram] += 1
                self.unigram_counts[tokens[i]] += 1
            self.unigram_counts[tokens[i+1]] += 1
        
        time_modify_dict = {}
        count =500000
        for i,j in self.bigram_counts.items():
            if j !=0:
              time_modify_dict[i]=j
            else:
              if count:
                time_modify_dict[i]=j
                count-=1
        print(len(time_modify_dict),"kkook")
        self.bigram_counts = time_modify_dict
        # Calculate initial bigram probabilities
        if smooth ==None:
          for bigram, count in tqdm(self.bigram_counts.items(), desc="Generating probalities of bigram"):
              previous_word = bigram[0]
              if count == 0:
                  self.bigram_probs[bigram] = 0
              else:
                  self.bigram_probs[bigram] = count / self.unigram_counts[previous_word]
        elif smooth =="laplace":
          for bigram, count in tqdm(self.bigram_counts.items(), desc="Generating probalities of bigram"):
            previous_word = bigram[0]
            self.bigram_probs[bigram] = (count + 1) / (self.unigram_counts[previous_word] + len(self.vocab))
        elif smooth == "kneser-ney":
#             bigram_counts = Counter(zip(corpus, corpus[1:]))
            c_star_1 = sum(1 for count in self.bigram_counts.values() if count == 1)
            c_star_2 = sum(1 for count in self.bigram_counts.values() if count == 2)
            d = c_star_1 / (c_star_1 + 2 * c_star_2)
#             unigram_counts = Counter(corpus)
#             total_bigrams = len(bigram_counts)
#             print(unigram_counts)
            for bigram in tqdm(self.bigram_counts.keys(), desc="Generating probabilities of bigram"):
#                 print(bigram[1],self.unigram_counts[bigram[1]])
                discounted_prob = max(self.bigram_counts.get(bigram, 0) - d, 0) / self.unigram_counts[bigram[1]]
                backoff_prob = sum(1 for key in self.bigram_counts.keys() if key[1] == bigram[1]) / len(self.bigram_counts)
                self.bigram_probs[bigram] = discounted_prob + 0.5 * backoff_prob
        
        
        self.scale_bigram_probabilities(self.bigram_probs)


    def generate_next_word(self, current_word,emotion="fear"):

        if not self.bigram_probs:
            raise ValueError("Model has not been trained yet.")
        bigram_probs = self.all_bigram_probs[emotion]
        possible_next_words = [w2 for w1, w2 in bigram_probs if w1 == current_word and w2 !="<start>"]
        emotional_normlization = []
        
        probabilities = [bigram_probs.get((current_word, w2), 0) for w2 in tqdm(possible_next_words, desc="Choosing the next word")]
        probabilities = np.array(probabilities)

        # Normalize probabilities
        norm = probabilities / sum(probabilities)

        # Choose the next word based on probabilities
        next_word = np.random.choice(possible_next_words, p=norm)

        return next_word


# Example usage:
# Assuming you have a dataset, you can create an instance of BigramLM and train it on the dataset
# For simplicity, let's consider a small dataset:
file_path = 'D:\Downloads\corpus.txt'
corpus=[]
try:
    with open(file_path, 'r') as file:
        for line in file:
            corpus.append( line.strip())
except FileNotFoundError:
    print(f"File '{file_path}' not found.")
except Exception as e:
    print(f"An error occurred: {e}")



bigram_model = BigramLM()
bigram_model.learn_model(corpus,"kneser-ney")

# Generate a sequence of words


# In[ ]:





# In[90]:


# import pickle

# with open('kneser-ney_vocab.pkl', 'wb') as file:
#     pickle.dump(bigram_model.vocab, file)

# with open('kneser-ney_bigram_counts.pkl', 'wb') as file:
#     pickle.dump(bigram_model.bigram_counts, file)

# with open('kneser-ney_unigram_counts.pkl', 'wb') as file:
#     pickle.dump(bigram_model.unigram_counts, file)

# with open('kneser-ney_bigram_probs.pkl', 'wb') as file:
#     pickle.dump(bigram_model.bigram_probs, file)

# with open('kneser-ney_model.pkl', 'wb') as file:
#     pickle.dump(bigram_model, file)

# with open('kneser-ney_total_prob.pkl', 'wb') as file:
#     pickle.dump(bigram_model.total_prob, file)

# with open('kneser-ney_all_bigram_probs.pkl', 'wb') as file:
#     pickle.dump(bigram_model.all_bigram_probs, file)


# In[8]:


import pickle
from transformers import pipeline
from tqdm import tqdm
import numpy as np
from collections import Counter
import math


# Load vocabulary
with open('kneser-ney_vocab.pkl', 'rb') as file:
    loaded_vocab = pickle.load(file)

# Load bigram counts
with open('kneser-ney_bigram_counts.pkl', 'rb') as file:
    loaded_bigram_counts = pickle.load(file)

# Load unigram counts
with open('kneser-ney_unigram_counts.pkl', 'rb') as file:
    loaded_unigram_counts = pickle.load(file)

# Load bigram probabilities
with open('kneser-ney_bigram_probs.pkl', 'rb') as file:
    loaded_bigram_probs = pickle.load(file)

# Load the entire bigram model
with open('kneser-ney_model.pkl', 'rb') as file:
    bigram_model = pickle.load(file)

# Load total probability
with open('kneser-ney_total_prob.pkl', 'rb') as file:
    loaded_total_prob = pickle.load(file)

# Load all bigram probabilities
with open('kneser-ney_all_bigram_probs.pkl', 'rb') as file:
    loaded_all_bigram_probs = pickle.load(file)


# In[26]:


z = []
for i,j in loaded_bigram_probs.items():
    if loaded_bigram_counts[i] and i[0]=='i' and i[1] in ['wanna', 'gain', 'guess', 'once', 'meant']:
        z.append((i,j))
for i in sorted(z,key = lambda x: x[1]):
    print(i)


# In[25]:


z = []
for i,j in loaded_bigram_probs.items():
    if loaded_bigram_counts[i]: #and i[0]=='i' and i[1] in ['wanna', 'gain', 'guess', 'once', 'meant', 'cry', 'how', 'left', 'stared', 'rely', 'kept', 'know', 'wake', 'invest', 'watch', 'exceptionally', 'figure', 'dream', 'took', 'confused']:
        z.append((i,j))
for i in sorted(z,key = lambda x: x[1],reverse=True):
    print(i)


# In[22]:


z = []

for em,tu in loaded_all_bigram_probs.items():
    count =15
    print(em)
    for a in tu:
        if count==0:
            break
        if a[0]=='i':
            print(a,": ", loaded_all_bigram_probs[em][a])
            count-=1
        
    # print(loaded_all_bigram_probs['joy'][('i','am')])


# In[13]:


def generate_word_sequences(bigram_model, emotion, num_sequences=1, max_length=50):
    all_sequences = []

    for _ in range(num_sequences):
        current_word = "<start>"
        generated_sequence = []

        for _ in range(max_length):
            try:
                if current_word!= "<start>":
                    generated_sequence.append(current_word)
                current_word = bigram_model.generate_next_word(current_word, emotion)
            
            except Exception as e:
                current_word = bigram_model.generate_next_word("<start>", emotion)

        all_sequences.append(' '.join(generated_sequence)[1:])

    return all_sequences

# Example usage:
emotion_input = 'surprise'
num_sequences = 1
generated_sequences = generate_word_sequences(bigram_model, emotion_input, num_sequences,20)
for i, sequence in enumerate(generated_sequences, start=1):
    print(f"Sequence {i}: {sequence}")


# In[14]:


def generate_word_sequences(bigram_model, emotion, num_sequences=1, max_length=50):
    all_sequences = []

    for _ in range(num_sequences):
        current_word = "<start>"
        generated_sequence = []

        for _ in range(max_length):
            try:
                if current_word != "<start>":
                    generated_sequence.append(current_word)
                current_word = bigram_model.generate_next_word(current_word, emotion)

            except Exception as e:
                current_word = bigram_model.generate_next_word("<start>", emotion)

        all_sequences.append(' '.join(generated_sequence)[1:])

    return all_sequences


# Example usage:
emotions = ['joy', 'sadness', 'love', 'anger', 'fear', 'surprise']
num_sequences_per_emotion = 50
generated_corpus = []
generated_labels = []

for emotion in emotions:
    generated_sequences = generate_word_sequences(bigram_model, emotion, num_sequences_per_emotion, 12)
    generated_corpus.extend(generated_sequences)
    generated_labels.extend([emotion] * num_sequences_per_emotion)

# Print or use the generated corpus and labels as needed
for i, (sequence, label) in enumerate(zip(generated_corpus, generated_labels), start=1):
    print(f"Sequence {i}: {sequence} - Emotion: {label}")


# In[15]:


import os
# Example usage:
emotions = ['joy', 'sadness', 'love', 'anger', 'fear', 'surprise']
# Create a directory to store generated samples
output_directory = "generated_samples_k"
os.makedirs(output_directory, exist_ok=True)

def generate_and_save_sequences(bigram_model, emotion, num_sequences=50, max_length=20):
    generated_sequences = generate_word_sequences(bigram_model, emotion, num_sequences, max_length)
    
    # Save the generated sequences to a file
    output_filename = f"{output_directory}/gen_{emotion.lower()}.txt"
    with open(output_filename, 'w') as file:
        for sequence in generated_sequences:
            file.write(f"{sequence}\n")

    return output_filename

# Example usage:
for emotion in emotions:
    generated_file = generate_and_save_sequences(bigram_model, emotion, num_sequences_per_emotion, 30)
    print(f"Generated {num_sequences_per_emotion} samples for {emotion}: {generated_file}")


# In[17]:


emotions = ['joy', 'sadness', 'love', 'anger', 'fear', 'surprise']
# output_directory = "generated_samples_SAVE"
def read_generated_sequences_and_labels(emotion):
    input_filename = f"{output_directory}/gen_{emotion.lower()}.txt"
    generated_sequences = []
    labels = []  # Assign the emotion label to all sequences

    try:
        with open(input_filename, 'r') as file:
            for line in file:
                generated_sequences.append(line.strip())
                labels.append(emotion)
    except FileNotFoundError:
        print(f"File '{input_filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return generated_sequences, labels

# Example usage:
all_generated_sequences = []
all_generated_labels = []

for emotion in emotions:
    generated_sequences, labels = read_generated_sequences_and_labels(emotion)
    all_generated_sequences.extend(generated_sequences)
    all_generated_labels.extend(labels)


# In[10]:


# with open(r"D:\Downloads\tfidf_svc_model.pkl", 'rb') as file:
#     loaded_label_encoder, loaded_tfidf_vectorizer, loaded_svc = pickle.load(file)


# In[18]:


from sklearn.metrics import accuracy_score, f1_score, classification_report


# Assuming you have loaded the components using pickle
# with open('model.pkl', 'rb') as file:
#     loaded_label_encoder, loaded_tfidf_vectorizer, loaded_svc, loaded_grid_search = pickle.load(file)

# Sample testing data
test_corpus = all_generated_sequences
test_labels = all_generated_labels

# Transform labels using the loaded label encoder
encoded_test_labels = loaded_label_encoder.transform(test_labels)

# Use the loaded TF-IDF vectorizer and SVC model for prediction
X_test = loaded_tfidf_vectorizer.transform(test_corpus)
y_pred = loaded_svc.predict(X_test)
# print(y_pred)
# Decode the predicted labels back to original labels
# decoded_pred_labels = loaded_label_encoder.inverse_transform(y_pred)
# print(decoded_pred_labels)
# Calculate accuracy
accuracy = accuracy_score(test_labels, y_pred)
print("Accuracy:", accuracy)

# Calculate F1 score
f1 = f1_score(test_labels, y_pred, average='macro')
print("F1 Score:", f1)

# Generate classification report
report = classification_report(test_labels, y_pred, target_names=loaded_label_encoder.classes_)
print("Classification Report:")
print(report)

