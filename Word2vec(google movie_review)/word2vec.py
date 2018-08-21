import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
import pickle
import logging
from gensim.models import word2vec

logging.basicConfig(format='%(asctime)s: %(levelname)s:%(message)s', level=logging.INFO)

# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review, "lxml").get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)


def save_sentences():
    # Read data from files
    train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
    unlabeled_train = pd.read_csv("unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

    # Verify the number of reviews that were read (100,000 in total)
    print("Read %d labeled train reviews, %d labeled test reviews, ""and %d unlabeled reviews\n"\
          %(train["review"].size, test["review"].size, unlabeled_train["review"].size))

    # Download the punkt tokenizer for senetence splitting
    # nltk.download()

    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    sentences = []

    print("Parsing sentences from training set...")
    for i, review in enumerate(train["review"]):
        if(i+1)%1000 == 0:
            print("[training set]{} of {}".format(i+1, train["review"].size))
        sentences += review_to_sentences(review, tokenizer)


    for i, review in enumerate(unlabeled_train["review"]):
        if(i+1) % 1000 == 0:
            print("[training set]{} of {}".format(i+1, train["review"].size))
        sentences += review_to_sentences(review, tokenizer)

    with open("sentences.pickle", "wb") as f:
        pickle.dump(sentences, f)

def load_sentences():
    print("Load sentences from pickle...")
    with open("sentences.pickle", "rb") as f:
        sentences = pickle.load(f)
    return sentences

def word2vec_model(num_features, min_word_count, num_workers, context_size, downsampling):
    print("Training model...")

    sentences = load_sentences()
    model = word2vec.Word2Vec(sentences,
                              workers=num_workers,
                              size=num_features,
                              min_count = min_word_count,
                              window=context_size,
                              sample=downsampling)

    # memory-efficient
    model.init_sims(replace=True)

    # save model
    model_name = "300features_40minwords_10context"
    model.save(model_name)

    return model


if __name__=="__main__":
    num_features = 300
    min_word_count=40
    num_workers=6
    context_size=10
    downsampling=1e-5

    # for the first time
    save_sentences()
    model = word2vec_model(num_features, min_word_count, num_workers, context_size, downsampling)

    # for the second time
    # model = word2vec.Word2Vec.load("300features_40minwords_10context")

    print("Q1. doesnt_match [dog cat rabbit fire]")
    print(model.doesnt_match("dog cat rabbit fire".split()))

    print("Q2. doesnt_match [china korea america japan]")
    print(model.doesnt_match("china korea america japan".split()))

    print("Q3. doesnt_match [bag hat pants apple]")
    print(model.doesnt_match("bag hat pants apple".split()))

    print("Q4.most_similar[boy]")
    print(model.most_similar("boy"))

    print("Q5.most_similar[girl]")
    print(model.most_similar("girl"))

    print("Q6.most_similar[lovely]")
    print(model.most_similar("lovely"))