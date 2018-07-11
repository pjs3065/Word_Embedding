# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
import pandas as pd
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

def review_to_words(raw_review):
    # 1. remove html
    review_text = BeautifulSoup(raw_review).get_text()

    # 2. remove non-letters
    letters_only = re.sub("[^a-zA-Z]"," ",  review_text)

    # 3. convert to lower case, split into individual words
    words = letters_only.lower().split()

    # 4. In python, searching a set is much faster than searching a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))

    #5. remove stop words
    meaningful_words = [w for w in words if not w in stops]

    #6. Join the words back into on string separated by space. and return the result.
    return(" ".join( meaningful_words))

# main
train = pd.read_csv("labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)

clean_review = review_to_words(train["review"][0])
print(clean_review)




