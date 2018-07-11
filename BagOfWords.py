# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
import pandas as pd
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords


train = pd.read_csv("labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)

example1 = BeautifulSoup(train["review"][0])

letters_only = re.sub("[^a-zA-Z]",
                      " ",
                      example1.get_text() )

lower_case = letters_only.lower()
words = lower_case.split()

#nltk.download()  # Download text data sets, including stop words

print (train["review"][0])
print(example1.get_text())
print(letters_only)
print(stopwords.words("korea"))