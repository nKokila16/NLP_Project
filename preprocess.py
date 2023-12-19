from autocorrect import Speller
from nltk.tokenize import word_tokenize
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from autocorrect import spell


def to_lower(text):
    spell  = Speller(lang='en')
    texts = spell(text)
    return ' '.join([w.lower() for w in word_tokenize(text)])

def clean_text(lower_case):
    words  = nltk.word_tokenize(lower_case)
    extra_words = re.sub("[^A-Za-z" "]+", " ", str(lower_case))
    with open(r"C:\NLP Project\stop.txt",'r') as sw:
        stop_words=sw.read()
    keywords = [word for word in words if not word in stop_words  and word in extra_words]
    
    return keywords
