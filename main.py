import gensim
from nltk.tokenize import word_tokenize
import os
import nltk


def setup():
    nltk.download('punkt')
    nltk.download('wordnet')
    porter = nltk.PorterStemmer()
    wnl = nltk.WordNetLemmatizer()
    return [porter, wnl]

def get_mappings():
    return {
        "shouldn't": "should not",
        "couldn't": "could not",
        "i'm": "i am",
        "don't": "do not",
        "can't": "cannot",
        "wouldn't": "would not"
    }

def normalise(text, mappings):
    text = text.lower()
    for key, value in mappings.items():
        text = text.replace(key, value)
    return text

def build_categories(porter, wnl):
    raw_documents = ["I'm taking the show on the road.",
                     "My socks are a force multiplier.",
                 "I am the barber who cuts everyone's hair who doesn't cut their own.",
                 "Legend has it that the mind is a mad monkey.",
                "I make my own fun.",
                "Socks are a good thing"]
    print("Number of documents:",len(raw_documents))

    mappings = get_mappings()
    norm_raw_documents = []
    for doc in raw_documents:
        norm_raw_documents.append(normalise(doc, mappings))

    tokens_array = []
    for doc in norm_raw_documents:
        tokens_array.append(word_tokenize(doc))
    print(tokens_array)


    stemmed_tokens_array = []
    for tokens in tokens_array:
        stemmed_tokens_array.append([porter.stem(t) for t in tokens])
    print(stemmed_tokens_array)


    lemmatized_tokens_array = []
    for stemmed_tokens in stemmed_tokens_array:
        lemmatized_tokens_array.append([wnl.lemmatize(t) for t in stemmed_tokens])
    print(lemmatized_tokens_array)


    dictionary = gensim.corpora.Dictionary(lemmatized_tokens_array)
    print(dictionary[5])
    print(dictionary.token2id['road'])
    print("Number of words in dictionary:",len(dictionary))
    for i in range(len(dictionary)):
        print(i, dictionary[i])


    corpus = [dictionary.doc2bow(stemmed_tokens) for stemmed_tokens in stemmed_tokens_array]
    print(corpus)


    tf_idf = gensim.models.TfidfModel(corpus)
    print(tf_idf)
    s = 0
    for i in corpus:
        s += len(i)
    print(s)


    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"
    sims = gensim.similarities.Similarity(dir_path,tf_idf[corpus],
                                          num_features=len(dictionary))
    print(sims)
    print(type(sims))

    return [dictionary, tf_idf, sims]

def categories(porter, wnl, dictionary, tf_idf, sims):
    mappings = get_mappings()
    input_text = normalise("found a sock thing", mappings)
    input_text_tokens = word_tokenize(input_text)
    input_text_stem = [porter.stem(t) for t in input_text_tokens]
    input_text_lem = [wnl.lemmatize(t) for t in input_text_stem]


    query_doc = input_text_lem
    print(query_doc)
    query_doc_bow = dictionary.doc2bow(query_doc)
    print(query_doc_bow)
    query_doc_tf_idf = tf_idf[query_doc_bow]
    print(query_doc_tf_idf)


    print(sims[query_doc_tf_idf])


[porter, wnl] = setup()
[dictionary, tf_idf, sims] = build_categories(porter, wnl)
categories(porter, wnl, dictionary, tf_idf, sims)
