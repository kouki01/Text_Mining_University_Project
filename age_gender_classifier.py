from nltk.corpus.reader.xmldocs import XMLCorpusReader
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from nltk.tag.mapping import map_tag
import numpy as np
import time
import pickle
from textstat.textstat import textstat
from bs4 import BeautifulSoup
import re
from pattern.en import parse
from nltk.util import bigrams


def save_classifier(classifier, tpe, n_feat, n_sample):
    """
    Save the classifier in a pickle file with a noun indicating date, number of documents
    and number of features used in to train the classifier
    :param classifier:
    :param tpe: Naive Bayesien
    :param n_feat: number of features
    :param n_sample: number of documents
    :return:
    """
    file_name = tpe+time.strftime("%H_%M_%d_%m_%Y")+'_feat'+str(n_feat)+'_sample'+str(n_sample)+'.pickle'
    f = open(file_name, 'wb')
    pickle.dump(classifier, f)
    f.close()


def gender_feature(text, feature_vect):
    """
    Extract the gender features
    :param text:
    :param feature_vect: contains a bag of words and a list of bigrams
    :return: a dictionary which contains the feature and its computed value
    """
    #sentence length and vocab features
    tokens = word_tokenize(text.lower())
    sentences = sent_tokenize(text.lower())
    words_per_sent = np.asarray([len(word_tokenize(s)) for s in sentences])

    #bag_of_word features
    bag_dict = {}
    for bag in feature_vect[:29]:
        bag_dict[bag] = bag in tokens

    #bigrams features
    bigram_dict = {}
    for big in feature_vect[29:]:
        bigram_dict[big] = big in bigrams(tokens)

    #POS tagging features
    POS_tag = ['ADJ', 'ADV', 'DET', 'NOUN', 'PRT', 'VERB', '.']
    tagged_word = parse(text, chunks=False, tagset='UNIVERSAL').split()
    simplified_tagged_word = [(tag[0], map_tag('en-ptb', 'universal', tag[1])) for s in tagged_word for tag in s]
    freq_POS = nltk.FreqDist(tag[1] for tag in simplified_tagged_word if tag[1] in POS_tag)

    d = dict({'sentence_length_variation': words_per_sent.std()}, **bag_dict)

    return dict(dict(d, **bigram_dict), **freq_POS)


def age_feature(text, feature_vect):
    """
    Extract age features
    :param text:
    :param feature_vect: contains a bag of words
    :return:a dictionary which contains the feature and its computed value
    """
    tokens = word_tokenize(text.lower())
    #tokens_len = float(len(tokens))
    features={}
    for word in feature_vect:
        features['contains(%s)' % word] = (word in set(tokens))
    return dict(features, **dict({'FRE': textstat.flesch_reading_ease(text), 'FKGL': textstat.flesch_kincaid_grade(text)}))


def create_feature_vect(file_name):
    """
    upload for each class list of element needed in the feature extractor function
    :param file_name: path of the pickle file containing the desired list
    :return:feature list
    """
    fp2 = open(file_name)
    feature_vect = pickle.load(fp2)
    fp2.close()
    print feature_vect
    return feature_vect


def feature_apply(feature_extractor, feature_vect, attrib, n_sample):
    """
    Read ,process the data set and extract features for each document
    :param feature_extractor: function that extract features
    :param feature_vect: contains a list of features
    :param attrib: indicate if the process for gender or age feature extraction
    :param n_sample: number of document to be processed
    :return:vector that contain the extracted features
    """
    corpus_root = 'en'
    newcorpus = XMLCorpusReader(corpus_root, '.*')

    feature_set = []
    doc_list = newcorpus.fileids()

    for doc in doc_list:
        doc = newcorpus.xml(doc)
        print(doc[0].attrib["count"])
        txt = " ".join([doc[0][j].text for j in range(int(doc[0].attrib["count"])) if doc[0][j].text is not None])
        txt = BeautifulSoup(txt).get_text()
        txt = re.sub(r"(\http://\S*)", '', txt)
        if textstat.sentence_count(txt) != 0:
            feature_set.append((feature_extractor(txt, feature_vect), doc.attrib[attrib]))

    return feature_set


def generate_classifier(feature_set, n_sample):
    """
    Divide the feature vector to a training set and development set
    Train a NaiveBayes classifier with the training set and evaluate the accuracy with development set.
    :param feature_set:
    :param n_sample: number of document
    :return:the trained classifier
    """
    train_len = int(len(feature_set)*0.9)
    train_set, dev_set = feature_set[:train_len], feature_set[train_len:]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    save_classifier(classifier, "Naive", len(feature_set[0][0]), n_sample)
    print nltk.classify.accuracy(classifier, dev_set)
    classifier.show_most_informative_features(20)
    return classifier



n_sample = 236600
feature_set = feature_apply(gender_feature, create_feature_vect('gender_words.txt'), 'gender', n_sample)
#feature_set = feature_apply(age_feature, create_feature_vect('age_words.txt'), 'age_group', n_sample)
classifier = generate_classifier(feature_set, n_sample)

