import pickle
from nltk.corpus.reader.xmldocs import XMLCorpusReader
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import numpy as np
import pickle
from textstat.textstat import textstat
from bs4 import BeautifulSoup
import HTMLParser
from pattern.en import parse
from nltk.util import bigrams

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

    features = {}
    for word in feature_vect:
        features['contains(%s)' % word] = (word in set(tokens))
    return dict(features, **dict({'FRE': textstat.flesch_reading_ease(text),
                                  'FKGL': textstat.flesch_kincaid_grade(text)}))


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


def fetch_text(doc):
    """
    Process the text contained in a document
    :param doc:
    :return:processed text
    """
    txt = " ".join([doc[0][j].text for j in range(int(doc[0].attrib["count"])) if doc[0][j].text is not None])
    try:
        txt = BeautifulSoup(txt).get_text()
    except HTMLParser.HTMLParseError:
        txt = ""
    return txt


def extract_true_pred(file):
    """
    Extract the true age and gender from truth-en.txt file provided with test corpus
    :param file: path of the truth-en.txt file
    :return:dictionary which has name of xml file as a key and a list value composed from real age and gender values
    """
    f = open(file)
    true_pred = {}
    for line in f.readlines():
        line = line.split(":::")
        true_pred[line[0]+"_en_XXX_XXX.xml"] = [line[1], line[2][:3]]
    f.close()
    return true_pred


def test_set(corpus_dir, feature_extrator, vect_path, i):
    """
    Read ,process the test set and extract features for each document
    :param corpus_dir:path of the test set
    :param feature_extrator: function that extract features
    :param vect_path:
    :param i:index of class in the true_pred dictionay values; if 0 it refers to the gender else it refers to the age
    :return:vector that contain the extracted features
    """
    vect = create_feature_vect(vect_path)
    newcorpus = XMLCorpusReader(corpus_dir, '.*')
    doc_list = newcorpus.fileids()
    test_feature_set = []
    true_pred = extract_true_pred(corpus_dir[:-2]+"truth-en.txt")
    for doc in doc_list:
        xml_name = doc
        doc = newcorpus.xml(doc)
        print(doc[0].attrib["count"])
        txt = fetch_text(doc)
        if (textstat.sentence_count(txt) != 0) and (txt != ""):
            test_feature_set.append((feature_extrator(txt, vect), true_pred[xml_name][i]))

    return test_feature_set


def evaluation(test_feature_set, classifier, classes):
    """
    Evaluate a classifier with a feature test list; for each class in classes
    it calculates the f-measure, recall and precision metrics and display the confusion matrix
    :param test_feature_set:
    :param classifier: trained classifier
    :param classes: list containing the classes of an attribute (gender or age)
    :return:
    """
    import collections
    import nltk.metrics
    import nltk
    ref_set = collections.defaultdict(set)
    test_set = collections.defaultdict(set)
    ref_matrix = []
    test_matrix = []
    for i, (feats, label) in enumerate(test_feature_set):
        ref_set[label].add(i)
        observed = classifier.classify(feats)
        test_set[observed].add(i)
        ref_matrix.append(label)
        test_matrix.append(observed)

    for cls in classes:
        print cls, ' precision:', nltk.metrics.precision(ref_set[cls], test_set[cls])
        print cls, ' recall:', nltk.metrics.recall(ref_set[cls], test_set[cls])
        print cls, ' F-measure:', nltk.metrics.f_measure(ref_set[cls], test_set[cls])


    cm = nltk.ConfusionMatrix(ref_matrix, test_matrix)
    print(cm)
    print(cm.pp(sort_by_count=True, show_percents=True, truncate=9))

#change the parameters for the age_classifier evaluation
# by changing file and uncomment the call of the evaluation function
f = open("gender_classifier.pickle")
classifier = pickle.load(f)
f.close()

test_feature_set = test_set("pan13-test-corpus1\\en", gender_feature, "gender_words.txt", 0)

f = open("test_feature_set.pickle", 'w')
pickle.dump(test_feature_set, f)
f.close

print nltk.classify.accuracy(classifier, test_feature_set)
#evaluation(test_feature_set, classifier, ['10s', '20s', '30s'])
evaluation(test_feature_set, classifier, ['female', 'male'])