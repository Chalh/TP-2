def gender_features(word):
     return {'last_letter': word[-1]}
import nltk

from nltk.corpus import names
labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
                 [(name, 'female') for name in names.words('female.txt')])
import random
random.shuffle(labeled_names)

def get_word_features(wordlist):

    wordlist = nltk.FreqDist(wordlist)

    word_features = wordlist.keys()

    return word_features
def word_feats(words):
    return dict([(word, True) for word in words])
def get_words_in_tweets(tweets):

    all_words = []

    for (words, sentiment) in tweets:

      all_words.extend(words)

    return all_words


def gender_features2(name):
    features = {}
    features["first_letter"] = name[0].lower()
    features["last_letter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count({})".format(letter)] = name.lower().count(letter)
        features["has({})".format(letter)] = (letter in name.lower())
    return features


featuresets = [(gender_features2(n), gender) for (n, gender) in labeled_names]



train_set, test_set = featuresets[500:], featuresets[:500]



classifier = nltk.NaiveBayesClassifier.train(train_set)


print(classifier.classify(gender_features('Neo')))
