# -*- coding: utf-8 -*-
# ########################################################################################################################
#                                                                                                                      #
# AUTEUR: CHIHEB EL OUEKDI
#  IFT - 7022
#  TRAVAIL-2
# DATE:                                                                                                 #
########################################################################################################################


import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.probability import *
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import os
from nltk.stem import WordNetLemmatizer
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier

import sys
reload(sys)
sys.setdefaultencoding('utf8')


rep_pos = "Book/pos_Bk/"
rep_neg = "Book/neg_Bk/"

ps = PorterStemmer()
# préparation de la liste stop_words
stop_words = set(stopwords.words('english'))

nb_fichier =0

# Récuperation de la liste des fichier
Liste_fich_positif = [(f, 'pos') for f in os.listdir(rep_pos)]

Liste_fich_negatif = [(f, 'neg') for f in os.listdir(rep_neg)]



classifier = NaiveBayesClassifier.train(Liste_fich_negatif)
# ######################################################################################################################
#   Entrainement 90%du corpus, Test 10% du corpus                                                                                                                   #
#
########################################################################################################################

index_debut_test = 1
index_fin_test = 100

#Entrainement NaiveBayes

def get_words_in_tweets(tweets):

    all_words = []

    for (words, sentiment) in tweets:

      all_words.extend(words)

    return all_words

def get_word_features(wordlist):

    wordlist = nltk.FreqDist(wordlist)

    word_features = wordlist.keys()

    return word_features
def word_feats(words):
    return dict([(word, True) for word in words])

while(index_fin_test<1001):
    for i in range(1,index_debut_test) + range (index_fin_test,1001):
        continue

#    print index_debut_test.__str__() + ":" + index_fin_test.__str__()+ ":" +nb_fichier.__str__()
    nb_fichier = 0
    index_debut_test = index_debut_test + 100
    index_fin_test = index_fin_test + 100

for fichier in os.listdir(rep_pos):
    if fichier.endswith(".text"):
        f = open(rep_pos+fichier,"r")
        tokens = word_tokenize(f.read())
        freq_dist_unigram = FreqDist(tokens)
        #  analyse grammaticale (POS tagging)
        liste_postagging = nltk.pos_tag(tokens)
        
        for t in tokens:

            try:
                # Normalisation des mots
                wordnet_lemmatizer = WordNetLemmatizer()
                w = wordnet_lemmatizer.lemmatize(ps.stem(t))
            except ValueError:
                print rep_pos + fichier
                nb_fichier=nb_fichier+1



print nb_fichier
#nltk.download('wordnet')

#abc = wn.synset('dog.n.1')

#print abc.definition()



