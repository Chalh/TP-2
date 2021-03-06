# -*- coding: utf-8 -*-
# ########################################################################################################################
#                                                                                                                      #
# AUTEUR: CHIHEB EL OUEKDI
#  IFT - 7022
#  TRAVAIL-2
# DATE:                                                                                                 #
########################################################################################################################

import nltk
from nltk.probability import *
from nltk.classify import NaiveBayesClassifier
from nltk.classify import MaxentClassifier
import re
import os

import sys
import importlib
reload(sys)



#Lecture du contenu du fichier d'entrainement
file = open('identification_langue/corpus_entrainement/english-training.txt', 'rb')
contenu_english_train = file.read()
file.close()

########################################################################################################################
#                                                                                                                      #
#               Preparation de la structure de calcul de probabilite avec NLTK
#                                                                                                                      #
########################################################################################################################


def Attributs_phrase(phrase,nbgrm):

    attribut = {}
    phrase = phrase.lower()
    ngram_train = list(nltk.ngrams(phrase,nbgrm))
    freq_dist_ngram = FreqDist(ngram_train)

    for t in ngram_train:
        attribut["count({})".format(t)] = freq_dist_ngram[t]

    return attribut


def Reconnaitre_langue(fichier, classificateur):

    count_langue = {"eng":0,"esp":0,"fr":0, "port":0}
    f = open(fichier, 'rb')

    p1 = re.compile(r"\b((?!\.).)+(.)\b")
    testset=[]
    lignes = f.readlines()
    for ligne in lignes:
        m1 = re.split(r"(.[^\.]+\.)", ligne.decode('utf-8'))
        for x in m1:
            if x is not None:
                testset.append(Attributs_phrase(x, 3))
#    testset = [Attributs_phrase(ligne, 3) for ligne in f]
    for atrb in testset:
        resultat = classificateur.classify(atrb)
        count_langue[resultat] +=1
    langue_texte = "eng"
    nb_phrase_langue =0
    for x in count_langue:
        if count_langue[x] > nb_phrase_langue:
            langue_texte = x
            nb_phrase_langue = count_langue[x]
    print (count_langue)
    return langue_texte

file = open('identification_langue/corpus_entrainement/english-training.txt', 'rb')

train_english = [(Attributs_phrase (ligne,3),"eng") for ligne in file]
file.close()

file = open('identification_langue/corpus_entrainement/espanol-training.txt', 'rb')
train_espagnol = [(Attributs_phrase (ligne,3),"esp") for ligne in file]
file.close()

file = open('identification_langue/corpus_entrainement/french-training.txt', 'rb')
train_french = [(Attributs_phrase (ligne,3),"fr") for ligne in file]
file.close()

file = open('identification_langue/corpus_entrainement/portuguese-training.txt', 'rb')
train_portuguese = [(Attributs_phrase (ligne,3),"port") for ligne in file]
file.close()


trainset = train_english + train_espagnol + train_french + train_portuguese

classifier = NaiveBayesClassifier.train(trainset)

rep_test = "identification_langue/corpus_test1/"
for fabc in os.listdir(rep_test):
    print (str(fabc) + "::" + Reconnaitre_langue(rep_test+fabc, classifier))

classifier.show_most_informative_features()


