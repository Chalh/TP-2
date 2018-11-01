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
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.probability import *
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import os
from nltk.stem import WordNetLemmatizer
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.classify import MaxentClassifier

import sys
reload(sys)
sys.setdefaultencoding('utf8')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('sentiwordnet')

rep_pos = "Book/pos_Bk/"
rep_neg = "Book/neg_Bk/"

ps = PorterStemmer()
# préparation de la liste stop_words
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

nb_fichier =0

def Attributs_mots(texte):
    attribut = {}
    type_a_considere = ["NN" , "VB" , "JJ" , "AD"]
    type_sentiword = {"NN":"n" , "VB":"v" , "JJ":"a" , "AD":"r"}
    tokens = word_tokenize(texte)
    freq_texte = FreqDist(tokens)
    texte_normalise=[]
    texte_frequent =[]
    for t in tokens:
        # Normalisation des mots
        if (freq_texte[t] > 1) and (not t in stop_words):
        #if not t in stop_words:
            texte_frequent.append(t)

    txt_pos_tag=nltk.pos_tag(texte_frequent)
    freq_texte_frequent = FreqDist(texte_frequent)
    nb_mot_positif = 0
    for bgr in txt_pos_tag:
        for x in type_a_considere:
            if x in bgr[1]:
                type_sw = type_sentiword[x]

                try:
                    w = wordnet_lemmatizer.lemmatize(ps.stem(bgr[0]))
                    try:
                        senti_result = swn.senti_synset(bgr[0] + "." + type_sw + ".01")
                        if senti_result.pos_score() > senti_result.neg_score():
                            nb_mot_positif += 1
                    except:
                        pass
                except :
                    w=bgr[0]

                attribut["count({})".format(w)] = freq_texte_frequent[bgr[0]]
                break
    attribut["nb_mot_positif"] = nb_mot_positif
    return attribut


# Récuperation de la liste des fichier
Liste_fich_positif = [rep_pos+f for f in os.listdir(rep_pos)]

Liste_fich_negatif = [rep_neg+f for f in os.listdir(rep_neg)]

#50 articles train 10 articles test
#listetrainpos = Liste_fich_positif[:50]
#listetrainneg = Liste_fich_negatif[:50]
#listetestpos = Liste_fich_positif[991:]
#listetestneg = Liste_fich_negatif[991:]

#90% train 10% test
listetrainpos = Liste_fich_positif[:950]
listetrainneg = Liste_fich_negatif[:950]
listetestpos = Liste_fich_positif[951:]
listetestneg = Liste_fich_negatif[951:]


#10% test 90% train
#listetrainpos = Liste_fich_positif[51:]
#listetrainneg = Liste_fich_negatif[51:]
#listetestpos = Liste_fich_positif[:50]
#listetestneg = Liste_fich_negatif[:50]

testneg = [Attributs_mots (open(f,"r").read()) for f in listetestneg]
testpos = [Attributs_mots (open(f,"r").read()) for f in listetestpos]

testset = testneg+testpos

testnegm = [(Attributs_mots (open(f,"r").read()),"neg") for f in listetestneg]
testposm = [(Attributs_mots (open(f,"r").read()),"pos") for f in listetestpos]

testsetm = testnegm+testposm


# ######################################################################################################################
#   Entrainement 90%du corpus, Test 10% du corpus                                                                                                                   #
#
########################################################################################################################

index_debut_test = 1
index_fin_test = 100


#Entrainement NaiveBayes
trainpos = []
for fichier in listetrainpos:
   f = open(fichier,"r")
   try:
       trainpos.append((Attributs_mots (f.read()),"pos"))
   except ValueError:
       continue

trainneg = []
for fichier in listetrainneg:
   f = open(fichier,"r")
   try:
       trainneg.append((Attributs_mots (f.read()),"neg"))
   except ValueError:
       continue


trainset = trainneg+trainpos

classifier = NaiveBayesClassifier.train(trainset)

print nltk.classify.util.accuracy(classifier, testsetm)
classifier.show_most_informative_features()

classifier = MaxentClassifier.train(trainset)

print nltk.classify.util.accuracy(classifier, testsetm)
classifier.show_most_informative_features()



