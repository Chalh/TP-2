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
#nltk.download('averaged_perceptron_tagger')

rep_pos = "Book/pos_Bk/"
rep_neg = "Book/neg_Bk/"

ps = PorterStemmer()
# préparation de la liste stop_words
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()
nb_fichier =0

def Attributs_mots(texte):
    tokens = word_tokenize(texte)
    freq_texte = FreqDist(tokens)
    texte_filtred = []
    for t in tokens:

        if (freq_texte[t] > 1) and (not t in stop_words):
            try:
                                # Normalisation des mots
                w = wordnet_lemmatizer.lemmatize(ps.stem(t))
            except ValueError:
                w=t

            texte_filtred.append((t, freq_texte[t],w))

    return texte_filtred


def Attributs_mots2(texte):
    attribut = {}
    tokens = word_tokenize(texte)
    txt_pos_tag=nltk.pos_tag(tokens)
    freq_texte = FreqDist(tokens)
    texte_normalise=[]
    for t in tokens:
        # Normalisation des mots
        if (freq_texte[t] > 1) and (not t in stop_words):
            try:
                w = wordnet_lemmatizer.lemmatize(ps.stem(t))
            except ValueError:
                w=t
            texte_normalise.append(w)


    freq_texte_norm = FreqDist(texte_normalise)

    attribut["mot"] = w


    return texte_filtred


# Récuperation de la liste des fichier
Liste_fich_positif = [rep_pos+f for f in os.listdir(rep_pos)]

Liste_fich_negatif = [rep_neg+f for f in os.listdir(rep_neg)]

#negfeats = [(word_feats(open(rep_pos+f,"r").read()), 'neg') for f in os.listdir(rep_pos)]

#classifier = NaiveBayesClassifier.train(negfeats)
# ######################################################################################################################
#   Entrainement 90%du corpus, Test 10% du corpus                                                                                                                   #
#
########################################################################################################################

index_debut_test = 1
index_fin_test = 100


#Entrainement NaiveBayes
#posfeats = [((Attributs_mots (f.read())), 'neg') for f in Liste_fich_positif]
#negfeats = [((Attributs_mots (f.read())), 'neg') for f in Liste_fich_negatif]

posfeats = []
for fichier in Liste_fich_positif:
   f = open(fichier,"r")
   posfeats.append((Attributs_mots2 (f.read()),"pos"))

trainpos = posfeats[:950]
testpos = posfeats[951:]

#classifier = NaiveBayesClassifier.train(trainpos)
##print 'accuracy:', nltk.classify.util.accuracy(classifier, testpos)
#classifier.show_most_informative_features()

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
        xyz=Attributs_mots2 (f.read())

print nb_fichier
#nltk.download('wordnet')

#abc = wn.synset('dog.n.1')

#print abc.definition()



