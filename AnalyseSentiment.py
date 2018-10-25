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

import sys
reload(sys)
sys.setdefaultencoding('utf8')


rep_pos = "Book/pos_Bk/"
rep_neg = "Book/neg_Bk/"

ps = PorterStemmer()
# préparation de la liste stop_words
stop_words = set(stopwords.words('english'))

nb_fichier =0
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



