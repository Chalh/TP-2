# -*- coding: utf-8 -*-
# ########################################################################################################################
#                                                                                                                      #
# AUTEUR: CHIHEB EL OUEKDI
#  IFT - 7022
#  TRAVAIL-2
# DATE:                                                                                                 #
########################################################################################################################


from nltk.corpus import wordnet as wn
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import os


import re
import sys
reload(sys)
sys.setdefaultencoding('utf8')


rep_pos = "Book/pos_Bk/"
rep_neg = "Book/neg_Bk/"
fichier = open("Book/pos_Bk/pos288.text")
contenu = fichier.read()
ps = PorterStemmer()
nb_fichier =0
for fichier in os.listdir(rep_pos):
    if fichier.endswith(".text"):
        f = open(rep_pos+fichier,"r")
        print rep_pos+fichier
        tokens = word_tokenize(f.read())

        for t in tokens:
            print rep_pos+fichier
            print xyz
            print(ps.stem(t))

print nb_fichier
#nltk.download('wordnet')

#abc = wn.synset('dog.n.1')

#print abc.definition()



