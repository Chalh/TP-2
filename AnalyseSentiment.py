from nltk.corpus import wordnet as wn
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import os


rep_pos = "Book/pos_Bk/"
rep_neg = "Book/neg_Bk/"
fichier = open("Book/pos_Bk/pos288.text")
contenu = fichier.read()

nb_fichier =0
for fichier in os.listdir(rep_pos):
    if fichier.endswith(".text"):
        f = open(rep_pos+fichier,"r")
        print fichier
        print os.path.dirname(fichier)
        contenu = f.read()
        print contenu
        nb_fichier = nb_fichier+1
#        words = word_tokenize(contenu)
#        ps = PorterStemmer()

#        for w in words:
#            print(w,ps.stem(w))

print nb_fichier
#nltk.download('wordnet')

#abc = wn.synset('dog.n.1')

#print abc.definition()



