
import re

rep_test = "identification_langue/corpus_test1/"
fihcier ="test20.txt"


p1 = re.compile(r".*\.")

file = open(rep_test+fihcier,"r")
n=0
for line in file:
    m1 = line #p1.search(line)
    if m1 is not None:
        print m1#.group(0)
        n+=1
print n