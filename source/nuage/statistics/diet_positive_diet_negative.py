import os
from infrastructure.file_system import get_path

path = get_path()

path_article = path + '/otu_random_forest.txt'
path_our = path + '/otu_diet.txt'

f = open(path_article)
otus_article = f.read().splitlines()
otus_article = [e[:-1] for e in otus_article]
f.close()

top_features_paper = []
f = open(path_article)
for line in f:
    top_features_paper.append(line.replace(' \n', ''))
f.close()

f = open(path_our)
otus_our = f.read().splitlines()
f.close()

intersection = set(otus_article).intersection(set(otus_our))

print(f'number in intersection = {len(intersection)}')