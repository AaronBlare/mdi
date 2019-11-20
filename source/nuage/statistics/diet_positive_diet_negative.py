import os
from infrastructure.file_system import get_path

path = get_path()

path_article = path + '/original/diet_negative.txt'
path_our = path + '/otu_diet.txt'

f = open(path_article)
otus_article = f.read().splitlines()
f.close()

f = open(path_our)
otus_our = f.read().splitlines()
f.close()

intersection = set(otus_article).intersection(set(otus_our))

print(f'number in intersection = {len(intersection)}')