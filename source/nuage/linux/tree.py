import csv
from collections import defaultdict
from pprint import pprint


def tree(): return defaultdict(tree)


def tree_add(t, path):
    for node in path:
        t = t[node]


def pprint_tree(tree_instance):
    def dicts(t): return {k: dicts(t[k]) for k in t}

    pprint(dicts(tree_instance))


def csv_to_tree(input):
    t = tree()
    for row in input:
        tree_add(t, row)
    return t


def tree_to_newick(root):
    items = []
    for k in root.keys():
        s = ''
        if len(root[k].keys()) > 0:
            sub_tree = tree_to_newick(root[k])
            if sub_tree != '':
                s += '(' + sub_tree + ')'
        s += k
        items.append(s)
    return ','.join(items)


def csv_to_weightless_newick(input):
    t = csv_to_tree(input)
    #pprint_tree(t)
    return tree_to_newick(t)


with open('/home/qiime2/Desktop/shared/nuage/OTU_classification.csv', mode='r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    data = list(reader)

for row in data:
    row.insert(0, 'root')

result = csv_to_weightless_newick(data)
result += ';'
result_file = open('/home/qiime2/Desktop/shared/nuage/linux/OTU_newick.txt', 'w')
result_file.write(result)
result_file.close()
