data_file_path = '/home/qiime2/Desktop/shared/nuage/linux/'
file_name = 'tree.nwk'

f = open(data_file_path + file_name, 'r+')
line = f.readline()
f.close()

tree = ''
tree_chars = ['(', ')', ':', ',', ';']
curr_otu = ''
for char in line:
    if char in tree_chars:
        if len(curr_otu) == 0:
            tree += char
        elif len(curr_otu.split('_')) == 3:
            tree += '\'OTU_' + curr_otu.split('_')[0] + '\'' + char
            curr_otu = ''
        else:
            tree += curr_otu + char
            curr_otu = ''
    else:
        curr_otu += char

f = open(data_file_path + 'otu_tree.txt', 'w')
f.write(tree + '\r\n')
f.close()
