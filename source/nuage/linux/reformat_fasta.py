data_file_path = '/home/qiime2/Desktop/shared/nuage/'
file_name = 'otusn.fasta'

otu_data = {}
last_otu = ''
f = open(data_file_path + file_name, 'r+')
for line in f:
    if line.startswith('>'):
        last_otu = line.rstrip()
        otu_data[last_otu] = ''
    else:
        otu_data[last_otu] += line.rstrip()
f.close()

length = {}
for key in otu_data:
    length[key] = len(otu_data[key])

f = open(data_file_path + 'otu_dna.fasta', 'w')
for key in otu_data:
    f.write(key + '\r\n')
    f.write(otu_data[key] + '\r\n')
f.close()
