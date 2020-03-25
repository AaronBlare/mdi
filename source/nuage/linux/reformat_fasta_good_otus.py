data_file_path = '/home/qiime2/Desktop/shared/nuage/'
file_name = 'otu_dna.fasta'

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

good_otus = []
data_file_path = '/home/qiime2/Desktop/shared/nuage/linux/'
file_name = 'good_otus.txt'
f = open(data_file_path + file_name, 'r+')
for line in f:
    good_otus.append('>' + line.rstrip())
f.close()

f = open(data_file_path + 'good_otu_dna.fasta', 'w')
for key in otu_data:
    if key in good_otus:
        f.write(key + '\r\n')
        f.write(otu_data[key] + '\r\n')
f.close()
