from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from Bio import AlignIO
from Bio import SeqIO
from Bio import Seq
import os


data_file_path = '/home/qiime2/Desktop/shared/nuage/'
records = SeqIO.parse(data_file_path + 'otu_dna.fasta', 'fasta')
records = list(records)
max_len = max(len(record.seq) for record in records)

for record in records:
    if len(record.seq) != max_len:
        sequence = str(record.seq).ljust(max_len, '.')
        record.seq = Seq.Seq(sequence)
assert all(len(record.seq) == max_len for record in records)

# write to temporary file and do alignment
output_file = '{}_padded.fasta'.format(os.path.splitext(data_file_path + 'otu_dna.fasta')[0])
with open(output_file, 'w') as f:
    SeqIO.write(records, f, 'fasta')
alignment = AlignIO.read(output_file, "fasta")

# Calculate the distance matrix
calculator = DistanceCalculator('identity')
dm = calculator.get_distance(alignment)

# Construct the phylogenetic tree using UPGMA algorithm
constructor = DistanceTreeConstructor()
tree = constructor.upgma(dm)

# Draw the phylogenetic tree
Phylo.write(tree, 'tree.txt', 'newick')
