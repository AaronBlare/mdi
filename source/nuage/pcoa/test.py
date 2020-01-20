from io import StringIO
from skbio import TreeNode
from skbio.diversity.beta import unweighted_unifrac

tree = TreeNode.read(StringIO('(((((OTU1:0.5,OTU2:0.5):0.5,OTU3:1.0):1.0):0.0,'
                              '(OTU4:0.75,(OTU5:0.5,((OTU6:0.33,OTU7:0.62):0.5'
                              ',OTU8:0.5):0.5):0.5):1.25):0.0)root;'))

u_counts = [1, 0, 0, 4, 1, 2, 3, 0]
v_counts = [0, 1, 1, 6, 0, 1, 0, 0]
otu_ids = ['OTU1', 'OTU2', 'OTU3', 'OTU4', 'OTU5', 'OTU6', 'OTU7', 'OTU8']
uu = unweighted_unifrac(u_counts, v_counts, otu_ids, tree)
print(round(uu, 2))


from ete3 import PhyloTree

t = PhyloTree('((H,I), A, (B,(C,D)))root;', format=1)
print(t)
D = t&"D"
# Get the path from B to the root
node = D
path = []
while node.up:
  path.append(node)
  node = node.up
# I substract D node from the total number of visited nodes
print("There are", len(path)-1, "nodes between D and the root")
A = t&"A"
# Get the path from B to the root
node = A
path = []
while node.up:
  path.append(node)
  node = node.up
print("There are", len(path)-1, "nodes between A and the root")
print(t.children)
print(t.get_children())
print(t.up)
print(t.name)
print(t.dist)
print(t.is_leaf())
print(t.get_tree_root())
print(t.children[0].get_tree_root())
print(t.children[0].children[0].get_tree_root())
# You can also iterate over tree leaves using a simple syntax
for leaf in t:
    print(leaf.name)
