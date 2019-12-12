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
