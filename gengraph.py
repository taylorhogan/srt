import graphviz
import json



def build_hierarchy (root, library):
    dot.node(root['name'], root['name'])
    if "children" in root:
        children = root['children']
        for child in children:
            dot.edge(root['name'], child)
            if child in library:
                child_template = library[child]
                build_hierarchy(child_template, library)
            else:
                print ("Could not find "+ child)


dot = graphviz.Digraph('hardware block diagram', comment='')
dot.attr('node', shape='box')
dot.attr(size='6,6')

with open('hardware.json', 'r') as f:
    data = json.load(f)

# Now you can work with the data
modules = data['modules']
for module in modules:
    fields = modules[module]
    if "root"in fields:
        build_hierarchy(fields, modules)







u = dot.unflatten(stagger=3)
#u.view()
dot.render(directory='doctest-output', view=True)
'doctest-output/round-table.gv.pdf'