import graphviz
import json

dot = graphviz.Digraph('hardware block diagram', comment='')



def build_hierarchy (root, library):
    dot.node(root['name'], root['name'])
    if "children" in root:
        children = root['children']
        for child in children:
            dot.edge(root['name'], child)
            child_template = library[child]
            build_hierarchy(child_template, library)



with open('hardware.json', 'r') as f:
    # Load the JSON data into a Python dictionary
    data = json.load(f)

# Now you can work with the data
modules = data['modules']
for module in modules:
    fields = modules[module]
    if "root"in fields:
        build_hierarchy(fields, modules)








dot.render(directory='doctest-output', view=True)
'doctest-output/round-table.gv.pdf'