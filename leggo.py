import inspect
import graphviz


class ASTNode (object):
    def __init__(self, name='root', carrier = False):
        self.name = name
        self.carrier = carrier
        self.node_map = {}

class AST(object):

    def __init__(self, name='root'):
        self.name = name
        self.children = []


    def __repr__(self):
        return self.name

    def add_child(self, node):
        assert isinstance(node, AST)
        self.children.append(node)
        self.node_map.update({node.name: node})



    def find_node (self, name):
        return self.node_map.get(name)



class Leggo():
    def __init__(self):
        dot = graphviz.Digraph('block diagram', comment='')
        dot.attr('node', shape='box')
        dot.attr(size='100,100')

        self.stack = []
        self.stack.append(dot)

        self.cluster_num = 0
        self.top = dot

    def next_cluster_name(self):
        self.cluster_num += 1
        return "cluster_" + str(self.cluster_num)

    def module(self, carrier):
        def module_decorator(func):
            def wrapper(*args, **kwargs):
                this_module_library_name = func.__name__
                this_module_instance_name = args[0]
                this_parent_name, parent_instance_name, hname = self.caller_name()
                my_parent_path = hname
                my_path = hname + "/" + this_module_instance_name
                print(hname + "/" + this_module_instance_name)

                current_dot = self.stack[-1]

                current_dot.node(my_parent_path, shape='box')
                if carrier is True:
                    next_cluster_name = self.next_cluster_name()
                    with current_dot.subgraph(name=next_cluster_name) as next_dot:
                        self.stack.append (next_dot)

                        current_dot = self.stack[-1]

                        #current_dot.attr(style='filled', color = 'lightgrey')
                        #current_dot.node_attr.update(style='filled', color='white')
                        current_dot.node(my_path, color='red')
                        current_dot.edge(my_parent_path, my_path)
                        print(carrier)
                        func(*args, **kwargs)
                        self.stack.pop()
                else:
                    current_dot.node(my_path, color='red')
                    current_dot.edge(my_parent_path, my_path)
                    func(*args, **kwargs)


            return wrapper

        return module_decorator


    def caller_name(self):
        cf = inspect.currentframe().f_back.f_back
        hname = ""
        while cf is not None:
            this_name = cf.f_locals.get('name', "root")
            hname = "/" + this_name + hname
            if this_name == "root":
                break;
            cf = cf.f_back.f_back

        caller_frame = inspect.currentframe().f_back.f_back
        caller_name = caller_frame.f_code.co_name
        caller_instance_name = caller_frame.f_locals.get('name', "root")
        return caller_name, caller_instance_name, hname


    def render(self):
        self.top.render('graph', format='png')

if __name__ == '__main__':
    dot = graphviz.Digraph('block diagram', comment='')

    dot.node ("/root")
    subgraph = None
    #with dot.subgraph(name="cluster_0") as c0:

        # c0.node ("/root/PCB",color="red")
        # c0.edge ("/root", "/root/PCB")
        # with c0.subgraph(name="cluster_1") as c1:
        #     with c0.subgraph(name="cluster_2") as c2:
        #         c2.node("/root/PCB/P2", color="red")
        #         c2.edge("/root/PCB", "/root/PCB/P2")
        #
        #         with c2.subgraph(name="cluster_4") as c4:
        #             c4.node("/root/PCB/P2/c2", color="red")
        #             c4.edge("/root/PCB/P2", "/root/PCB/P2/c2")
        #
        #     with c0.subgraph(name="cluster_2") as c9:
        #         c9.node("/root/PCB/P1", color="red")
        #         c9.edge("/root/PCB", "/root/PCB/P1")
        #         with c9.subgraph(name="cluster_3") as c3:
        #             c3.node("/root/PCB/P1/c1", color="red")
        #             c3.edge("/root/PCB/P1", "/root/PCB/P1/c1")

    print ("dome")
    dot.render('graph', format='png')



