import inspect
import graphviz


class AST_Data (object):
    def __init__(self, name, library_name, carrier = False):
        self.name = name
        self.library_name = library_name
        self.carrier = carrier



class AST_Node(object):

    def __init__(self, data):
        self.data = data
        self.children = []


    def print_me (self):
        print (self.data.name)
        for child in self.children:
            child.print_me()

    def add_child(self, child_data):
        child_node = AST_Node(child_data)
        self.children.append(child_node)
        return child_node






class Leggo():
    def __init__(self):
        self.cluster_num = 0
        self.node_map = {}
        self.parent_stack = []
        self.ast = None
        #self.dot = graphviz.Graph('block diagram', engine='neato')
        self.dot= graphviz.Digraph()
        self.dot.attr(overlap='false')
        self.dot.attr(fontsize='16')
        self.dot.attr(size='6,6')
        self.dot.attr(rankdir='LR')




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



                parent_tree = self.node_map.get(my_parent_path, None)
                if parent_tree is None:
                    parent_data = AST_Data(my_parent_path, this_module_instance_name, True)
                    parent_tree = AST_Node(parent_data)
                    self.ast = parent_tree
                    self.node_map[my_parent_path] = parent_tree

                child_data= AST_Data(my_path, this_module_library_name, carrier)

                child_tree = parent_tree.add_child(child_data)
                self.node_map[my_path] = child_tree





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


    def print_me (self):
        ast = self.ast
        ast.print_me()

    def render_tree (self, parent, child):


        if child.data.carrier:
            self.dot.attr('node',color='lightblue')
            self.dot.attr('node', style='filled')
        else:
            self.dot.attr('node',color='red')
            self.dot.attr('node', style='')

        self.dot.node(child.data.name,shape='box', label=child.data.name+"\n"+child.data.library_name)

        if (parent != None):
            self.dot.edge(parent.data.name, child.data.name)

        for g_child in child.children:
            self.render_tree (child,g_child)

    def render(self, name='graph'):
        ast = self.ast
        self.render_tree (None, ast)
        self.dot.render(name, format='png')
        self.dot.view()



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



