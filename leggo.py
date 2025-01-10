import inspect
import graphviz


class Leggo():
    def __init__(self):
        dot = graphviz.Digraph('block diagram', comment='')
        dot.attr('node', shape='box')
        dot.attr(size='6,6')

        self.my_dot = dot


    def module(self, func):
        def wrapper(*args, **kwargs):
            this_module_library_name = func.__name__
            this_module_instance_name = args[0]
            this_parent_name, parent_instance_name, hname = self.caller_name()
            my_parent_path = hname
            my_path = hname + "/" + this_module_instance_name
            print(hname + "/" + this_module_instance_name)
            self.my_dot.edge(my_parent_path, my_path)
            func(*args, **kwargs)

        return wrapper

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

    def render (self):
        self.my_dot.render(directory='doctest-output', view=True)
        'doctest-output/round-table.gv.pdf'

