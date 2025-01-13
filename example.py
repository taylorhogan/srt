
import leggo

l = leggo.Leggo()


@l.module(False)
def gate (name, signal)->{}:
    return True

@l.module(False)
def chip (name, i1)->{}:
    out = gate ("not0", i1)
    not_not = gate("not1", out)
    return not_not

@l.module(True)
def package (name, i1, i2)->{}:
    chip ("c1", i1)
    chip ("c2", i2)
    return None

@l.module(True)
def pcb (name, i1, i2) -> {}:
    package ("p1", i1, i2)
    package ("p2", i1, i2)
    return None



pcb ("my_pcb", 1,2)
l.print_me()
l.render('eda')





