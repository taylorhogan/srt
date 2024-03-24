from lark import Lark

with open('syntax.lark', 'r') as file:
    data = file.read()
l = Lark(data)


tree =  (l.parse ("@tmhobservatory image m32 thanks"))

print (tree)
