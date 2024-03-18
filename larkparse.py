from lark import Lark

with open('syntax.lark', 'r') as file:
    data = file.read()
l = Lark(data)


print (l.parse ("image m32"))

