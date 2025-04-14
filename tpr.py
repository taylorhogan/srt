from astropy.io.votable.converters import Float

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return
        last_node = self.head
        while last_node.next:
            last_node = last_node.next
        last_node.next = new_node

    def print_list(self):
        current_node = self.head
        while current_node:
            print(current_node.data, end=" ")
            current_node = current_node.next
        print()

class FrontierNode ():
    def __init__ (self):
        self.cost = 0
        self.previous = None

class Frontier ():
    def __init__ (self):
        self.ll = LinkedList()


    def insert_sorted(self, new_node):
        if self.ll.head is None:
            self.ll.head = new_node
            return

        if new_node.data <= self.ll.head.data:
            new_node.next = self.ll.head
            self.ll.head = new_node
            return

        current = self.ll.head
        while current.next is not None and current.next.cost < new_node.cost:
            current = current.next

        new_node.next = current.next
        current.next = new_node

cl



