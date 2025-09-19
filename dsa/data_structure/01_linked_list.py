# Linked Lists - Singly & Doubly Linked List

from hmac import new
from traceback import print_tb


class SinglyNode:
    def __init__(self, data, next=None):
        self.data = data
        self.next = next

    def __str__(self):
        return str(self.data)


head = SinglyNode(1)
a = SinglyNode(2)
b = SinglyNode(3)
c = SinglyNode(4)

head.next = a
a.next = b 
b.next = c

# print(head)

# Traverse the list - o(n)
current = head
while current:
    print(current)
    current = current.next

# Diplay - O(n)
def display(head):
  curr = head
  elements = []
  while curr:
    elements.append(str(curr.data))
    curr = curr.next
  print(' -> '.join(elements))

display(head)

def search(head, data):
    current = head
    while current:
        if data == current.data:
            return True
        current = current.next
    return False

print(search(head, 4))

# Doubly Linked Lists

class DoublyNode:
    def __init__(self, data, next=None, prev=None):
        self.data = data
        self.next = next
        self.prev = prev

    def __str__(self):
        return str(self.data)

head = tail = DoublyNode(1)

print(f'{head}  { tail}')


# Diplay  - O(n)
def display(head):
  curr = head
  elements = []
  while curr:
    elements.append(str(curr.data))
    curr = curr.next
  print(' -> '.join(elements))

display(head)

# Insert at beginning - O(1)
def insert_at_beginning(head, tail, data):
    new_node = DoublyNode(data, next=head)
    head.prev = new_node
    return new_node, tail

head , tail  = insert_at_beginning(head, tail, 3)
display(head)

# Insert at end - O(n)
def insert_at_end(head, tail, data):
    new_node = DoublyNode(data, prev=tail)
    tail.next = new_node
    return head , new_node

head , tail = insert_at_end(head, tail, 4)
display(head)