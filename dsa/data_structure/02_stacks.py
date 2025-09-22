# Stacks - a fundamental data structure that represents a dynamic collection of elements.
# A stack follows the Last-In, First-Out (LIFO) principle:
#   - The last element added (pushed) to the stack is the first one to be removed (popped).
# Common stack operations:
#   - push: Add an element to the top of the stack.
#   - pop: Remove and return the top element from the stack.
#   - peek/top: View the top element without removing it.
#   - isEmpty: Check if the stack is empty.
# Stacks are widely used in:
#   - Function call management (call stack)
#   - Undo mechanisms in editors
#   - Expression evaluation and syntax parsing
class Stack:
    def __init__(self):
        self.data = []

    def push(self, node):
        self.data.append(node)

    def pop(self):
        self.data.pop()

    def print_stack(self):
        print(self.data)


def main():
    stack =  Stack()
    stack.print_stack()
    
    stack.push(1)
    stack.print_stack()
    stack.push(2)
    stack.print_stack()
    stack.push(3)
    stack.print_stack()

    stack.pop()
    stack.print_stack()

    stack.push(4)
    stack.print_stack()

main()