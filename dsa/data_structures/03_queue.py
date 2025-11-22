# Queues - data structure that represents a dynamic set of data

# A queue is a linear data structure that follows the First-In, First-Out (FIFO) principle.
# This means that the first element added to the queue will be the first one to be removed.
# Queues are commonly used in scenarios where order needs to be preserved, such as:
# - Managing tasks in a printer queue
# - Handling requests in web servers
# - Breadth-first search in graphs
# Basic operations of a queue include:
# - Enqueue: Add an element to the end of the queue
# - Dequeue: Remove an element from the front of the queue
# - Peek/Front: View the element at the front without removing it
# - isEmpty: Check if the queue is empty

from collections import deque


class Queue:
    def __init__(self):
        self.data = deque()

    def enqueue(self, node):
        self.data.append(node)

    def dequeue(self):
        self.data.popleft()

    def print_queue(self):
        print(self.data)


def main():
    queue = Queue()
    queue.print_queue()

    queue.enqueue(1)
    queue.print_queue()
    queue.enqueue(2)
    queue.print_queue()
    queue.enqueue(3)
    queue.print_queue()

    queue.dequeue()
    queue.print_queue()

    queue.enqueue(4)
    queue.print_queue()

    queue.dequeue()
    queue.print_queue()


main()
