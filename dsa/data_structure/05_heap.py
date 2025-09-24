# heaps

def left(i):
    return 2*i

def right(i):
    return 2*i + 1

def perent(i):
    return i // 2

def max_heapify(a, heap_size, i):
    l = left(i)
    r = right(i)

    largest = i

    if l < heap_size and a[l] > a[i]:
        largest = l

    if r < heap_size and a[r] > a[i]:
        largest = r

    if largest!= i:
        # swap elements
        a[i], a[largest] = a[largest], a[i]
        max_heapify(a, heap_size, i)

def build_max_heap(a):
    heap_size = len(a)

    for i in range(heap_size // 2, 0, -1):
        max_heapify(a, heap_size, i)

def main():
    # root is at index 1
    # it can be at index zero but see here: https://www.quora.com/Why-do-indexes-for-heaps-start-at-1
    # and: https://stackoverflow.com/questions/22900388/why-in-a-heap-implemented-by-array-the-index-0-is-left-unused

    a = [None, 0,5,10,15,20,25,28,35,40,45,50]

    build_max_heap(a)

    # print heap staring with the root at index 1
    print(f'Heap: {a[1:]}')


main()