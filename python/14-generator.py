# yield vs return

# `yield` pauses the function and returns a value,
# while `return` terminates the fuction


def count_up_to(n):
    i = 0
    while i < n:
        yield i
        i += 1


counter = count_up_to(5)
print(next(counter))
print(next(counter))
