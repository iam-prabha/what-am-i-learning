def my_dec(func):
    def wrapper():
        print(f'something is happening before the function is called')
        func()
        print(f'something is happening after the function is called')
    return wrapper

@my_dec
def say_hello():
    print(f'Arigato')

say_hello()