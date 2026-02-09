def simple_decorator(func):
    def wrapper(name):
        print("good morning")
        result = func(name)
        print("thanks for using the function")
        return result
    return wrapper


@simple_decorator
def display(name):
    print(f"hello my name is {name}")


display("ananda")
