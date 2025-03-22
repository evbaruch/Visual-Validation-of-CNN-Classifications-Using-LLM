from logger import setup_logger, log_function_call, log_class_methods

# Set up logger with a custom name (e.g., "experiment_1")
setup_logger("experiment_2")

@log_function_call
def add(a, b):
    return a + b

@log_function_call
def divide(a, b):
    return a / b  # Will raise an exception if b == 0

@log_class_methods
class Calculator:
    def multiply(self, x, y):
        return x * y

    def subtract(self, x, y):
        return x - y

if __name__ == "__main__":
    print(add(3, 5))
    try:
        print(divide(10, 0))  # This will log an exception
    except ZeroDivisionError:
        print("Caught a division by zero error!")

    calc = Calculator()
    print(calc.multiply(4, 6))
    print(calc.subtract(10, 3))
