# Converted from 05-oops_concept.ipynb

# ======================================================================
# ### **Core Concept of oop**
# ======================================================================

# ======================================================================
# There are four pillars of OOP, often referred to as **P.I.E.A**.:
# 
# * Polymorphism
# 
# * Inheritance
# 
# * Encapsulation
# 
# * Abstraction
# ======================================================================

# ======================================================================
# **Class**
# 
# A class is a blueprint, a template, or a prototype from which objects are created. It defines a set of attributes (data) and methods (functions) that the created objects will have.
# 
# * **Definition:** You define a class using the class keyword.
# 
# * **Naming Convention:** Class names typically use `PascalCase` (e.g., `MyClass`, `Car`, `Person`).
# ======================================================================

# %%
# Defining a simple class
class Dog:
    # class attribute (shared by all instances of Dog)
    species = "Canis familiaris"

    # The constructor method: __init__
    # 'self' refers to the instance of the class being created
    def __init__(self, name, breed, age):
        # Instance attributes (unique to each object)
        self.name = name
        self.breed = breed
        self.age = age
        self.is_hungry = True  # Default state

    # Instance method (behavior of the object)
    def bark(self):
        return f"{self.name} says woof!"

    def eat(self):
        if self.is_hungry:
            self.is_hungry = False
            return f"{self.name} is full now"
        else:
            return f"{self.name} is not hungry right now!"

    # Special method for string representation (human-readable)
    def __str__(self):
        return f"{self.name} ({self.breed}, {self.age} years old)"

    # Special method for "official" string representation (for debugging)
    def __repr__(self):
        return f"Dog(name={self.name}, breed={self.breed}, age={self.age})"

# ======================================================================
# **Object (Instance)**
# 
# An object is an instance of a class. It's a concrete realization of the blueprint. When you create an object, memory is allocated to store its specific attributes.
# 
# **Creation:** You create an object by calling the class as if it were a function.
# ======================================================================

# %%
# Creating object (instance) of the Dog class
my_dog = Dog("Buddy", "Golden Retriever", 5)
your_dog = Dog("Max", "Labrador", 3)

# %%
# Accessing attributes
print(f"{my_dog.name} is a {my_dog.breed} and is {my_dog.age} years old.")
print(f"{your_dog.name} is a {your_dog.breed} and is {your_dog.age} years old.")

# %%
# Calling methods
print(my_dog.bark())
print(your_dog.eat())
print(your_dog.eat())  # Try eating again
print(f"Is {my_dog.name} hungry? {my_dog.is_hungry}")

# %%
# using __str__ and __repr__
print(my_dog)  # Calls __str__
print(repr(your_dog))  # Calls __repr__

# ======================================================================
# **Inheritance**
# 
# **Concept:** Inheritance allows a new class (the **subclass** or **child class**) to inherit properties (attributes and methods) from an existing class (the **superclass** or **parent class**). This promotes code reusability and establishes an "is-a" relationship (e.g., a `GoldenRetriever` is a `Dog`).
# 
# **Syntax:** `class ChildClass(ParentClass):`
# 
# `super():` Used to call methods of the parent class, especially the parent's  `__init__` method.
# ======================================================================

# %%
class Animal:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def speak(self):
        raise NotImplementedError("Subclasses must implement this method")

    def __str__(self):
        return f"{self.name} (Age: {self.age})"


# Dog inherits from Animal
class Dog(Animal):
    def __init__(self, name, age, breed):
        super().__init__(name, age)  # call parent's constructor
        self.breed = breed

    def speak(self):  # overiding the speak method
        return f"{self.name} says woof!"


# Cat inherits from Animal
class Cat(Animal):
    def __init__(self, name, age, color):
        super().__init__(name, age)
        self.color = color

    def speak(self):  # overiding the speak method
        return f"{self.name} says meow!"


my_dog = Dog("Max", 7, "German Shepherd")
my_cat = Cat("Whiskers", 4, "Tabby")

print(my_dog)
print(my_dog.speak())

print(my_cat)
print(my_cat.speak())

# ======================================================================
# **Multiple Inheritance:** Python supports inheriting from multiple parent classes. While powerful, it can lead to complex method resolution orders (MRO) and potential diamond problems. Use with caution.
# ======================================================================

# %%
class Flyer:
    def fly(self):
        return "I can fly!"


class Swimmer:
    def swim(self):
        return "I can swim!"


class Duck(Flyer, Swimmer):  # Inherits from both
    def quack(self):
        return "Quack!"


d = Duck()
print(d.fly())
print(d.swim())
print(d.quack())

# ======================================================================
# **Polymorphism**
# 
# **Concept:** Polymorphism means "many forms." In OOP, it refers to the ability of objects of different classes to respond to the same method call in their own way. This is often achieved through method overriding (as seen with `speak()` above) and "duck typing."
# 
# **Duck Typing:** A core Python concept. If an object "walks like a duck and quacks like a duck," then it's treated as a duck. Python doesn't care about the object's explicit type, only whether it has the necessary methods or attributes.
# ======================================================================

# %%
# Reusing our Dog and Cat classes from Inheritance example


def make_animal_speak(animal):
    # We don't care if 'animal' is a Dog or a Cat, only that it has a 'speak' method.
    print(f"{animal.name}: {animal.speak()}")


make_animal_speak(my_dog)  # Output: Max: Max says Woof!
make_animal_speak(my_cat)  # Output: Whiskers: Whiskers says Meow!


# Example with a non-Animal class that has a 'speak' method
class Robot:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return "Beep boop!"


my_robot = Robot("R2D2")
make_animal_speak(my_robot)  # Output: R2D2: Beep boop! (Demonstrates duck typing)

# ======================================================================
# **Encapsulation**
# 
# **Concept:** Encapsulation means bundling data (attributes) and the methods that operate on that data within a single unit (the class). It also involves restricting direct access to some of an object's components, promoting data integrity.
# 
# In Python, strict "private" members don't exist as they do in Java/C++. Instead, we use conventions:
# 
# **Public:** `my_attribute` - Accessible from anywhere.
# 
# **Protected (Convention):** `_my_attribute` - Single leading underscore. Indicates it's intended for internal use within the class or its subclasses, but can still be accessed directly. Programmers are "asked" to respect this convention.
# 
# **Private (Name Mangling):** `__my_attribute` - Double leading underscore. Python "mangles" the name (e.g., `_ClassName__my_attribute`) to make it harder (but not impossible) to access directly from outside the class. This is primarily to avoid naming conflicts in inheritance.
# ======================================================================

# %%
class BankAccount:
    def __init__(self, initial_balance):
        self._balance = initial_balance  # Protected attribute (by convention)

    def deposit(self, amount):
        if amount > 0:
            self._balance += amount
            print(f"Deposited {amount}. New balance: {self._balance}")
        else:
            print("Deposit amount must be positive.")

    def withdraw(self, amount):
        if amount > 0 and amount <= self._balance:
            self._balance -= amount
            print(f"Withdrew {amount}. New balance: {self._balance}")
        else:
            print("Invalid withdrawal amount or insufficient funds.")

    def get_balance(self):  # Public method to access balance
        return self._balance

    # Attempt to demonstrate name mangling (though direct access is still possible if you know the mangled name)
    def __private_method(self):
        print("This is a private-ish method.")

    def public_method_calling_private(self):
        self.__private_method()  # Can be called internally


my_account = BankAccount(1000)
my_account.deposit(500)
my_account.withdraw(200)
print(f"Current balance: {my_account.get_balance()}")

# Accessing protected attribute directly (discouraged but possible)
# print(f"Direct access to balance (discouraged): {my_account._balance}")

# Attempting to call the "private" method directly (will fail)
# my_account.__private_method() # AttributeError

# Calling via a public method
my_account.public_method_calling_private()  # Works

# ======================================================================
# **Abstraction**
# 
# **Concept:** Abstraction means showing only essential information and hiding the complex implementation details. It focuses on "what" an object does rather than "how" it does it. This allows users to interact with objects without needing to understand the internal complexities.
# 
# In Python, abstraction is achieved through:
# 
# **Classes and Objects:** The very act of defining a class is abstraction. Users interact with methods like `deposit()` or `bark()` without needing to know the internal logic.
# 
# **Abstract Base Classes (ABCs):** Using the `abc` module, you can define abstract classes and abstract methods. Abstract methods must be implemented by concrete subclasses. This enforces a contract.
# ======================================================================

# %%
from abc import ABC, abstractmethod


# Abstract Base Class
class Shape(ABC):  # Inherit from ABC
    @abstractmethod
    def area(self):
        """Calculates and returns the area of the shape."""
        pass  # Must be implemented by subclasses

    @abstractmethod
    def perimeter(self):
        """Calculates and returns the perimeter of the shape."""
        pass  # Must be implemented by subclasses

    def describe(self):  # Concrete method, can be inherited or overridden
        return "This is a geometric shape."


class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):  # Must implement abstract method
        return self.width * self.height

    def perimeter(self):  # Must implement abstract method
        return 2 * (self.width + self.height)


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):  # Must implement abstract method
        return 3.14159 * self.radius**2

    def perimeter(self):  # Must implement abstract method (circumference)
        return 2 * 3.14159 * self.radius


# shape = Shape() # TypeError: Can't instantiate abstract class Shape

rect = Rectangle(10, 5)
circle = Circle(7)

print(f"Rectangle area: {rect.area()}")
print(f"Rectangle perimeter: {rect.perimeter()}")
print(rect.describe())

print(f"Circle area: {circle.area()}")
print(f"Circle perimeter: {circle.perimeter()}")
print(circle.describe())

# %%


