from abc import ABC, abstractmethod

class Person(ABC):
    def __init__(self, name, age, gender, address):
        super().__init__()
        self.name = name
        self.age = age
        self.gender = gender
        self.address = address

    def __str__(self):
        return f"Name:{self.name}, Age: {self.age}, Gender:{self.gender}, Address: {self.address}"
    def greet(self, other_person):
        return f"Hello {other_person.name} my name is {self.name}."
    @abstractmethod
    def introduce(self):
        pass
        # static method
    @staticmethod
    def is_adult(age):
        return age >= 18
class Student(Person):
    def __init__(self,name,age,gender,address,course):
        super().__init__(name,age,gender,address)
        self.course = course

    def introduce(self):
        return f"hi,i am {self.name} and i study in {self.course}."
p1 = Student("Prasant", 21,"male","Kathmandu","Computer Science")     
p2 = Student("Rahul",15,"male","Pokhara","Mathematics")      
print(p1)
print(p1.greet(p2))
print(p1.introduce()) 
print(Person.is_adult(p1.age))
print(Person.is_adult(p2.age))