# print("Hi Folks!")

# salary = int(input("Enter your income: "))
# if salary < 50000:
#     print("You are Poor")
# else:
#     print("You are Rich")


# arr = [10, 50, 30, 20, 40]

# for i in arr:
#     if i == 30:
#         print("Found 30")
#         break

# details = {
#     "name": "Rahi",
#     "age": 24,
#     "job": "Backend Developer"
# }

# print(details["name"])
# print(details["age"])


# list_of_dict = [
#     {"name": "Rahi", "age": 24, "job": "Backend Developer"},
#     {"name": "Avik", "age": 25, "job": "App Developer"},
# ]

# for person in list_of_dict:
#     print(person["name"])
#     print(person["age"])
#     print(person["job"])
#     print("-----")


# def add(a, b):
#     return a + b

# def i_am(name):
#     print(f"My name is {name}")

# name = input("Enter your name: ")
# i_am(name)


class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def display(self):
        print(f"Name: {self.name}, Age: {self.age}")
    
obj = Person(name="Saitama", age=30)

obj.display()

obj.name = "Goku"
obj.age = 35
obj.display()