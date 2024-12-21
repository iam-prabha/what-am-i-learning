# number = [1,2,3,4,5]
# number.append(6) # [1, 2, 3, 4, 5, 6]
# number.pop(2)    # [1, 2, 4, 5, 6]
# print(number)
# fruits = ["apple","orange","grapes"]

# for i in range(len(fruits)):
#     print(fruits[i])

# dictories key-pairs like object js


# function always top


import json


def add_person():
    name = input("Name: ")
    age = input("Ages: ")
    email = input("Email: ")

    person = {"name":name,"age":age,"email":email}
    return person

def display_people(people):
    for i , person in enumerate(people):
        print(i + 1 ,"-",person["name"],"|",person["age"],"|",person["email"])

# enumerate give me index and element for every list element inside of list 
def delete_contact(people):
    display_people(people)

    while True:
        number = input("Enter a number to delete: ")
        try:
            number = int(number)
            if number <= 0 or number > len(people):
                print("Invalid number, out of range.")
            else:
                break
        except:
            print("invalid number")
    people.pop(number - 1)
    print("people deleted")


def search(people):
    search_name = input("Search for a name: ").lower()
    results = []

    for person in people:
        name = person["name"]
        if search_name in name.lower():
            results.append(person)
   
    display_people(results)


print("Hi, Welcome to the system contect management")
print()

with open("contact.json","r") as f:
    people = json.load(f)["contacts"]

while True:
    print()
    print("Contact list size: ",len(people))
    command = input("You can 'Add', 'Delete', or 'Search' and 'Q' for quit: ").lower()

    if command == "add":
        person = add_person()
        people.append(person)
        print("person added")
    elif command == "delete":
        delete_contact(people)
    elif command == "search":
        search(people)
    elif command == 'q':
        break
    else:
        print("Invalid input ")
    
with open("contact.json","w") as f:
    json.dump({"contacts": people},f)