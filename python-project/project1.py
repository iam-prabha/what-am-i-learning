name= input("Hey type your name: ")
print("Hello " + name," welcome back")

should_we_play = input("Do yo wanna play? ").lower()


if should_we_play == "y" or should_we_play == "yes":
    print("we wanna a play")

    weapon = input("Choice a weapon (axe/sword)?")
    direction = input("Do you want to go left or right? ").lower()
    if direction == "l":
        print("okay, Let go left")
    elif direction == "r":
        print("okay, Let GO RIGHT")
    else:
        print("Sorry, you die")
else:
    print("we not wanna play")