def get_number(number):
    while True:
        opernd = input("Number "+ str(number)+ ": ")
        try:
           return float(opernd)
        except:
            print("Invalid number,try again.")


opernd = get_number(1)
opernd2 = get_number(2)

sign = input("Sign: ")

result=0
if sign == "+":
    result = float(opernd) + float(opernd2)
elif sign == "-":
    result = float(opernd) - float(opernd2)
elif sign == "/":
    if float(opernd2) != 0:
        result = float(opernd) / float(opernd2)
    else:
        result = print("divided by Zero")
elif sign == "*":
    result = float(opernd) * float(opernd2);
print(result)


