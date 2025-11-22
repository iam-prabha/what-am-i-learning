import requests
# making a GET request

req = requests.get("https://www.geeksforgeeks.org/python-programming-language/")

# checking the status code
# successful response 200
print(req.status_code)
print("\n")
# checking the content of the response
print(req.content)
