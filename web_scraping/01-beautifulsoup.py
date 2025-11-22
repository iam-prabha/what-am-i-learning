import requests
from bs4 import BeautifulSoup

# Making a GET request
r = requests.get("https://www.geeksforgeeks.org/python-programming-language/")

soup = BeautifulSoup(r.content, "html.parser")
# print(soup.prettify())  # Print the prettified HTML content

s = soup.find("div", class_="entry-content")
content = soup.find_all("p")  # Find all <p> tags within the content

print(content)
