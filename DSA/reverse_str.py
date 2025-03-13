# 逆順にする
def reverse(text):
    return text[::-1]

def reverse2(text):
    return ''.join(reversed(text))

def reverse3(text):
    return ''.join(text[i] for i in range(len(text)-1, -1, -1))

def reverse4(text):
    tight_str= list(text)
    return ''.join(tight_str[::-1])
word = 'katenkyoukotsukaramatsushinjuu'

print(reverse(word))
print(reverse2(word))
print(reverse3(word))
print(reverse4(word))