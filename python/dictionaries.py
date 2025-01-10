# list
# - a linear collection of values that stay in order

# dictionaries
# - a bag of value each with it own label on it 
# eg: key-value pairs

# dictionary literals (constants)
# variable = dist() or {}
# key not exist in dist has get method return a default is 0

# counts = dict()
# names = ['luffy','zoro','sanji','robin']
# for name in names :
#     counts[name] = counts.get(name,0) + 1 // get(obj or value,setValue)
# print(counts)
fname = input('Enter the file: ')
if len(fname) < 1 : fname = 'words.txt'
hand = open(fname)

di = dict()
for lin in hand :
    lin =  lin.rstrip()
    wds = lin.split()
    for w in wds :
        # idiom: retrieve/create/update counter
        di[w] = di.get(w,0) + 1

print(di)

# now we want to find most common
theword = None
largest = -1
for k,v in di.items() :
    if v > largest :
        largest = v
        theword = k

print('Done',theword,largest)
