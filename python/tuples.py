# Tuples are like lists 
# - have elements which are indexed started at 0
# - its a immutable and a contents
# dont have sort(),append(),reverse() => traceback err:attr Err
# has only - index,count 
# operators work with tuples
# sorting list of tuples
# d = {'a':20,'b':2,'c':34}
# t = sorted(d.items())
d = {'a', 'b', 'c'}
# for k,v in t :
#    print(k,v)  
# print(c)
# sort by values instead of key
print(tuple(d.item()))