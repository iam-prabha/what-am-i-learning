# Hash Tables
#
# A hash table is a data structure that stores key-value pairs and allows for fast insertion, deletion, and lookup of values based on their keys.
#
# How it works:
# - A hash function is used to convert the key into an index in an underlying array (the "table").
# - The value is stored at the computed index.
# - When retrieving a value, the hash function is applied to the key to find the correct index.
#
# Key features:
# - Average-case time complexity for insert, delete, and search is O(1).
# - Handles collisions (when two keys hash to the same index) using techniques like chaining (linked lists at each index) or open addressing (finding another open slot).
#
# Common uses:
# - Implementing dictionaries/maps in programming languages (e.g., Python's dict).
# - Caching, database indexing, and sets.


def hashing_by_division(k, m):
    return k % m


def main():
    dictionary = {"a": 1, "b": 2, "c": "gen z", "d": True}

    print(dictionary)

    # insert
    dictionary["e"] = False
    print(dictionary)

    # delete
    dictionary.pop("a")
    print(dictionary)

    # search
    print(dictionary["b"])

    # key = 50 , table size = 13
    k = 50
    m = 13

    print(f"hash of 50 with table size 13 --> {hashing_by_division(k, m)}")


main()
