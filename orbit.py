import random

def orbit_string(length, seed = -1):
    r = random
    if seed != -1:
        r.seed(seed)

    string = ""
    for i in range(length):
        string += str(r.randint(0,1))

    return string

def equal_length(str1, str2):
    count = 0
    for i in range(min(len(str1), len(str2))):
        if str1[i] == str2[i]:
            count += 1
    return count
