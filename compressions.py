import math
from collections import Counter
import zlib
import sys

def entropy(s):
    freq = Counter(s)
    total_symbols = len(s)
    entropy_value = 0.0

    for count in freq.values():
        p = count / total_symbols
        entropy_value -= p * math.log2(p)

    return entropy_value

def run_length_encoding(s):
    encoding = []
    prev_char = s[0]
    count = 1
    for char in s[1:]:
        if char == prev_char:
            count += 1
        else:
            encoding.append((prev_char, count))
            prev_char = char
            count = 1
    encoding.append((prev_char, count))
    return encoding

def rle_compression_ratio(s):
    rle_encoded = run_length_encoding(s)
    compressed_size = sum([len(str(count)) + 1 for char, count in rle_encoded])
    return (compressed_size / len(s)) - 1


def zlib_compression(s):
    s = s.encode()
    original = sys.getsizeof(s)
    compressed = sys.getsizeof(zlib.compress(s))

    return original/compressed