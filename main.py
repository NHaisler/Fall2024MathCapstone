from random_generators import lorenz_random_number
from compressions import *

#Entropy | frequency of symbols | 0 is low, 1 is high
#RLE compression ratio | long sequences of the same symbol | 0 is low, 1 is high

number = "0000000000000000"
print("Entropy:", entropy(number))
print("RLE compression ratio:", rle_compression_ratio(number))

number = "0101010101010101"
print("Entropy:", entropy(number))
print("RLE compression ratio:", rle_compression_ratio(number))




# #Generate a bunch of numbers
# for number in lorenz_random_number(10,length = 14):
#     print("Entropy:", entropy(number))
#     print("RLE compression ratio:", rle_compression_ratio(number))