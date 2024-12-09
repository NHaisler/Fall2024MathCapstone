from random_generators import lorenz_random_number
from compressions import *

#Entropy | frequency of symbols | 0 is low, 1 is high
#RLE compression ratio | long sequences of the same symbol | 0 is low, 1 is high
# print(f"Zlib Score {zlib_compression(number)}")

# number = "0000000000000000"
# print("Entropy:", entropy(number))
# print("RLE compression ratio:", rle_compression_ratio(number))
# print(f"Zlib Score {zlib_compression(number)}")

# number = "0101010101010101"
# print("Entropy:", entropy(number))
# print("RLE compression ratio:", rle_compression_ratio(number))
# print(f"Zlib Score {zlib_compression(number)}")



# source_list = [_ for _ in lorenz_random_number(10,length = 14)]
# sampled_list = [_ for _ in lorenz_random_number(10, sampled=True, length = 14)]

#Generate a bunch of numbers
total_diff = 0
n = 10
for o_num, s_num in zip(lorenz_random_number(n,length = 14), lorenz_random_number(n, sampled=True, length = 14)):
    print(f"-------------------------------------------------")
    org_comp  = zlib_compression(o_num)
    sam_comp  = zlib_compression(s_num)
    diff      = org_comp - sam_comp
    print(f"Source System:  {o_num} | Sampled System: {s_num}")
    print(f"Original: {org_comp}")
    print(f"Sampled: {org_comp}")
    print(f"Difference: {diff}")

    total_diff += diff
else:
    print(f"-------------------------------------------------")

print(f"Mean Error over {n} points: {total_diff/n}")
print(f"======================================================")






curr_diff = zlib_compression(o_num) - zlib_compression(s_num)
print(f"Source System:  {o_num} | Sampled System: {s_num}")
print(f"Difference in zlib compression: {curr_diff}")