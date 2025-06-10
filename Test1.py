import math

E = 131e3
width = 40.8
thickness = 0.24
length = 93.1
n = 1

kth = (314 * n * E * width * thickness ** 3) / (24 * length ** 3)
k_ref = 0.914470347
ratio = kth / k_ref

print(kth)
print(ratio)
