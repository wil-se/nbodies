#!/usr/bin/python

import random

max_value = 100
bodies_number = input("bodies number: ")
print(bodies_number)

for b in range(int(bodies_number)):
	print("{} {} {} 1000000000000 0 0 0".format(random.randint(0, max_value), random.randint(0, max_value), random.randint(0, max_value)))

