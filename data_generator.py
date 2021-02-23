import random

max_value = 1000000
bodies_number = input("bodies number: ")
print(bodies_number)

for b in range(int(bodies_number)):
	print("{} {} {} 100 0 0 0".format(random.randint(0, max_value), random.randint(0, max_value), random.randint(0, max_value)))

