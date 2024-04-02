import random

# File path where the number is stored
file_path = "number.txt"

# Read the current number from the file
with open(file_path, "r") as file:
    current_number = int(file.read().strip())

# Generate a random number
random_number = random.randint(1, 100)

# Add the random number to the current number
new_number = current_number + random_number

# Write the new number back to the file
with open(file_path, "w") as file:
    file.write(str(new_number))

print(f"Updated number: {new_number}")
