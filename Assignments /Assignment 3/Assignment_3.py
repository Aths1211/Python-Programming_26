
# Program to check whether a triangle is right-angled
# using a user-defined function

def check_right_triangle(a, b, c):
    sides = sorted([a, b, c])

    if sides[0] <= 0 or sides[0] + sides[1] <= sides[2]:
        return False

    if abs(sides[0]**2 + sides[1]**2 - sides[2]**2) < 0.000001:
        return True
    else:
        return False


# Accept input from user
a = float(input("Enter first side: "))
b = float(input("Enter second side: "))
c = float(input("Enter third side: "))

# Function calling
if check_right_triangle(a, b, c):
    print("It is a Right-Angled Triangle")
else:
    print("It is NOT a Right-Angled Triangle")
