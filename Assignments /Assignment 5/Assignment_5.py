
import re

# Accept a string from the user
text = input("Enter a string: ")

# Check whether string contains only letters and digits
if re.fullmatch(r'[a-zA-Z0-9]+', text):
    print("Valid String")
else:
    print("Invalid String")
