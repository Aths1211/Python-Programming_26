# -*- coding: utf-8 -*-
"""ASSIGNMENT 1

# Assignment 1: Operations on List, Tuple, and Dictionary using Student data

# LIST OF STUDENTS
student_list = ["Amit", "Priya", "Rahul", "Sneha"]
print(student_list)

student_list.append("Kunal")
print(student_list)

student_list[1] = "Priyanka"
print(student_list)

student_list.remove("Rahul")
print(student_list)

print()


# 2. TUPLE OF STUDENTS


student_tuple = ("Amit", "Priya", "Rahul", "Sneha")
print(student_tuple)


student_tuple = student_tuple + ("Kunal",)
print(student_tuple)


temp = list(student_tuple)
temp[1] = "Priyanka"
student_tuple = tuple(temp)
print(student_tuple)


temp = list(student_tuple)
temp.remove("Rahul")
student_tuple = tuple(temp)
print(student_tuple)

print()


# 3. DICTIONARY OF STUDENTS (roll_no: name)

student_dict = {101: "Amit", 102: "Priya", 103: "Rahul", 104: "Sneha"}
print(student_dict)

student_dict[105] = "Kunal"
print(student_dict)


student_dict[102] = "Priyanka"
print(student_dict)

del student_dict[103]
print(student_dict)

