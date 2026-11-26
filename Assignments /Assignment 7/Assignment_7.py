
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create sample dataset
data = {
    'Name': ['A', 'B', 'C', 'D', 'E'],
    'Study_Hours': [2, 4, 5, 7, 8],
    'Marks': [40, 55, 65, 80, 90],
    'Attendance': [60, 70, 75, 85, 95]
}

df = pd.DataFrame(data)

# Save dataset as CSV
df.to_csv("students.csv", index=False)

# Load dataset
df = pd.read_csv("students.csv")

# Display dataset
print("Dataset:")
print(df)

# 1. Line Plot
plt.figure()
plt.plot(df['Name'], df['Marks'], marker='o')
plt.title("Student Marks")
plt.xlabel("Students")
plt.ylabel("Marks")
plt.show()

# 2. Bar Plot
plt.figure()
plt.bar(df['Name'], df['Marks'])
plt.title("Student Marks Comparison")
plt.xlabel("Students")
plt.ylabel("Marks")
plt.show()

# 3. Scatter Plot
plt.figure()
plt.scatter(df['Study_Hours'], df['Marks'])
plt.title("Study Hours vs Marks")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.show()

# 4. Heatmap
plt.figure()
correlation = df[['Study_Hours', 'Marks', 'Attendance']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
