
import pandas as pd

# Create sample student dataset
data = {
    'Name': ['Akshat', 'Rahul', 'Aman', 'Priya', 'Rahul'],
    'Marks': [85, 70, None, 90, 70],
    'City': ['Pune', 'Mumbai', 'Delhi', None, 'Mumbai'],
    'Age': [18, 19, 18, 20, 19]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# Display first five rows
print("\nFirst Five Rows:")
print(df.head())

# Display dataset information
print("\nDataset Information:")
df.info()

# Display statistical summary
print("\nStatistical Description:")
print(df.describe())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove rows where Name is missing
df = df.dropna(subset=['Name'])

# Fill missing Marks with mean
df['Marks'] = df['Marks'].fillna(df['Marks'].mean())

# Fill missing City with mode
df['City'] = df['City'].fillna(df['City'].mode()[0])

# Remove duplicate records
df = df.drop_duplicates()

# Drop irrelevant Age column
df = df.drop(columns=['Age'])

# Filter students having marks greater than or equal to 75
filtered_df = df.loc[df['Marks'] >= 75]

print("\nCleaned Dataset:")
print(df)

print("\nFiltered Dataset:")
print(filtered_df)

# Save cleaned data
df.to_csv('cleaned_students.csv', index=False)

print("\nPreprocessing Completed Successfully!")
