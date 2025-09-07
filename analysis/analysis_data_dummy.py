from os import path
import pandas as pd

def analyze_student_performance():
    """
    Reads the 'students_performance.csv' file from the 'data' directory
    and performs initial exploratory data analysis.
    :return:
    """
    file_path = 'data/student_habits_performance.csv'

    # checking if file exist
    if not path.exists(file_path):
        print(f"File not present on given path {file_path}")

    # Read csv file
    try:
        df = pd.read_csv(file_path)
    except pd.errors.ParseError as e:
        print(f"Error reading the CSV file: {e}")
        return

# Initial Data Exploration------
    print(f"1. Initial Data Exploration")
    print(df.head())

    print(f"\n 2. Data Info ")
    df.info()

    print("\n3. Descriptive Analysis of statistical numeric column")
    print(df.describe())

# Data Cleaning and Preprocessing:
    print("\n Data Cleaning and Preprocessing:")
    print("\n4. Missing Values per Column:")
    print(df.isnull().sum())


    # check duplicate rows
    print("5. check duplicate rows")
    duplicate_rows = df.duplicated().sum()
    print(f"\n check duplicate rows {duplicate_rows}")


# Call the function to run above analysis
if __name__ == "__main__":
    analyze_student_performance()






