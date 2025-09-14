import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set page configuration for a wider layout
st.set_page_config(
    page_title="Student Performance Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Step 1: Project Setup and Data Acquisition ---
# create a function to load data
# cache the data to avoid re-loading on every interaction
@st.cache_data
def load_data(file_path):
    """
    Loads the student performance dataset from a given file path and preprocesses it.
    :param file_path: The path to the CSV file.
    :return: A pandas DataFrame.
    """
    try:
        # Load the data
        df = pd.read_csv(file_path)
        # Print all column names
        print("Columns in the DataFrame:")
        print(df.columns)

        # --- Make column names consistent and clean ---
        # Convert to lowercase, strip whitespace, and replace spaces with underscores
        df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

        # Print the cleaned columns for easy debugging in the terminal
        print("DataFrame columns after cleaning:", df.columns.tolist())

        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' not found.")
        st.error("Please make sure the file is in the same directory as this script.")
        st.stop()

# Define the file path
file_path = "/Users/shaveta.bali/Downloads/eda-project-ShavetaBali/data/students_performance.csv"

# Load the data
df = load_data(file_path)

# check if the data was loaded successfully
if df is not None:
    #---Initial Inspection -----
    st.sidebar.header("Data Inspection")
    if st.sidebar.checkbox("Show Raw Data"):
        st.subheader("Raw Data (First 5 Rows)")
        st.write(df.head())
        st.subheader("Data Shape")
        st.write(df.shape)
        st.subheader("Descriptive Statistics")
        st.write(df.describe(include='all'))

    # --------------Data Cleaning and preprocessing ---------------
    # Check for missing values and duplicates
    st.sidebar.subheader("Data Quality Check")
    if st.sidebar.checkbox("Check for Missing Values and Duplicates"):
        st.subheader("Missing values")
        st.write(df.isnull().sum())
        st.subheader("Duplicate Rows")
        st.write(f"Number of duplicate rows: {df.duplicated().sum()}")


    # ---------Step 3. Building the Streamlit application -----
    # set up the app layout
    st.title("Student Performance Analysis & Visualization")
    st.markdown("This interactive dashboard explores the **Student Performance in Exams** dataset from Kaggle. "
                "Use the filters in the sidebar to uncover insights!")

    # ============================================================================
    # STEP 2.4: UNIVARIATE ANALYSIS
    # ============================================================================
    st.header("Univariate Analysis")
    st.markdown("Analyzing the distribution of individual variables.")

    # Create a figure with 3 subplots for the scores
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.suptitle('Distribution of Student Test Scores (Univariate Analysis)', fontsize=16, y=1.02)

    # Plot histogram for math score
    sns.set_style("whitegrid")
    sns.histplot(data=df, x='math_score', kde=True, bins=20, ax=axes[0])
    axes[0].set_title('Distribution of Math Score', fontsize=12)
    axes[0].set_xlabel('Score')
    axes[0].set_ylabel('Frequency')

    # Plot histogram for reading score
    sns.histplot(data=df, x='reading_score', kde=True, bins=20, ax=axes[1])
    axes[1].set_title('Distribution of Reading Score', fontsize=12)
    axes[1].set_xlabel('Score')
    axes[1].set_ylabel('Frequency')

    # Plot histogram for writing score (note: the column name is 'writing_score' after renaming)
    sns.histplot(data=df, x='writing_score', kde=True, bins=20, ax=axes[2])
    axes[2].set_title('Distribution of Writing Score', fontsize=12, y=1.02)
    axes[2].set_xlabel('Score')
    axes[2].set_ylabel('Frequency')

    # Display the plots in Streamlit
    st.pyplot(fig)

    # ============================================================================
    # STEP 2.5: BIVARIATE ANALYSIS
    # ============================================================================
    st.header("BIVARIATE ANALYSIS")
    st.markdown("Exploring the relationships between pairs of variables.")

    #create a figure with 3 sub plots
    fig,axes_biv = plt.subplots(1,3, figsize=(18,5))
    plt.suptitle('Relationship Between Test Scores (Bivariate Analysis)', fontsize=16 ,y=1.02 )

    # Scatter plot: Math vs. Reading Score
    sns.scatterplot(data=df, x='math_score', y='reading_score', ax=axes_biv[0])
    axes_biv[0].set_title('Math vs. Reading Score', fontsize=12)
    axes_biv[0].set_xlabel("Math_Score")
    axes_biv[0].set_ylabel("Reading_score")

    # Scatter plot: Math vs. Writing Score
    sns.scatterplot(data=df, x='math_score', y='writing_score', ax=axes_biv[1])
    axes_biv[1].set_title('Math vs. Writing Score', fontsize=12)
    axes_biv[1].set_xlabel('Math Score')
    axes_biv[1].set_ylabel('Writing Score')

    # Scatter plot: Reading vs. Writing Score
    sns.scatterplot(data=df, x='reading_score', y='writing_score', ax=axes_biv[2])
    axes_biv[2].set_title('Reading vs. Writing Score', fontsize=12)
    axes_biv[2].set_xlabel('Reading Score')
    axes_biv[2].set_ylabel('Writing Score')

    st.pyplot(fig)

    # --- Box plots for scores by categorical variables ---
    st.subheader("Scores by categorical variable")
    st.markdown("Compare score distribution among different groups")

     # Get a list of categorical columns for the user to select
    categorical_cols = ['gender', 'race/ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
    score_cols = ['math_score', 'reading_score', 'writing_score']

    # create a selection box for the user to select teh category
    selected_category = st.sidebar.selectbox("Select categorical Variable",categorical_cols)

    #Create a figure with 3 subplots for box plot
    fig_box,axes_box=plt.subplots(1,3, figsize=(18,5))
    plt.suptitle(f'Distribution of Scores by {selected_category.replace("_", " ").title()}', fontsize=16,y=1.02)



    # Box plot for Math Score
    sns.boxplot(data=df, x=selected_category, y='math_score', ax=axes_box[0])
    axes_box[0].set_title('Math Score', fontsize=12)
    axes_box[0].set_xlabel('')

    # Box plot for reading score
    sns.boxplot(data=df, x=selected_category, y='reading_score', ax=axes_box[1])
    axes_box[1].set_title('Reading Score',fontsize=12)
    axes_box[1].set_xlabel('')

    # Box plot for Writing Score
sns.boxplot(data=df, x=selected_category, y='writing_score', ax=axes_box[2])
axes_box[2].set_title('Writing Score', fontsize=12)
axes_box[2].set_xlabel('')

# Rotate x-axis labels for better readability if the labels are long
for ax in axes_box:
    ax.tick_params(axis='x', rotation=45)

# Adjust layout and display plots
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
st.pyplot(fig_box)


# ============================================================================
# STEP 2.6: MULTIVARIATE ANALYSIS
# ============================================================================
st.header("Multivariate Analysis")
st.markdown("Analyzing relationships between three or more variables.")

# --- Correlation Heatmap ---
st.subheader("Correlation Between Scores")
st.markdown("A heatmap showing the correlation matrix of the three scores.")

# Calculate the correlation matrix
correlation_matrix = df[['math_score', 'reading_score', 'writing_score']].corr()

# create a heatmap
fig_corr,ax_corr= plt.subplots(figsize=(18,5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f",linewidths=.5,ax=ax_corr)
ax_corr.set_title('Correlation Matrix  of student scores')
st.pyplot(fig_corr)

# --- Grouped Bar Charts ---
st.subheader("Mean Scores Across Multiple Categories")
st.markdown("Comparing the average scores based on two categorical variables.")

# Create sidebar filters for grouped bar chart
st.sidebar.subheader("Grouped Bar Chart Filters")
# Get a list of the categorical columns
grouped_cat_cols = ['gender', 'test_preparation_course', 'lunch', 'parental_level_of_education', 'race/ethnicity']

# Get a list of scores
score_cols = ['math_score', 'reading_score', 'writing_score']

# Let the user select two categorical variables
x_var = st.sidebar.selectbox("Select first grouping variable (x-axis)", grouped_cat_cols, index=0)
hue_var = st.sidebar.selectbox("Select second grouping variable (hue)", grouped_cat_cols, index=1)

# Make sure the user selects two different variables
if x_var == hue_var:
    st.warning("Please select two different variables to compare.")
else:
    # Group the data and calculate the mean of the scores
    grouped_data = df.groupby([x_var, hue_var])[score_cols].mean().reset_index()

    # Reshape the data for a grouped bar chart
    grouped_data_melt = grouped_data.melt(id_vars=[x_var, hue_var], var_name='score_type', value_name='mean_score')

    # Create the grouped bar chart
    fig_grouped, ax_grouped = plt.subplots(figsize=(12, 6))
    sns.barplot(data=grouped_data_melt, x=x_var, y='mean_score', hue=hue_var, ax=ax_grouped)
    ax_grouped.set_title(f'Mean Scores by {x_var.replace("_", " ").title()} and {hue_var.replace("_", " ").title()}')
    ax_grouped.set_xlabel(x_var.replace("_", " ").title())
    ax_grouped.set_ylabel("Mean Score")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_grouped)

st.sidebar.markdown("---")
st.sidebar.info("Dashboard created by an experienced data science student using Streamlit.")







