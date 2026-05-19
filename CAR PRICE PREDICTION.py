import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# =========================================
# CODEALPHA DATA SCIENCE INTERNSHIP
# TASK 3 : CAR PRICE PREDICTION
# =========================================

def main():

    print("\n======================================")
    print(" CAR PRICE PREDICTION PROJECT ")
    print("======================================\n")

    # -----------------------------------
    # FILE PATH
    # -----------------------------------

    file_path = r"C:\Users\HP\Downloads\car data.csv"

    # -----------------------------------
    # CHECK FILE EXISTS
    # -----------------------------------

    if not os.path.exists(file_path):

        print(" ERROR : Dataset file not found")
        print("\nCheck your file path.")
        return

    # -----------------------------------
    # LOAD DATASET
    # -----------------------------------

    try:
        df = pd.read_csv(file_path)

    except Exception as e:

        print(" ERROR LOADING DATASET")
        print(e)
        return

    print("Dataset Loaded Successfully\n")

    # -----------------------------------
    # SHOW COLUMN NAMES
    # -----------------------------------

    print("Dataset Columns:\n")
    print(df.columns)

    # -----------------------------------
    # DISPLAY DATASET
    # -----------------------------------

    print("\nFirst 5 Rows:\n")
    print(df.head())

    print("\n-----------------------------------")
    print("Dataset Information")
    print("-----------------------------------\n")

    print(df.info())

    print("\n-----------------------------------")
    print("Missing Values")
    print("-----------------------------------\n")

    print(df.isnull().sum())

    # -----------------------------------
    # REMOVE MISSING VALUES
    # -----------------------------------

    df.dropna(inplace=True)

    # -----------------------------------
    # REMOVE EXTRA SPACES FROM COLUMN NAMES
    # -----------------------------------

    df.columns = df.columns.str.strip()

    # -----------------------------------
    # ENCODING CATEGORICAL COLUMNS
    # -----------------------------------

    le = LabelEncoder()

    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    # -----------------------------------
    # FIND TARGET COLUMN AUTOMATICALLY
    # -----------------------------------

    possible_targets = [
        "Selling_Price",
        "selling_price",
        "Price",
        "price"
    ]

    target_column = None

    for col in possible_targets:

        if col in df.columns:
            target_column = col
            break

    # -----------------------------------
    # IF TARGET COLUMN NOT FOUND
    # -----------------------------------

    if target_column is None:

        print("\n ERROR : Price column not found")
        print("\nAvailable Columns:\n")

        for col in df.columns:
            print(col)

        return

    print(f"\n Target Column Found : {target_column}")

    # -----------------------------------
    # FEATURES & TARGET
    # -----------------------------------

    X = df.drop(target_column, axis=1)

    y = df[target_column]

    # -----------------------------------
    # TRAIN TEST SPLIT
    # -----------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # -----------------------------------
    # MODEL TRAINING
    # -----------------------------------

    model = LinearRegression()

    model.fit(X_train, y_train)

    print("\n Model Training Completed")

    # -----------------------------------
    # PREDICTION
    # -----------------------------------

    y_pred = model.predict(X_test)

    # -----------------------------------
    # EVALUATION
    # -----------------------------------

    mae = mean_absolute_error(y_test, y_pred)

    r2 = r2_score(y_test, y_pred)

    print("\n-----------------------------------")
    print("MODEL PERFORMANCE")
    print("-----------------------------------\n")

    print("Mean Absolute Error :", round(mae, 2))

    print("R2 Score :", round(r2, 2))

    # -----------------------------------
    # VISUALIZATION SETTINGS
    # -----------------------------------

    sns.set_style("whitegrid")

    # -----------------------------------
    # HISTOGRAM
    # -----------------------------------

    plt.figure(figsize=(8,6))

    sns.histplot(df[target_column], kde=True)

    plt.title("Car Price Distribution")

    plt.xlabel("Price")

    plt.ylabel("Count")

    plt.savefig("car_histogram.png")

    plt.show()

    # -----------------------------------
    # ACTUAL VS PREDICTED
    # -----------------------------------

    plt.figure(figsize=(8,6))

    plt.scatter(y_test, y_pred)

    plt.xlabel("Actual Price")

    plt.ylabel("Predicted Price")

    plt.title("Actual vs Predicted Price")

    plt.savefig("car_prediction.png")

    plt.show()

    # -----------------------------------
    # HEATMAP
    # -----------------------------------

    plt.figure(figsize=(12,8))

    correlation = df.corr(numeric_only=True)

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm"
    )

    plt.title("Correlation Heatmap")

    plt.savefig("car_heatmap.png")

    plt.show()

    # -----------------------------------
    # BAR GRAPH
    # -----------------------------------

    plt.figure(figsize=(10,6))

    df[target_column].head(10).plot(kind='bar')

    plt.title("Top 10 Car Prices")

    plt.xlabel("Index")

    plt.ylabel("Price")

    plt.savefig("car_bargraph.png")

    plt.show()

    # -----------------------------------
    # FINAL MESSAGE
    # -----------------------------------

    print("\n======================================")
    print(" PROJECT COMPLETED SUCCESSFULLY ")
    print("======================================\n")

    print("Generated Graph Files:\n")

    print("1. car_histogram.png")
    print("2. car_prediction.png")
    print("3. car_heatmap.png")
    print("4. car_bargraph.png")


# =========================================
# MAIN FUNCTION
# =========================================

if __name__ == "__main__":
    main()