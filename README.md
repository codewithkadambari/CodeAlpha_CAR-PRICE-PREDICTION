# Car Price Prediction with Machine Learning

## 📌 Project Overview
This project involves building a machine learning regression model to predict the selling price of cars based on various features. It demonstrates an end-to-end data science pipeline, including data cleaning, categorical encoding, feature selection, and model training.

## 🎯 Objectives
- **Data Preprocessing**: Handling missing values, stripping column formatting issues, and applying Label Encoding to transform categorical data into numeric variables.
- **Machine Learning**: Splitting the dataset, training a `LinearRegression` model, and making predictions on test data.
- **Performance Evaluation**: Using Mean Absolute Error (MAE) and R-squared (R2 Score) to measure the accuracy of the regression model.
- **Data Visualization**: Exploring the data distribution and correlation, as well as visualizing the model's predictive performance.

## 🛠️ Technologies & Libraries Used
- **Python 3.x**
- **Pandas**: Data manipulation and cleaning.
- **Scikit-Learn**: For LabelEncoding, data splitting, Linear Regression modeling, and generating evaluation metrics (MAE, R2).
- **Matplotlib & Seaborn**: For robust data visualization.

## 📊 Visualizations Generated
The script automatically generates and saves the following analytical plots:
1.  **`car_histogram.png`**: Displays the distribution of car selling prices.
2.  **`car_prediction.png`**: A scatter plot comparing the model's predicted prices against the actual prices.
3.  **`car_heatmap.png`**: A correlation matrix highlighting relationships between numeric features (e.g., Year, Present Price, Driven_kms).
4.  **`car_bargraph.png`**: A bar chart illustrating the top 10 car price records.

## 🚀 How to Run the Project
1.  **Prerequisites**: Ensure you have the required libraries installed:
    ```bash
    pip install pandas matplotlib seaborn scikit-learn
    ```
2.  **Dataset Configuration**: Ensure the `car data.csv` dataset is located at the path specified in the script (`C:\Users\HP\Downloads\car data.csv`) or update the `file_path` variable.
3.  **Execution**: Run the Python script:
    ```bash
    python car_price_prediction.py
    ```
4.  **Results**: The script outputs the target column logic, R2/MAE scores, and saves 4 `.png` files to the local directory.

## 📝 License
This project is open-source and available under the [MIT License](LICENSE).
