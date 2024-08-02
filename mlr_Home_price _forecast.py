import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import f_regression


class HousePricePredictor:
    def __init__(self):
        self.data = None
        self.model = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
    
    def load_data(self, filepath):
        # Load the dataset
        self.data = pd.read_csv(filepath)
        print('Data loaded successfully. Shape:', self.data.shape)

    def preprocess_data(self):
        # select features for multiple linear regression
        self.features = ['sqft_living', 'bedrooms', 'bathrooms', 'floors', 'grade', 'year_built']
        x = self.data[self.features]
        y = self.data['price']

        # Handle missing values (if any)
        #x = x.fillna(x.mean())
        
        # split the data into training and test sets
        X_train, X_test, self.Y_train, self.Y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        
        # scale the futures
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)
    
    def train_model(self):
        # Initialize and train the multiple linear regression
        self.model = LinearRegression()
        self.model.fit(self.X_train,self.Y_train)
        print('Model trained successfully')

    def evaluate_model(self):
        # Making predictions on the test set
        y_pred = self.model.predict(self.X_test)

        #calculate metrics
        mse = mean_squared_error(self.Y_test, y_pred)
        r2 = r2_score(self.Y_test,y_pred)
        
        print(f'Mean Squared Error: {mse}')
        print(f'R--squared Score: {r2}')

        # calculate and display features more importance
        f_statistic, _ = f_regression(self.X_train, self.Y_train)
        feature_importance = pd.Series(f_statistic, index=self.features).sort_values(ascending=False)
        print('\nFeature Importance:')
        print(feature_importance)

    def visualize_residuals(self):
        # Plot residuals
        y_pred = self.model.predict(self.X_test)
        residuals = self.Y_test - y_pred

        plt.figure(figsize=(10,6))
        plt.scatter(y_pred, residuals)
        plt.xlabel('Predicted prices')
        plt.ylabel('Residuals')
        plt.title('REsiduals Plot')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.show()

    def visualize_actual_vs_predicted(self):
        # Plot actual vs predicted values
        y_pred = self.model.predict(self.X_test)

        plt.figure(figsize=(10,6))
        plt.scatter(self.Y_test, y_pred, alpha = 0.5)
        plt.plot([self.Y_test.min(), self.Y_test.max()], [self.Y_test.min(), self.Y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.title('Actual Prices vs Predicted Prices')
        plt.show()

    def predict_price(self, features_dict):
        # predict price for given features
        features = np.array([[features_dict.get(f, 0) for f in self.features]])
        scaled_features = self.scaler.transform(features)
        price = self.model.predict(scaled_features)[0]
        print(f'Predicted Price: ${price}')

# usage example
if __name__ == '__main__':
    predictor = HousePricePredictor()

    #loading date
    data_file = dataset_file = os.getcwd()+'\house_data.csv'
    predictor.load_data(data_file)
    
    predictor.preprocess_data()
    predictor.train_model()
    predictor.evaluate_model()

    predictor.visualize_residuals()
    predictor.visualize_actual_vs_predicted()

    # predicting price for a house
    sample_house = { 'sqft_living': 2000,
        'bedrooms': 3,
        'bathrooms': 2,
        'floors': 1,
        'grade': 7,
        'year_built': 2003}
    
    predictor.predict_price(sample_house)