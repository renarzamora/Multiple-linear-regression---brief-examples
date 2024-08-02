import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

class MarketingAnalizer:
    def __init__(self):
        self.data = None
        self.model = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.features = None
    
    def load_data(self, filepath):
        # loading the dataset
        self.data = pd.read_csv(filepath)
        print('Data loaded successfully. Shape:', self.data.shape)

    def preprocess_data(self):
        # selecting features for multiple linear regression
        self.features = ['TV','radio','newspaper']
        x = self.data[self.features]
        y = self.data['sales']

        # split the data into training and testing sets
        X_train, X_test, self.Y_train, self.Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        #scale the features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)

    def train_model(self):
        # Initializing and training the ultiple linear regression
        self.model = LinearRegression()
        self.model.fit(self.X_train,self.Y_train)
        print('Model trained successfully')

    def evaluate_model(self):
        # making predictions on the test set
        y_pred = self.model.predict(self.X_test)

        #calculate metrics
        mse = mean_squared_error(self.Y_test, y_pred)
        r2 = r2_score(self.Y_test,y_pred)

        print(f'Mean Suared Error: {mse}')
        print(f'R2-Squared Erro: {r2}')

        # print model coefficients
        for feature, coef in zip(self.features, self.model.coef_):
            print(f'Coefficient for {feature}: {coef}')
        print(f'Intercept: {self.model.intercept_}')

    def visualize_pairplot(self):
        # creating a pairplot to visualize relationships between variables
        sns.pairplot(self.data[self.features+['sales']], height=2)
        plt.tight_layout()
        plt.show()

    def visualize_residuals(self):
        # ploting residuals
        y_pred = self.model.predict(self.X_test)
        residuals = self.Y_test - y_pred

        plt.figure(figsize=(10,6))
        plt.scatter(y_pred, residuals)
        plt.xlabel('Predicted sales')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.show()

    def predict_sales(self, ad_spend):
        # Predicting the sales for given advertising spend
        scaled_spend = self.scaler.transform([ad_spend])
        sales = self.model.predict(scaled_spend)[0]
        print(f'Predicted sales: ${sales:,.2f}')
        

    def calculate_correlations(self):
        # calculating the correlation matrix
        corr_matrix = self.data[self.features + ['sales']].corr()

        #creating a heatmap of the correlation matrix
        plt.figure(figsize=(10,6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Correlation Heatmap')
        plt.show()

# usage example
if __name__ == '__main__':
    analyzer = MarketingAnalizer()

    #loading data
    data_file = dataset_file = os.getcwd()+'\Advertising.csv'
    analyzer.load_data(data_file)     
    #analyzer.load_data('D:\Datascienceprojects\Regresiones lineales\Data\Advertising.csv')     


    analyzer.preprocess_data()
    analyzer.train_model()
    analyzer.evaluate_model()

    analyzer.visualize_pairplot()
    analyzer.visualize_residuals()

    # predict sales for given ad spend
    sample_ad_spend = [200000, 37000, 39000]  # TV, Radio, Newspaper
    analyzer.predict_sales(sample_ad_spend)
    analyzer.calculate_correlations()


