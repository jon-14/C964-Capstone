from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import data.transform_data as td

transaction_data = td.import_from_csv()
'''all_columns = ['client_id', 'transaction_id', 'timestamp', 'transaction_amount',
       'merchant_type', 'transaction_type', 'currency', 'city',
       'email_address', 'Fraud_Label', 'credit_limit',
       'avg_transaction_amount', 'transaction_average_ratio',
       'merchant_type_Clothing', 'merchant_type_Electronics',
       'merchant_type_Entertainment', 'merchant_type_Gas Station',
       'merchant_type_Grocery', 'merchant_type_Health',
       'merchant_type_Online Retail', 'merchant_type_Online Stores',
       'merchant_type_Restaurant', 'merchant_type_Utilities',
       'transaction_type_Digital Wallet', 'transaction_type_Online',
       'transaction_type_POS', 'currency_CDN', 'currency_EURO', 'currency_JPN',
       'currency_RMB', 'currency_USD']'''

# Dependend variable
y = transaction_data['Fraud_Label']

# Independend variables

parameters = [
    'transaction_amount',
    #'credit_limit',
    #'avg_transaction_amount', 
    #'transaction_average_ratio',
    #'merchant_type_Clothing', 'merchant_type_Electronics', 'merchant_type_Entertainment', 'merchant_type_Gas Station', 'merchant_type_Grocery', 'merchant_type_Health', 'merchant_type_Online Retail', 'merchant_type_Online Stores', 'merchant_type_Restaurant', 'merchant_type_Utilities',
    #'transaction_type_Digital Wallet', 'transaction_type_Online', 'transaction_type_POS', 
    #'currency_CDN', 'currency_EURO', 'currency_JPN', 'currency_RMB', 'currency_USD'
    ]
X = transaction_data[parameters]

# Split X, y into training data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.1, stratify=y, random_state=7)

# Create the model
logistic_regression_model = LogisticRegression(random_state=10, max_iter=1000)
logistic_regression_model.fit(X_train, y_train)

# Make predictions 
y_prediction = logistic_regression_model.predict(X_test)

# Measure accuracy
score1 = metrics.accuracy_score(y_true = y_test, y_pred = y_prediction)
print(score1)

