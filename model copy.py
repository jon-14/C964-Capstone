from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import data.transform_data as td
import pandas as pd

# How to drop columns from a dataframe that are in another dataframe - merge the dataframes and keep left only https://dnmtechs.com/deleting-rows-from-a-pandas-data-frame-based-on-another-data-frame/#google_vignette

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

all_fraud_transactions = transaction_data[transaction_data['Fraud_Label'] == 1]
all_real_transactions = transaction_data[transaction_data['Fraud_Label'] == 0]


fraud_transactions_train = all_fraud_transactions.head(1000)
fraud_transactions_test = pd.merge(all_fraud_transactions, fraud_transactions_train, on='transaction_id', how='left', indicator=True) 
fraud_transactions_test = fraud_transactions_test[fraud_transactions_test['_merge'] == 'left_only']
fraud_transactions_test.set_axis(['client_id', 'transaction_id', 'timestamp', 'transaction_amount',
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
       'currency_RMB', 'currency_USD'], axis='columns')

real_transactions_train = all_real_transactions.head(1000)
real_transactions_test = pd.merge(all_real_transactions, real_transactions_train, on='transaction_id', how='left', indicator=True) 
real_transactions_test = real_transactions_test[real_transactions_test['_merge'] == 'left_only']

X_train = pd.concat([fraud_transactions_train, real_transactions_train])
X_test = pd.concat([fraud_transactions_test, real_transactions_test])

print(X_train.columns)
print(X_test.columns)

y_train = X_train['Fraud_Label']
y_test = X_test['Fraud_Label']

# Choose independent parameters
parameters = [
    'transaction_amount',
    'credit_limit',
    #'avg_transaction_amount', 
    #'transaction_average_ratio',
    'merchant_type_Clothing', 'merchant_type_Electronics', 'merchant_type_Entertainment', 'merchant_type_Gas Station', 'merchant_type_Grocery', 'merchant_type_Health', 'merchant_type_Online Retail', 'merchant_type_Online Stores', 'merchant_type_Restaurant', 'merchant_type_Utilities',
    #'transaction_type_Digital Wallet', 'transaction_type_Online', 'transaction_type_POS', 
    'currency_CDN', 'currency_EURO', 'currency_JPN', 'currency_RMB', 'currency_USD'
    ]

X_train = X_train[parameters]
X_test = X_test[parameters]

# Create the model
logistic_regression_model = LogisticRegression(random_state=10, max_iter=1000)
logistic_regression_model.fit(X_train, y_train)

# Make predictions 
y_prediction = logistic_regression_model.predict(X_test)

y_pred_list = list(y_prediction)
y_test_list = list(y_test)

print(y_pred_list[0])
print(sum(y_pred_list))

# Measure Accuracy - Count correctly labeled non-fraud transactions
non_fraud_correct = 0
non_fraud_test_count = 0
for i in range(0, len(y_test_list)):
    if y_test_list[i] == 0:
        non_fraud_test_count += 1
        if y_pred_list[i] == 0:
            non_fraud_correct += 1
rate_non_fraud_correct = non_fraud_correct / non_fraud_test_count
print(rate_non_fraud_correct)

# Measure Accuracy - Count correctly labeled fraud transactions
fraud_correct = 0
fraud_test_count = 0
for i in range(0, len(y_test_list)):
    if y_test_list[i] == 1:
       fraud_test_count += 1
       if int(y_pred_list[i]) == 1:
            fraud_correct += 1
rate_fraud_correct = fraud_correct / fraud_test_count
print(rate_fraud_correct)

# Measure Accuracy - Fraud transactions
'''fraud_total = y_prediction
fraud_correct = y_prediction
fraud_incorrect = y_prediction'''

