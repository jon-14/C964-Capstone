import numpy
import pandas
import matplotlib.pyplot as plt
import scipy.stats as stats

transaction_data = pandas.read_csv('data/raw_data.csv')
avg_amount_by_customer = transaction_data.groupby('client_id')['transaction_amount'].mean()

# Add customers' average transaction amount to transaction_data as new column
transaction_data = pandas.merge(transaction_data, avg_amount_by_customer, on = 'client_id', how = 'left')
transaction_data = transaction_data.rename(columns = {'transaction_amount_y': 'avg_transaction_amount'})

# Convert timestamp column to datetime
transaction_data['timestamp'] = pandas.to_datetime(transaction_data['timestamp'])

average_amount_by_customer = pandas.read_csv('data/avg_transaction_by_customer.csv')
fraud_transactions = transaction_data[transaction_data['Fraud_Label'] == 1]

# VISUALIZATION 1 - Bar Graph: Fraud Count by Merchant Type
fraud_count_by_merchant_type = fraud_transactions.groupby(fraud_transactions['merchant_type'])['Fraud_Label'].sum()
transaction_count_by_merchant_type = transaction_data.groupby('merchant_type')['transaction_id'].count()
percent_fraud_by_merchant_type = fraud_count_by_merchant_type / transaction_count_by_merchant_type

percent_fraud_by_merchant_type = percent_fraud_by_merchant_type.sort_values(ascending = False)
percent_fraud_by_merchant_type.plot(x='merchant_type', kind='bar')
#@TODO: uncomment plot when ready to share
#plt.show()

# VISUALIZATION 2 - Line Plot: Fraud Count by Month
fraud_count_by_month = fraud_transactions.groupby(fraud_transactions['timestamp'].dt.month)['Fraud_Label'].sum() # https://www.statology.org/pandas-group-by-month/
transaction_count_by_month = transaction_data.groupby(transaction_data['timestamp'].dt.month)['timestamp'].count()
percent_fraud_by_month = fraud_count_by_month / transaction_count_by_month

percent_fraud_by_month.plot(x = 'timestamp')
# @TODO: uncomment plot when ready to share
#plt.show() 

# VISUALIZATION 3 - Correlation Matrix
columns = transaction_data.columns
parameters = ['transaction_amount_x', 'credit_limit', 'avg_transaction_amount']
print(transaction_data[['client_id', 'Fraud_Label', 'merchant_type', 'transaction_type']])

print(transaction_data.shape)
transaction_data.dropna(how='any', axis=0)
print(transaction_data.shape)
