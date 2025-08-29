import numpy
import pandas
import matplotlib.pyplot as plt

transaction_data = pandas.read_csv('data/raw_data.csv')

# Convert timestamp column to datetime
transaction_data['timestamp'] = pandas.to_datetime(transaction_data['timestamp'])

average_amount_by_customer = pandas.read_csv('data/avg_transaction_by_customer.csv')
fraud_transactions = transaction_data[transaction_data['Fraud_Label'] == 1]

# VISUALIZATION 1 - Bar Graph: Fraud Count by Merchant Type
fraud_count_by_merchant_type = fraud_transactions.groupby('merchant_type')
fraud_count_by_merchant_type = fraud_count_by_merchant_type['transaction_id'].count()

transaction_count_by_merchant_type = transaction_data.groupby('merchant_type')
transaction_count_by_merchant_type = transaction_count_by_merchant_type['transaction_id'].count()

percent_fraud_by_merchant_type = fraud_count_by_merchant_type / transaction_count_by_merchant_type
percent_fraud_by_merchant_type = percent_fraud_by_merchant_type.sort_values(ascending=False)

print(percent_fraud_by_merchant_type)
percent_fraud_by_merchant_type.plot(x='merchant_type', kind='bar')
plt.show()

# VISUALIZATION 2 - Line Plot: Fraud Count by Month
fraud_count_by_month = fraud_transactions.groupby(fraud_transactions['timestamp'].dt.month)['Fraud_Label'].sum() # https://www.statology.org/pandas-group-by-month/
transaction_count_by_month = transaction_data.groupby(transaction_data['timestamp'].dt.month)['timestamp'].count()
percent_fraud_by_month = fraud_count_by_month / transaction_count_by_month
print(percent_fraud_by_month)
percent_fraud_by_month.plot(x = 'timestamp')
plt.show()

# VISUALIZATION 3 - Correlation Matrix