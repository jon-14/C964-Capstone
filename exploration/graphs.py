import numpy
import pandas
import matplotlib.pyplot as plt

transaction_data = pandas.read_csv('../data/raw_data.csv')
average_amount_by_customer = pandas.read_csv('../data/avg_transaction_by_customer.csv')
fraud_transactions = transaction_data[transaction_data['Fraud_Label'] == 1]

# Graph 1 - Bar Graph: Fraud Count by Merchant Type
fraud_count_by_merchant_type = fraud_transactions.groupby('merchant_type')
fraud_count_by_merchant_type = fraud_count_by_merchant_type['transaction_id'].count()

transaction_count_by_merchant_type = transaction_data.groupby('merchant_type')
transaction_count_by_merchant_type = transaction_count_by_merchant_type['transaction_id'].count()

percent_fraud_by_merchant_type = fraud_count_by_merchant_type / transaction_count_by_merchant_type
percent_fraud_by_merchant_type = percent_fraud_by_merchant_type.sort_values(ascending=False)

print(percent_fraud_by_merchant_type)
percent_fraud_by_merchant_type.plot(x='merchant_type', kind='bar')
plt.show()
# Graph 2 - Line Plot: Fraud Count by Month

# Graph 3 - Correlation Matrix