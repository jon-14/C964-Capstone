import matplotlib.pyplot as plt
import scipy.stats as stats
from data import transform_data as td
import seaborn as sb

transaction_data = td.import_from_csv()
fraud_transactions = transaction_data[transaction_data['Fraud_Label'] == 1]

# VISUALIZATION 0 - Percent of fraudulent transactions
percent_fraudulent = 100 * len(fraud_transactions) / len(transaction_data)
print(percent_fraudulent)

# VISUALIZATION 1 - Bar Graph: Fraud Rate by Merchant Type
fraud_count_by_merchant_type = fraud_transactions.groupby(fraud_transactions['merchant_type'])['Fraud_Label'].sum()
transaction_count_by_merchant_type = transaction_data.groupby('merchant_type')['transaction_id'].count()
percent_fraud_by_merchant_type = fraud_count_by_merchant_type / transaction_count_by_merchant_type

percent_fraud_by_merchant_type = percent_fraud_by_merchant_type.sort_values(ascending = False)
'''@TODO: uncomment plot when ready to share
percent_fraud_by_merchant_type.plot(x='merchant_type', kind='bar', title="Fraud Rate by Merchant Type")
plt.show()'''

# VISUALIZATION 2 - Line Plot: Fraud Count by Month
fraud_count_by_month = fraud_transactions.groupby(fraud_transactions['timestamp'].dt.month)['Fraud_Label'].sum() # https://www.statology.org/pandas-group-by-month/
transaction_count_by_month = transaction_data.groupby(transaction_data['timestamp'].dt.month)['timestamp'].count()
percent_fraud_by_month = fraud_count_by_month / transaction_count_by_month
'''@TODO: uncomment plot when ready to share
percent_fraud_by_month.plot(x = 'timestamp', title="Fraud Rate by Month")
plt.show()'''

# VISUALIZATION 3 - Correlation Matrix - numerical parameters only
parameters = ['transaction_amount', 'Fraud_Label', 'credit_limit', 'avg_transaction_amount', 'transaction_average_ratio']
corr_matrix = transaction_data[parameters].corr()

# VISUALIZATION 4 - Bar Graph: Fraud Rate by Transaction Type
fraud_count_by_transaction_type = fraud_transactions.groupby(fraud_transactions['transaction_type'])['Fraud_Label'].sum()
transaction_count_by_transaction_type = transaction_data.groupby('transaction_type')['transaction_id'].count()
percent_fraud_by_transaction_type = fraud_count_by_transaction_type / transaction_count_by_transaction_type

percent_fraud_by_transaction_type = percent_fraud_by_transaction_type.sort_values(ascending = False)
'''@TODO: Uncomment when ready to share
percent_fraud_by_transaction_type.plot(x='transaction_type', kind='bar', title="Fraud Rate by Transaction Type")
plt.tight_layout()
plt.show()'''

