import matplotlib.pyplot as plt
import scipy.stats as stats
from data import transform_data as td
import seaborn as sb

def fraud_rate_bar_graph(category: str):
    fraud_count_by_category = fraud_transactions.groupby(fraud_transactions[f'{category}'])['Fraud_Label'].sum()
    transaction_count_by_category = transaction_data.groupby(f'{category}')['transaction_id'].count()
    percent_fraud_by_category = fraud_count_by_category / transaction_count_by_category
    percent_fraud_by_category.sort_values(ascending=False, inplace=True)
    percent_fraud_by_category.plot(x=f'{category}', kind='bar', title=f'Fraud Rate by {category}')    
    plt.tight_layout()
    plt.show()

def transaction_rate_bar_graph(category: str):
    transaction_count_by_category = transaction_data.groupby(f'{category}')['transaction_id'].count()
    transaction_count_by_category.sort_values(ascending=False, inplace=True)
    transaction_count_by_category.plot(x=f'{category}', kind='bar', title=f'Transaction Count by {category}')
    plt.tight_layout()
    plt.show()

transaction_data = td.import_from_csv()
fraud_transactions = transaction_data[transaction_data['Fraud_Label'] == 1]

# VISUALIZATION 0 - Percent of fraudulent transactions
percent_fraudulent = 100 * len(fraud_transactions) / len(transaction_data)
print(percent_fraudulent)

# VISUALIZATION 1 - Bar Graph: Fraud Rate by Merchant Type
# @TODO: Uncomment when ready to share
#fraud_rate_bar_graph('merchant_type')

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
# @TODO: Uncomment when ready to share
#fraud_rate_bar_graph('transaction_type')

print(transaction_data.columns)
for x in ['merchant_type', 'transaction_type', 'currency', 'city', 'Fraud_Label']:
    fraud_rate_bar_graph(x)
    transaction_rate_bar_graph(x)