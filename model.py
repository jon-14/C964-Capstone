from sklearn.linear_model import LogisticRegression
import data.transform_data as td

transaction_data = td.import_from_csv()

# Dependend variable
Y = transaction_data['Fraud_Label']

#Independend variables
parameters = ['transaction_amount_x', 'credit_limit', 'avg_transaction_amount', 'transaction_average_ratio']
X = transaction_data[parameters]


