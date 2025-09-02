import numpy as np
import pandas

# Import data from CSV and transform the data for use in the model.

def import_from_csv():
    """
    Import data from CSV and prepare the data for use in the machine learning model.
        * Remove rows with null values
        * Add derived fields
            - Average transaction amount by customer
            - Transaction-to-Average ratio (transaction amount / average transaction amount)
            - One-hot encoding for categorical parameters
    Input: none
    Returns: transaction_data (dataframe)
    """

    transaction_data = pandas.read_csv('data/raw_data.csv')    
    
    # Delete rows with null values
    transaction_data.dropna(how='any', axis=0)

    # Convert timestamp column to datetime
    transaction_data['timestamp'] = pandas.to_datetime(transaction_data['timestamp'])

    # Add new column -- avg_transaction_amount (customers' average transaction amount by client_id)
    #   https://stackoverflow.com/questions/56799202/pandas-groupby-and-cumulative-mean-of-previous-rows-in-group
    transaction_data['avg_transaction_amount'] = (
        transaction_data
            .groupby('client_id')['transaction_amount']
            .transform(lambda x: x.shift().expanding().mean()))

    # Add new column - transaction_average_ratio (transaction amount by customer's average transaction amount)
    transaction_data['transaction_average_ratio'] = transaction_data['transaction_amount'] / transaction_data['avg_transaction_amount']

    # One-hot encode categorical variables (merchant_type, transaction_type, currency) 
    #   https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/
    #   https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python
    categories = ['merchant_type', 'transaction_type', 'currency']
    one_hot_encoding = pandas.get_dummies(transaction_data[categories]).astype(float)
    transaction_data = transaction_data.join(one_hot_encoding)
    
    # Replace any NaN values created from expanding average or one-hot encoding with 0
    transaction_data.fillna(0, inplace=True)

    return transaction_data
