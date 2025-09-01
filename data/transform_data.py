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
    Input: none
    Returns: transaction_data (dataframe)
    """

    transaction_data = pandas.read_csv('data/raw_data.csv')    
    
    # Delete rows with null values
    transaction_data.dropna(how='any', axis=0)

    # Convert timestamp column to datetime
    transaction_data['timestamp'] = pandas.to_datetime(transaction_data['timestamp'])

    # Add new column -- avg_transaction_amount (customers' average transaction amount by client_id)
    avg_amount_by_customer = transaction_data.groupby('client_id')['transaction_amount'].mean()
    transaction_data = pandas.merge(transaction_data, avg_amount_by_customer, on = 'client_id', how = 'left')
    transaction_data = transaction_data.rename(columns = {'transaction_amount_y': 'avg_transaction_amount'})

    # Add new column - transaction_average_ratio (transaction amount by customer's average transaction amount)
    transaction_data['transaction_average_ratio'] = transaction_data['transaction_amount_x'] / transaction_data['avg_transaction_amount']

    return transaction_data
