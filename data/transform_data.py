import pandas


def standardize_currency(currency_type, amount):
    """Convert a currency into USD. Conversion rates current as of 2025-08-24
    
    Args: 
        currency_type: the input currency type ('USD' 'CDN' 'RMB' 'EURO' 'JPN')
        amount:        amount of currency

    Returns:
        amount converted to USD 
    """
    conversion_rates = {
        'CDN': 1.38,
        'RMB': 7.16,
        'EURO': 0.85,
        'JPN': 147.37
    }
    if currency_type not in ['CDN', 'RMB', 'EURO', 'JPN']:
        raise ValueError("currency_type must be 'CDN', 'RMB', 'EURO', or 'JPN'")
        return -1
    