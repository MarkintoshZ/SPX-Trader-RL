from alpha_vantage.timeseries import TimeSeries
ts = TimeSeries(key='YOUR_API_KEY', output_format='pandas')
data, meta_data = ts.get_intraday(symbol='MSFT',interval='5min', outputsize='full')
print(data)
print(meta_data)
print(data.describe())
data.to_csv('data/SPX5min.csv')