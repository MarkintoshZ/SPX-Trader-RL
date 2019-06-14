import pandas as pd

df = pd.read_csv("data/DAT_MT_SPXUSD_M1_2018.csv")
df.Time = pd.to_datetime(df.Time, format='%Y%m%d %H%M%S')
df.set_index('Time', inplace=True)

ndf = df.resample('10Min').agg({
    'Open':'first',
    'High':'max',
    'Low':'min',
    'Close': ['last', 'std', 'median'],
    'Volume': 'sum'
    })

ndf.columns = ['Open', 'High', 'Low', 'Close', 'STD', 'Median', 'Volume']
print(ndf)
ndf.reset_index(inplace=True)

ndf.to_csv('SPX10min.csv', 
           header=['Date', 'Open', 'High', 'Low', 'Close',
                   'STD', 'Median', 'Volume'],
           index=False)