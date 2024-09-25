from config import tushare_token
import tushare as ts
import pandas as pd

pro = ts.pro_api(tushare_token)

start_date = '20100101'
end_date = '20240201'
ticker = '601288.SH'

df_fina = pro.fina_indicator(ts_code = ticker, start_date = start_date, end_date = end_date)
df_price = pro.daily(ts_code = ticker, start_date = start_date, end_date = end_date)

df_fina['end_date'] = pd.to_datetime(df_fina['end_date'])
df_price['trade_date'] = pd.to_datetime(df_price['trade_date'])

# plot df_fina column bps and df_price column close in one plot.
import matplotlib.pyplot as plt
plt.plot(df_fina['end_date'], df_fina['bps'], 'g-')
plt.plot(df_price['trade_date'], df_price['close'], 'b-')
plt.legend(['bps', 'close'])
# 设置纵坐标从0开始
plt.ylim(0, max(df_fina['bps']))
# 保存图片
plt.savefig('bank_bps.png')
plt.show()