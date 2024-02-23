#!/usr/bin/env /home/alex/anaconda3/bin/python
# 可以考虑放在crontab里面，每天定时运行，打在log里面. 例如
# 30 18 * * 1-5 /path/to/your/repo/calculate_premium.py >> /var/log/premium.log 2>&1
# 注意给脚本加上可执行权限 chmod +x calculate_premium.py
# 给log文件加上可写权限 chmod 666 /var/log/premium.log

import tushare as ts
import pandas as pd
import datetime
from tabulate import tabulate
from config import tushare_token

def get_futures_data(contract_code):
    # 替换成你自己的Tushare Token
    ts.set_token(tushare_token)
    
    # 初始化Tushare
    pro = ts.pro_api()
    
    # 获取股指期货合约信息
    futures_info = pro.fut_basic(exchange='CFFEX', fut_type='1')
    
    # 根据合约代码筛选对应的合约数据
    futures_info = futures_info[futures_info['ts_code'] == contract_code]
    
    # 获得当前日期
    today = datetime.datetime.now()
    today = today.strftime("%Y%m%d")

    # 获取最近一个交易日的日期
    trade_cal = pro.trade_cal(exchange='CFFEX', is_open='1', start_date='20230701', end_date=today)
    today = datetime.datetime.now().strftime('%Y%m%d')
    hour = datetime.datetime.now().hour

    if (today in set(trade_cal['pretrade_date']) or today in set(trade_cal['cal_date'])) and hour > 17:
        last_trade_day = today
    else:
        date_map = dict(zip(trade_cal['cal_date'], trade_cal['pretrade_date']))
        last_trade_day = date_map[today]
    # print("today:", today, "last_trade_day:", last_trade_day)    
    # 获取合约价格数据
    futures_prices = pro.fut_daily(ts_code=contract_code, trade_date=last_trade_day)
    
    # 获取股指数据
    index_code = {'IC': '000905.SH', 'IH': '000016.SH', 'IF': '000300.SH', 'IM': '000852.SH'}
    stock_index_data = pro.index_daily(ts_code=index_code[contract_code[:2]], trade_date=last_trade_day)
    
    return futures_prices, stock_index_data, futures_info

def calculate_premium(futures_prices, stock_index_data, futures_info):
#    print(futures_prices)
#    print(stock_index_data)
#    print(futures_info)
    futures_info['expire_date'] = futures_info['last_ddate']
    # 合并数据，以方便后续计算
    merged_data = pd.merge(futures_prices, stock_index_data, on='trade_date', suffixes=('_futures', '_index'))
    merged_data = pd.merge(merged_data, futures_info, left_on = 'ts_code_futures', right_on='ts_code', suffixes=('_futures', '_index'))

    # 计算贴水（贴水 = 期货价格 - 股指价格）
    merged_data['premium'] = merged_data['close_futures'] - merged_data['close_index']

    # 计算贴水百分比
    if len(merged_data['premium']) != 0:
        merged_data['premium_percentage'] = "%.2f%%"%(merged_data['premium'] / merged_data['close_index'] * 100)
    else:
        merged_data['premium_percentage'] = 0
    
    return merged_data[['ts_code_futures', 'trade_date', 'close_futures', 'close_index', 'premium', 'premium_percentage', 'expire_date']]

def calculate_annualized_premium(premium_data):
    # 计算剩余到期日天数
    # print(premium_data)
    premium_data['expiry_date'] = pd.to_datetime(premium_data['expire_date'])
    premium_data['remaining_days'] = (premium_data['expiry_date'] - datetime.datetime.now()).dt.days
    
    # 计算年化贴水
    if len(premium_data['remaining_days']) != 0:
        premium_data['annualized_premium'] = (premium_data['premium'] / premium_data['remaining_days']) * 365
        premium_data['ann_premium_percentage'] = "%.2f%%"%(premium_data['annualized_premium'] / premium_data['close_index'] * 100)
    else:
        premium_data['annualized_premium'] = 0
        premium_data['ann_premium_percentage'] = 0

    return premium_data[['ts_code_futures', 'trade_date', 'close_futures', 'close_index', 'premium', 'premium_percentage', 'annualized_premium', 'ann_premium_percentage']]

def get_demonth_list():
    exchange = 'CFFEX'
    fut_type = '1'
    pro = ts.pro_api(tushare_token)
    df = pro.fut_basic(exchange=exchange, fut_type=fut_type)
    # filter out delisted contracts
    today_str = datetime.datetime.now().strftime('%Y%m%d')
    df = df[df['delist_date'] > today_str]
    # keep contracts with ts_code starting with 'IC'
    df = df[df['ts_code'].str.startswith('IC')]
    d_months = df['d_month'].unique()
    d_months.sort()
    return d_months

if __name__ == "__main__":
    yearmonths = get_demonth_list()
    print(yearmonths)
    suffix = '.CFX'

    # 获取合约的数据并计算年化贴水
    premiums_list = []
    products = ['IC', 'IH', 'IF', 'IM']
    for product in products:
        instruments = []
        for yearmonth in yearmonths:
            yearmonth = int(yearmonth[2:])
            instruments.append("%s%04d%s" % (product, yearmonth, suffix))
        premiums = []
        for instrument in instruments:
            ic_futures_data, ic_stock_index_data, futures_info = get_futures_data(instrument)
            ic_premium_data = calculate_premium(ic_futures_data, ic_stock_index_data, futures_info)
            ic_premium_data_with_annualized = calculate_annualized_premium(ic_premium_data)
            premiums.append(ic_premium_data_with_annualized)
        premiums = pd.concat(premiums)
        premiums.set_index(['ts_code_futures'], inplace=True)
        premiums_list.append(premiums)
    premiums = pd.concat(premiums_list)
    # remove suffix from ts_code_futures
    premiums.reset_index(inplace=True)
    premiums['ts_code_futures'] = premiums['ts_code_futures'].str[:-4]
    # rename columns ts_code_futures   |   trade_date |   close_futures |   close_index |   premium | premium_percentage   |   annualized_premium | ann_premium_percentage
    premiums.rename(columns={'ts_code_futures':'期货', 'trade_date': '日期', 'close_futures': '期货价格', 'close_index': '股指价格', 'premium': '贴水', 
                             'premium_percentage': '贴水百分比', 'annualized_premium': '年化贴水', 'ann_premium_percentage': '年化百分比'}, inplace=True)
    # set index to ts_code_futures
    premiums.set_index(['期货'], inplace=True)
    # split table by product, i.e. the prefix of ts_code_futures
    for product in products:
        product_premiums = premiums[premiums.index.str.startswith(product)]
        # print with tabulate format and headers
        print(tabulate(product_premiums, headers='keys', tablefmt='psql'))
    print('\n\n')
