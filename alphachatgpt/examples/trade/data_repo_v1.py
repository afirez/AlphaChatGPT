import pandas as pd
import numpy as np

import time

from datasource import * 


def get_new_cycle_records(source_df, target_cycle):
    ret = []
    
    if source_df is None or len(source_df) < 2:
        return None
    
    source_len = len(source_df)
    source_cycle = source_df.iloc[-1]['date'] - source_df.iloc[-2]['date']

    # source_cycle: pd.Timedelta
    # ms = source_cycle.total_seconds() * ms
    source_cycle =  source_cycle.total_seconds() * 1000

    if target_cycle % source_cycle != 0:
        raise ValueError("target_cycle is not an integral multiple of source_cycle.")
    
    if (1000 * 60 * 60) % target_cycle != 0 and (1000 * 60 * 60 * 24) % target_cycle != 0:
        raise ValueError("target_cycle cannot complete the cycle.")
    
    multiple = target_cycle / source_cycle
    
    is_begin = False
    count = 0
    high = 0
    low = 0
    open = 0
    close = 0
    date = 0
    vol = 0
    
    for i in range(source_len):
        # source_df.iloc[i]['date']: pd.Timedelta
        # ms = source_df.iloc[i]['date'].timestamp() * 1000
        if ((1000 * 60 * 60 * 24) - (source_df.iloc[i]['date'].timestamp() * 1000) % (1000 * 60 * 60 * 24) +
            (pd.to_datetime('now').tz_localize('UTC').tz_convert('Asia/Shanghai').utcoffset().total_seconds() * 1000 * 60)) % target_cycle == 0:
            is_begin = True
        
        if is_begin:
            if count == 0:
                date = source_df.iloc[i]['date']
                open = source_df.iloc[i]['open']
                close = source_df.iloc[i]['close']
                high = source_df.iloc[i]['high']
                low = source_df.iloc[i]['low']
                vol = source_df.iloc[i]['volume']
                count += 1
            elif count < multiple:
                close = source_df.iloc[i]['close']
                high = max(high, source_df.iloc[i]['high'])
                low = min(low, source_df.iloc[i]['low'])
                vol += source_df.iloc[i]['volume']
                count += 1
            
            if count == multiple or i == source_len - 1:
                ret.append({
                    'date': date,
                    'open': open,
                    'close': close,
                    'high': high,
                    'low': low,
                    'volume': vol,
                })
                count = 0
    columns = [
        "date",
        "open",
        "close",
        "high",
        "low",
        "volume"
    ]
    return pd.DataFrame(ret, columns=columns)

code_list = [
    ['518880.XSHG','518880.XSHG'], #黄金ETF  518880
    ['513100.XSHG','513100.XSHG'], #纳指ETF  513100
    ['399006.XSHE','159915.XSHE'], #创业板 399006 159915 159952 512900
    # ['513050.XSHG','513050.XSHG'], #中概互联 ETF  
    # ['399673.XSHE','399673.XSHE'], #创业板50 399673 159949
    ['000016.XSHG','000016.XSHG'], #上证50 000016 510050 510710 510850
    # ['000300.XSHG','510300.XSHG'], #沪深300
    # ['000905.XSHG','510500.XSHG'] #中证500
    # ['399975.XSHE','399975.XSHE'], #证劵公司 399975 512880 512000 512900
    # ['399987.XSHE','399987.XSHE'], #中证酒
    # ['399932.XSHE','159928.XSHE'], #消费ETF
    # ['000913.XSHG','000913.XSHG'], #300医药 000913 399913 000913
    # ['515000.XSHG','515000.XSHG'], #科技 515000
    # ['000015.XSHG','510880.XSHG'], #红利ETF
]

code_list_ = [l[1].split(".")[0] for l in code_list]

def get_source_df():
    fund_etf_hist_min_em_df = fund_etf_hist_min_em(
        symbol = code_list_[2],
        period="1",
        adjust="qfq",
        start_date="2024-01-02 09:30:00",
        end_date="2024-01-08 15:00:00",
    )

    fund_etf_hist_min_em_df.columns = [
        "date",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "price",
    ]

    source_df = fund_etf_hist_min_em_df
    source_df["date"] = pd.to_datetime(source_df["date"])
    # source_df = fund_etf_hist_min_em_df.set_index(["date"], drop=True, inplace=False)
    print(source_df)
    return source_df

source_df = get_source_df()

def case_1():
    """
    # Example usage:
    # Assuming source_df is a pandas DataFrame with columns: 'high', 'low', 'open', 'close', 'Time', 'volume'
    # target_cycle is the desired target cycle in milliseconds

    # source_df = pd.DataFrame(...)  # Replace ... with your data
    # target_cycle = 1000 * 60 * 60 * 4  # 4 hours in milliseconds

    # new_cycle_df = get_new_cycle_records(source_df, target_cycle)
    # print(new_cycle_df)

    """ 

    # while True:
    #     r = exchange.get_records()  # 原始数据，作为合成K线的基础K线数据，例如要合成4小时K线，可以用1小时K线作为原始数据。
    #     r2 = get_new_cycle_records(r, 1000 * 60 * 60 * 4)  # 通过 GetNewCycleRecords 函数 传入 原始K线数据 r , 和目标周期， 1000 * 60 * 60 * 4 即 目标合成的周期 是4小时K线数据。

    #     # 在这里插入你的画图逻辑

    #     time.sleep(1)  # 每次循环间隔 1 秒，防止访问K线接口获取数据过于频繁，导致交易所限制

    new_df = get_new_cycle_records(source_df=source_df, target_cycle=1000 * 60 * 3)
    # print(new_df)

    return new_df

def case_2():
    new_df = get_new_cycle_records(source_df=source_df, target_cycle=1000 * 60 * 5)
    # print(new_df)
    return new_df

def case_3():
    new_df = get_new_cycle_records(source_df=source_df, target_cycle=1000 * 60 * 15)
    # print(new_df)
    return new_df

def case_4():
    code_list = [
        ['518880.XSHG','518880.XSHG'], #黄金ETF  518880
        ['513100.XSHG','513100.XSHG'], #纳指ETF  513100
        ['399006.XSHE','159915.XSHE'], #创业板 399006 159915 159952 512900
        # ['513050.XSHG','513050.XSHG'], #中概互联 ETF  
        # ['399673.XSHE','399673.XSHE'], #创业板50 399673 159949
        ['000016.XSHG','000016.XSHG'], #上证50 000016 510050 510710 510850
        # ['000300.XSHG','510300.XSHG'], #沪深300
        # ['000905.XSHG','510500.XSHG'] #中证500
        # ['399975.XSHE','399975.XSHE'], #证劵公司 399975 512880 512000 512900
        # ['399987.XSHE','399987.XSHE'], #中证酒
        # ['399932.XSHE','159928.XSHE'], #消费ETF
        # ['000913.XSHG','000913.XSHG'], #300医药 000913 399913 000913
        # ['515000.XSHG','515000.XSHG'], #科技 515000
        # ['000015.XSHG','510880.XSHG'], #红利ETF
    ]

    code_list_ = [l[1].split(".")[0] for l in code_list]

    fund_etf_hist_hfq_em_df = fund_etf_hist_em(
        symbol=code_list_[2],
        period="daily",
        start_date="20000101",
        end_date="20240108",
        adjust="hfq",
    )
    print(fund_etf_hist_hfq_em_df)

    fund_etf_spot_em_df = fund_etf_spot_em()
    # print(fund_etf_spot_em_df)

    fund_etf_hist_min_em_df = fund_etf_hist_min_em(
        symbol = code_list_[2],
        period="1",
        adjust="hfq",
        start_date="2024-01-02 09:30:00",
        end_date="2024-01-08 15:00:00",
    )

    df_spot_em = fund_etf_hist_em_df[fund_etf_hist_em_df["代码"] == code_list_[2]]
    print(df_spot_em)

    df_all_spot_em = fund_etf_spot_em_df
    
    # numpy groupby 
    a = df_all_spot_em
    a = a[a[:, 0].argsort()]  # 按照user_id排序（排序是使用np.split进行分组的前提）
    a = np.split(a, np.unique(a[:, 0], return_index=True)[1][1:])  # 按照user_id分组
    a = np.array([tmp[tmp[:, 2].argsort()[-1], :2] for tmp in a])  # 取分组下分值最高的item_id
    a = a[a[:, 1].argsort()]
    item_ids, ind = np.unique(a[:, 1], return_index=True)
    a = np.split(a[:, 0], ind[1:])
    result = dict(zip(item_ids, a))


if __name__ == "__main__":
    new_1_df = case_1()
    print(new_1_df)
    new_2_df = case_2()
    print(new_2_df)
    new_3_df = case_3()
    print(new_3_df)
    
    # case_4()