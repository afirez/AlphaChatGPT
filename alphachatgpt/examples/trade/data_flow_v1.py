import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
import zmq

from datasource import *

"""

pip install pandas 

# 消息队列除了 pyzmq， 还可以使用 redis，kafka
pip install pyzmq

# 定时器除了apscheduler 还可以使用 schedule, pip install schedule
# 或者使用线程， asyncio 轮询 
pip install apscheduler 

"""

# ZeroMQ 上下文
context = zmq.Context()

# Tick 数据发布者
tick_publisher = context.socket(zmq.PUB)
tick_publisher.bind("tcp://*:5555")

# K 线数据发布者
kline_publisher = context.socket(zmq.PUB)
kline_publisher.bind("tcp://*:5556")


def agg_ticks_to_klines(source_df: pd.DataFrame, target_interval='5S'):
    """
    根据 Tick 数据生成 K 线数据
    :param tick_data: 实时 Tick 数据 (pandas DataFrame)
    :param interval: K 线的时间间隔，默认为 5S 钟
    :return: K 线数据 (pandas DataFrame)
    """

    # 确保'time'列是datetime类型
    source_df['datetime'] = pd.to_datetime(source_df['datetime'])

    # 将标的和时间作为分组关键字，使用groupby聚合
    grouped_df = source_df.groupby(['code', pd.Grouper(key='datetime', freq=target_interval)])

    # 对每个分组应用聚合操作，例如计算OHLC和总成交量
    kline_df = grouped_df.agg({
        'price': ['first', 'max', 'min', 'last'],  # open, high, low, close
        'volume': 'sum'
    })

    # 重置索引，将分组关键字还原为列
    kline_df = kline_df.reset_index()

    kline_df.columns = [
        "code",
        "date",
        "open",
        "high",
        "low",      
        "close",
        "volume",
    ]

    # 打印结果
    # print(kline_df)

    return kline_df

code_list = [
    ['518880.XSHG','518880.XSHG'], #黄金ETF  518880 159934
    ['513100.XSHG','159941.XSHG'], #纳指ETF  159941 513100 
    ['399006.XSHE','159915.XSHE'], #创业板 159915 399006 159915 159952 512900
    # ['513050.XSHG','159607.XSHG'], #中概互联 ETF 513050 159607
    # ['399673.XSHE','159949.XSHE'], #创业板50 399673 159949
    ['000016.XSHG','510050.XSHG'], #上证50 000016 510050 510710 510850
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

def tick_data_mock(code):
    # 设置交易时间范围
    trading_hours_1 = pd.date_range("2024-01-01 09:30:00", "2024-01-01 11:30:00", freq='S')
    trading_hours_2 = pd.date_range("2024-01-01 13:00:00", "2024-01-01 15:00:00", freq='S')
    trading_hours = trading_hours_1.union(trading_hours_2)

    # 生成tick数据
    tick_data = pd.DataFrame({
        'datetime': trading_hours,
        'code': [code] * len(trading_hours),
        'price': np.random.rand(len(trading_hours)) * 10 + 100,  # 生成随机价格，假设在100到110之间
        'volume': np.random.randint(1, 100, size=len(trading_hours))  # 生成随机成交量
    })

    # 打印生成的tick数据
    print(tick_data)
    return tick_data

def tick_data_list_mock():
    df = pd.concat([tick_data_mock(code=code) for code in code_list_])
    return df

df_all = tick_data_list_mock()

def send_to_zeromq(socket, data_key, data):
    """
    发送数据到 ZeroMQ socket
    :param socket: ZeroMQ socket
    :param data_key: 用于识别数据类型的键
    :param data: 要发送的数据
    """
    message = {data_key: data}
    socket.send_multipart([bytes(data_key, 'utf-8'), json.dumps(message).encode('utf-8')])

def process_tick_and_kline():
    # 获取实时 Tick 数据
    tick_data = fund_etf_spot_em()
    
    if '日期' in tick_data.columns:
        tick_data['datetime'] = pd.to_datetime(tick_data['日期'])
    else:
        if 'datetime' not in tick_data.columns:
            # 获取本地当前时间
            current_datetime = datetime.datetime.now()   
            # 在 Tick 数据中插入本地当前时间列到第一列
            tick_data['datetime'] = current_datetime
            tick_data['datetime'] = pd.to_datetime(tick_data['datetime'])
        else:
            tick_data['datetime'] = pd.to_datetime(tick_data['datetime'])

    # 将 "datetime" 列转换为 datetime 类型
    # tick_data['datetime'] = pd.to_datetime(tick_data['datetime'])

    # 将 "datetime" 列移到第一列
    tick_data = tick_data[['datetime'] + [col for col in tick_data.columns if col != 'datetime']]

    tick_data.columns = [
        "datetime",
        "code",
        "name"
        "price",
        "涨跌额",
        "涨跌幅",
        "volume",
        "成交额",
        "开盘价",
        "最高价",
        "最低价",
        "昨收",
        "换手率",
        "流通市值",
        "总市值",
    ]

    # 生成 K 线数据（默认：每 5 分钟生成一个 K 线）
    kline_data = agg_ticks_to_klines(tick_data)

    # 将 Tick 数据发送到 ZeroMQ
    send_to_zeromq(tick_publisher, 'tick_data', tick_data.to_dict(orient='records'))

    # 将 K 线数据发送到 ZeroMQ
    send_to_zeromq(kline_publisher, 'kline_data', kline_data.to_dict(orient='records'))

def case_1():
    # 使用 APScheduler 定时每 5 分钟执行任务
    scheduler = BackgroundScheduler()
    scheduler.add_job(process_tick_and_kline, 'interval', minutes=5)
    scheduler.start()

def case_1_consume():
    # ZeroMQ Tick 和 K 线数据的消费者
    def consume_from_zeromq(socket, data_key):
        while True:
            message = socket.recv_multipart()
            data = json.loads(message[1].decode('utf-8'))
            print(f"从 {data_key} 收到数据: {data}")

    # 启动 Tick 和 K 线数据的消费者
    consume_tick_data = consume_from_zeromq(context.socket(zmq.SUB), 'tick_data')
    consume_tick_data.connect("tcp://localhost:5555")
    consume_tick_data.setsockopt_string(zmq.SUBSCRIBE, 'tick_data')

    consume_kline_data = consume_from_zeromq(context.socket(zmq.SUB), 'kline_data')
    consume_kline_data.connect("tcp://localhost:5556")
    consume_kline_data.setsockopt_string(zmq.SUBSCRIBE, 'kline_data')

def case_1_init():
    # 运行 ZeroMQ 事件循环
    try:
        zmq.device(zmq.FORWARDER, tick_publisher, kline_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        tick_publisher.close()
        kline_publisher.close()
        context.term()
