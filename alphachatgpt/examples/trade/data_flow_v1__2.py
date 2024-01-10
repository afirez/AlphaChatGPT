import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
import zmq

import time

from datasource import *

"""

pip install pandas 

# 消息队列除了 pyzmq， 还可以使用 redis，kafka
pip install pyzmq

# 定时器除了apscheduler 还可以使用 schedule, pip install schedule
# 或者使用线程， asyncio 轮询
# 这里用 轮询 代替 apscheduler 测试 case
pip install apscheduler 

"""

class StaticScope:
    
    # 缓存 tick，应该用存入数据库代替
    tick_df = pd.DataFrame()

    # 缓存 kline，应该用存入数据库代替
    kline_df = pd.DataFrame()
    # kline_df = pd.DataFrame([], columns=[
    #     "code",
    #     "date",
    #     "open",
    #     "high",
    #     "low",      
    #     "close",
    #     "volume",
    # ])

staticScope = StaticScope()

class ZMQ:

    # ZeroMQ 上下文
    context = zmq.Context()

    # Tick 数据发布者
    tick_publisher = context.socket(zmq.PUB)
    # tick_publisher.bind("tcp://*:5555")

    # K 线数据发布者
    kline_publisher = context.socket(zmq.PUB)
    # kline_publisher.bind("tcp://*:5556")

    def bind_publisher(self):
        self.tick_publisher.bind("tcp://*:5555")
        self.kline_publisher.bind("tcp://*:5556")
    
    def send_to_zeromq(self, socket, data_key, data):
        """
        发送数据到 ZeroMQ socket
        :param socket: ZeroMQ socket
        :param data_key: 用于识别数据类型的键
        :param data: 要发送的数据
        """
        message = {data_key: data}
        # print(f"data_key: {data_key} \n{data}")
        socket.send_multipart([bytes(data_key, 'utf-8'), json.dumps(message).encode('utf-8')])

mq = ZMQ()


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


def process_tick_and_kline(tick_data):
    # 获取实时 Tick 数据
    # tick_data = fund_etf_spot_em()

    if 0:
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

    # 将 "datetime" 列转换为 datetime 类型
    tick_data['datetime'] = pd.to_datetime(tick_data['datetime'])
    
    staticScope.tick_df = pd.concat([staticScope.tick_df, tick_data])
    
    # 生成 K 线数据（默认：每 5 分钟生成一个 K 线）
    kline_data = agg_ticks_to_klines(staticScope.tick_df)

    # 这有个判断数据是不是有更新，待完善
    if len(staticScope.kline_df) == 0:
        staticScope.kline_df = kline_data   
    else:
        if not kline_data.iloc[-1].equals(staticScope.kline_df.iloc[-1]):
            print(kline_data.iloc[-1])

        # diff = kline_data - staticScope.kline_df
        # print(f"diff: \n{diff}")
        staticScope.kline_df = kline_data  

    
    tick_data_ = tick_data.copy()
    kline_data_ = kline_data.copy()

    tick_data_['datetime'] = tick_data_['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    kline_data_['date'] = kline_data_['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    print(tick_data_)

    # 将 Tick 数据发送到 ZeroMQ
    mq.send_to_zeromq(mq.tick_publisher,'tick_data', tick_data_.to_dict(orient='records'))

    # 将 K 线数据发送到 ZeroMQ
    mq.send_to_zeromq(mq.kline_publisher, 'kline_data', kline_data_.to_dict(orient='records'))



def case_1():
    # 使用 APScheduler 定时每 5 分钟执行任务
    # scheduler = BackgroundScheduler()
    # scheduler.add_job(process_tick_and_kline, 'interval', minutes=5)
    # scheduler.start()

    # or 使用 schedule 定时执行任务

    mq.bind_publisher()

    datetime_list = df_all[df_all["code"] == code_list_[0]]["datetime"].values

    for dt in datetime_list:
        process_tick_and_kline(df_all[df_all["datetime"] == dt])
        time.sleep(0.5)

def case_1_consume():
    import zmq
    import json
    import time

    # ZeroMQ 上下文
    # mq.context = zmq.Context()

    # Tick 数据订阅者
    tick_subscriber = mq.context.socket(zmq.SUB)
    tick_subscriber.connect("tcp://localhost:5555")
    tick_subscriber.setsockopt_string(zmq.SUBSCRIBE, "tick_data")

    # K 线数据订阅者
    kline_subscriber = mq.context.socket(zmq.SUB)
    kline_subscriber.connect("tcp://localhost:5556")
    kline_subscriber.setsockopt_string(zmq.SUBSCRIBE, "kline_data")

    def process_tick_data(data):
        # 处理 Tick 数据的逻辑
        print("Received Tick Data:")
        print(json.dumps(data, indent=2))
        # Add your logic here

    def process_kline_data(data):
        # 处理 K 线数据的逻辑
        print("Received K-line Data:")
        print(json.dumps(data, indent=2))
        # Add your logic here

    try:
        while True:
            # 接收并处理 Tick 数据
            tick_message = tick_subscriber.recv_multipart()
            tick_data_key, tick_data_json = tick_message
            tick_data = json.loads(tick_data_json)
            process_tick_data(tick_data)

            # 接收并处理 K 线数据
            kline_message = kline_subscriber.recv_multipart()
            kline_data_key, kline_data_json = kline_message
            kline_data = json.loads(kline_data_json)
            process_kline_data(kline_data)
            time.sleep(0.1)

    except KeyboardInterrupt:
        # 在用户按下 Ctrl+C 时，关闭 ZeroMQ 连接
        tick_subscriber.close()
        kline_subscriber.close()
        mq.context.term()


def case_1_init():
    # 运行 ZeroMQ 事件循环
    try:
        zmq.device(zmq.FORWARDER, mq.tick_publisher, mq.kline_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        mq.tick_publisher.close()
        mq.kline_publisher.close()
        mq.context.term()

if __name__ == "__main__":

    import argparse

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="data flow: agg ticks to klines")

    # 添加参数
    parser.add_argument("mode", help="mode: 0 producer, 1 consume", default=0)
    # parser.add_argument("--flag", help="flag: 0 producer, 1 consume")

    # 解析命令行参数
    args = parser.parse_args()

    # 访问参数
    mode = args.mode
    # flag = args.flag

    print("Argument 1:", mode)
    # print("Argument 2:", flag)

    if mode and str(mode) == "1":
        case_1_consume()
    else:
        case_1()    
