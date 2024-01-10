import pandas as pd
import json
import redis
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler

from datasource import *

"""

pip install pandas 
pip install redis
pip install apscheduler

"""

# 初始化 Redis 连接
redis_conn = redis.StrictRedis(host='localhost', port=6379, db=0)

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


def send_to_redis(data, key):
    """
    将数据发送到 Redis 消息队列
    :param data: 要发送的数据
    :param key: Redis 队列的键名
    """
    redis_conn.publish(key, json.dumps(data))

def process_tick_and_k_line():
    # 获取实时 Tick 数据
    tick_data = fund_etf_spot_em()

    # 生成 K 线数据，默认为每5分钟生成一个K线
    k_line_data = agg_ticks_to_klines(tick_data)

    # 发送 Tick 数据到 Redis
    send_to_redis(tick_data.to_dict(orient='records'), 'tick_data')

    # 发送 K 线数据到 Redis
    send_to_redis(k_line_data.to_dict(orient='records'), 'k_line_data')

# 使用定时任务库定时执行处理 Tick 和 K 线数据
scheduler = BackgroundScheduler()
scheduler.add_job(process_tick_and_k_line, 'interval', minutes=5)  # 每5分钟执行一次
scheduler.start()

# 消费 Redis 中的 Tick 数据和 K 线数据
def consume_from_redis(queue_key):
    pubsub = redis_conn.pubsub()
    pubsub.subscribe(queue_key)

    for message in pubsub.listen():
        if message['type'] == 'message':
            data = json.loads(message['data'])
            print(f"Received data from {queue_key}: {data}")

# 启动两个消费者，分别消费 Tick 数据和 K 线数据
consume_tick_data = consume_from_redis('tick_data')
consume_k_line_data = consume_from_redis('k_line_data')
