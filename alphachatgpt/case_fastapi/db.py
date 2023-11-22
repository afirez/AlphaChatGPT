from sqlalchemy import create_engine, Column, Integer, String, MetaData, Table
from databases import Database

# 配置数据库连接信息
DATABASE_URL = "sqlite:///./test.db"


# 创建 SQLAlchemy 引擎
engine = create_engine(DATABASE_URL)

# 创建数据库实例
metadata = MetaData()

# 创建 users 表
drama_subtitles = Table(
    "tb_drama_subtitles",
    metadata,
    Column("id", Integer, primary_key=True, index=True),
    # Column("sub_id", Integer, unique=True),
    Column("sub_id", Integer),
    Column("drama_id", Integer),
    Column("episode_id", Integer),
    Column("language", String),
    Column("language_id", Integer),
    Column("url", String),
    Column("expire", Integer),
)


engine = create_engine(DATABASE_URL)
metadata.create_all(engine)

# 创建数据库连接池
database = Database(DATABASE_URL)
