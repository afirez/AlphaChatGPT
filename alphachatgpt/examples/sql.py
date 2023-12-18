from sqlalchemy import create_engine, text
"""
pip install sqlalchemy
pip install mysql-connector
"""

host = "192.168.0.18"
port = "3306"
db =  "db_x"
user = "root"
pwd = "123456"

engine = create_engine(f"mysql+mysqlconnector://{user}:{pwd}@{host}:{port}/{db}")

user_id = "000007"

with engine.connect() as conn:
    result = conn.execute(text(f"DELETE FROM accounts WHERE id={user_id}"))
    conn.commit()