from urllib.parse import quote
from sqlalchemy import create_engine
from sqlalchemy import text


user ="bkh"
pwd = "bkh"
host = "10.28.224.136"
port = 30010
db_url = f"mysql+pymysql://{user}:{quote(pwd)}@{host}:{port}/cv06_database"

engine = create_engine(
    db_url, pool_size=5, max_overflow=5, echo=True
)


conn = engine.connect()
result = conn.execute(text("show databases"))
try:
    conn.execute(text("CREATE TABLE test_table (x int, y int)"))
except: 
    conn.close()
    exit()
conn.execute(text("INSERT INTO test_table (x, y) VALUES (:x, :y)"),
            [{"x": 1, "y":1}, {"x":2, "y":4}])
conn.commit()
print(result.all())
conn.close()