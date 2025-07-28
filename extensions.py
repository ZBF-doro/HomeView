from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

# 创建数据库实例
db = SQLAlchemy(model_class=Base)