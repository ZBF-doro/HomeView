from extensions import db


class User(db.Model):
    __tablename__ = 'user'

    # 关键修改：使用正确的列名（带下划线）
    user_id = db.Column('user_id', db.Integer, primary_key=True)
    user_name = db.Column('user_name', db.String(80), unique=True, nullable=False)
    user_password = db.Column('user_password', db.String(128), nullable=False)

    def __repr__(self):
        return f'<User {self.user_name}>'

    # 添加密码验证方法（可选）
    def check_password(self, password):
        return self.user_password == password

