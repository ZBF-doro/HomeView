# app.py
from flask import Flask, url_for, redirect, session
from extensions import db
from views.page.page import page_bp
from views.user.user import ub as user_bp
import os
from config import Config


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # 初始化数据库
    db.init_app(app)

    # 创建数据库表
    with app.app_context():
        db.create_all()

    # 注册蓝图
    app.register_blueprint(page_bp)
    app.register_blueprint(user_bp, url_prefix='/user')

    # 根路由重定向
    @app.route('/home')
    def index():
        if session.get('username') and session.get('auth_token'):
            return redirect(url_for('page.home'))
        else:
            return redirect(url_for('user.login'))

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)