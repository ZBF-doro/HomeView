from flask import Blueprint, render_template, request, redirect, session, url_for
from model.User import User
from extensions import db
import secrets
import logging
from sqlalchemy import text

logger = logging.getLogger(__name__)

ub = Blueprint('user', __name__, template_folder='templates', url_prefix='/user')


@ub.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('username') and session.get('auth_token'):
        return redirect(url_for('page.home'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        logger.info(f"登录请求: username={username}")

        try:
            # 使用原始SQL查询避免列名问题
            sql = text("SELECT * FROM user WHERE user_name = :username")
            result = db.session.execute(sql, {'username': username}).fetchone()

            if result:
                # 创建用户对象
                user = User(
                    user_id=result[0],
                    user_name=result[1],
                    user_password=result[2]
                )

                if user.user_password == password:
                    token = secrets.token_hex(16)
                    session['username'] = user.user_name
                    session['user_id'] = user.user_id
                    session['auth_token'] = token
                    logger.info(f"用户 {username} 登录成功，重定向到首页")
                    return redirect(url_for('page.home'))
                else:
                    logger.warning(f"登录失败: 密码错误 (username={username})")
                    return render_template('login.html', error='用户名或密码错误')
            else:
                logger.warning(f"登录失败: 用户不存在 (username={username})")
                return render_template('login.html', error='用户名或密码错误')

        except Exception as e:
            logger.error(f"数据库查询失败: {str(e)}")
            return render_template('login.html', error='系统错误，请重试')

    return render_template('login.html')


@ub.route('/register', methods=['GET', 'POST'])
def register():
    if session.get('username') and session.get('auth_token'):
        return redirect(url_for('page.home'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        logger.info(f"注册请求: username={username}")

        if not username:
            return render_template('register.html', error='用户名不能为空')
        if not password:
            return render_template('register.html', error='密码不能为空')
        if not confirm_password:
            return render_template('register.html', error='请确认密码')
        if password != confirm_password:
            return render_template('register.html', error='两次输入的密码不一致')

        try:
            # 检查用户名是否存在
            sql = text("SELECT * FROM user WHERE user_name = :username")
            result = db.session.execute(sql, {'username': username}).fetchone()

            if result:
                return render_template('register.html', error='用户名已存在')

        except Exception as e:
            logger.error(f"数据库查询失败: {str(e)}")
            return render_template('register.html', error='系统错误，请重试')

        try:
            # 使用text显式声明SQL表达式
            insert_sql = text("INSERT INTO user (user_name, user_password) VALUES (:username, :password)")
            db.session.execute(insert_sql, {'username': username, 'password': password})
            db.session.commit()

            logger.info(f"用户 {username} 注册成功")

            # 注册成功后跳转到登录页面
            return redirect(url_for('user.login'))

        except Exception as e:
            db.session.rollback()
            logger.error(f"用户注册失败: {str(e)}")
            return render_template('register.html', error='注册失败，请重试')

    return render_template('register.html')


@ub.route('/logout')
def logout():
    username = session.get('username', '未知用户')
    logger.info(f"用户登出: username={username}")
    session.pop('username', None)
    session.pop('user_id', None)
    session.pop('auth_token', None)
    return redirect(url_for('user.login'))