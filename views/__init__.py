# views/__init__.py
from .user.user import ub as user_blueprint
from .page.page import page_bp as  page_blueprint

__all__ = ['user_blueprint', 'page_blueprint']