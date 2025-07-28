from model.Hourse_info import Hourse_info
from extensions import db


def get_filtered_data(query):
    """获取过滤后的数据，排除脏数据"""
    # 过滤掉无效的on_time
    query = query.filter(
        db.or_(
            Hourse_info.on_time != "0000-00-00 00:00:00",
            Hourse_info.on_time.is_(None)
        )
    )

    # 过滤掉无效的room_desc
    query = query.filter(
        db.or_(
            Hourse_info.room_desc.notin_(['[""]', '[]', '""']),
            Hourse_info.room_desc.is_(None)
        )
    )

    # 过滤掉空价格
    query = query.filter(Hourse_info.price != '')

    return query