# from flask import Flask, session, render_template, redirect, Blueprint, request, jsonify
# from model.Hourse_info import Hourse_info
# from extensions import db
# from sqlalchemy.exc import SQLAlchemyError
# import logging
#
# # 配置日志
# logger = logging.getLogger(__name__)
#
# pb = Blueprint('page', __name__, url_prefix='/', template_folder='templates')
#
# @pb.route('/home')
# def home():
#     username = session.get('username')
#     return render_template('index.html', username=username)
#
# @pb.route('/data_operation', methods=['GET'])
# def data_operation():
#     username = session.get('username')
#     keyword = request.args.get('keyword', '')
#     page = request.args.get('page', 1, type=int)
#
#     query = Hourse_info.query
#     if keyword:
#         query = query.filter(
#             db.or_(
#                 Hourse_info.title.like(f'%{keyword}%'),
#                 Hourse_info.city.like(f'%{keyword}%'),
#                 Hourse_info.region.like(f'%{keyword}%')
#             )
#         )
#
#     pagination = query.paginate(page=page, per_page=10)
#     houses = pagination.items
#
#     return render_template(
#         'data_operation.html',
#         username=username,
#         houses=houses,
#         pagination=pagination,
#         keyword=keyword
#     )
#
# @pb.route('/api/houses', methods=['GET'])
# def get_houses():
#     page = request.args.get('page', 1, type=int)
#     per_page = request.args.get('per_page', 10, type=int)
#     keyword = request.args.get('keyword', '')
#
#     query = Hourse_info.query
#     if keyword:
#         query = query.filter(
#             db.or_(
#                 Hourse_info.title.like(f'%{keyword}%'),
#                 Hourse_info.city.like(f'%{keyword}%'),
#                 Hourse_info.region.like(f'%{keyword}%')
#             )
#         )
#
#     pagination = query.paginate(page=page, per_page=per_page)
#     houses = [
#         {
#             'id': house.id,
#             'title': house.title,
#             'city': house.city,
#             'region': house.region,
#             'address': house.address,
#             'price': house.price,
#             'hourseType': house.hourseType,
#             'area_range': house.area_range,
#             'hourseDecoration': house.hourseDecoration,
#             'tags': house.tags
#         } for house in pagination.items
#     ]
#
#     return jsonify({
#         'items': houses,
#         'total': pagination.total,
#         'pages': pagination.pages,
#         'current_page': pagination.page
#     })
#
# @pb.route('/api/houses/<int:house_id>', methods=['GET'])
# def get_house(house_id):
#     house = Hourse_info.query.get_or_404(house_id)
#     return jsonify({
#         'id': house.id,
#         'title': house.title,
#         'city': house.city,
#         'region': house.region,
#         'address': house.address,
#         'price': house.price,
#         'hourseType': house.hourseType,
#         'area_range': house.area_range,
#         'hourseDecoration': house.hourseDecoration,
#         'tags': house.tags
#     })
#
# @pb.route('/api/houses', methods=['POST'])
# def add_house():
#     data = request.get_json()
#     try:
#         new_house = Hourse_info(
#             title=data.get('title'),
#             city=data.get('city'),
#             region=data.get('region'),
#             address=data.get('address'),
#             price=data.get('price'),
#             hourseType=data.get('hourseType'),
#             area_range=data.get('area_range'),
#             hourseDecoration=data.get('hourseDecoration'),
#             tags=data.get('tags'),
#             cover=data.get('cover'),
#             room_desc=data.get('room_desc'),
#             all_ready=data.get('all_ready'),
#             company=data.get('company'),
#             on_time=data.get('on_time'),
#             open_date=data.get('open_date'),
#             totalPrice_range=data.get('totalPrice_range'),
#             sale_status=data.get('sale_status'),
#             detail_url=data.get('detail_url')
#         )
#         db.session.add(new_house)
#         db.session.commit()
#         return jsonify({'id': new_house.id}), 201
#     except Exception as e:
#         db.session.rollback()
#         logger.error(f"添加房屋失败: {str(e)}")
#         return jsonify({'error': '添加失败'}), 500
#
# @pb.route('/api/houses/<int:house_id>', methods=['PUT'])
# def update_house(house_id):
#     try:
#         house = Hourse_info.query.get_or_404(house_id)
#         data = request.get_json()
#
#         for key, value in data.items():
#             if hasattr(house, key):
#                 setattr(house, key, value)
#
#         db.session.commit()
#         return jsonify({'message': '更新成功'})
#     except Exception as e:
#         db.session.rollback()
#         logger.error(f"更新房屋失败: {str(e)}")
#         return jsonify({'error': '更新失败'}), 500
#
# @pb.route('/api/houses/<int:house_id>', methods=['DELETE'])
# def delete_house(house_id):
#     try:
#         house = Hourse_info.query.get_or_404(house_id)
#         db.session.delete(house)
#         db.session.commit()
#         return jsonify({'message': '删除成功'})
#     except Exception as e:
#         db.session.rollback()
#         logger.error(f"删除房屋失败: {str(e)}")
#         return jsonify({'error': '删除失败'}), 500
#
# @pb.route('/detail_analysis')
# def detail_analysis():
#     username = session.get('username')
#     return render_template('detail_analysis.html', username=username)
#
# @pb.route('/other_analysis')
# def other_analysis():
#     username = session.get('username')
#     return render_template('other_analysis.html', username=username)
#
# @pb.route('/prediction')
# def prediction():
#     username = session.get('username')
#     return render_template('prediction.html', username=username)
#
# @pb.route('/type_analysis')
# def type_analysis():
#     username = session.get('username')
#     return render_template('type_analysis.html', username=username)
#
# @pb.route('/wordcloud')
# def wordcloud():
#     username = session.get('username')
#     return render_template(, username=username)