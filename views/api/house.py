from flask import Blueprint, request, jsonify
from app import db
from model.Hourse_info import Hourse_info
from sqlalchemy.exc import SQLAlchemyError

house_api = Blueprint('house_api', __name__)


@house_api.route('/api/houses', methods=['GET'])
def get_houses():
    """获取所有房屋数据（分页）"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        keyword = request.args.get('keyword', '')

        # 构建查询
        query = Hourse_info.query

        # 应用搜索过滤
        if keyword:
            query = query.filter(
                db.or_(
                    Hourse_info.title.ilike(f'%{keyword}%'),
                    Hourse_info.city.ilike(f'%{keyword}%'),
                    Hourse_info.region.ilike(f'%{keyword}%'),
                    Hourse_info.address.ilike(f'%{keyword}%'),
                    Hourse_info.tags.ilike(f'%{keyword}%')
                )
            )

        # 执行分页查询
        houses = query.paginate(page=page, per_page=per_page)

        # 准备响应数据
        result = {
            'items': [{
                'id': house.id,
                'title': house.title,
                'city': house.city,
                'region': house.region,
                'address': house.address,
                'hourseType': house.hourseType,
                'area_range': house.area_range,
                'price': house.price,
                'hourseDecoration': house.hourseDecoration,
                'tags': house.tags
            } for house in houses.items],
            'total': houses.total,
            'pages': houses.pages,
            'current_page': houses.page
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@house_api.route('/api/houses/<int:house_id>', methods=['GET'])
def get_house(house_id):
    """获取单个房屋详情"""
    try:
        house = Hourse_info.query.get_or_404(house_id)
        return jsonify({
            'id': house.id,
            'title': house.title,
            'city': house.city,
            'region': house.region,
            'address': house.address,
            'room_desc': house.room_desc,
            'area_range': house.area_range,
            'price': house.price,
            'hourseDecoration': house.hourseDecoration,
            'hourseType': house.hourseType,
            'tags': house.tags,
            'detail_url': house.detail_url
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@house_api.route('/api/houses', methods=['POST'])
def create_house():
    """创建新房屋"""
    try:
        data = request.json
        required_fields = ['title', 'city', 'region', 'address', 'price']

        # 验证必填字段
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # 创建新房屋对象
        new_house = Hourse_info(
            title=data['title'],
            city=data['city'],
            region=data['region'],
            address=data['address'],
            price=data['price'],
            hourseType=data.get('hourseType', ''),
            area_range=data.get('area_range', ''),
            hourseDecoration=data.get('hourseDecoration', ''),
            tags=data.get('tags', '')
        )

        # 保存到数据库
        db.session.add(new_house)
        db.session.commit()

        return jsonify({'message': 'House created successfully', 'id': new_house.id}), 201

    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({'error': 'Database error: ' + str(e)}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@house_api.route('/api/houses/<int:house_id>', methods=['PUT'])
def update_house(house_id):
    """更新房屋信息"""
    try:
        house = Hourse_info.query.get_or_404(house_id)
        data = request.json

        # 更新字段
        if 'title' in data: house.title = data['title']
        if 'city' in data: house.city = data['city']
        if 'region' in data: house.region = data['region']
        if 'address' in data: house.address = data['address']
        if 'price' in data: house.price = data['price']
        if 'hourseType' in data: house.hourseType = data['hourseType']
        if 'area_range' in data: house.area_range = data['area_range']
        if 'hourseDecoration' in data: house.hourseDecoration = data['hourseDecoration']
        if 'tags' in data: house.tags = data['tags']

        # 保存更改
        db.session.commit()

        return jsonify({'message': 'House updated successfully'}), 200

    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({'error': 'Database error: ' + str(e)}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@house_api.route('/api/houses/<int:house_id>', methods=['DELETE'])
def delete_house(house_id):
    """删除房屋"""
    try:
        house = Hourse_info.query.get_or_404(house_id)
        db.session.delete(house)
        db.session.commit()
        return jsonify({'message': 'House deleted successfully'}), 200
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({'error': 'Database error: ' + str(e)}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500