import json
from collections import defaultdict
from flask import session, render_template, request, jsonify, redirect, url_for, flash
from sqlalchemy import func, or_
from werkzeug.security import generate_password_hash

from model.Hourse_info import Hourse_info
from model.User import User
from model.History import History
from extensions import db
import logging
from flask import Blueprint
from functools import wraps
import re
from datetime import datetime

logger = logging.getLogger(__name__)
page_bp = Blueprint('page', __name__, template_folder='templates')


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('username') or not session.get('auth_token'):
            return redirect(url_for('user.login'))
        return f(*args, **kwargs)

    return decorated_function


@page_bp.route('/')
def root():
    return redirect(url_for('page.home'))


@page_bp.route('/home')
@login_required
def home():
    username = session.get('username', '游客')
    total_houses = Hourse_info.query.count()
    region_count = db.session.query(Hourse_info.region).distinct().count()
    avg_price = 0
    try:
        avg_price_result = db.session.query(
            func.avg(
                func.cast(
                    func.replace(Hourse_info.price, '元/㎡', ''),
                    db.Float
                )
            )
        ).filter(
            Hourse_info.price.isnot(None),
            Hourse_info.price != '',
            func.cast(func.replace(Hourse_info.price, '元/㎡', ''), db.Float) > 0
        ).scalar()
        avg_price = round(avg_price_result, 2) if avg_price_result else 0
    except Exception as e:
        logger.error(f"平均价格计算失败: {str(e)}")
    recent_houses = Hourse_info.query.filter(
        Hourse_info.open_date.isnot(None),
        Hourse_info.open_date != ''
    ).order_by(Hourse_info.open_date.desc()).limit(5).all()

    # 创建房屋字典副本（不修改原始对象）
    cleaned_houses = []
    for house in recent_houses:
        house_data = {
            'id': house.id,
            'title': house.title,
            'region': house.region,
            'price': house.price,
            'area_range': house.area_range,
            'open_date': house.open_date
        }
        # 仅对副本进行清理
        if house_data['price']:
            house_data['price'] = house_data['price'].replace('元/㎡', '').strip()
        if house_data['area_range']:
            house_data['area_range'] = house_data['area_range'].replace('㎡', '').strip()
        if house_data['open_date'] and '开盘时间：' in house_data['open_date']:
            house_data['open_date'] = house_data['open_date'].replace('开盘时间：', '').strip()
        cleaned_houses.append(house_data)

    region_counts = []
    try:
        region_counts = db.session.query(
            Hourse_info.region,
            func.count(Hourse_info.id).label('count')
        ).filter(
            Hourse_info.region.isnot(None),
            Hourse_info.region != ''
        ).group_by(Hourse_info.region).order_by(func.count(Hourse_info.id).desc()).limit(5).all()
    except Exception as e:
        logger.error(f"热门区域查询失败: {str(e)}")
    decoration_counts = []
    try:
        decoration_counts = db.session.query(
            Hourse_info.hourseDecoration,
            func.count(Hourse_info.id).label('count')
        ).filter(
            Hourse_info.hourseDecoration.isnot(None),
            Hourse_info.hourseDecoration != ''
        ).group_by(Hourse_info.hourseDecoration).all()
    except Exception as e:
        logger.error(f"装修类型查询失败: {str(e)}")
    price_distribution = []
    try:
        price_ranges = [
            (0, 10000), (10000, 20000), (20000, 30000),
            (30000, 40000), (40000, 50000), (50000, 100000)
        ]
        for min_price, max_price in price_ranges:
            count = Hourse_info.query.filter(
                func.cast(
                    func.replace(Hourse_info.price, '元/㎡', ''),
                    db.Float
                ).between(min_price, max_price)
            ).count()
            price_distribution.append({
                '价格区间': f"{min_price / 10000:.1f}-{max_price / 10000:.1f}万",
                '数量': count
            })
    except Exception as e:
        logger.error(f"价格分布查询失败: {str(e)}")
    return render_template(
        'index.html',
        username=username,
        total_houses=total_houses,
        region_count=region_count,
        avg_price=avg_price,
        recent_houses=cleaned_houses,
        region_counts=region_counts,
        decoration_counts=decoration_counts,
        price_distribution=price_distribution,
        active_page='index'
    )


# 在 data_operation 路由中添加历史记录保存逻辑
@page_bp.route('/data_operation', methods=['GET'])
@login_required
def data_operation():
    username = session.get('username', '游客')
    keyword = request.args.get('keyword', '')
    page = request.args.get('page', 1, type=int)

    # 创建基础查询
    query = Hourse_info.query

    # 添加搜索条件
    if keyword:
        # 支持多字段搜索：标题、城市、区域、地址、户型、装修类型
        query = query.filter(
            or_(
                Hourse_info.title.ilike(f'%{keyword}%'),
                Hourse_info.city.ilike(f'%{keyword}%'),
                Hourse_info.region.ilike(f'%{keyword}%'),
                Hourse_info.address.ilike(f'%{keyword}%'),
                Hourse_info.hourseType.ilike(f'%{keyword}%'),
                Hourse_info.hourseDecoration.ilike(f'%{keyword}%')
            )
        )

        # 保存搜索历史记录
        try:
            user = User.query.filter_by(user_name=username).first()
            if user:
                new_history = History(
                    city=keyword,  # 使用搜索关键词作为城市
                    price="",  # 价格字段设为空
                    user_id=user.user_id
                )
                db.session.add(new_history)
                db.session.commit()
        except Exception as e:
            logger.error(f"保存历史记录失败: {str(e)}")
            db.session.rollback()

    # 添加排序（按ID倒序）
    query = query.order_by(Hourse_info.id.desc())

    # 分页处理
    pagination = query.paginate(page=page, per_page=10)
    houses = pagination.items

    return render_template(
        'data_operation.html',
        username=username,
        houses=houses,
        pagination=pagination,
        keyword=keyword
    )

@page_bp.route('/price_analysis')
@login_required
def price_analysis():
    username = session.get('username', '游客')
    return render_template('price_analysis.html', username=username)


@page_bp.route('/api/price_analysis/region_avg_price')
@login_required
def region_avg_price():
    try:
        houses = Hourse_info.query.all()
        region_prices = defaultdict(list)
        for house in houses:
            if house.region and house.price:
                try:
                    price_value = house.price
                    if '元/㎡' in price_value:
                        price_value = price_value.split('元/㎡')[0].strip()
                    price_float = float(price_value)
                    if price_float > 0:
                        region_prices[house.region].append(price_float)
                except (ValueError, TypeError):
                    continue
        result = []
        for region, prices in region_prices.items():
            if prices:
                avg_price = round(sum(prices) / len(prices), 2)
                result.append({
                    'region': region,
                    'avg_price': avg_price,
                    'house_count': len(prices)
                })
        result.sort(key=lambda x: x['avg_price'], reverse=True)
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        logger.error(f"区域平均价格分析失败: {str(e)}")
        return jsonify({'success': False, 'message': '区域平均价格分析失败'}), 500


@page_bp.route('/api/price_analysis/price_distribution')
@login_required
def price_distribution_api():
    try:
        prices = []
        for house in Hourse_info.query.all():
            if house.price:
                try:
                    price_value = house.price
                    if '元/㎡' in price_value:
                        price_value = price_value.split('元/㎡')[0].strip()
                    price_float = float(price_value)
                    if price_float > 0:
                        prices.append(price_float)
                except (ValueError, TypeError):
                    continue
        if not prices:
            return jsonify({'success': False, 'message': '没有可用的价格数据'}), 404
        min_price = int(min(prices))
        max_price = int(max(prices))
        interval = max(1, (max_price - min_price) // 10)
        bins = list(range(min_price, max_price + interval, interval))
        labels = [f'{bins[i]}-{bins[i + 1]}' for i in range(len(bins) - 1)]
        price_groups = defaultdict(int)
        for price in prices:
            for i in range(len(bins) - 1):
                if bins[i] <= price < bins[i + 1]:
                    price_groups[labels[i]] += 1
                    break
        data = [{'price_range': k, 'count': v} for k, v in price_groups.items()]
        avg_price = round(sum(prices) / len(prices), 2)
        sorted_prices = sorted(prices)
        median_price = sorted_prices[len(sorted_prices) // 2] if len(sorted_prices) % 2 == 1 else (
                                                                                                          sorted_prices[
                                                                                                              len(sorted_prices) // 2 - 1] +
                                                                                                          sorted_prices[
                                                                                                              len(sorted_prices) // 2]
                                                                                                  ) / 2
        return jsonify({
            'success': True,
            'data': data,
            'stats': {
                'min_price': min_price,
                'max_price': max_price,
                'avg_price': avg_price,
                'median_price': round(median_price, 2),
                'total_houses': len(prices)
            }
        })
    except Exception as e:
        logger.error(f"价格分布分析失败: {str(e)}")
        return jsonify({'success': False, 'message': '价格分布分析失败'}), 500


@page_bp.route('/api/price_analysis/decoration_price')
@login_required
def decoration_price():
    try:
        houses = Hourse_info.query.all()
        decoration_prices = defaultdict(list)
        for house in houses:
            if house.hourseDecoration and house.price:
                try:
                    price_value = house.price
                    if '元/㎡' in price_value:
                        price_value = price_value.split('元/㎡')[0].strip()
                    price_float = float(price_value)
                    if price_float > 0:
                        decoration_prices[house.hourseDecoration].append(price_float)
                except (ValueError, TypeError):
                    continue
        result = []
        for decoration, prices in decoration_prices.items():
            if prices:
                avg_price = round(sum(prices) / len(prices), 2)
                result.append({
                    'decoration': decoration,
                    'avg_price': avg_price,
                    'house_count': len(prices)
                })
        result.sort(key=lambda x: x['avg_price'], reverse=True)
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        logger.error(f"装修类型价格分析失败: {str(e)}")
        return jsonify({'success': False, 'message': '装修类型价格分析失败'}), 500


@page_bp.route('/api/price_analysis/area_price')
@login_required
def area_price():
    try:
        houses = Hourse_info.query.all()
        data = []
        for house in houses:
            if house.area_range and house.price:
                try:
                    price_value = house.price
                    if '元/㎡' in price_value:
                        price_value = price_value.split('元/㎡')[0].strip()
                    price_float = float(price_value)
                    area_range_str = house.area_range
                    if '㎡' in area_range_str:
                        area_range_str = area_range_str.split('㎡')[0].strip()
                    try:
                        area_range = json.loads(area_range_str.replace("'", '"'))
                        if isinstance(area_range, list) and len(area_range) >= 2:
                            avg_area = (float(area_range[0]) + float(area_range[1])) / 2
                        else:
                            if '-' in area_range_str:
                                min_area, max_area = area_range_str.split('-')[:2]
                                avg_area = (float(min_area) + float(max_area)) / 2
                            else:
                                avg_area = float(area_range_str)
                    except:
                        avg_area = float(area_range_str)
                    if price_float > 0 and avg_area > 0:
                        data.append({'area': avg_area, 'price': price_float})
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        logger.error(f"面积价格关系分析失败: {str(e)}")
        return jsonify({'success': False, 'message': '面积价格关系分析失败'}), 500


@page_bp.route('/detail_analysis')
@login_required
def detail_analysis():
    username = session.get('username', '游客')
    return render_template('detail_analysis.html', username=username)


@page_bp.route('/api/detail_analysis/region_distribution')
@login_required
def region_distribution():
    try:
        houses = Hourse_info.query.all()
        region_counts = defaultdict(int)
        for house in houses:
            if house.region:
                region_counts[house.region] += 1
        data = [{'name': region, 'value': count} for region, count in region_counts.items()]
        data.sort(key=lambda x: x['value'], reverse=True)
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        logger.error(f"区域分布分析失败: {str(e)}")
        return jsonify({'success': False, 'message': '区域分布分析失败'}), 500


@page_bp.route('/api/detail_analysis/hourse_type_distribution')
@login_required
def hourse_type_distribution():
    try:
        houses = Hourse_info.query.all()
        type_counts = defaultdict(int)
        for house in houses:
            if house.hourseType:
                types = house.hourseType.split(',')
                if types:
                    main_type = types[0].strip()
                    type_counts[main_type] += 1
        data = [{'name': hourse_type, 'value': count} for hourse_type, count in type_counts.items()]
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        logger.error(f"户型分布分析失败: {str(e)}")
        return jsonify({'success': False, 'message': '户型分布分析失败'}), 500


@page_bp.route('/api/detail_analysis/decoration_distribution')
@login_required
def decoration_distribution():
    try:
        houses = Hourse_info.query.all()
        decoration_counts = defaultdict(int)
        for house in houses:
            if house.hourseDecoration:
                decoration_counts[house.hourseDecoration] += 1
        data = [{'name': decoration, 'value': count} for decoration, count in decoration_counts.items()]
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        logger.error(f"装修类型分布分析失败: {str(e)}")
        return jsonify({'success': False, 'message': '装修类型分布分析失败'}), 500


@page_bp.route('/api/detail_analysis/price_region')
@login_required
def price_region():
    try:
        houses = Hourse_info.query.all()
        region_prices = defaultdict(list)
        for house in houses:
            house.clean_data(for_analysis=True)
            if house.region and house.price:
                try:
                    price_float = float(house.price) if house.price else None
                    if price_float and price_float > 0:
                        region_prices[house.region].append(price_float)
                except (ValueError, TypeError):
                    continue
        result = []
        for region, prices in region_prices.items():
            if prices:
                prices_sorted = sorted(prices)
                n = len(prices_sorted)
                q1 = prices_sorted[int(n * 0.25)]
                median = prices_sorted[int(n * 0.5)]
                q3 = prices_sorted[int(n * 0.75)]
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = [p for p in prices if p < lower_bound or p > upper_bound]
                min_val = max(lower_bound, min(prices))
                max_val = min(upper_bound, max(prices))
                result.append({
                    'region': region,
                    'min': min_val,
                    'q1': q1,
                    'median': median,
                    'q3': q3,
                    'max': max_val,
                    'outliers': outliers,
                    'count': n
                })
        result.sort(key=lambda x: x['median'], reverse=True)
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        logger.error(f"价格与区域关系分析失败: {str(e)}")
        return jsonify({'success': False, 'message': '价格与区域关系分析失败'}), 500


@page_bp.route('/api/detail_analysis/sale_status_distribution')
@login_required
def sale_status_distribution():
    try:
        houses = Hourse_info.query.all()
        status_counts = defaultdict(int)
        status_mapping = {
            "1": "在售",
            "2": "已售",
            "3": "待售",
            "4": "停售",
            1: "在售",
            2: "已售",
            3: "待售",
            4: "停售"
        }
        for house in houses:
            if house.sale_status:
                status_value = str(house.sale_status).strip()
                status = status_mapping.get(status_value, status_value)
                if status == status_value:
                    try:
                        int_value = int(status_value)
                        status = status_mapping.get(int_value, status_value)
                    except ValueError:
                        pass
                status_counts[status] += 1
        data = [{'name': status, 'value': count} for status, count in status_counts.items()]
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        logger.error(f"销售状态分布分析失败: {str(e)}")
        return jsonify({'success': False, 'message': '销售状态分布分析失败'}), 500


@page_bp.route('/type_analysis')
@login_required
def type_analysis():
    username = session.get('username', '游客')
    return render_template('type_analysis.html', username=username)


@page_bp.route('/api/type_analysis/type_stats', methods=['GET'])
def get_type_stats():
    try:
        # 查询每种房屋类型的价格最小值和最大值
        query = db.session.query(
            Hourse_info.hourseType,
            func.count(Hourse_info.id).label('count'),
            func.avg(Hourse_info.price).label('avg_price'),
            func.min(Hourse_info.price).label('min_price'),
            func.max(Hourse_info.price).label('max_price')
        ).group_by(Hourse_info.hourseType)
        results = query.all()

        data = []
        for result in results:
            # 清理价格数据（移除单位并转换为数字）
            try:
                min_price = float(result.min_price.replace('元/㎡', '')) if result.min_price else 0
                max_price = float(result.max_price.replace('元/㎡', '')) if result.max_price else 0
                price_range = f"{min_price:.2f}-{max_price:.2f} 元/㎡"
            except (ValueError, TypeError):
                price_range = "--"

            data.append({
                'type': result.hourseType,
                'count': result.count,
                'avg_price': round(result.avg_price, 2),
                'price_range': price_range
            })

        return jsonify({'success': True, 'data': data, 'message': '数据获取成功'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'获取数据失败: {str(e)}'}), 500

@page_bp.route('/api/houses', methods=['GET'])
@login_required
def get_houses():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    keyword = request.args.get('keyword', '')
    query = Hourse_info.query
    if keyword:
        query = query.filter(
            or_(
                Hourse_info.title.like(f'%{keyword}%'),
                Hourse_info.city.like(f'%{keyword}%'),
                Hourse_info.region.like(f'%{keyword}%')
            )
        )
    pagination = query.paginate(page=page, per_page=per_page)
    houses = [{
        'id': house.id,
        'title': house.title,
        'city': house.city,
        'region': house.region,
        'address': house.address,
        'price': house.price,
        'hourseType': house.hourseType,
        'area_range': house.area_range,
        'hourseDecoration': house.hourseDecoration
    } for house in pagination.items]
    return jsonify({
        'items': houses,
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': pagination.page
    })


@page_bp.route('/api/houses/<int:house_id>', methods=['GET'])
@login_required
def get_house(house_id):
    house = Hourse_info.query.get_or_404(house_id)
    return jsonify({
        'id': house.id,
        'title': house.title,
        'city': house.city,
        'region': house.region,
        'address': house.address,
        'price': house.price,
        'hourseType': house.hourseType,
        'area_range': house.area_range,
        'hourseDecoration': house.hourseDecoration
    })


@page_bp.route('/api/houses', methods=['POST'])
@login_required
def add_house():
    data = request.get_json()
    try:
        # 为所有可选字段提供空字符串默认值
        new_house = Hourse_info(
            title=data.get('title', ''),
            city=data.get('city', ''),
            region=data.get('region', ''),
            address=data.get('address', ''),
            price=data.get('price', ''),
            hourseType=data.get('hourseType', ''),
            area_range=data.get('area_range', ''),
            hourseDecoration=data.get('hourseDecoration', ''),
            cover=data.get('cover', ''),
            room_desc=data.get('room_desc', ''),
            all_ready=data.get('all_ready', ''),
            company=data.get('company', ''),
            on_time=data.get('on_time', ''),
            open_date=data.get('open_date', ''),
            tags=data.get('tags', ''),
            totalPrice_range=data.get('totalPrice_range', ''),
            sale_status=data.get('sale_status', ''),
            detail_url=data.get('detail_url', '')
        )
        db.session.add(new_house)
        db.session.commit()
        return jsonify({'id': new_house.id}), 201
    except Exception as e:
        db.session.rollback()
        logger.error(f"添加房屋失败: {str(e)}")
        return jsonify({'error': '添加失败'}), 500


@page_bp.route('/api/houses/<int:house_id>', methods=['PUT'])
@login_required
def update_house(house_id):
    try:
        house = Hourse_info.query.get_or_404(house_id)
        data = request.get_json()

        # 添加封面图更新（允许为空）
        if 'cover' in data:
            house.cover = data['cover']  # 可以是None

        # 其他更新字段...
        update_fields = ['title', 'city', 'region', 'address', 'price',
                         'hourseType', 'area_range', 'hourseDecoration']
        for key in update_fields:
            if key in data:
                setattr(house, key, data[key])

        db.session.commit()
        return jsonify({'message': '更新成功'})
    except Exception as e:
        db.session.rollback()
        logger.error(f"更新房屋失败: {str(e)}")
        return jsonify({'error': '更新失败'}), 500


@page_bp.route('/api/houses/<int:house_id>', methods=['DELETE'])
@login_required
def delete_house(house_id):
    try:
        house = Hourse_info.query.get_or_404(house_id)
        db.session.delete(house)
        db.session.commit()
        return jsonify({'message': '删除成功'})
    except Exception as e:
        db.session.rollback()
        logger.error(f"删除房屋失败: {str(e)}")
        return jsonify({'error': '删除失败'}), 500


@page_bp.route('/other_analysis')
@login_required
def other_analysis():
    username = session.get('username', '游客')
    return render_template('other_analysis.html', username=username)


def parse_open_date(date_str):
    if not date_str:
        return None

    patterns = [
        r'(\d{4})年(\d{1,2})月',
        r'(\d{4})-(\d{1,2})',
        r'(\d{4})\.(\d{1,2})',
        r'(\d{4})/(\d{1,2})'
    ]

    for pattern in patterns:
        match = re.search(pattern, date_str)
        if match:
            year = match.group(1)
            month = match.group(2).zfill(2)
            return f"{year}-{month}"

    return None


@page_bp.route('/api/other_analysis/open_time_trend')
@login_required
def open_time_trend():
    try:
        houses = Hourse_info.query.filter(Hourse_info.open_date.isnot(None)).all()
        monthly_counts = defaultdict(int)

        # 解析日期格式
        for house in houses:
            if not house.open_date:
                continue

            date_str = house.open_date
            # 处理多种日期格式
            if '年' in date_str and '月' in date_str:
                year = date_str.split('年')[0]
                month = date_str.split('年')[1].split('月')[0]
                date_key = f"{year}-{month.zfill(2)}"
                monthly_counts[date_key] += 1
            elif '-' in date_str:
                parts = date_str.split('-')
                if len(parts) >= 2:
                    year = parts[0]
                    month = parts[1][:2]  # 取前两位作为月份
                    date_key = f"{year}-{month.zfill(2)}"
                    monthly_counts[date_key] += 1

        # 如果没有数据，返回示例数据
        if not monthly_counts:
            return jsonify({
                'success': True,
                'data': [
                    {'date': '2023-01', 'count': 15},
                    {'date': '2023-02', 'count': 22},
                    {'date': '2023-03', 'count': 18},
                    {'date': '2023-04', 'count': 25}
                ]
            })

        # 转换为前端需要的格式
        data = [{'date': date, 'count': count} for date, count in monthly_counts.items()]
        data.sort(key=lambda x: x['date'])
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        logger.error(f"开盘时间趋势分析失败: {str(e)}")
        return jsonify({'success': False, 'message': '开盘时间趋势分析失败'}), 500


@page_bp.route('/api/other_analysis/tag_cloud')
@login_required
def tag_cloud():
    try:
        houses = Hourse_info.query.filter(Hourse_info.tags.isnot(None)).all()
        tag_counts = defaultdict(int)

        # 处理标签数据
        for house in houses:
            if house.tags:
                # 清理标签并分割
                tags = house.tags.replace(' ', '').replace('，', ',')
                tags = tags.split(',')

                for tag in tags:
                    if tag:  # 跳过空标签
                        tag_counts[tag] += 1

        # 如果没有数据，返回示例数据
        if not tag_counts:
            return jsonify({
                'success': True,
                'data': [
                    {'name': '学区房', 'value': 42},
                    {'name': '地铁房', 'value': 38},
                    {'name': '精装修', 'value': 35},
                    {'name': '南北通透', 'value': 28},
                    {'name': '拎包入住', 'value': 25}
                ]
            })

        # 转换为前端需要的格式
        data = [{'name': tag, 'value': count} for tag, count in tag_counts.items()]
        data.sort(key=lambda x: x['value'], reverse=True)
        return jsonify({'success': True, 'data': data[:20]})  # 最多返回20个标签
    except Exception as e:
        logger.error(f"标签词云分析失败: {str(e)}")
        return jsonify({'success': False, 'message': '标签词云分析失败'}), 500


def parse_price_range(price_str):
    if not price_str:
        return None

    price_str = price_str.replace('万', '').replace('元', '').strip()

    if '-' in price_str:
        parts = price_str.split('-')
        if len(parts) == 2:
            try:
                min_price = float(parts[0])
                max_price = float(parts[1])
                return (min_price, max_price)
            except (ValueError, TypeError):
                return None

    try:
        price = float(price_str)
        return (price, price)
    except (ValueError, TypeError):
        return None


@page_bp.route('/api/other_analysis/total_price_distribution')
@login_required
def total_price_distribution():
    try:
        houses = Hourse_info.query.filter(Hourse_info.totalPrice_range.isnot(None)).all()
        price_buckets = defaultdict(int)

        buckets = [
            (0, 100), (100, 200), (200, 300), (300, 400),
            (400, 500), (500, 600), (600, 700), (700, 800),
            (800, 900), (900, 1000), (1000, 1500), (1500, float('inf'))
        ]

        for house in houses:
            price_range = parse_price_range(house.totalPrice_range)
            if not price_range:
                continue

            min_price, max_price = price_range
            avg_price = (min_price + max_price) / 2

            for bucket_min, bucket_max in buckets:
                if bucket_min <= avg_price < bucket_max:
                    if bucket_max == float('inf'):
                        bucket_label = f"{bucket_min}万以上"
                    else:
                        bucket_label = f"{bucket_min}-{bucket_max}万"
                    price_buckets[bucket_label] += 1
                    break

        if not price_buckets:
            return jsonify({
                'success': True,
                'data': [
                    {'price_range': '0-100万', 'count': 35},
                    {'price_range': '100-200万', 'count': 42},
                    {'price_range': '200-300万', 'count': 28},
                    {'price_range': '300-400万', 'count': 18}
                ]
            })

        data = []
        for bucket_min, bucket_max in buckets:
            if bucket_max == float('inf'):
                bucket_label = f"{bucket_min}万以上"
            else:
                bucket_label = f"{bucket_min}-{bucket_max}万"

            count = price_buckets.get(bucket_label, 0)
            data.append({'price_range': bucket_label, 'count': count})

        return jsonify({'success': True, 'data': data})
    except Exception as e:
        logger.error(f"总价分布分析失败: {str(e)}")
        return jsonify({'success': False, 'message': '总价分布分析失败'}), 500


@page_bp.route('/api/other_analysis/status_decoration_relation')
@login_required
def status_decoration_relation():
    try:
        houses = Hourse_info.query.filter(
            Hourse_info.sale_status.isnot(None),
            Hourse_info.hourseDecoration.isnot(None)
        ).all()

        status_mapping = {
            "1": "在售",
            "2": "已售",
            "3": "待售",
            "4": "停售",
            "在售": "在售",
            "已售": "已售",
            "待售": "待售",
            "停售": "停售",
            "sale": "在售",
            "sold": "已售",
            "pending": "待售",
            "stop": "停售"
        }

        status_list = set()
        decoration_list = set()
        relation_data = defaultdict(lambda: defaultdict(int))

        for house in houses:
            status = str(house.sale_status).strip()
            status = status_mapping.get(status, status)

            decoration = str(house.hourseDecoration).strip()

            if not status or not decoration:
                continue

            status_list.add(status)
            decoration_list.add(decoration)
            relation_data[status][decoration] += 1

        if not relation_data:
            return jsonify({
                'success': True,
                'data': [
                    {"status": "在售", "decoration": "精装修", "value": 15},
                    {"status": "在售", "decoration": "简装修", "value": 8},
                    {"status": "已售", "decoration": "精装修", "value": 20},
                    {"status": "待售", "decoration": "毛坯", "value": 5}
                ],
                'status_list': ["在售", "已售", "待售"],
                'decoration_list': ["精装修", "简装修", "毛坯"]
            })

        data = []
        for status in status_list:
            for decoration in decoration_list:
                count = relation_data[status].get(decoration, 0)
                if count > 0:
                    data.append({
                        "status": status,
                        "decoration": decoration,
                        "value": count
                    })

        return jsonify({
            'success': True,
            'data': data,
            'status_list': sorted(list(status_list)),
            'decoration_list': sorted(list(decoration_list))
        })
    except Exception as e:
        logger.error(f"销售状态与装修类型关系分析失败: {str(e)}")
        return jsonify({'success': False, 'message': '销售状态与装修类型关系分析失败'}), 500


@page_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    username = session.get('username')
    user = User.query.filter_by(user_name=username).first_or_404()

    if request.method == 'POST':
        new_username = request.form.get('username', '').strip()
        new_password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()

        # 验证用户名
        if new_username and new_username != username:
            if User.query.filter_by(user_name=new_username).first():
                flash('用户名已存在', 'danger')
                return redirect(url_for('page.profile'))
            user.user_name = new_username

        # 验证密码
        if new_password:
            if new_password != confirm_password:
                flash('两次输入的密码不一致', 'danger')
                return redirect(url_for('page.profile'))
            # 直接存储明文密码（最简单实现）
            user.user_password = new_password

        try:
            db.session.commit()
            session['username'] = new_username or username
            flash('资料更新成功', 'success')
        except Exception as e:
            db.session.rollback()
            logger.error(f"更新用户资料失败: {str(e)}")
            flash('更新失败，请重试', 'danger')
        return redirect(url_for('page.profile'))

    return render_template('profile.html', user=user)


@page_bp.route('/history')
@login_required
def history():
    username = session.get('username')
    user = User.query.filter_by(user_name=username).first_or_404()

    histories = History.query.filter_by(user_id=user.user_id).all()

    return render_template('history.html', username=username, histories=histories)


@page_bp.route('/api/history/add', methods=['POST'])
@login_required
def add_history():
    try:
        username = session.get('username')
        user = User.query.filter_by(user_name=username).first_or_404()

        data = request.get_json()
        city = data.get('city', '')
        price = data.get('price', '')

        if not city or not price:
            return jsonify({'success': False, 'message': '参数不完整'}), 400

        new_history = History(
            city=city,
            price=price,
            user_id=user.user_id
        )
        db.session.add(new_history)
        db.session.commit()

        return jsonify({'success': True, 'message': '历史记录添加成功'})
    except Exception as e:
        db.session.rollback()
        logger.error(f"添加历史记录失败: {str(e)}")
        return jsonify({'success': False, 'message': '添加失败'}), 500


@page_bp.route('/api/history/clear', methods=['DELETE'])
@login_required
def clear_history():
    try:
        username = session.get('username')
        user = User.query.filter_by(user_name=username).first_or_404()

        # 删除当前用户的所有历史记录
        History.query.filter_by(user_id=user.user_id).delete()
        db.session.commit()

        return jsonify({'success': True, 'message': '历史记录已清空'})
    except Exception as e:
        db.session.rollback()
        logger.error(f"清空历史记录失败: {str(e)}")
        return jsonify({'success': False, 'message': '清空失败'}), 500