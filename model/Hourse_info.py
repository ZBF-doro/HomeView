from extensions import db
import json

class Hourse_info(db.Model):
    __tablename__ = 'hourse_info'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255))
    cover = db.Column(db.String(255))
    city = db.Column(db.String(255))
    region = db.Column(db.String(255))
    address = db.Column(db.String(255))
    room_desc = db.Column(db.String(255))
    area_range = db.Column(db.String(255))
    all_ready = db.Column(db.String(255))
    price = db.Column(db.String(255))
    hourseDecoration = db.Column(db.String(255))
    company = db.Column(db.String(255))
    hourseType = db.Column(db.String(255))
    on_time = db.Column(db.String(255))
    open_date = db.Column(db.String(255))
    tags = db.Column(db.String(255))
    totalPrice_range = db.Column(db.String(255))
    sale_status = db.Column(db.String(255))
    detail_url = db.Column(db.String(255))

    def clean_data(self, for_analysis=False):
        """清理数据用于显示或分析"""
        # 清理价格数据
        if self.price:
            try:
                # 移除单位并转换为浮点数
                cleaned_price = self.price.replace('元/㎡', '').strip()
                price_float = float(cleaned_price)

                # 检查有效价格
                if price_float <= 0:
                    if for_analysis:
                        self.price = None
                else:
                    self.price = str(price_float)
            except (ValueError, TypeError):
                # 转换失败，保留原始值
                if for_analysis:
                    self.price = None

        # 清理面积数据
        if self.area_range:
            try:
                # 移除单位
                cleaned_area = self.area_range.replace('㎡', '').strip()

                # 尝试解析为范围或单个值
                if '-' in cleaned_area:
                    min_area, max_area = cleaned_area.split('-', 1)
                    min_area = float(min_area.strip())
                    max_area = float(max_area.strip())

                    # 检查有效面积
                    if min_area <= 0 or max_area <= 0:
                        if for_analysis:
                            self.area_range = None
                    else:
                        self.area_range = f"{min_area}-{max_area}"
                else:
                    area_val = float(cleaned_area)
                    if area_val <= 0:
                        if for_analysis:
                            self.area_range = None
                    else:
                        self.area_range = str(area_val)
            except (ValueError, TypeError):
                if for_analysis:
                    self.area_range = None

        # 清理总价数据
        if self.totalPrice_range and '万' in self.totalPrice_range:
            self.totalPrice_range = self.totalPrice_range.split('万')[0].strip()

        # 清理开盘时间
        if self.open_date and '开盘时间：' in self.open_date:
            self.open_date = self.open_date.replace('开盘时间：', '').strip()

        # 清理标签数据
        if self.tags:
            self.tags = self.tags.replace(' ', '').replace(',,', ',').strip(',')