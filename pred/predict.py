import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib
import optuna
from optuna.samplers import TPESampler
import shap
import matplotlib.pyplot as plt


class AdvancedHousePricePredictor:
    def __init__(self, data_path=None, db_connection=None):
        """
        高级房价预测模型

        参数:
        data_path (str): 数据文件路径 (CSV格式)
        db_connection: 数据库连接对象 (SQLAlchemy)
        """
        self.data_path = data_path
        self.db_connection = db_connection
        self.models = {}
        self.feature_importance = {}
        self.scaler = StandardScaler()
        self.preprocessor = None
        self.best_model = None
        self.stacked_model = None
        self.explainer = None

    def load_data(self):
        """从CSV文件或数据库加载数据"""
        if self.data_path:
            df = pd.read_csv(self.data_path)
        elif self.db_connection:
            query = "SELECT * FROM hourse_info"
            df = pd.read_sql(query, self.db_connection)
        else:
            raise ValueError("必须提供数据路径或数据库连接")

        # 数据清洗
        df = self._clean_data(df)
        return df

    def _clean_data(self, df):
        """数据清洗和预处理"""
        # 价格处理
        df['price'] = df['price'].str.replace('元/㎡', '').astype(float)

        # 面积处理 - 取平均值
        def process_area(area_str):
            if pd.isna(area_str):
                return np.nan
            try:
                # 尝试解析为范围
                if '[' in area_str and ']' in area_str:
                    area_list = eval(area_str)
                    return sum(area_list) / len(area_list)
                elif '-' in area_str:
                    parts = area_str.split('-')
                    return (float(parts[0]) + float(parts[1])) / 2
                else:
                    return float(area_str)
            except:
                return np.nan

        df['area'] = df['area_range'].apply(process_area)

        # 总价处理
        df['total_price'] = df['totalPrice_range'].str.replace('万', '').astype(float)

        # 处理开盘时间
        df['open_date'] = pd.to_datetime(
            df['open_date'].str.replace('开盘时间：', ''),
            errors='coerce'
        )
        df['open_year'] = df['open_date'].dt.year
        df['open_month'] = df['open_date'].dt.month

        # 标签处理
        df['tags'] = df['tags'].str.split(',')

        # 特征工程
        df['region_level'] = df['region'].apply(self._region_to_level)
        df['decoration_level'] = df['hourseDecoration'].apply(self._decoration_to_level)

        # 选择特征和目标变量
        features = df[[
            'region', 'hourseDecoration', 'hourseType', 'area',
            'total_price', 'open_year', 'open_month', 'region_level',
            'decoration_level', 'sale_status'
        ]]

        # 添加标签特征 - 热门标签计数
        popular_tags = ['地铁房', '学区房', '公园房', '商圈房', '江景房', '湖景房', '精装修', '现房']
        for tag in popular_tags:
            df[f'tag_{tag}'] = df['tags'].apply(lambda x: 1 if tag in x else 0)
            features[f'tag_{tag}'] = df[f'tag_{tag}']

        target = df['price']

        return features, target

    def _region_to_level(self, region):
        """区域分级 (基于经济水平)"""
        premium_regions = ['市中心', '金融区', '高新区']
        standard_regions = ['商业区', '住宅区', '开发区']

        if region in premium_regions:
            return 3
        elif region in standard_regions:
            return 2
        else:
            return 1

    def _decoration_to_level(self, decoration):
        """装修分级"""
        if '豪装' in decoration:
            return 4
        elif '精装' in decoration:
            return 3
        elif '简装' in decoration:
            return 2
        elif '毛坯' in decoration:
            return 1
        else:
            return 2  # 默认中等装修

    def create_preprocessor(self, categorical_features, numerical_features):
        """创建数据预处理管道"""
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features),
                ('num', numerical_transformer, numerical_features)
            ])
        return self.preprocessor

    def train_base_models(self, X, y):
        """训练多个基础模型"""
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 定义特征类型
        categorical_features = ['region', 'hourseDecoration', 'hourseType', 'sale_status']
        numerical_features = ['area', 'total_price', 'open_year', 'open_month',
                              'region_level', 'decoration_level'] + [c for c in X.columns if 'tag_' in c]

        # 创建预处理器
        preprocessor = self.create_preprocessor(categorical_features, numerical_features)

        # 定义模型
        models = {
            'xgb': xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'lgbm': lgb.LGBMRegressor(
                objective='regression',
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=7,
                num_leaves=50,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'catboost': CatBoostRegressor(
                iterations=1000,
                learning_rate=0.05,
                depth=6,
                loss_function='RMSE',
                verbose=0,
                random_state=42
            ),
            'rf': RandomForestRegressor(
                n_estimators=500,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gbr': GradientBoostingRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=5,
                random_state=42
            ),
            'svr': SVR(kernel='rbf', C=1.0, epsilon=0.1),
            'mlp': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=42
            )
        }

        # 训练模型并评估
        results = {}
        for name, model in models.items():
            # 创建完整管道
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', model)
            ])

            # 训练模型
            pipeline.fit(X_train, y_train)

            # 评估模型
            y_pred = pipeline.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            # 存储结果和模型
            results[name] = {'rmse': rmse, 'r2': r2}
            self.models[name] = pipeline

            print(f"{name.upper()} - RMSE: {rmse:.2f}, R2: {r2:.4f}")

        return results

    def hyperparameter_tuning(self, X, y, model_name='xgb'):
        """使用Optuna进行超参数优化"""
        # 划分数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 定义特征类型
        categorical_features = ['region', 'hourseDecoration', 'hourseType', 'sale_status']
        numerical_features = ['area', 'total_price', 'open_year', 'open_month',
                              'region_level', 'decoration_level'] + [c for c in X.columns if 'tag_' in c]

        # 创建预处理器
        preprocessor = self.create_preprocessor(categorical_features, numerical_features)

        # 定义目标函数
        def objective(trial):
            if model_name == 'xgb':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 1),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'random_state': 42
                }
                model = xgb.XGBRegressor(**params)
            elif model_name == 'lgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'random_state': 42
                }
                model = lgb.LGBMRegressor(**params)
            elif model_name == 'catboost':
                params = {
                    'iterations': trial.suggest_int('iterations', 500, 2000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                    'random_strength': trial.suggest_float('random_strength', 0, 1),
                    'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                    'random_state': 42
                }
                model = CatBoostRegressor(**params, verbose=0)
            else:
                raise ValueError(f"不支持的模型类型: {model_name}")

            # 创建完整管道
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', model)
            ])

            # 使用交叉验证
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = []

            for train_idx, val_idx in kf.split(X_train):
                X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

                pipeline.fit(X_train_fold, y_train_fold)
                y_pred = pipeline.predict(X_val_fold)
                score = np.sqrt(mean_squared_error(y_val_fold, y_pred))
                scores.append(score)

            return np.mean(scores)

        # 创建Optuna研究
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=50)

        # 使用最佳参数训练最终模型
        best_params = study.best_params
        if model_name == 'xgb':
            model = xgb.XGBRegressor(**best_params, random_state=42)
        elif model_name == 'lgbm':
            model = lgb.LGBMRegressor(**best_params, random_state=42)
        elif model_name == 'catboost':
            model = CatBoostRegressor(**best_params, random_state=42, verbose=0)

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        # 在整个训练集上训练
        pipeline.fit(X_train, y_train)

        # 评估测试集
        y_pred = pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"优化后的{model_name.upper()} - RMSE: {rmse:.2f}, R2: {r2:.4f}")

        # 保存最佳模型
        self.best_model = pipeline
        self.models[f'{model_name}_tuned'] = pipeline

        return best_params, rmse, r2

    def train_stacked_model(self, X, y):
        """训练堆叠模型 (元学习器)"""
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 使用基础模型生成预测作为新特征
        base_predictions_train = pd.DataFrame()
        base_predictions_test = pd.DataFrame()

        for name, model in self.models.items():
            if name.endswith('_tuned'):  # 只使用调优后的模型
                base_predictions_train[name] = model.predict(X_train)
                base_predictions_test[name] = model.predict(X_test)

        # 训练元学习器
        meta_learner = ElasticNet(
            alpha=0.001,
            l1_ratio=0.7,
            max_iter=10000,
            random_state=42
        )

        meta_learner.fit(base_predictions_train, y_train)

        # 评估堆叠模型
        y_pred = meta_learner.predict(base_predictions_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"堆叠模型 - RMSE: {rmse:.2f}, R2: {r2:.4f}")

        self.stacked_model = meta_learner
        return rmse, r2

    def train_ensemble(self, X, y):
        """训练完整模型集合"""
        # 训练基础模型
        print("训练基础模型...")
        self.train_base_models(X, y)

        # 对最佳模型进行超参数调优
        print("\n超参数优化...")
        self.hyperparameter_tuning(X, y, 'xgb')
        self.hyperparameter_tuning(X, y, 'lgbm')
        self.hyperparameter_tuning(X, y, 'catboost')

        # 训练堆叠模型
        print("\n训练堆叠模型...")
        self.train_stacked_model(X, y)

        # 选择最佳单一模型
        best_model_name = None
        best_rmse = float('inf')
        for name, model in self.models.items():
            if name.endswith('_tuned'):
                y_pred = model.predict(X)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model_name = name

        self.best_model = self.models[best_model_name]
        print(f"\n最佳单一模型: {best_model_name}, RMSE: {best_rmse:.2f}")

    def predict(self, X, use_ensemble=False):
        """预测房价"""
        if use_ensemble and self.stacked_model:
            # 使用堆叠模型进行预测
            base_predictions = pd.DataFrame()
            for name, model in self.models.items():
                if name.endswith('_tuned'):
                    base_predictions[name] = model.predict(X)
            return self.stacked_model.predict(base_predictions)
        elif self.best_model:
            # 使用最佳单一模型进行预测
            return self.best_model.predict(X)
        else:
            raise RuntimeError("模型尚未训练")

    def explain_prediction(self, X_sample):
        """解释模型预测 (使用SHAP)"""
        if not self.best_model:
            raise RuntimeError("模型尚未训练")

        # 创建SHAP解释器
        if self.explainer is None:
            # 提取预处理后的特征名称
            preprocessor = self.best_model.named_steps['preprocessor']
            X_processed = preprocessor.transform(X_sample)

            # 获取特征名称
            cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
            num_features = preprocessor.named_transformers_['num'].feature_names_in_
            all_features = list(cat_features) + list(num_features)

            # 创建解释器
            model = self.best_model.named_steps['regressor']
            self.explainer = shap.TreeExplainer(model)
            self.feature_names = all_features

        # 解释预测
        preprocessor = self.best_model.named_steps['preprocessor']
        X_processed = preprocessor.transform(X_sample)
        shap_values = self.explainer.shap_values(X_processed)

        # 可视化解释
        plt.figure()
        shap.summary_plot(shap_values, X_processed, feature_names=self.feature_names, show=False)
        plt.tight_layout()
        plt.savefig('shap_summary.png')

        return shap_values

    def save_model(self, path='house_price_model.pkl'):
        """保存整个预测器"""
        joblib.dump(self, path)

    @staticmethod
    def load_model(path='house_price_model.pkl'):
        """加载预测器"""
        return joblib.load(path)

    def evaluate_model(self, X, y):
        """评估模型性能"""
        if not self.best_model:
            raise RuntimeError("模型尚未训练")

        y_pred = self.predict(X)

        metrics = {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'explained_variance': explained_variance_score(y, y_pred)
        }

        # 残差分析
        residuals = y - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('预测值')
        plt.ylabel('残差')
        plt.title('残差分析')
        plt.savefig('residual_analysis.png')

        return metrics


# 使用示例
if __name__ == "__main__":
    # 初始化预测器 (使用数据库连接)
    predictor = AdvancedHousePricePredictor(db_connection=db_engine)

    # 加载数据
    features, target = predictor.load_data()

    # 训练完整模型集合
    predictor.train_ensemble(features, target)

    # 保存模型
    predictor.save_model('advanced_house_price_model.pkl')

    # 预测示例
    sample = pd.DataFrame([{
        'region': '高新区',
        'hourseDecoration': '精装',
        'hourseType': '住宅',
        'area': 120,
        'total_price': 360,
        'open_year': 2023,
        'open_month': 6,
        'region_level': 3,
        'decoration_level': 3,
        'sale_status': '在售',
        'tag_地铁房': 1,
        'tag_学区房': 1,
        'tag_公园房': 0,
        'tag_商圈房': 1,
        'tag_江景房': 0,
        'tag_湖景房': 0,
        'tag_精装修': 1,
        'tag_现房': 1
    }])

    price_prediction = predictor.predict(sample)
    print(f"预测房价: {price_prediction[0]:.2f} 元/平方米")

    # 解释预测
    shap_values = predictor.explain_prediction(sample)