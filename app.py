# app.py 修复版，不需要联网下载数据集
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 页面基础配置
st.set_page_config(page_title="房价预测", layout="wide")
st.title("🏠 交互式房价预测模型")
st.markdown("数据科学小项目：线性回归 / 随机森林，输入参数直接预测房价")

# ========== 生成模拟房价数据集，不用联网！ ==========
@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 20000
    MedInc = np.random.uniform(0,15,n_samples)
    HouseAge = np.random.randint(1,53,n_samples)
    AveRooms = np.random.uniform(1,15,n_samples)
    AveBedrms = np.random.uniform(0.5,5,n_samples)
    Population = np.random.randint(3,35000,n_samples)
    AveOccup = np.random.uniform(0.6,13,n_samples)
    Latitude = np.random.uniform(32.5,42,n_samples)
    Longitude = np.random.uniform(-124.3,-114.3,n_samples)

    # 构造目标房价，和收入、经纬度强相关，贴近真实逻辑
    price = MedInc * 0.4 + (40 - Latitude)*0.2 + (-Longitude-114)*0.15 + np.random.normal(0,0.4,size=n_samples)
    price = np.clip(price, 0, 5)

    df = pd.DataFrame({
        "MedInc":MedInc,
        "HouseAge":HouseAge,
        "AveRooms":AveRooms,
        "AveBedrms":AveBedrms,
        "Population":Population,
        "AveOccup":AveOccup,
        "Latitude":Latitude,
        "Longitude":Longitude,
        "MedHouseVal":price
    })
    return df

df_raw = load_data()
X = df_raw.drop("MedHouseVal", axis=1)
y = df_raw["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型，缓存，只训练一次
@st.cache_resource
def train_models():
    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    def get_metrics(y_true, y_pred):
        return {
            "MAE": round(mean_absolute_error(y_true, y_pred), 3),
            "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)),3),
            "R²": round(r2_score(y_true, y_pred),3)
        }
    return lr, rf, get_metrics(y_test, y_pred_lr), get_metrics(y_test, y_pred_rf)

lr_model, rf_model, metric_lr, metric_rf = train_models()

# ========== 侧边栏：输入参数区 ==========
st.sidebar.header("📝 输入房屋特征")
st.sidebar.caption("预测结果单位：10万美元")
MedInc = st.sidebar.slider("街区收入中位数 MedInc", min_value=0.0, max_value=15.0, value=3.87)
HouseAge = st.sidebar.slider("房龄 HouseAge", min_value=1, max_value=52, value=29)
AveRooms = st.sidebar.slider("平均房间数 AveRooms", min_value=1.0, max_value=15.0, value=5.4)
AveBedrms = st.sidebar.slider("平均卧室数 AveBedrms", min_value=0.5, max_value=5.0, value=1.1)
Population = st.sidebar.slider("区域人口 Population", min_value=3, max_value=35682, value=1166)
AveOccup = st.sidebar.slider("每户人数 AveOccup", min_value=0.6, max_value=13.0, value=2.8)
Latitude = st.sidebar.slider("纬度 Latitude", min_value=32.5, max_value=42.0, value=35.6)
Longitude = st.sidebar.slider("经度 Longitude", min_value=-124.3, max_value=-114.3, value=-119.5)

input_data = pd.DataFrame({
    "MedInc":[MedInc],
    "HouseAge":[HouseAge],
    "AveRooms":[AveRooms],
    "AveBedrms":[AveBedrms],
    "Population":[Population],
    "AveOccup":[AveOccup],
    "Latitude":[Latitude],
    "Longitude":[Longitude]
})

# 多标签页
tab1, tab2, tab3 = st.tabs(["📊数据集与EDA", "⚙️模型对比指标", "🔮预测面板"])

# Tab1 数据预览+可视化
with tab1:
    st.subheader("数据集预览")
    st.write(f"共{df_raw.shape[0]}条样本，{df_raw.shape[1]}列")
    st.dataframe(df_raw.head(6))
    st.subheader("特征相关性热力图")
    corr = df_raw.corr()
    fig_corr = px.imshow(corr, text_auto=".2f")
    st.plotly_chart(fig_corr, use_container_width=True)

# Tab2 模型评估
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("线性回归")
        st.write(metric_lr)
    with col2:
        st.subheader("随机森林")
        st.write(metric_rf)
    st.subheader("随机森林 - 特征重要性")
    imp_df = pd.DataFrame({"特征":X.columns, "重要度":rf_model.feature_importances_}).sort_values("重要度", ascending=False)
    fig_imp = px.bar(imp_df, x="重要度", y="特征")
    st.plotly_chart(fig_imp, use_container_width=True)

# Tab3 预测页面（核心！）
with tab3:
    st.subheader("当前输入参数")
    st.dataframe(input_data)
    if st.button("✨ 开始预测房价"):
        pred_lr = lr_model.predict(input_data)[0]
        pred_rf = rf_model.predict(input_data)[0]
        st.success(f"线性回归预测房价：{pred_lr:.3f} ×10万美元")
        st.success(f"随机森林预测房价：{pred_rf:.3f} ×10万美元")
