import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, json, os

st.set_page_config(page_title='AQI Predictor', page_icon='🌫️',
                   layout='wide', initial_sidebar_state='expanded')

# ── Helpers ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

def path(f): return os.path.join(BASE, f)

@st.cache_resource
def load_model():  return joblib.load(path('aqi_model.pkl'))

@st.cache_resource
def load_scaler(): return joblib.load(path('scaler.pkl'))

@st.cache_data
def load_data():   return pd.read_csv(path('aqi_clean.csv'), parse_dates=['Date'])

@st.cache_data
def load_config():
    with open(path('app_config.json')) as f: return json.load(f)

model        = load_model()
scaler       = load_scaler()
df           = load_data()
cfg          = load_config()
FEATURES     = cfg['features']
city_mapping = cfg['city_mapping']
city_names   = sorted(city_mapping.keys())

BUCKETS = [(0,50,'Good','#00e400','😊'),
           (51,100,'Satisfactory','#9acd32','🙂'),
           (101,200,'Moderate','#ffff00','😐'),
           (201,300,'Poor','#ff7e00','😷'),
           (301,400,'Very Poor','#ff0000','🤢'),
           (401,9999,'Severe','#99004c','☠️')]

def aqi_info(v):
    for lo,hi,lbl,col,icon in BUCKETS:
        if lo <= v <= hi: return lbl, col, icon
    return 'Severe','#99004c','☠️'

def make_prediction(city, pm25, pm10, no2, so2, co, o3, lag1, month, dow):
    season = (int(month)-1)//3+1
    inp = pd.DataFrame([{
        'PM2.5':pm25,'PM10':pm10,'NO2':no2,'SO2':so2,'CO':co,'O3':o3,
        'City_enc':city_mapping[city],'month':month,
        'dayofweek':dow,'season':season,'AQI_lag1':lag1
    }])[FEATURES]
    return float(model.predict(scaler.transform(inp))[0])

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image('https://i.imgur.com/kQbSMGl.png', width=60)
st.sidebar.title('🌫️ AQI Predictor')
st.sidebar.markdown('India Air Quality — ML Dashboard')
st.sidebar.markdown('---')
page = st.sidebar.radio('', [
    '🔮 Predict AQI',
    '📊 EDA Dashboard',
    '🏙️ City Rankings',
    '🎛️ What-If Simulator',
    '📋 About'
])
st.sidebar.markdown('---')
st.sidebar.caption('Built with XGBoost + Streamlit')

# ════════════════════════════════════════════════════
# PAGE 1 — Predict
# ════════════════════════════════════════════════════
if page == '🔮 Predict AQI':
    st.title('🔮 Predict Air Quality Index')
    st.markdown('Fill in the current pollutant readings to get an instant AQI prediction.')
    st.markdown('---')

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader('Location & Time')
        city  = st.selectbox('City', city_names)
        month = st.slider('Month', 1, 12, 6)
        dow   = st.slider('Day of Week (0=Mon)', 0, 6, 2)
        lag1  = st.number_input("Yesterday's AQI", 0.0, 1000.0, 120.0)
    with c2:
        st.subheader('Particulates')
        pm25 = st.number_input('PM2.5 (µg/m³)', 0.0, 1000.0, 60.0)
        pm10 = st.number_input('PM10  (µg/m³)', 0.0, 1000.0, 90.0)
        no2  = st.number_input('NO2   (µg/m³)', 0.0,  500.0, 30.0)
    with c3:
        st.subheader('Gases')
        so2 = st.number_input('SO2 (µg/m³)',  0.0, 500.0, 15.0)
        co  = st.number_input('CO  (mg/m³)',  0.0, 100.0,  1.0, step=0.1)
        o3  = st.number_input('O3  (µg/m³)',  0.0, 500.0, 40.0)

    st.markdown('---')
    if st.button('🔮 Predict AQI', type='primary', use_container_width=True):
        pred = make_prediction(city,pm25,pm10,no2,so2,co,o3,lag1,month,dow)
        lbl, col, icon = aqi_info(pred)
        season_name = ['','Winter','Spring','Summer','Autumn'][(month-1)//3+1]

        r1, r2, r3, r4 = st.columns(4)
        r1.metric('Predicted AQI', f'{pred:.1f}')
        r2.metric('City', city)
        r3.metric('Season', season_name)
        r4.metric('Category', f'{icon} {lbl}')

        st.markdown(
            f'<div style="background:{col};padding:18px;border-radius:12px;'
            f'text-align:center;font-size:1.5em;font-weight:bold;color:#111;margin-top:12px">'
            f'{icon} {lbl} — AQI {pred:.0f}</div>',
            unsafe_allow_html=True
        )

# ════════════════════════════════════════════════════
# PAGE 2 — EDA
# ════════════════════════════════════════════════════
elif page == '📊 EDA Dashboard':
    st.title('📊 Exploratory Data Analysis')
    t1, t2, t3, t4 = st.tabs(['📦 Distribution','🔗 Correlation','📈 Trend','🌆 City View'])

    with t1:
        fig, axes = plt.subplots(1, 2, figsize=(14,5))
        axes[0].hist(df['AQI'], bins=50, color='steelblue', edgecolor='white')
        axes[0].set_title('AQI Distribution'); axes[0].set_xlabel('AQI')
        df.boxplot(column='AQI', by='season', ax=axes[1])
        axes[1].set_title('AQI by Season')
        axes[1].set_xlabel('Season (1=Winter 2=Spring 3=Summer 4=Autumn)')
        plt.suptitle('')
        st.pyplot(fig)

    with t2:
        num = ['PM2.5','PM10','NO2','SO2','CO','O3','month','season','AQI_lag1','AQI']
        corr = df[num].select_dtypes(include='number').corr()
        fig, ax = plt.subplots(figsize=(11,8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                    mask=mask, linewidths=0.5, ax=ax, annot_kws={'size':9})
        ax.set_title('Correlation Heatmap')
        st.pyplot(fig)

    with t3:
        selected_city = st.selectbox('Select city for trend', ['All Cities'] + city_names)
        d = df if selected_city == 'All Cities' else df[df['City']==selected_city]
        mn = d.groupby(d['Date'].dt.to_period('M'))['AQI'].mean().reset_index()
        mn['Date'] = mn['Date'].dt.to_timestamp()
        fig, ax = plt.subplots(figsize=(13,4))
        ax.plot(mn['Date'], mn['AQI'], color='darkorange', linewidth=1.8)
        ax.fill_between(mn['Date'], mn['AQI'], alpha=0.15, color='darkorange')
        ax.set_title(f'Monthly AQI Trend — {selected_city}')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        st.pyplot(fig)

    with t4:
        sel = st.selectbox('City', city_names, key='city_view')
        cdf = df[df['City']==sel]
        col1, col2, col3 = st.columns(3)
        col1.metric('Mean AQI',   f"{cdf['AQI'].mean():.1f}")
        col2.metric('Max AQI',    f"{cdf['AQI'].max():.0f}")
        col3.metric('Total Days', len(cdf))
        fig, ax = plt.subplots(figsize=(13,4))
        ax.plot(cdf['Date'], cdf['AQI'], alpha=0.7, linewidth=0.8, color='steelblue')
        ax.set_title(f'Daily AQI — {sel}')
        ax.set_xlabel('Date'); ax.set_ylabel('AQI')
        st.pyplot(fig)

# ════════════════════════════════════════════════════
# PAGE 3 — City Rankings
# ════════════════════════════════════════════════════
elif page == '🏙️ City Rankings':
    st.title('🏙️ City AQI Rankings')
    stats = df.groupby('City')['AQI'].agg(['mean','median','std','max','min']).round(1)
    stats.columns = ['Mean AQI','Median','Std Dev','Max','Min']
    stats = stats.sort_values('Mean AQI', ascending=False)
    st.dataframe(
        stats.style.background_gradient(cmap='RdYlGn_r', subset=['Mean AQI']),
        use_container_width=True
    )
    fig, ax = plt.subplots(figsize=(12,5))
    cols = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(stats)))
    bars = ax.bar(stats.index, stats['Mean AQI'], color=cols, edgecolor='white')
    ax.set_title('Average AQI by City', fontweight='bold')
    ax.set_xlabel('City'); ax.set_ylabel('Average AQI')
    plt.xticks(rotation=30, ha='right')
    for bar, val in zip(bars, stats['Mean AQI']):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                f'{val:.0f}', ha='center', fontsize=9)
    st.pyplot(fig)

# ════════════════════════════════════════════════════
# PAGE 4 — What-If
# ════════════════════════════════════════════════════
elif page == '🎛️ What-If Simulator':
    st.title('🎛️ What-If Pollution Simulator')
    st.markdown('Drag sliders to simulate different pollution scenarios in real time.')

    c1, c2 = st.columns(2)
    with c1:
        city  = st.selectbox('City', city_names, key='wi_city')
        pm25  = st.slider('PM2.5', 0.0, 500.0, 60.0)
        pm10  = st.slider('PM10',  0.0, 500.0, 90.0)
        no2   = st.slider('NO2',   0.0, 200.0, 30.0)
    with c2:
        so2   = st.slider('SO2',   0.0, 200.0, 15.0)
        co    = st.slider('CO',    0.0,  50.0,  1.0, step=0.5)
        o3    = st.slider('O3',    0.0, 300.0, 40.0)
        lag1  = st.slider("Yesterday's AQI", 0.0, 600.0, 120.0)

    month  = st.slider('Month', 1, 12, 6)
    dow    = st.slider('Day of Week', 0, 6, 2)

    pred = make_prediction(city,pm25,pm10,no2,so2,co,o3,lag1,month,dow)
    lbl, col, icon = aqi_info(pred)

    st.markdown('---')
    a, b = st.columns(2)
    a.metric('Live Predicted AQI', f'{pred:.1f}')
    b.markdown(
        f'<div style="background:{col};padding:16px;border-radius:10px;'
        f'text-align:center;font-weight:bold;font-size:1.4em;color:#111">'
        f'{icon} {lbl}</div>', unsafe_allow_html=True
    )

# ════════════════════════════════════════════════════
# PAGE 5 — About
# ════════════════════════════════════════════════════
elif page == '📋 About':
    st.title('📋 About This Project')
    st.markdown("""
    ### 🌫️ AQI Prediction — Machine Learning Project

    **Dataset:** Air Quality Data in India (Kaggle)  
    **Records:** ~29,500 daily city observations across 26 Indian cities  
    **Target Variable:** Air Quality Index (AQI)

    ---
    #### Models Trained
    | Model | Description |
    |---|---|
    | Linear Regression | Baseline model |
    | Random Forest | Ensemble of decision trees |
    | Gradient Boosting | Sequential boosting |
    | **XGBoost** ✅ | Best performer — used in this app |

    #### Features Used
    PM2.5, PM10, NO2, SO2, CO, O3, City, Month, Day of Week, Season, Previous Day AQI

    #### Tech Stack
    Python · Pandas · NumPy · Scikit-learn · XGBoost · Streamlit · Matplotlib · Seaborn
    """)
