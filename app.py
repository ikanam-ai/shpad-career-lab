import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
from scipy import stats
from src.func import *
from nltk.corpus import stopwords
from pymorphy3 import MorphAnalyzer
import nltk
import folium
from catboost import CatBoostRegressor, Pool
import shap
import pymorphy3
import re
import pickle
import numpy as np
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import zipfile

DEFAULT_TARGET_SALARY = 70000
CACHE_TTL = 3600

def configure_page():
    st.set_page_config(
        layout="wide",
        page_title="β EkanaMmM",
        page_icon="📈",
        menu_items={
            'Get Help': 'https://github.com/your_repo',
            'Report a bug': "mailto:your@email.com",
            'About': "EkanaMmM"
        }
    )

@st.cache_data(ttl=CACHE_TTL)
def load_data():
    try:
        if not os.path.exists(r'hh.zip'):
            raise FileNotFoundError("Файл hh.zip не найден")
        
        if not zipfile.is_zipfile(r'hh.zip'):
            raise ValueError("Файл не является ZIP-архивом")
        
        with zipfile.ZipFile(r'hh.zip', 'r') as zipf:
            if 'hh.csv' not in zipf.namelist():
                raise ValueError("Файл hh.csv не найден в архиве")
            
            with zipf.open('hh.csv') as file:
                try:
                    df = pd.read_csv(file, encoding='utf-8')
                except UnicodeDecodeError:
                    file.seek(0)
                    df = pd.read_csv(file, encoding='cp1251')
                return df
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {str(e)}")
        st.stop()

@st.cache_resource
def load_model_artifacts():
    try:
        model = CatBoostRegressor()
        model.load_model(r'data/catboost_model.cbm')
        vectorizer = pickle.load(open(r'data/tfidf_vectorizer.pkl', 'rb'))
        return model, vectorizer
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {str(e)}")
        return None, None

def create_sidebar_filters(df, skills):
    with st.sidebar:
        st.title("🔍 Фильтры")

        with st.expander("🔧 Основные параметры", expanded=True):
            experience = st.selectbox(
                "Опыт работы",
                options=sorted(df['experience_category'].unique()),
                index=0
            )

            target_salary = st.number_input(
                "Сравнить с зарплатой (руб)",
                min_value=int(df['salary_from'].min()),
                max_value=int(df['salary_to'].max()),
                value=DEFAULT_TARGET_SALARY,
                step=1000
            )

        with st.expander("⚙️ Условия работы"):
            schedule_type = st.multiselect(
                "График работы",
                options=sorted(df['schedule_label'].unique()),
                default=["Полный день"]
            )
            
            employment_type = st.multiselect(
                "Тип занятости",
                options=sorted(df['employment_label'].unique()),
                default=["Полная занятость"]
            )

        with st.expander("🌍 География"):
            regions = st.multiselect(
                "Регионы",
                options=sorted(df['region_name'].unique()),
                default=["Москва", "Санкт-Петербург"]
            )

            roles = st.multiselect(
                "Должности",
                options=sorted(df['role_name'].str.lower().unique()),
                default=[]
            )
            
            skills = st.multiselect(
                "Ключевые навыки",
                options=skills,
                default=[]
            )
            
            only_active = st.checkbox(
                "Только активные вакансии",
                value=True
            )

    return {
        'experience': experience,
        'target_salary': target_salary,
        'schedule_type': schedule_type,
        'employment_type': employment_type,
        'regions': regions,
        'roles': roles,
        'only_active': only_active,
        'skills': skills
    }

@st.cache_data
def get_region_coordinates(geo_data):
    from data.coordinates import cordes
    city_coords = cordes
    
    geo_data['latitude'] = geo_data['region_name'].map(lambda x: city_coords.get(x, (None, None))[0])
    geo_data['longitude'] = geo_data['region_name'].map(lambda x: city_coords.get(x, (None, None))[1])
    
    missing_coords = geo_data[geo_data['latitude'].isna()]
    if not missing_coords.empty:
        geolocator = Nominatim(user_agent="geoapiExercises")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        
        for idx, row in missing_coords.iterrows():
            try:
                location = geocode(row['region_name'] + ', Россия')
                if location:
                    geo_data.at[idx, 'latitude'] = location.latitude
                    geo_data.at[idx, 'longitude'] = location.longitude
            except:
                continue
                
    return geo_data.dropna(subset=['latitude', 'longitude'])

def create_metric_card(title, value, color="#4a6bdf"):
    return f"""
    <div class="metric-card" style="border-left-color: {color};">
        <h3>{title}</h3>
        <h1>{value}</h1>
    </div>
    """

def show_market_overview_tab(filtered_df, target_salary):
    st.header("📊 Обзор рынка")
    
    col1, col2, col3 = st.columns(3)
    
    total_vacancies = len(filtered_df)
    with col1:
        st.markdown(create_metric_card(
            "Вакансий в регионе", 
            f"{total_vacancies:,}" if total_vacancies > 0 else "N/A"
        ), unsafe_allow_html=True)
    
    if total_vacancies > 0:
        where_min = filtered_df[filtered_df['salary_from'] > 0]
        median_min = int(where_min['salary_from'].median()) if not where_min.empty else 0
        
        with col2:
            st.markdown(create_metric_card(
                "Медианная минимальная зарплата",
                f"{median_min:,} ₽" if median_min > 0 else "N/A",
                "#27ae60"
            ), unsafe_allow_html=True)
        
        where_max = filtered_df[filtered_df['salary_to'] > 0]
        median_max = int(where_max['salary_to'].median()) if not where_max.empty else 0
        
        with col3:
            st.markdown(create_metric_card(
                "Медианная максимальная зарплата",
                f"{median_max:,} ₽" if median_max > 0 else "N/A",
                "#e74c3c"
            ), unsafe_allow_html=True)
        
        st.subheader("📈 Распределение зарплат")
        
        salary_col1, salary_col2 = st.columns(2)
        with salary_col1:
            salary_type = st.radio(
                "Тип зарплаты для анализа",
                ["Минимальная зарплата", "Максимальная зарплата"],
                horizontal=True
            )
        
        with salary_col2:
            bins = st.slider(
                "Количество интервалов",
                min_value=5, max_value=50, value=20
            )
        
        salary_data = (filtered_df[filtered_df['salary_from'] > 0]['salary_from'] 
                      if salary_type == "Минимальная зарплата" 
                      else filtered_df[filtered_df['salary_to'] > 0]['salary_to'])
        
        if not salary_data.empty:
            fig = px.histogram(
                salary_data,
                nbins=bins,
                labels={'value': 'Зарплата (руб)', 'count': 'Количество вакансий'},
                color_discrete_sequence=['#4a6bdf']
            )
            
            fig.add_vline(
                x=target_salary,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Ваша зарплата: {target_salary:,} руб"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            percentile = stats.percentileofscore(salary_data, target_salary)
            st.info(f"💡 Ваши зарплатные ожидания ({target_salary:,} руб) превышают зарплаты вакансий на {percentile:.1f}%")

def show_geography_tab(filtered_df):
    st.header("🌍 География вакансий")
    
    if filtered_df.empty:
        st.warning("Нет данных для отображения")
        return
    
    geo_data = filtered_df.groupby('region_name').agg(
        количество_вакансий=('id', 'count'),
        средняя_зарплата=('salary_from', 'mean')
    ).reset_index()
    
    geo_data = get_region_coordinates(geo_data)
    
    m = folium.Map(location=[62, 94], zoom_start=3)
    
    heat_data = [[row['latitude'], row['longitude'], row['количество_вакансий']] 
               for _, row in geo_data.iterrows()]
    HeatMap(heat_data, radius=15, blur=10).add_to(m)
    
    max_vacancies = geo_data['количество_вакансий'].max()
    for _, row in geo_data.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5 + 15 * (row['количество_вакансий'] / max_vacancies),
            popup=folium.Popup(
                f"<b>{row['region_name']}</b><br>"
                f"Вакансий: {row['количество_вакансий']}<br>"
                f"Средняя зарплата: {row['средняя_зарплата']:,.0f} руб",
                max_width=250
            ),
            color='#3186cc',
            fill=True
        ).add_to(m)
    
    st_folium(m, width=1200, height=600, returned_objects=[])
    
    st.subheader("Данные по регионам")
    st.dataframe(
        geo_data[['region_name', 'количество_вакансий', 'средняя_зарплата']]
        .sort_values('количество_вакансий', ascending=False)
        .style.format({'средняя_зарплата': '{:,.0f} руб.'}),
        use_container_width=True
    )

def show_prediction_tab(model, vectorizer, filters):
    def normalize_filter(value, is_list=False):
        if value is None:
            return [] if is_list else ''
        if isinstance(value, (list, tuple, set)):
            return [str(x) for x in value] if is_list else str(value[0]) if value else ''
        return [str(value)] if is_list else str(value)
    
    try:
        nltk.download('stopwords')
        russian_stopwords = stopwords.words('russian')

        morph = MorphAnalyzer()
        safe_filters = {
            'experience': normalize_filter(filters.get('experience')),
            'target_salary': float(filters.get('target_salary', 0)),
            'schedule_type': normalize_filter(filters.get('schedule_type')),
            'employment_type': normalize_filter(filters.get('employment_type')),
            'regions': normalize_filter(filters.get('regions'), is_list=True),
            'roles': normalize_filter(filters.get('roles'), is_list=True),
            'only_active': bool(filters.get('only_active', False)),
            'skills': normalize_filter(filters.get('skills'), is_list=True)
        }

        input_str = (
            f"{safe_filters['experience']},"
            f"{safe_filters['schedule_type']},"
            f"{safe_filters['employment_type']},"
            f"{','.join(safe_filters['regions'])},"
            f"{','.join(list(map(preprocess_text_2,safe_filters['roles'])))},"
            f"{','.join(list(map(preprocess_text,safe_filters['skills'])))}"
        )
        print(input_str)
        
        X = vectorizer.transform([input_str])
        pool = Pool(X)
        prediction = model.predict(pool)[0]

        shap_values = model.get_feature_importance(pool, type='ShapValues')
        feature_shap = shap_values[0][:-1]
        feature_names = vectorizer.get_feature_names_out()
        
        def get_contributions(prefixes, is_prefix=True):
            if not isinstance(prefixes, list):
                prefixes = [prefixes]
            contributions = []
            for prefix in prefixes:
                if not prefix:
                    continue
                for f, v in zip(feature_names, feature_shap):
                    if is_prefix and str(f).startswith(str(prefix)):
                        contributions.append(v)
                    elif not is_prefix and str(f) == str(prefix):
                        contributions.append(v)
            return contributions
        
        categories = {
            'Опыт': sum(get_contributions(safe_filters['experience'])),
            'График': sum(get_contributions(safe_filters['schedule_type'])),
            'Занятость': sum(get_contributions(safe_filters['employment_type'])),
            'Регионы': sum(get_contributions(safe_filters['regions'])),
            'Роли': sum(get_contributions(safe_filters['roles'])),
            'Навыки': sum(get_contributions(safe_filters['skills'], is_prefix=False))
        }
    except Exception as e:
        st.error(f"Ошибка: {str(e)}")
        return

    st.subheader("Параметры запроса")
    cols = st.columns(2)
    with cols[0]:
        st.markdown(f"**Опыт:** {safe_filters['experience'] or 'Не указан'}")
        st.markdown(f"**График:** {safe_filters['schedule_type'] or 'Не указан'}")
        st.markdown(f"**Занятость:** {safe_filters['employment_type'] or 'Не указан'}")
    with cols[1]:
        st.markdown(f"**Регионы:** {', '.join(safe_filters['regions']) or 'Не указаны'}")
        st.markdown(f"**Роли:** {', '.join(safe_filters['roles']) or 'Не указаны'}")
        st.markdown(f"**Навыки:** {', '.join(safe_filters['skills']) or 'Не указаны'}")
    
    st.success(f"### Прогнозируемая зарплата: {prediction:,.0f} руб.")
    
    st.subheader("Влияние факторов на зарплату")
    influence_df = pd.DataFrame({
        'Фактор': list(categories.keys()),
        'Влияние': list(categories.values()),
        'Абсолютное влияние': [abs(x) for x in categories.values()]
    }).sort_values('Абсолютное влияние', ascending=False)
    
    fig = px.bar(
        influence_df,
        x='Влияние',
        y='Фактор',
        orientation='h',
        color='Влияние',
        color_continuous_scale='RdBu'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    if safe_filters['skills']:
        st.subheader("Детализация по навыкам")
        skill_impacts = {skill: 0.0 for skill in safe_filters['skills']}
    
        for skill in safe_filters['skills']:
            for f, v in zip(feature_names, feature_shap):
                if str(f) == str(skill):
                    skill_impacts[skill] += v
    
        skill_details = pd.DataFrame({
            'Навык': list(skill_impacts.keys()),
            'Влияние на зарплату': list(skill_impacts.values()),
            'Абсолютное влияние': [abs(v) for v in skill_impacts.values()]
        }).sort_values('Абсолютное влияние', ascending=False)
    
        if not skill_details.empty:
            st.dataframe(
                skill_details.style.format({
                    'Влияние на зарплату': '{:,.2f} руб.',
                    'Абсолютное влияние': '{:,.2f}'
                }),
                height=min(400, 35 * len(skill_details)),
                use_container_width=True
            )

def main():
    configure_page()
    df = fill_none_1(load_data())

    with open(r'data\\unique_list.txt', 'r', encoding='utf-8') as f:
        loaded_list = [line.strip() for line in f.readlines()]

    clear_df = remove_salary_outliers(df)
    model, vectorizer = load_model_artifacts()
    filters = create_sidebar_filters(clear_df, loaded_list)
    
    filtered_df = get_median(
        clear_df,
        experience=filters['experience'],
        schedule_type=filters['schedule_type'],
        employment_type=filters['employment_type'],
        regions=filters['regions'],
        roles=filters['roles'],
        only_active=filters['only_active']
    )
    filtered_df = filtered_df[(filtered_df['salary_from'] > 0) & (filtered_df['salary_to'] > 0)]
    
    st.markdown("""
        <style>
            .metric-card {
                background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 20px;
                border-left: 4px solid #4a6bdf;
                text-align: center;
                height: 120px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }
            .metric-card h3 {
                color: #555;
                font-size: 14px;
                margin: 0 0 8px 0;
            }
            .metric-card h1 {
                color: #222;
                font-size: 28px;
                margin: 0;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("β EkanaMmM")
    
    tab1, tab2, tab3 = st.tabs(["📊 Обзор рынка", "🌍 География вакансий", "🔮 Прогноз зарплаты"])
    
    with tab1:
        show_market_overview_tab(filtered_df, filters['target_salary'])
    
    with tab2:
        show_geography_tab(filtered_df)
    
    with tab3:
        if model and vectorizer:
            show_prediction_tab(model, vectorizer, filters)
        else:
            st.warning("Модель не загружена")

if __name__ == "__main__":
    main()