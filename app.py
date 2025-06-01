# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from func import *
import plotly.graph_objects as go
from streamlit_tags import st_tags
from datetime import datetime, timedelta
import os
from scipy import stats
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import zipfile

# --- КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(
    layout="wide",
    page_title="JobMarket Analytics Pro",
    page_icon="📈",
    menu_items={
        'Get Help': 'https://github.com/your_repo',
        'Report a bug': "mailto:your@email.com",
        'About': "### JobMarket Analytics Pro v2.0\nАнализ рынка вакансий с расширенными возможностями"
    }
)

# --- ФУНКЦИИ ДЛЯ ЗАГРУЗКИ ДАННЫХ ---
@st.cache_data
def load_data():
    """Загрузка данных из ZIP-архива с расширенной обработкой ошибок"""
    try:
        # Проверка существования файла
        if not os.path.exists('hh.zip'):
            raise FileNotFoundError("Файл hh.zip не найден в рабочей директории")
        
        # Проверка, что это действительно ZIP-архив
        if not zipfile.is_zipfile('hh.zip'):
            raise ValueError("Файл не является ZIP-архивом или поврежден")
        
        with zipfile.ZipFile('hh.zip', 'r') as zipf:
            # Проверка наличия CSV-файла в архиве
            if 'hh.csv' not in zipf.namelist():
                available_files = "\n".join(zipf.namelist())
                raise ValueError(f"Файл hh.csv не найден в архиве. Доступные файлы:\n{available_files}")
            
            with zipf.open('hh.csv') as file:
                # Чтение с указанием кодировки (частая проблема)
                try:
                    df = pd.read_csv(file, encoding='utf-8')
                except UnicodeDecodeError:
                    file.seek(0)  # Сбрасываем позицию чтения
                    df = pd.read_csv(file, encoding='cp1251')  # Альтернативная кодировка
                
                # Проверка, что DataFrame не пустой
                if df.empty:
                    raise ValueError("CSV-файл загружен, но не содержит данных")
                
                return df
    except FileNotFoundError as e:
        st.error(f"Ошибка: {str(e)}. Рабочая директория: {os.getcwd()}")
        st.stop()
    except zipfile.BadZipFile:
        st.error("Файл поврежден или не является ZIP-архивом")
        st.stop()
    except Exception as e:
        st.error(f"Критическая ошибка при загрузке данных: {str(e)}")
        st.stop()

# Загрузка данных
df = fill_none_1(load_data())

# --- БОКОВАЯ ПАНЕЛЬ (ФИЛЬТРЫ) ---
def create_sidebar_filters():
    """Создание панели фильтров в сайдбаре"""
    with st.sidebar:
        st.title("🔍 Расширенные фильтры")

        # Основные параметры
        with st.expander("🔧 Основные параметры", expanded=True):
            experience = st.selectbox(
                "Опыт работы",
                options=list(df['experience_category'].unique()),
                index=0,
                help="Выберите требуемый уровень опыта"
            )

            target_salary = st.number_input(
                "Сравнить с зарплатой (руб)",
                min_value=int(df['salary_from'].min()),
                max_value=int(df['salary_to'].max()),
                value=70000,
                step=1000
            )

        # Условия работы
        with st.expander("⚙️ Условия работы"):
            schedule_type = st.multiselect(
                "График работы",
                options=list(df['schedule_label'].unique()),
                default=["Полный день"]
            )
            
            employment_type = st.multiselect(
                "Тип занятости",
                options=list(df['employment_label'].unique()),
                default=["Полная занятость"]
            )

        # География и специализация
        with st.expander("🌍 География и специализация"):
            regions = st.multiselect(
                "Регионы",
                options=list(df['region_name'].unique()),
                default=["Москва", "Санкт-Петербург"]
            )

            roles = st.multiselect(
                "Должности",
                options=list(df['role_name'].unique()),
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
        'only_active': only_active
    }

# Функция для получения координат регионов
@st.cache_data
def get_region_coordinates(geo_data):
    """Добавляем координаты для регионов"""
    # Координаты основных городов (можно расширить)
    from cordes import cordes
    city_coords = cordes
    geo_data['latitude'] = geo_data['region_name'].map(lambda x: city_coords.get(x, (None, None))[0])
    geo_data['longitude'] = geo_data['region_name'].map(lambda x: city_coords.get(x, (None, None))[1])
    
    # Для регионов без координат используем геокодинг
    geolocator = Nominatim(user_agent="geoapiExercises")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    
    for idx, row in geo_data.iterrows():
        if pd.isna(row['latitude']):
            try:
                location = geocode(row['region_name'] + ', Россия')
                if location:
                    geo_data.at[idx, 'latitude'] = location.latitude
                    geo_data.at[idx, 'longitude'] = location.longitude
            except:
                continue
                
    return geo_data.dropna(subset=['latitude', 'longitude'])

# Создаем фильтры и получаем их значения
filters = create_sidebar_filters()

# --- ОСНОВНАЯ ОБЛАСТЬ ---
def create_main_content(filters):
    """Создание основного контента страницы"""
    st.title("📈 JobMarket Analytics Pro")
    
    # Стили CSS
    st.markdown("""
        <style>
            .metric-card {
                background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 20px;
                border-left: 4px solid #4a6bdf;
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
            .stPlotlyChart {
                margin-bottom: -30px !important;
            }
            .plot-container {
                padding-bottom: 0 !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # Создаем вкладки
    tab1, tab2 = st.tabs(["📊 Обзор рынка", "🌍 География вакансий"])

    # ВКЛАДКА 1: ОБЗОР РЫНКА
    with tab1:
        show_market_overview_tab(filters)

    # ВКЛАДКА 2: ГЕОГРАФИЯ ВАКАНСИЙ
    with tab2:
        show_geography_tab(filters)

сlear_df = remove_salary_outliers(df)
spec = get_median(
                сlear_df,
                experience=filters['experience'],
                schedule_type=filters['schedule_type'],
                employment_type=filters['employment_type'],
                regions=filters['regions'],
                roles=filters['roles'],
                only_active=filters['only_active'])
filter_df = spec[(spec['salary_from'] > 0) & (spec['salary_to'] >0)]

total_vacancies = len(spec)

def show_market_overview_tab(filters):
    """Вкладка с обзором рынка"""
    st.header("📊 Обзор текущего рынка вакансий")
    
    # Метрики в колонках
    col1, col2, col3 = st.columns(3)

    metric_style = """
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
    """
    st.markdown(metric_style, unsafe_allow_html=True)
    with col1:
        if len(spec)>0:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>Вакансий в регионе</h3>
                    <h1>{total_vacancies:,}</h1>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="metric-card">
                    <h3>Вакансий в регионе</h3>
                    <h1>N/A</h1>
                    <p style='color: #7f8c8d; font-size: 14px; margin: 8px 0 0 0;'>Выберите регионы</p>
                </div>
            """, unsafe_allow_html=True)

    with col2:
        if total_vacancies > 0:
            where = spec[spec['salary_from'] > 0]
            median_min = where['salary_from'].median()
            
            st.markdown(f"""
                <div class="metric-card" style="border-left-color: #27ae60;">
                    <h3>Медианная минимальная зарплата</h3>
                    <h1>{int(median_min):,} ₽</h1>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="metric-card" style="border-left-color: #27ae60;">
                    <h3>Медианная минимальная зарплата</h3>
                    <h1>N/A</h1>
                </div>
            """, unsafe_allow_html=True)

    with col3:
        if total_vacancies > 0:
            where = spec[spec['salary_to'] > 0]
            median_max = where['salary_to'].median()
            
            st.markdown(f"""
                <div class="metric-card" style="border-left-color: #e74c3c;">
                    <h3>Медианная максимальная зарплата</h3>
                    <h1>{int(median_max):,} ₽</h1>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="metric-card" style="border-left-color: #e74c3c;">
                    <h3>Медианная максимальная зарплата</h3>
                    <h1>N/A</h1>
                </div>
            """, unsafe_allow_html=True)

    # Добавляем гистограмму распределения зарплат
    if len(spec) > 0:
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
                min_value=5,
                max_value=50,
                value=20,
                help="Выберите количество столбцов на гистограмме"
            )
        
        if salary_type == "Минимальная зарплата":
            salary_data = spec[spec['salary_from'] > 0]['salary_from']
            title = "Распределение минимальных зарплат"
        else:
            salary_data = spec[spec['salary_to'] > 0]['salary_to']
            title = "Распределение максимальных зарплат"
        
        fig = px.histogram(
            salary_data,
            nbins=bins,
            title=title,
            labels={'value': 'Зарплата (руб)', 'count': 'Количество вакансий'},
            color_discrete_sequence=['#4a6bdf']
        )
        
        # Добавляем вертикальную линию для target_salary
        fig.add_vline(
            x=filters['target_salary'],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Ваша зарплата: {filters['target_salary']:,} руб",
            annotation_position="top"
        )
        
        fig.update_layout(
            bargap=0.1,
            plot_bgcolor='white',
            xaxis_title="Зарплата, руб",
            yaxis_title="Количество вакансий",
            hovermode="x"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Добавляем анализ положения зарплаты
        if len(salary_data) > 0:
            percentile = stats.percentileofscore(salary_data, filters['target_salary'])
            st.info(f"💡 Ваши зарплатные ожидания ({filters['target_salary']:,} руб) превышают зарплаты вакансий на {percentile:.1f}%")

def show_geography_tab(filters):
    """Вкладка с географическим распределением вакансий (однократное отображение)"""
    st.header("🌍 Географическое распределение вакансий")
    # Создаем карту только один раз при первом вызове
    if not filter_df.empty:
        # Подготовка данных
        geo_data = filter_df.groupby('region_name').agg(
            количество_вакансий=('id', 'count'),
            средняя_зарплата=('salary_from', 'mean')
        ).reset_index()
        geo_data = get_region_coordinates(geo_data)
        
        # Создаем новую карту каждый раз, но без сохранения состояния
        m = folium.Map(location=[62, 94], zoom_start=3)
        
        # 1. Добавляем тепловую карту
        heat_data = [[row['latitude'], row['longitude'], row['количество_вакансий']] 
                   for _, row in geo_data.iterrows()]
        HeatMap(heat_data, radius=15, blur=10).add_to(m)
        
        # 2. Добавляем маркеры (кружки)
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
                fill=True,
                fill_color='#3186cc',
                fill_opacity=0.7
            ).add_to(m)
        
        # Отображаем карту без сохранения состояния
        st_folium(
            m,
            width=1200,
            height=600,
            returned_objects=[]  # Отключаем возврат взаимодействий
        )
        
        # Таблица с данными
        st.subheader("Данные по регионам")
        st.dataframe(
            geo_data[['region_name', 'количество_вакансий', 'средняя_зарплата']]
            .sort_values('количество_вакансий', ascending=False)
            .style.format({'средняя_зарплата': '{:,.0f} руб.'})
        )
    else:
        st.warning("Нет данных для отображения. Измените параметры фильтрации.")

if __name__ == "__main__":
    create_main_content(filters)