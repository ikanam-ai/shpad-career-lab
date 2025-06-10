import pandas as pd
from typing import List
from nltk.corpus import stopwords
from pymorphy3 import MorphAnalyzer
import nltk
import re

def get_median(df, experience:str, schedule_type:List[str], employment_type:List[str], regions:List[str], roles:List[str], only_active : bool):
    # Применяем фильтры
    if only_active:
        filters = df[df['is_open'] == int(only_active)] 
    else:
        filters = df

    filters = filters[filters['experience_category'] == experience]

    if len(schedule_type) > 0:
        filters = filters[filters['schedule_label'].isin(schedule_type)]

    if len(employment_type) > 0:
        filters = filters[filters['employment_label'].isin(employment_type)]
    
    if len(regions) > 0:
        filters = filters[filters['region_name'].isin(regions)]

    if len(roles) > 0:
        filters = filters[filters['role_name'].isin(roles)]

    
    return filters



def remove_salary_outliers(df):
    # Создаем копию и фильтруем аномалии
    df_clean = df.copy()
    
    # Удаляем 1% выбросов по каждой профессии
    def filter_group(group):
        lower_from = group['salary_from'].quantile(0.01)
        upper_from = group['salary_from'].quantile(0.99)
        lower_to = group['salary_to'].quantile(0.01)
        upper_to = group['salary_to'].quantile(0.99)
        
        return group[
            (group['salary_from'] >= lower_from) & 
            (group['salary_from'] <= upper_from) &
            (group['salary_to'] >= lower_to) & 
            (group['salary_to'] <= upper_to)
        ]
    
    return filter_group(df_clean)


def fill_none_1(df):
    df['region_district_name'] = df['region_district_name'].fillna('Не указан')
    df['soft_skills'] = df['soft_skills'].fillna('Не указаны')
    df['experience_category'] = df['experience_category'].fillna('Не указан')
    for column in df.columns:
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())

    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)
    
    return df



def clean_and_filter_salaries_3(df):
    # Поверхностная фильтрация
    df_clean = df[
        (df['salary_to'] <= 2_000_000)].copy()
    
    
    def filter_group(group):
        # Отдельно сохраняем строки, где зарплата == 0
        zeros = group[(group['salary_from'] == 0) | (group['salary_to'] == 0)]
        non_zeros = group[(group['salary_from'] != 0) & (group['salary_to'] != 0)]
        # Для редких вакансий 
        if len(non_zeros) < 20:
            filtered = non_zeros[
                (non_zeros['salary_from'] >= 10_000) & 
                (non_zeros['salary_to'] <= 1_000_000)
            ]
        else: # Для остальных вакансий
            q1_from, q3_from = non_zeros['salary_from'].quantile([0.04, 0.96])
            q1_to, q3_to = non_zeros['salary_to'].quantile([0.04, 0.96])
            iqr_from = q3_from - q1_from
            iqr_to = q3_to - q1_to
            lower_bound_from = q1_from - 1.5 * iqr_from
            upper_bound_from = q3_from + 1.5 * iqr_from
            lower_bound_to = q1_to - 1.5 * iqr_to
            upper_bound_to = q3_to + 1.5 * iqr_to
            
            filtered = non_zeros[
                (non_zeros['salary_from'] >= lower_bound_from) & 
                (non_zeros['salary_from'] <= upper_bound_from) &
                (non_zeros['salary_to'] >= lower_bound_to) & 
                (non_zeros['salary_to'] <= upper_bound_to)
            ]

        # Объединяем отфильтрованные и нулевые строки
        return pd.concat([filtered, zeros], ignore_index=True)

    df_filtered = df_clean.groupby('role_name', group_keys=False).apply(filter_group)

    #  СПЕЦИАЛЬНЫЕ ПРАВИЛА ДЛЯ ОТДЕЛЬНЫХ ПРОФЕССИЙ 
    # Расширяем границы для стажёровок и подработки
    intern_mask = df_filtered['role_name'].str.contains(
        'Стажёр|Intern|Подработка|подработка|ПОДРАБОТКА',
        case=False,
        na=False
    )
    df_filtered.loc[intern_mask, 'salary_from'] = df_filtered.loc[intern_mask, 'salary_from'].clip(lower=5_000)
     # Расширяем границы для топ-менеджеров
    executive_mask = df_filtered['role_name'].str.contains(
        r'\b(CEO|CTO|Директор|Генеральный директор|Head of)\b',
        case=False,
        na=False
    )
    df_filtered.loc[executive_mask, 'salary_to'] = df_filtered.loc[executive_mask, 'salary_to'].clip(upper=2_000_000)


    return df_filtered


def comma_tokenizer(text):
    return text.split(',')
               
def preprocess_text(text):
    return text.strip()


nltk.download('stopwords')
russian_stopwords = stopwords.words('russian')

morph = MorphAnalyzer()

def preprocess_text(text):
    words = text.split(', ')
    lemmas = [morph.parse(word)[0].normal_form for word in words]
    return ' '.join(lemmas)

def preprocess_text_2(text):
    text = re.sub(r'[^а-яёА-ЯЁ ]', '', text)
    words = text.split()
    lemmas = [morph.parse(word)[0].normal_form for word in words]
    return ' '.join(lemmas)