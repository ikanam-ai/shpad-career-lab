import pandas as pd
from typing import List, Dict, Tuple, Set, Optional

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

               