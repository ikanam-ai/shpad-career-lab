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

               