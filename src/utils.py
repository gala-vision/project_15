import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel


# функция для вывода основной информации о датафрейме
def information_data(data):
    print('Первые десять строк датафрейма:')
    display(data.head(10))
    print('Общая информация о датафрейме:')
    display(data.info())
    print('Описание данных:')
    display(data.describe())
    print('Количество пропусков:')
    display(data.isna().sum())
    print('Количество дубликатов:')
    display(data.duplicated().sum())

# функция для рассчета агрегированной оценки экспертов    
def aggregate_expert_ratings(row):
    ratings = [row['expert_1'], row['expert_2'], row['expert_3']]
    if len(set(ratings)) == 3:  # Если все три разные, считаем их противоречивыми и отбрасываем
        return None
    return sorted(ratings)[1]  # Медиана большинства
