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
    
def aggregate_expert_ratings(row):
    ratings = [row['expert_1'], row['expert_2'], row['expert_3']]
    if len(set(ratings)) == 3:  # Если все три разные, считаем их противоречивыми и отбрасываем
        return None
    return sorted(ratings)[1]  # Медиана большинства

# Функция для векторизации изображений
def extract_image_features(image_path, model):
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)  # Добавляем размерность batch
        with torch.no_grad():
            features = model_resnet(image)  # Получаем векторное представление
        return features.squeeze().numpy()  # Преобразуем в numpy
    except Exception as e:
        print(f"Ошибка с файлом {image_path}: {e}")
        
# Функция для векторизации текста
def extract_text_features(text):
    """Токенизирует текст, прогоняет через BERT и возвращает средний эмбеддинг."""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        outputs = model_bert(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Усредняем по всем токенам
    return embeddings.squeeze().numpy()  # Преобразуем в numpy-массив

# Функция для отображения изображения
def show_image(image_path, title):
    """Выводит изображение с подписью."""
    image = Image.open(image_path)
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()
        