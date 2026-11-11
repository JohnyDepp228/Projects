import os
import shutil
import pandas as pd

# --- НАСТРОЙКИ ПУТЕЙ И ОГРАНИЧЕНИЙ ---
# Укажите путь к корневой папке репозитория датасета (где лежат CSV-файлы и папка Images)
DATASET_ROOT = "fish-vista" 

# Специфичный CSV-файл для разметки классификации
CSV_NAME = "classification_train.csv"

# Папка, куда скрипт разложит картинки по видам
OUTPUT_DIR = "fish-vista_sorted_by_species"

# СКОЛЬКО ВИДОВ РЫБ ОСТАВИТЬ
NUM_SPECIES_TO_KEEP = 100
# -------------------------------------

def sort_limited_species():
    csv_path = os.path.join(DATASET_ROOT, CSV_NAME)
    
    if not os.path.exists(csv_path):
        print(f"Ошибка: Не найден CSV-файл по пути: '{os.path.abspath(csv_path)}'")
        print("Проверьте переменную DATASET_ROOT в начале скрипта.")
        return

    # Загружаем таблицу разметки
    df = pd.read_csv(csv_path, low_memory=False)
    
    if 'filename' not in df.columns or 'standardized_species' not in df.columns:
        print("Ошибка: В CSV-файле отсутствуют колонки 'filename' или 'standardized_species'!")
        return

    # Отбираем ТОП-100 видов, у которых больше всего картинок
    print(f"Выбираем {NUM_SPECIES_TO_KEEP} самых популярных видов рыб...")
    top_species = df['standardized_species'].value_counts().head(NUM_SPECIES_TO_KEEP).index.tolist()
    
    # Фильтруем датафрейм
    df_filtered = df[df['standardized_species'].isin(top_species)]
    
    print(f"Отобрано видов: {len(top_species)}")
    print(f"Всего изображений для этих видов в таблице: {len(df_filtered)}")
    print("Начинаем копирование...")

    copied_count = 0
    missing_count = 0

    # Очищаем старую папку вывода, если скрипт упал ранее, чтобы начать начисто
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    for idx, row in df_filtered.reset_index(drop=True).iterrows():
        img_name = row['filename']
        species_name = row['standardized_species']

        # Очищаем название вида для безопасного имени папки
        safe_species_name = "".join([c if c.isalnum() or c in (' ', '_', '-') else '_' for c in species_name])
        safe_species_name = safe_species_name.strip().replace(' ', '_')

        # УМНЫЙ ПОИСК КАРТИНКИ:
        # Проверяем 3 возможных места нахождения файла:
        src_paths_to_try = []
        
        # 1. Если колонка file_name заполнена (например, 'Images/chunk_1/имя.jpg')
        if 'file_name' in row and pd.notna(row['file_name']):
            src_paths_to_try.append(os.path.join(DATASET_ROOT, row['file_name']))
            
        # 2. Если файлы уже были объединены в общую папку Images
        src_paths_to_try.append(os.path.join(DATASET_ROOT, "Images", img_name))
        
        # 3. Ручной перебор чанков на случай, если структура папок оригинальная
        for chunk_id in range(1, 8):
            src_paths_to_try.append(os.path.join(DATASET_ROOT, "Images", f"chunk_{chunk_id}", img_name))

        # Выбираем первый реально существующий путь
        src_path = None
        for path in src_paths_to_try:
            if os.path.exists(path):
                src_path = path
                break

        # Целевая папка и путь
        species_folder = os.path.join(OUTPUT_DIR, safe_species_name)
        dst_path = os.path.join(species_folder, img_name)

        # Копируем файл
        if src_path is not None:
            os.makedirs(species_folder, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        else:
            missing_count += 1

        # Прогресс
        if (idx + 1) % 2000 == 0:
            print(f"Проверено строк: {idx + 1}..., Успешно скопировано: {copied_count}")

    print("\nСортировка завершена без ошибок!")
    print(f"Успешно скопировано картинок в новые папки: {copied_count} шт.")
    if missing_count > 0:
        print(f"Файлов не найдено на диске (пропущено): {missing_count} шт.")
        print(f"Примечание: Убедитесь, что вы скачали сами картинки (LFS-файлы) из HuggingFace!")
    
    if os.path.exists(OUTPUT_DIR):
        print(f"Итого создано папок (видов рыб): {len(os.listdir(OUTPUT_DIR))}")
    else:
        print("Папки не созданы, так как ни один файл изображений не был найден.")

if __name__ == "__main__":
    sort_limited_species()
