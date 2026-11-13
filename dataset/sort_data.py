import os
import shutil
import pandas as pd

# --- НАСТРОЙКИ ПУТЕЙ И ОГРАНИЧЕНИЙ ---
# Укажите путь к корневой папке репозитория датасета (где лежат CSV-файлы и папка Images)
DATASET_ROOT = "fish-vista" 

# Специфичный CSV-файл для разметки классификации
CSV_NAME = "classification_train.csv"

# Главная папка для вывода отсортированных данных
OUTPUT_DIR = "fish-vista_split"

# СКОЛЬКО ВИДОВ РЫБ ОСТАВИТЬ
NUM_SPECIES_TO_KEEP = 100

# ПАРАМЕТРЫ РАЗДЕЛЕНИЯ
TRAIN_COUNT = 100
TEST_COUNT = 50
TOTAL_REQUIRED = TRAIN_COUNT + TEST_COUNT  # 150
# -------------------------------------

def sort_and_split_species():
    csv_path = os.path.join(DATASET_ROOT, CSV_NAME)
    
    if not os.path.exists(csv_path):
        print(f"Ошибка: Не найден CSV-файл по пути: '{os.path.abspath(csv_path)}'")
        return

    # Загружаем таблицу разметки
    df = pd.read_csv(csv_path, low_memory=False)
    
    if 'filename' not in df.columns or 'standardized_species' not in df.columns:
        print("Ошибка: В CSV-файле отсутствуют колонки 'filename' или 'standardized_species'!")
        return

    # Подсчитываем количество картинок для каждого вида
    species_counts = df['standardized_species'].value_counts()
    
    # Фильтруем: оставляем только те виды, у которых ЕСТЬ ХОТЯ БЫ 150 картинок
    suitable_species = species_counts[species_counts >= TOTAL_REQUIRED].index.tolist()
    
    if len(suitable_species) < NUM_SPECIES_TO_KEEP:
        print(f"Внимание: В датасете всего {len(suitable_species)} видов, у которых есть >= {TOTAL_REQUIRED} картинок.")
        print(f"Скрипт соберет максимально возможное количество видов: {len(suitable_species)}")
        chosen_species = suitable_species
    else:
        # Берем ровно 100 видов из подходящих (самых наполненных)
        chosen_species = suitable_species[:NUM_SPECIES_TO_KEEP]

    # Очищаем старую папку вывода, если она была, чтобы начать начисто
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    print(f"Отобрано видов рыб для обработки: {len(chosen_species)}")
    print(f"Для каждого вида будет скопировано: {TRAIN_COUNT} в train и {TEST_COUNT} в test.")
    print("Начинаем копирование...")

    total_copied = 0
    missing_count = 0

    # Проходим по каждому выбранному виду отдельно
    for species_idx, species_name in enumerate(chosen_species):
        # Вытаскиваем все строки для этого конкретного вида
        species_df = df[df['standardized_species'] == species_name].reset_index(drop=True)
        
        # Безопасное имя для папки класса
        safe_species_name = "".join([c if c.isalnum() or c in (' ', '_', '-') else '_' for c in species_name])
        safe_species_name = safe_species_name.strip().replace(' ', '_')

        # Будем перебирать строки, пока успешно не скопируем ровно 150 картинок на этот класс
        current_species_copied = 0
        
        for _, row in species_df.iterrows():
            if current_species_copied >= TOTAL_REQUIRED:
                break  # Собрали нужные 150 штук для этого вида, переходим к следующему

            img_name = row['filename']

            # УМНЫЙ ПОИСК ИСХОДНОЙ КАРТИНКИ
            src_paths_to_try = []
            if 'file_name' in row and pd.notna(row['file_name']):
                src_paths_to_try.append(os.path.join(DATASET_ROOT, row['file_name']))
            src_paths_to_try.append(os.path.join(DATASET_ROOT, "Images", img_name))
            for chunk_id in range(1, 8):
                src_paths_to_try.append(os.path.join(DATASET_ROOT, "Images", f"chunk_{chunk_id}", img_name))

            src_path = None
            for path in src_paths_to_try:
                if os.path.exists(path):
                    src_path = path
                    break

            if src_path is not None:
                # Определяем, куда положить картинку (первые 100 идут в train, следующие 50 в test)
                if current_species_copied < TRAIN_COUNT:
                    sub_folder = "train"
                else:
                    sub_folder = "test"

                # Собираем итоговый путь: fish-vista_split/train/название_вида/картинка.jpg
                target_folder = os.path.join(OUTPUT_DIR, sub_folder, safe_species_name)
                dst_path = os.path.join(target_folder, img_name)

                os.makedirs(target_folder, exist_ok=True)
                shutil.copy2(src_path, dst_path)
                
                current_species_copied += 1
                total_copied += 1
            else:
                missing_count += 1

        # Выводим мини-прогресс по классам
        if (species_idx + 1) % 10 == 0 or (species_idx + 1) == len(chosen_species):
            print(f"Обработано видов рыб: {species_idx + 1}/{len(chosen_species)}...")

    print("\nСортировка и разделение завершены успешно!")
    print(f"Всего картинок распределено: {total_copied} шт.")
    if missing_count > 0:
        print(f"Предупреждение: {missing_count} файлов из таблицы отсутствовали на вашем диске.")
    
    # Финальная проверка структуры
    train_dir = os.path.join(OUTPUT_DIR, "train")
    test_dir = os.path.join(OUTPUT_DIR, "test")
    if os.path.exists(train_dir):
        print(f"Создана папка обучения: {os.path.abspath(train_dir)} (внутри {len(os.listdir(train_dir))} классов)")
    if os.path.exists(test_dir):
        print(f"Создана папка теста: {os.path.abspath(test_dir)} (внутри {len(os.listdir(test_dir))} классов)")

if __name__ == "__main__":
    sort_and_split_species()
