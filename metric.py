import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Функция расчёта метрики для рекомендатеьной системы
def metric(df_users,  # Предварительно рассчитанный дата фрейм с интересами всех пользователей
           games_popularity,  # Предварительно рассчитанная матрица популярности для всех игр
           games_streamers,  # Предварительно рассчитанный датафрейм с парами Игра - Стример
           rec_system,      # Объект класса рекомендательной системы, имеющий функцию pridict(user_id :int, . . .) и возвращающий list индексов предсказанных стримеров
           interface=False,  # Нужно ли вам выводить даннные для каждого пользователя отдельно?
           save_path='',  # Путь для сохранения расчитанных матриц
           time_calculate=1  # Время в минутах, которое вы хотите выделить на рассчёты
           ):
    # Инициализация матрицы корреляций
    corellation_matrix = pd.DataFrame(data=0, index=pd.unique(games_streamers['game']),
                                      columns=pd.unique(games_streamers['game']))
    # Счётчики для усреднения различных характеристик
    n = 0
    m = 0
    # Начало отсчёта времени для завершения функции по времени
    start_time = time.time()
    # Рассчитываемые характеристики предсказаний
    mean_procent_new_streamers = 0
    mean_procent_new_games = 0
    mean_procent_interest = 0

    # Начало рассчёта метрики
    for user in df_users['ID_user']:
        n += 1

        # Делаем предикт для конкретного пользователя
        # ++++++++++++++++++++++++++++++++++++++++
        predict = rec_system.predict(user, n=30, interface=False, debug_mode=False)  # Ввести свои данные для работы данного метода, но user должен передаваться
        # ++++++++++++++++++++++++++++++++++++++++

        # Поиск пересечения между интересами пользователя и предсказанием (Поиск известных пользователю стримеров)
        old_streamers = set(predict).intersection(
            set(df_users[df_users['ID_user'] == user]['array_streamers'].values[0]))
        if interface:
            print(f'Количсетво стримеров в предикте: {len(predict)}')
            print(
                f"Количество стримеров в истории пользователя: {len(set(df_users[df_users['ID_user'] == user]['array_streamers'].values[0]))}")
            print(f'Процент новых стримеров: {round((1 - len(old_streamers) / len(predict)) * 100, 2)}%')

        # Счётчик для усреднённых результатов
        mean_procent_new_streamers += round((1 - len(old_streamers) / len(predict)) * 100, 2)

        # =================================================================================================================#
        # Поиск игр, которые стримят спредикченные стримеры
        predict_games = []
        for i in predict:
            game = games_streamers[games_streamers["streamId"] == int(i)]["game"]
            if game.shape[0] > 0:
                predict_games.append(game.values[0])

        # Воборка игр, которые пользователь смотрел ранее
        old_games = set(df_users[df_users['ID_user'] == user]['game'].values[0])

        if len(old_games) != 0:  # Проверка, что у пользователя не нулевая история найденных игр
            m += 1
            # Поиск пересечения игр интересных пользователю и полученных из предикта
            intersection_games = set(predict_games).intersection(old_games)

            if interface:
                print(
                    f'Процент новых игр в предикте: {round((1 - len(intersection_games) / len(predict_games)) * 100, 2)}%')

            # Счётчик для усреднённых результатов
            mean_procent_new_games += round((1 - len(intersection_games) / len(predict_games)) * 100, 2)

            if interface:
                print(
                    f'Насколько игры совпадают с интересами: {round(len(intersection_games) / len(old_games) * 100, 2)}%')

            # Счётчик для усреднённых результатов
            mean_procent_interest += round(len(intersection_games) / len(old_games) * 100, 2)

        # =================================================================================================================#
        # Рассчёт метрики корреляции
        corellation_matrix.loc[(corellation_matrix.index.isin(
            df_users[df_users['ID_user'] == user]['game'].values[0])), corellation_matrix.columns.isin(
            predict_games)] += 1

        # if n == 100: # Вариант, если захотите заканчивать рассчёт метрики по количеству проверенных пользователей
        if time.time() - start_time > time_calculate * 60:  # Вариант, если захотите заканчивать рассчёт метрики по времени работы
            print(f"Кол-во пройденых пользователй: {n}")
            break

    # Вывод результатов
    print('==============================================================================')
    print(f'Средний процент новых стримеров в предсказании: {mean_procent_new_streamers / n}%')
    print(f'Средний процент новых игр в предсказании: {mean_procent_new_games / m}%')
    print(f'Средний процент совпадения предсказания с интересами пользователя: {mean_procent_new_games / m}%')
    print('==============================================================================')

    # Вывод не нормированной матрицы корреляций
    scale = 100
    sns.set(rc={'figure.figsize': (30, 25)})
    games = games_streamers.groupby('game').count().sort_values(by='streamId', ascending=False).index
    sns.heatmap(corellation_matrix.loc[(games[1:scale]), games[1:scale]], annot=False, cmap="YlGnBu", linecolor='white',
                linewidth=1)
    plt.show()

    df_normalize = pd.DataFrame(corellation_matrix.values / games_popularity.values)

    df_normalize.index = games_popularity.index
    df_normalize.columns = games_popularity.columns

    # Сохранение матриц
    corellation_matrix.to_json(save_path + 'corellation_matrix_60.json')
    df_normalize.to_json(save_path + 'df_normalize_60.json')
    # Вывод нормированной матрицы корреляций
    sns.heatmap(df_normalize.loc[(games[1:scale]), games[1:scale]], annot=False, cmap="YlGnBu", linecolor='white',
                linewidth=1)
    plt.show()

    df1 = pd.read_json(save_path + 'df_normalize_60.json')
    df1.drop(columns=['Другое'], inplace=True)

    df_vals = df1.values
    indexes_v = np.where(np.sum(df_vals, axis=0) > np.mean(np.sum(df_vals, axis=0)) / 3)[0]
    indexes_g = np.where(np.sum(df_vals, axis=1) > np.mean(np.sum(df_vals, axis=1)) / 3)[0]

    df_vals = df_vals[:, indexes_v]
    df_vals = df_vals[indexes_g, :]

    indexes_df = df1.index.values[indexes_g]
    columns = df1.columns.values[indexes_v]

    scale = min(scale, df_vals.shape[1])

    for column_id in range(scale):
        # ======================================
        if column_id == 0:
            replace_column = np.argmax(np.sum(df_vals, axis=0))
        else:
            replace_column = np.argmin(
                np.sum(np.abs(df_vals[:, column_id:].T - df_vals[:, column_id - 1]).T, axis=0)) + column_id
        # ======================================
        df_vals[:, replace_column], df_vals[:, column_id] = np.copy(df_vals[:, column_id]), np.copy(
            df_vals[:, replace_column])
        # ======================================
        columns[replace_column], columns[column_id] = columns[column_id], columns[replace_column]

        # Вывод матрицы отсортированной по предсказаниям
    df2 = pd.DataFrame(df_vals, columns=columns, index=indexes_df)
    sns.heatmap(df2.iloc[0:scale, 0:scale], annot=False, cmap="YlGnBu", linecolor='white', linewidth=1)
    plt.show()

    for column_id in range(scale * 2):
        # ======================================
        if column_id == 0:
            replace_column = np.argmax(np.sum(df_vals, axis=1))
        else:
            replace_column = np.argmin(
                np.sum(np.abs(df_vals[column_id:, :] - df_vals[column_id - 1, :]), axis=1)) + column_id
        # ======================================
        df_vals[replace_column, :], df_vals[column_id, :] = np.copy(df_vals[column_id, :]), np.copy(
            df_vals[replace_column, :])
        # ======================================
        indexes_df[replace_column], indexes_df[column_id] = indexes_df[column_id], indexes_df[replace_column]

        # Вывод матрицы отсортированной по предсказаниям и пользовательским интересам
    df2 = pd.DataFrame(df_vals, columns=columns, index=indexes_df)
    sns.heatmap(df2.iloc[0:scale, 0:scale], annot=False, cmap="YlGnBu", linecolor='white', linewidth=1)
    plt.show()
    return None


# Ввести пути к соответствующим файлам
path = 'datasets/'
df_users_preprocessing = pd.read_json(path + 'df_users_preprocessing.json')
df_game_popularity = pd.read_json(path + 'df_game_popularity.json')
games_streamers = pd.read_csv(path + 'games_streamers.csv')
games_streamers.drop(columns=['Unnamed: 0'], inplace=True)

# Вызов функции
metric(df_users_preprocessing, df_game_popularity, games_streamers, rec_system, save_path='datasets/', time_calculate=60)