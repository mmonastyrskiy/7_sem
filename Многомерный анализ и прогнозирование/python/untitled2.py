# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from scipy.stats import f
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial.distance import mahalanobis


# install openpyxl

def get_train_data(data, features, train_samples=None):  #объявление функции принимает 3 параметра последние не обязательно вводить (3 это обучающая выборка) фичерс это Х загаловки а дата это таблица с данными
    train_data = pd.DataFrame() # создаётся переменная трейн дата( новый дата фрейм дата фрейм это таблица )
    for cls, samples in train_samples.items(): # заходим в цикл переменная cls будет последовательно принемать значения номеров классов то есть каждый цикл это следующий номер класса, sempls бедет принимать список номеров строк которые принадлежат этому классу   точка айтемс это пара значений (ключ и значение) для словаря train samples ключ это номер класса из обучающей выборки а значение это номера строк этого класса из обучающей выборки
        train_samps = data[features].loc[samples] #train_samps это новая таблица в которую мы записываем значения(Х) которое входят в обучающаю выборку (значение номер класса и так до конца)
        train_samps["Class"] = cls # это доп колонка номер класса в которую вносится номер класса к которому принадлежит внесённый X
        train_data = pd.concat([train_data, train_samps]) # сохраняем полоученные строчки в train_data а train_samps в котором они были до этого очищаем
    train_data = train_data.astype({"Class": 'int32'}) #  в train_data присваиваем итоговую таблицу с обучающей выборкой и показываем что в колонке класс данные типа интедгер
    return train_data


def scatter_matrix(samples):
    # является ли подклассом?
    if isinstance(samples, pd.Series):  # проверка если по каким то причинам наши значения признаков имеют тип данных series то их конвертирует в data frame
        samples = samples.to_frame()
    d = samples - samples.mean() # вычитает из значений признаков средние значения признаков
    res = np.zeros((d.shape[1], d.shape[1])) #создаёт матрицу нулей размерностью 9 на 9
    # приводит к виду int: 32, 24, ...
    for _, row in d.iterrows(): # проходимся циклом по каждой строке матрицы D датафрейм(там где от значений - срзнач)
        col = row.to_frame() # берём строчку из таблицы D( которая сейчас имеет вид seria и мы приводим её к виду dataframe)
        res += col @ col.T # матрица из нулей 9х9 берём строчку из датафрейма и умножаем её на неё же только транспонированную и так проходим по всем строчкам
    return res


def classes_scatter_matrix(samples, labels): # передаём samples (значение признаков в обучающей выборке табличкой) и labels (номера классов) shape если 0 то число строк если 1 то число колонок
    A = np.zeros((samples.shape[1], samples.shape[1])) # zeros создаёт матрицу shape на shape то есть создаёт матрицу число признаков на число призноков  (9 на 9) заполненую нулями
    for cls in labels.unique(): # переменная cls счётчик по классам принимает значения уникальных классов то есть у нас от 1 до 7
        A += scatter_matrix(samples[labels == cls]) # В матрицу А прибавляем соответствующие значения из матрицы полученые в результате работы skater matrix для текущего класса
    return A


def find_mahl_sqr_dist(centers, samples, covr):                                                # функция принемает в себя дважды средние значения и один раз матрицу ковариций
    res = pd.DataFrame(index=samples.index, columns=centers.index)                             # создаём новый датафрейм вверху заголовки это номера классов и слева заголовки тоже номера классов
    for i in centers.index:                                                                    # двойной цикл идёт по одной и той же таблице
        for j in samples.index:
            res[i][j] = mahalanobis(centers.loc[i], samples.loc[j], np.linalg.inv(covr)) ** 2  # вычисляется растояние махаланобиса в квадрате.      np.linalg.inv(covr)-возвращает матрицу обратную матрице ковариации       centers.loc[i] и samples.loc[j] возвращают i и j строки таблицы means(ср знач) и это значение записывается в ячейку  ij
    return res


def get_def_coef(lda, features):
    return pd.DataFrame(
        np.vstack([lda.intercept_, lda.coef_.T]),                                # Сложение массивов  intercept и coef( транспонированный ) и получаем датафрейм содержащий в себе данный полученные в результате сложения массивов
        index=["Const"] + features,                                              # по строкам признаки Х1...Х9, coef
        columns=lda.classes_                                                     # по столбцам по номерам классов из дискр анализа
    )


def LDA_predict(lda, x):# принимает в себя результаты линейного дискр анализа и значения признаков всех объектов( исходная таблица только с значениями X1...X9)
    return pd.DataFrame(    # функция считает распределение по классам (классификация масива тестовых векторов Х  )
        lda.predict(x),
        columns=["Class"],
        index=x.index
    )


def LDA_predict_probab(lda, x): # принимает в себя результаты линейного дискр анализа и значения признаков всех объектов( исходная таблица только с значениями X1...X9)
    return pd.DataFrame(       # возвращает апостериорные вероятности классификации в соответствии с каждым классом в массиве тестовых векторов
        lda.predict_proba(x),
        columns=lda.classes_,
        index=x.index
    )


def wilks_lambda(samples, labels):  # samples  - на каждой итерации мы передаём в функцию wilks lmbd dataframe сначала с одной колонкой Х1 постепенно увеличивая число колонок пока не дойдём до конца   labels - колонка с номерами классов
    if isinstance(samples, pd.Series): #Метод isinstance () в Python используется для проверки принадлежности объекта к определенному классу или типу данных. Он принимает два аргумента: объект, который нужно проверить, и класс или тип данных, к которому нужно проверить принадлежность.
        samples = samples.to_frame() #to_frame() — это встроенная функция в библиотеке Pandas, которая используется для конвертации серии в DataFrame
    # определитель матрицы рассеивания
    dT = np.linalg.det(scatter_matrix(samples))
    # определитель классовой матрицы рассеивания
    dE = np.linalg.det(classes_scatter_matrix(samples, labels))
    return dE / dT # их частное и есть Лямбда Уилкса


def f_p_value(lmbd, n_obj, n_sign, n_cls): #  sign это число признаков вошедших в модель lmbd - значение лямбды( число)   n_obj - число объектов в обучающей выборке  n_cls - число классов в обучающей выборке
    num = (1-lmbd)*(n_obj - n_cls - n_sign)
    den = lmbd * (n_cls - 1)
    f_value = num / den
    p = f.sf(f_value, n_cls-1, n_obj-n_cls-n_sign)
    return f_value, p


def forward(samples, labels, f_in = 1e-4):                                            # samples - набор объектов в обучающей выборке !!! labels - колонка с номерами классов  f_in точность( это F to enter в статистике)
    st_columns = ["Wilk's lmbd", "Partial lmbd", "F to enter", "P value"]             # создаётся список названий колонок таблицы
    n_cls = labels.unique().size                                                      #число уникальных классов нашей обучающей выборки
    n_obj = samples.shape[0]                                                          # число  объектов в обучающей выборке
                                                                                      # хранение пременных вне и в модели
    out = {0: pd.DataFrame(columns=st_columns, index=samples.columns, dtype=float)}   # создаётся словарик ключу(ключ показывает какой у нас шаг метода) 0 ставится в соответствие дата фрейм(пустой) с колонками из переменной st_colums и индексами(строки таблицы) Х1,,,Х9
    into = {0: pd.DataFrame(columns=st_columns, dtype=float)}                         # создаётся словарик ключу(ключ показывает какой у нас шаг метода) 0 ставится в соответствие дата фрейм(пустой) с колонками из переменной st_colums
    step = 0                                                                          # шаг нашего метода

    while True:
        model_lmbd = wilks_lambda(samples[into[step].index], labels)  #посчитали Лямбду Уилкса для модели на данном шаге
        #into[step].index представляет собой набор индексов , которые указывают на признаки, выбранные на текущем шаге (step) в процессе отбора признаков.
   # samples[into[step].index] извлекает подмножество DataFrame samples, содержимое которого соответствует только тем признакам, которые были выбраны в текущей итерации
        # Далее проходим по переменным вне модели на данном шаге
        # далее рассчитываем характеристики для данных переменных и записываем их в таблицу
        for el in out[step].index: # el переменная счётчик по списку индексов датафрейма внутри out (Х1 Х2,,, Х9)
            lmbda = wilks_lambda(samples[into[step].index.tolist() + [el]], labels)   # мы с датафрейма samples  берём значение из колонок в квадратных скобках список колонок который есть в into для текущего шага + текущая колонку из счётчика el и эти значения помещаются в функцию вилкс лямбда
            partial_lmbd = lmbda / model_lmbd   #
            f_lmbd, p_value = f_p_value(partial_lmbd, n_obj, into[step].index.size+1, n_cls) #into[step].index.size число признаков вошедших в модель
            out[step].loc[el] = lmbda, partial_lmbd, f_lmbd, p_value
        # расчёт характеристик элементов в модели
        for el in into[step].index:
            lmbda = wilks_lambda(samples[into[step].index.drop(el)], labels)
            partial_lmbd = model_lmbd / lmbda
            f_lmbd, p_value = f_p_value(partial_lmbd, n_obj, into[step].index.size-1, n_cls)
            into[step].loc[el] = lmbda, partial_lmbd, f_lmbd, p_value

        if out[step].index.size == 0 or out[step]["F to enter"].max() < f_in:         # критерий для остановки цикла
        # если вне модели нет переменных ИЛИ новая пере-менная обладает f_to_enter меньше порогового значения, цикл остановлен

            break
        # добавление нового элемента
        el_to_enter = out[step]["F to enter"].idxmax() # ищем элемент с max f_to_enter
                # переносим его из элементов "вне модели" в эле-менты "в модели"
        into[step+1] = into[step]._append(out[step].loc[el_to_enter])
        out[step+1] = out[step].drop(index=el_to_enter)

        step += 1
    return into, out


def backward(samples, labels, f_r = 10.00):
    st_columns = ["Wilk's lmbd", "Partial lmbd", "F to remove", "P value"]
    n_cls = labels.unique().size
    n_obj = samples.shape[0]
    # хранение пременных вне и в модели(е)
    into = {0: pd.DataFrame(columns=st_columns, index=samples.columns, dtype=float)}
    out = {0: pd.DataFrame(columns=st_columns, dtype=float)}
    step = 0

    while True:
        model_lmbd = wilks_lambda(samples[into[step].index], labels)
        # расчёт характеристик элементов вне модели
        for el in out[step].index:
            lmbda = wilks_lambda(samples[into[step].index.tolist() + [el]], labels)
            partial_lmbd = lmbda / model_lmbd
            f_lmbd, p_value = f_p_value(partial_lmbd, n_obj, into[step].index.size, n_cls)
            out[step].loc[el] = lmbda, partial_lmbd, f_lmbd, p_value
        # расчёт характеристик элементов в моделе
        for el in into[step].index:
            lmbda = wilks_lambda(samples[into[step].index.drop(el)], labels)
            partial_lmbd = model_lmbd / lmbda
            f_lmbd, p_value = f_p_value(partial_lmbd, n_obj, into[step].index.size-1, n_cls)
            into[step].loc[el] = lmbda, partial_lmbd, f_lmbd, p_value

        if into[step].index.size == 0 or into[step]["F to remove"].min() > f_r:
            break
        # удаление элемента
        el_to_remove = into[step]["F to remove"].idxmin()
        out[step+1] = out[step]._append(into[step].loc[el_to_remove])
        into[step+1] = into[step].drop(index=el_to_remove)

        step += 1
    return into, out


FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9"]

TRAIN_SAMPLES = {
    1: [3, 5, 7, 9, 13,14,16,17,23,24,26],
    2: [32, 82],
    3: [20,25,28,39,45,68],
    4: [15,19,43,47,60],
    5: [8,21,22,41,42,51]
}

data = pd.read_excel(r'norm_data.xlsx', usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
data = data.loc[range(0, 85)]
print("Данные")
print(data.head())
data_to_excel = data[FEATURES]

train_data = get_train_data(data, FEATURES, TRAIN_SAMPLES)
print("Обучающая выборка")
print(train_data)
data_to_excel["Train sample"] = train_data.Class

cov = pd.DataFrame(
    classes_scatter_matrix(train_data[FEATURES], train_data.Class) / (train_data.shape[0] - train_data.Class.unique()  #  classes_scatter_matrix классовая матрица рассеивания
                                                                      .size),
    index=FEATURES,
    columns=FEATURES
)
print("Ковариационная матрица")
print(cov)

lda = LinearDiscriminantAnalysis().fit(train_data[FEATURES], train_data.Class)  # присваиваем в переменную lda результаты дискриминантного анализа нашей модели
means = pd.DataFrame(lda.means_, index=lda.classes_, columns=FEATURES)          # Полученные в результате дискриминантного анализа средние значения в виде датафрейма с строчками классы и колонками признаки помещаем в переменную means
print("Средние значения")
print(means)

cen_dis = find_mahl_sqr_dist(means, means, cov)
print("Расстояние Махланобиса (обучающая выборка)")
print(cen_dis)

df_coef = get_def_coef(lda, FEATURES)
print("Функции Фишера")
print(df_coef)

print("Pi: ", lda.priors_)

lda_predict = LDA_predict(lda, data[FEATURES])
print("Распределение по классам")
print(lda_predict)
data_to_excel["Result Lda"] = lda_predict

samp_dist = find_mahl_sqr_dist(means, data[FEATURES], cov)
print("Расстояние Махланобиса")
print(samp_dist)

lda_post_prob = LDA_predict_probab(lda, data[FEATURES])
print("Probabilities")
print(lda_post_prob)
#Вызываем результаты работы заданной выше функции:
into, out = forward(train_data[FEATURES], train_data.Class)
print("Forward stepwise")
for i, tab in into.items(): #Выводим результаты работы для переменных "в модели" на экран.
    print("Step: ", i)
    print(tab, end="\n\n")

forw_stepwise = into[len(into) - 4].index.tolist()   #Выводим названия признаков в модели.
              # смотрим каждый шаг  выбираем последний шаг где p-value не превышает 0,05. смотрим сколько признаков на этом шаге вошло в модель
print(forw_stepwise)
#Проводим дискриминантный анализ для отобранных признаков.
forw_stepwise_lda = LinearDiscriminantAnalysis().fit(train_data[forw_stepwise], train_data.Class)
forw_stepwise_coef = get_def_coef(forw_stepwise_lda, forw_stepwise)
print("Функции Фишера")
print(forw_stepwise_coef)
print("Pi: ", forw_stepwise_lda.priors_)
forw_stepwise_pred = LDA_predict(forw_stepwise_lda, data[forw_stepwise])
print("Распределение")
print(forw_stepwise_pred.head())
data_to_excel["Result forward"] = forw_stepwise_pred

into, out = backward(train_data[FEATURES], train_data.Class)
print("Backward stepwise")
for i, tab in into.items():
    print("Step: ", i)
    print(tab, end="\n\n")

back_stepwise = into[len(into) - 1].index.tolist()
print(back_stepwise)
back_stepwise_lda = LinearDiscriminantAnalysis().fit(train_data[back_stepwise], train_data.Class)
back_stepwise_coef = get_def_coef(back_stepwise_lda, back_stepwise)
print("Функции Фишера")
print(back_stepwise_coef)
print("Pi: ", back_stepwise_lda.priors_)
back_stepwise_pred = LDA_predict(back_stepwise_lda, data[back_stepwise])
print("Распределение")
print(back_stepwise_pred.head())
data_to_excel["Result backward"] = back_stepwise_pred

data_to_excel.to_excel(r'/content/Пустой Эксель.xlsx')

