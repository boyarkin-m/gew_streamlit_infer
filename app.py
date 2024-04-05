# импортируем библиотеки pandas, numpy
import pandas as pd
import numpy as np
# импортируем библиотеку streamlit
import streamlit as st
# импортируем пакет dill
import dill

from scipy import stats
from PIL import Image

with open('model.dl', 'rb') as model_file:
    model = dill.load(model_file)


# функция запуска web-интерфейса
def run():
    image = Image.open('logo.jpg')
    st.sidebar.image(image)
    question = ("В каком режиме вы хотели сделать прогноз, Онлайн\n"
                "(Online) или загрузкой файла данных(Batch)?")
    add_selectbox = st.sidebar.selectbox(question, ("Online", "Batch"))
    sidebar_ttl = ("Прогнозирование просрочки с использованием\n"
                   "метода логистической регрессии.")
    st.sidebar.info(sidebar_ttl)
    st.title("Прогнозирование просрочки:")

    if add_selectbox == "Online":
        RevolvingUtilizationOfUnsecuredLines = st.number_input("RevolvingUtilizationOfUnsecuredLines")
        age = st.number_input('age', step=1)
        NumberOfTime30_59DaysPastDueNotWorse = st.number_input("NumberOfTime30-59DaysPastDueNotWorse", step=1)
        DebtRatio = st.number_input("DebtRatio")
        MonthlyIncome = st.number_input("MonthlyIncome")
        NumberOfOpenCreditLinesAndLoans = st.number_input("NumberOfOpenCreditLinesAndLoans", step=1)
        NumberOfTimes90DaysLate = st.number_input("NumberOfTimes90DaysLate", step=1)
        NumberRealEstateLoansOrLines = st.number_input("NumberRealEstateLoansOrLines", step=1)
        NumberOfTime60_89DaysPastDueNotWorse = st.number_input("NumberOfTime60-89DaysPastDueNotWorse", step=1)
        NumberOfDependents = st.number_input("NumberOfDependents", step=1)
        output = ""

        input_dict = {
            'RevolvingUtilizationOfUnsecuredLines': float(RevolvingUtilizationOfUnsecuredLines),
            'age': int(age),
            'NumberOfTime30-59DaysPastDueNotWorse': int(NumberOfTime30_59DaysPastDueNotWorse),
            'DebtRatio': float(DebtRatio),
            'MonthlyIncome': float(MonthlyIncome),
            'NumberOfOpenCreditLinesAndLoans': int(NumberOfOpenCreditLinesAndLoans),
            'NumberOfTimes90DaysLate': int(NumberOfTimes90DaysLate),
            'NumberRealEstateLoansOrLines': int(NumberRealEstateLoansOrLines),
            'NumberOfTime60-89DaysPastDueNotWorse': int(NumberOfTime60_89DaysPastDueNotWorse),
            'NumberOfDependents': float(NumberOfDependents)
            }

        input_df = pd.DataFrame([input_dict])

        if st.button("Спрогнозировать вероятность просрочки"):
            # выполняем предварительную обработку новых данных
    #         input_df = preprocessing(input_df)
            # вычисляем вероятности для новых данных
            output = model.predict_proba(input_df)[:, 1]
            output = str(output)
            st.success("Вероятность просрочки: {}".format(output))

    if add_selectbox == "Batch":
        file_upload_ttl = ("Загрузите csv-файл с новыми данными\n"
        "для вычисления вероятностей:")
        file_upload = st.file_uploader(file_upload_ttl, type=['csv'])

        if file_upload is not None:
            newdata = pd.read_csv(file_upload)
            # выполняем предварительную обработку новых данных
            # newdata = preprocessing(newdata)
            # вычисляем вероятности для новых данных
            prob = model.predict_proba(newdata)[:, 1]
            # вывод вероятностей на веб-странице
            st.success("Вероятности просрочки для загруженных данных:")
            st.write(prob)


if __name__ == '__main__':
    run()