import pandas as pd
import numpy as np
import sklearn

from sklearn.linear_model import LogisticRegression #Логистическая регрессия
from sklearn.model_selection import train_test_split #Для разбиения на тестовый и тренинговый сеты
from sklearn.tree import DecisionTreeClassifier #Древо решений
from sklearn.neural_network import MLPClassifier #Для нейронной сети
from sklearn.model_selection import GridSearchCV#Для настройки Древа
from sklearn.ensemble import RandomForestClassifier #Рандомный лес
df = pd.read_csv('dataset5.csv')
x = df[['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10','Q11','Q13']].values
y = df['Q12'].values

x_train, x_test, y_train, y_test = train_test_split(x,y) #c рандомным разбиением
print('-- Разбиение на тестовый и тренировочный сеты задано! --')
j = 0

while(j < 1):
    print('--------------Меню--------------')
    print('best - наглядно показывает лучшую модель для данного датасета\nfull - обучение всех алгоритмов на данном разбиении\nlbfgs - обучение алгоритма lbfgs\nresplit - переразбиение тестового и тренировочного сетов\nexit - выход')
    choice = input()
    if (choice == 'full'):
        
        print('Логическая регрессия')
        model1 = LogisticRegression(max_iter=1000)
        model1.fit(x_train,y_train)
        s1 = model1.score(x_test,y_test)

        print(model1.score(x_test,y_test))
        print(model1.predict([[1,1,1,1,1,1,1,1,1,1,1,1]]))
        print(model1.predict([[2,2,2,2,2,2,2,2,2,2,2,2]]))

        print('Древо решений')
        model2 = DecisionTreeClassifier()
        model2.fit(x_train,y_train)
        s2 = model2.score(x_test,y_test)

        print(model2.score(x_test,y_test))
        print(model2.predict([[1,1,1,1,1,1,1,1,1,1,1,1]]))
        print(model2.predict([[2,2,2,2,2,2,2,2,2,2,2,2]]))

        print('Рандомный лес')
        model3 = RandomForestClassifier(n_estimators=8)
        model3.fit(x_train,y_train)
        s3 = model3.score(x_test,y_test)

        print(model3.score(x_test,y_test))
        print(model3.predict([[1,1,1,1,1,1,1,1,1,1,1,1]]))
        print(model3.predict([[2,2,2,2,2,2,2,2,2,2,2,2]]))

        print('Нейронная сеть adam')
        model4 = MLPClassifier(max_iter=1000,alpha=0.001,solver='adam')
        model4.fit(x_train,y_train)
        s4 = model4.score(x_test,y_test)

        print(model4.score(x_test,y_test))
        print(model4.predict([[1,1,1,1,1,1,1,1,1,1,1,1]]))
        print(model4.predict([[2,2,2,2,2,2,2,2,2,2,2,2]]))

        print('Нейронная сеть sgd')
        model5 = MLPClassifier(max_iter=5000,alpha=0.001,solver='sgd')
        model5.fit(x_train,y_train)
        s5 = model5.score(x_test,y_test)

        print(model5.score(x_test,y_test))
        print(model5.predict([[1,1,1,1,1,1,1,1,1,1,1,1]]))
        print(model5.predict([[2,2,2,2,2,2,2,2,2,2,2,2]]))

        print('Нейронная сеть lbfgs')
        model6 = MLPClassifier(max_iter=1000,alpha=0.001,solver='lbfgs')
        model6.fit(x_train,y_train)
        s6 = model6.score(x_test,y_test)

        print(model6.score(x_test,y_test))
        print(model6.predict([[1,1,1,1,1,1,1,1,1,1,1,1]]))
        print(model6.predict([[2,2,2,2,2,2,2,2,2,2,2,2]]))



        if ((s1 >= s2) and (s1>=s2) and (s1>=s4) and (s1>=s5) and (s1>=s6)):
            model = LogisticRegression(max_iter=1000)
            s = s1
            model = model1

        if ((s2 >= s1) and (s2>=s3) and (s2>=s4) and (s2>=s5) and (s2>=s6)):
            model = DecisionTreeClassifier()
            s = s2
            model = model2

        if ((s3 >= s1) and (s3>=s2) and (s3>=s4) and (s3>=s5) and (s3>=s6)):
            model = MLPClassifier(max_iter=1000)
            s = s3
            model = model3

        if ((s4 >= s1) and (s4>=s2) and (s4>=s3) and (s4>=s5) and (s4>=s6)):
            model = MLPClassifier(max_iter=1000,alpha=0.01,solver='adam')
            s = s4
            model = model4
        
        if ((s5 >= s1) and (s5>=s2) and (s5>=s3) and (s5>=s4) and (s5>=s6)):
            model = MLPClassifier(max_iter=5000,alpha=0.01,solver='sgd')
            s = s5
            model = model5

        if ((s6 >= s1) and (s6>=s2) and (s6>=s3) and (s6>=s4) and (s6>=s5)):
            model = MLPClassifier(max_iter=1000,alpha=0.01,solver='lbfgs')
            s = s6
            model = model6
    

        print('Лучшая модель этого разбиения:', model)

        i = 0
        cl = 0
        f = open('data.txt','a')
    
        while (i < 1):
            print('clear - очистка данных\ninp - поштучный ввод\nreset - сброс счётчика\nback - выход из режима')
            ex = input()
            if (ex == 'inp'):
                cl = cl + 1
                f = open('data.txt','a')
                print('Ввод параметров клиента '+str(cl))
                q1 = int(input('Номер ответа на вопрос 1 = '))
                q2 = int(input('Номер ответа на вопрос 2 = '))
                q3 = int(input('Номер ответа на вопрос 3 = '))
                q4 = int(input('Номер ответа на вопрос 4 = '))
                q5 = int(input('Номер ответа на вопрос 5 = '))
                q6 = int(input('Номер ответа на вопрос 6 = '))
                q7 = int(input('Номер ответа на вопрос 7 = '))
                q8 = int(input('Номер ответа на вопрос 8 = '))
                q9 = int(input('Номер ответа на вопрос 9 = '))
                q10 = int(input('Номер ответа на вопрос 10 = '))
                q11 = int(input('Номер ответа на вопрос 11 = '))
                q13 = int(input('Номер ответа на вопрос 13 = '))
                res = (model.predict([[q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q13]]))
                if(res == 1):
                    print('Клиент '+str(cl)+' тип: 1 - Решительный')
                    f.write("Клиент "+str(cl)+" тип: 1 - Решительный\n")
                    
                    
                elif (res == 2):
                    print('Клиент '+str(cl)+' тип: 2 - Нерешительный')
                    f.write("Клиент "+str(cl)+" тип: 2 - Нерешительный\n")
                    
                    
                elif (res == 3):
                    print('Клиент '+str(cl)+' тип: 3 - Импульсивный')
                    f.write("Клиент "+str(cl)+" тип: 3 - Импульсивный\n")
                    
                
            elif (ex == 'back'):
                i = 1;
                f.close()
            elif (ex == 'reset'):
                cl = 0;
                print('-- Счётчик сброшен! --')
            
                
            elif (ex == 'clear'):
                print('Вы уверены, что хотите стереть все данные? y/n')
                e = input()
                if (e=='y'):
                    f.close()
                    f = open('data.txt','w')
                    f.write('')
                    f.close()
                    print('')
                    print('-- Данные были стёрты --')
                    print('')
                elif(e == 'n'):
                    i = 0

    elif (choice == 'resplit'):
        x_train, x_test, y_train, y_test = train_test_split(x,y)
        print('')
        print('!-- Датасет был заново разбит на тестовый и тренировочный сеты --!')
        print('')
        
    elif(choice == 'lbfgs'):
        print('Нейронная сеть lbfgs')
        
        model =  MLPClassifier(max_iter=1000,alpha=0.001,solver='lbfgs')
        model.fit(x,y)
        
        print(model.predict([[1,1,1,1,1,1,1,1,1,1,1,1]]))
        print(model.predict([[2,2,2,2,2,2,2,2,2,2,2,2]]))

        i = 0
        cl = 0
        f = open('data.txt','a')
    
        while (i < 1):
            print('dataset - анализ датасета\nclear - очистка данных\ninp - поштучный ввод\nreset - сброс счётчика\nback - выход из режима')
            ex = input()
            if (ex == 'inp'):
                cl = cl + 1
                f = open('data.txt','a')
                print('Ввод параметров клиента '+str(cl))
                q1 = int(input('Номер ответа на вопрос 1 = '))
                q2 = int(input('Номер ответа на вопрос 2 = '))
                q3 = int(input('Номер ответа на вопрос 3 = '))
                q4 = int(input('Номер ответа на вопрос 4 = '))
                q5 = int(input('Номер ответа на вопрос 5 = '))
                q6 = int(input('Номер ответа на вопрос 6 = '))
                q7 = int(input('Номер ответа на вопрос 7 = '))
                q8 = int(input('Номер ответа на вопрос 8 = '))
                q9 = int(input('Номер ответа на вопрос 9 = '))
                q10 = int(input('Номер ответа на вопрос 10 = '))
                q11 = int(input('Номер ответа на вопрос 11 = '))
                q13 = int(input('Номер ответа на вопрос 13 = '))
                res = (model.predict([[q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q13]]))
                if(res == 1):
                    print('Клиент '+str(cl)+' тип: 1 - Решительный')
                    f.write("Клиент "+str(cl)+" тип: 1 - Решительный\n")
                    
                    
                elif (res == 2):
                    print('Клиент '+str(cl)+' тип: 2 - Нерешительный')
                    f.write("Клиент "+str(cl)+" тип: 2 - Нерешительный\n")
                    
                    
                elif (res == 3):
                    print('Клиент '+str(cl)+' тип: 3 - Импульсивный')
                    f.write("Клиент "+str(cl)+" тип: 3 - Импульсивный\n")

                f.close()    
                
            elif (ex == 'back'):
                i = 1;
                f.close()
            elif (ex == 'reset'):
                cl = 0;
                print('-- Счётчик сброшен! --')

            elif(ex == 'dataset'):
                f = open('data.txt','a')
                datas = pd.read_csv('dataset4.csv')
                x_datas = datas[['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10','Q11','Q13']].values
                print(model.predict(x))
                f.write(str(model.predict(x)))
                f.close()

            elif (ex == 'clear'):
                print('Вы уверены, что хотите стереть все данные? y/n')
                e = input()
                if (e=='y'):
                    f.close()
                    f = open('data.txt','w')
                    f.write('')
                    f.close()
                    print('')
                    print('-- Данные были стёрты --')
                    print('')
                elif(e == 'n'):
                    i = 0
        

    elif(choice == 'best'):
        print('Примерное время до окончания: 30 минут')
        m1 = []
        m2 = []
        m3 = []
        m4 = []
        m5 = []
        m6 = []
        m = 0
        sum1 = 0
        sum2 = 0
        sum3 = 0
        sum4 = 0
        sum5 = 0
        sum6 = 0
        score1 = 0
        score2 = 0
        score3 = 0
        score4 = 0
        score5 = 0
        score6 = 0
        model1 = LogisticRegression(max_iter = 1000) 
        model2 = DecisionTreeClassifier()
        model3 = RandomForestClassifier(n_estimators=8)
        model4 = MLPClassifier(max_iter=3000,alpha=0.0001,solver='adam')
        model5 = MLPClassifier(max_iter=5000,alpha=0.0001,solver='sgd')
        model6 = MLPClassifier(max_iter=3000,alpha=0.0001,solver='lbfgs')
        
        for m in range(1000): #Рекомендуемое число итераций = 1000

            x_train, x_test, y_train, y_test = train_test_split(x,y)
            
            model1.fit(x_train,y_train)
            s1 = model1.score(x_test,y_test)
            m1.append(s1)

            
            model2.fit(x_train,y_train)
            s2 = model2.score(x_test,y_test)
            m2.append(s2)

            
            model3.fit(x_train,y_train)
            s3 = model3.score(x_test,y_test)
            m3.append(s3)

            
            model4.fit(x_train,y_train)
            s4 = model4.score(x_test,y_test)
            m4.append(s4)

            
            model5.fit(x_train,y_train)
            s5 = model5.score(x_test,y_test)
            m5.append(s5)

            
            model6.fit(x_train,y_train)
            s6 = model6.score(x_test,y_test)
            m6.append(s6)

            print(str((m+1)/10) + '%')
            
        for m in range(1000):
            sum1 = sum1 + m1[m]
            sum2 = sum2 + m2[m]
            sum3 = sum3 + m3[m]
            sum4 = sum4 + m4[m]
            sum5 = sum5 + m5[m]
            sum6 = sum6 + m6[m]
        

        score1 = sum1 / 1000
        score2 = sum2 / 1000
        score3 = sum3 / 1000
        score4 = sum4 / 1000
        score5 = sum5 / 1000
        score6 = sum6 / 1000

        print('-- Точности моделей машинного обучения --')

        print('Логистическая регрессия: '+str(score1))
        print('Древо решений: '+str(score2))
        print('Рандомный лес: '+str(score3))
        print('Нейронная сеть алгоритм adam: '+str(score4))
        print('Нейронная сеть алгоритм sgd: '+str(score5))
        print('Нейронная сеть алгоритм lbfgs: '+str(score6))
    
    elif (choice == 'exit'):
        j = 1;







