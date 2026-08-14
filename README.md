# Projects
0. - [x] !!!!Hash function
   - [x] Connect logs with targets(remove connection by epoch) hiddenToOutError, MSE & RMSE
   - [x] TF-IDF + униграммы/биграммы || BOW + биграммы
1. - [x] Learning
2. - [x] Saving perceptron config
3. - [ ] # Optimization
4. - [x] Dataset
5. - [x] Reading dataset from file
6. - [x] Dividing code into separate parts(*.hpp, *.cpp)
7. - [x] Упорядочить входной слой в соответствии с порядком idf(в структуру idf добавит индекс в входной слое)
   - [x]       double LeakyReLu(const double &data) {
             return data > 0 ? data : 0.01 * data;
               }
    - [x]    double directiveLeakyReLu(const double &data) {
             return data > 0 ? 1.0 : 0.01;
         }
- [x] Вернуть реальный TF-IDF вместо 1.0
- [x] Слишком большой разброс [-1, 1]
- [x] Создать Dataset в конструкторе Perceptron
- [x] Инициализировать поле vectorize этими данными
- [x] Использовать ЭТУ же vectorize в Learning() и в ProccedString()
- [x] Сделать сервер с этим пересептроном и клиента дающего логи для проверки
- [ ] # Сделать тесты нормально(отдельно от всего проекта) 
