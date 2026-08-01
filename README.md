# Projects
0. - [x] !!!!Hash function
   - [ ] Connect logs with targets(remove connection by epoch) hiddenToOutError, MSE & RMSE
   - [x] TF-IDF + униграммы/биграммы || BOW + биграммы
1. - [x] Learning
2. - [x] Saving perceptron config
3. - [ ] OPtimization
4. - [x] Dataset
5. - [x] Reading dataset from file
6. - [ ] Dividing code into separate parts(*.hpp, *.cpp)
7. - [x] Упорядочить входной слой в соответствии с порядком idf(в структуру idf добавит индекс в входной слое)
   - [ ]       double LeakyReLu(const double &data) {
             return data > 0 ? data : 0.01 * data;
               }
    - [ ]    double directiveLeakyReLu(const double &data) {
             return data > 0 ? 1.0 : 0.01;
         }
- [ ] Вернуть реальный TF-IDF вместо 1.0
- [ ] Слишком большой разброс [-1, 1]
- [ ] Создать Dataset в конструкторе Perceptron
- [ ] Инициализировать поле vectorize этими данными
- [ ] Использовать ЭТУ же vectorize в Learning() и в ProccedString()
