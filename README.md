# Plan 

## Filters
- [x] Filters slide 

- [x] Filters init

- [x] Biases init / getters / setters

- [x]Learning

### Maps of signs

- [x] GeLu 

- [x] GMP 

- [x] Pooling

- [x] Map of errors for prev layer

### Channels 

- [ ] Padding & Stride logic

- [ ] Combine Filters + Maps of signs (Forward)

- [ ] Conv Backward (Расчет градиентов фильтров dW и сдвигов db)

- [ ] Input error pass to convolution layer (dx для предыдущего слоя)

## Classifier

- [ ] Flatten layer (Maps to 1D Vector)

- [ ] FC Weights & Biases Init

- [ ] Forward

- [ ] Softmax / Sigmoid activation (Выходной слой)

- [ ] Backpropagation (FC layer gradients)

## Loss & Training

- [ ] Cross-Entropy Loss (Прямой ход и градиент)

- [ ] Optimizer (SGD / Adam)

- [ ] Train Loop (Эпохи)


## Tests

- [x] FilterClassTests.FilterSlideSize

- [x] FilterClassTests.FilterSlideValues

- [x] MapClassTests.PooledMapSize

- [x] MapClassTests.PooledMapValues

- [x] MapClassTests.PooledMapValuesIdx

- [x] MapClassTests.GMP

- [x] Forward.MapSizeOF16AfterForward

- [x] Forward.MapSizeOF32AfterForward

- [x] Forward.MapSizeOF32After2Forward

- [x] ActivateFunction.GeLu

- [x] ActivateFunction.GeLuNegative

- [x] ActivateFunction.GeLuZero

- [x] Overrided operator in maps

- [x] FindMapForLearning in

- [x] FindMapForPrevLayer(Filters)

- [x] Padding(Filters)