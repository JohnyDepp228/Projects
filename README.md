# Projects

# Speed 
04.09.26 - 19 sec 

# TODO list
1. - [x] Сonvolution
2. - [x] Pooling(Min,Max)
3. - [X] GAP
4. - [X] Block of Сonvolution and MaxPooling
5. - [x] Add image decomposition
6. - [x] Save map of signs into channel
7. - [x] First channel done
8. - [x] Check names of functions and variables(Grammar)
9. - [x] Padding
10. - [x] Bilinear interpolation
11. - [x] Change pooling with new function
12. - [ ] Increase speed of convulation
13. - [ ] Add learning
14. - [ ] Test new architecture of storing channels

# New architecture of storing channels
                        struct Channel[m]
                       /                 \
                      /                   \
              struct Filter[n]      struct MapOfSigns[n]
              /                                  \
             /                                    \
    std::vector<double> filter               std::vector<double> mapOfSigns



# Test LIST
- [ ] Test Сonvolution and MaxPooling/MinPooling
- [ ] Test GAP
