# Projects

# Speed 
04.09.26 - 19 sec 

09.09.26 - 3 sec

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
12. - [ ] Increase speed of convulation(Add Multi threads)
13. - [ ] Add learning
14. - [x] Test new architecture of storing channels
15.  - [x] Pooling and conv of blocks 2,3...(sum of channel) in new architecture

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
