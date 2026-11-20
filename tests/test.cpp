#include "../src/filters.h"
#include "../src/maps.h"
#include <gtest/gtest.h>
#include <cmath>
#include <iostream>


TEST(FilterClassTests, FilterSlideSize) {
    Filters f;
    std::vector<double> bigMat(16,2.0);

    std::vector<double> smallMat(4,4.0);

    f.SetStride(1);

    std::vector<double> res = f.Slide(bigMat, smallMat,1);

    int size = ((std::sqrt(bigMat.size()) - std::sqrt(smallMat.size())) / f.GetStride()) + 1;

    std::vector<double> target(size * size,32.0);

    ASSERT_EQ(res.size(), target.size());
    double bias = 0.01;

    for (size_t i = 0; i < target.size(); ++i) {
        EXPECT_DOUBLE_EQ(res[i] , target[i] + bias) << "Mismatch at index " << i;
    }
}


TEST(MapClassTests, PooledMapSize) {
    Maps m;

    m.SetStride(2);

    std::vector<double> map(16,1.0);

    std::vector<double> target(4,1.0);

    std::vector<double> res = m.Pooling(map);

    ASSERT_EQ(res.size(), target.size());
}

TEST(MapClassTests, PooledMapValues) {
    Maps m;

    m.SetStride(2);

    std::vector<double> map = { 1.0,1.0,1.0,1.0,
                                1.0,2.0,2.0,1.0,
                                1.0,1.0,1.0,1.0,
                                1.0,2.0,2.0,1.0,
                              };

    std::vector<double> target(4,2.0);

    std::vector<double> res = m.Pooling(map);

    for (size_t i = 0; i < target.size(); ++i) {
        EXPECT_DOUBLE_EQ(res[i], target[i]) << "Mismatch at index " << i;
    }
}


TEST(MapClassTests, PooledMapValuesIdx) {
    Maps m;

    m.SetStride(2);

    std::vector<double> map = { 1.0,1.0,1.0,1.0,
                                1.0,2.0,2.0,1.0,
                                1.0,1.0,1.0,1.0,
                                1.0,2.0,2.0,1.0,
                              };

    std::vector<int> target = {5,6,13,14};

    m.Pooling(map);

    std::vector<int> res = m.GetPooledIdx();

    for (size_t i = 0; i < target.size(); ++i) {
        EXPECT_EQ(res[i], target[i]) << "Mismatch at index " << i;
    }
}


TEST(Forward, MapSizeOF16AfterForward) {
    std::vector<double> map(16,1.0);
    Maps m;
    Filters f;
    f.SetStride(1);
    f.SetHeight(3);
    f.SetWidth(3);

    
    f.FillFilter();
    m.SetStride(2);

    std::vector<double> temp = f.Slide(map, f.GetFilter(),1);

    temp =  m.ApplyActivateFunction(temp);

    std::vector<double> res =  m.Pooling(temp);

    ASSERT_EQ(res.size(), 1);
}

TEST(Forward, MapSizeOF32AfterForward) {
    std::vector<double> map(49,1.0);
    Maps m;
    Filters f;
    f.SetStride(1);
    f.SetHeight(3);
    f.SetWidth(3);

    
    f.FillFilter();
    m.SetStride(2);

    std::vector<double> temp = f.Slide(map, f.GetFilter(),1);

    temp =  m.ApplyActivateFunction(temp);

    std::vector<double> res =  m.Pooling(temp);

    ASSERT_EQ(res.size(), 4);
}

TEST(Forward, MapSizeOF32After2Forward) {
    std::vector<double> map(49,1.0);
    Maps m;
    Filters f;
    f.SetStride(1);
    f.SetHeight(3);
    f.SetWidth(3);

    
    f.FillFilter();
    m.SetStride(2);

    std::vector<double> temp = f.Slide(map, f.GetFilter(),1);

    temp =  m.ApplyActivateFunction(temp);

    temp = f.Slide(temp,f.GetFilter(),1);

    temp =  m.ApplyActivateFunction(temp);

    std::vector<double> res =  m.Pooling(temp);

    ASSERT_EQ(res.size(), 1);
}

TEST(MapClassTests, GMP) {
    Maps m;

    std::vector<double> map = { 1.0,1.0,1.0,1.0,
                                1.0,2.0,1.0,1.0,
                                1.0,1.0,1.0,1.0,
                                1.0,1.0,1.0,1.0,
                              };

    int idx = 5;
    double GMPtarget = 2.0;
    m.SetMap(map);
    m.GMP();
    m.GetGMPIdx();
    
    EXPECT_DOUBLE_EQ( m.GMP(),GMPtarget);
    EXPECT_EQ( m.GetGMPIdx(), idx);
}

TEST(ActivateFunction, GeLu){
    Maps m;

    double data = 3.2;

    double res = m.GeLu(data);

    double target = 3.198;

    EXPECT_NEAR(res,target,0.01);
}

TEST(ActivateFunction, GeLuNegative){
    Maps m;

    double data = -3.2;

    double res = m.GeLu(data);

    double target = -0.002199;

    EXPECT_NEAR(res,target,0.01);
}

TEST(ActivateFunction, GeLuZero){
    Maps m;

    double data = 0;

    double res = m.GeLu(data);

    double target = 0;

    EXPECT_NEAR(res,target,0.01);
}

TEST(OverrideOperators, Multiplication){
    Maps m;

    std::vector<double> testMap(4,2.0);

    std::vector<double> testMap2(4,2.0);

    std::vector<double> expected(4, 4.0);

    m.SetMapBeforeActivate(testMap);

    std::vector<double> res = m * testMap2;

    ASSERT_EQ(res.size(), expected.size());
    for (size_t i = 0; i < res.size(); ++i) {
        EXPECT_NEAR(res[i], expected[i], 0.01);
    }
}

TEST(MapClassTests, FindMapForLearningFiltersSize) {
    Maps m;
        Filters f;
        f.SetStride(1);
        f.SetHeight(3);
        f.SetWidth(3);
        f.FillFilter();

       
        std::vector<double> inputMap = {
            1.0, 1.0, 1.0, 1.0,
            1.0, 3.0, 1.0, 3.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 3.0, 1.0, 3.0
        };

      
        std::vector<double> convMap = f.Slide(inputMap, f.GetFilter(),1);
        m.ApplyActivateFunction(convMap);

        
        std::vector<double> pooledMap = m.Pooling(convMap);

       
        std::vector<double> errorMap = { 0.5 };
        std::vector<double> temp1 = m.ReversePooling(errorMap); 

        
        std::vector<double> filtersError = f.Slide(inputMap, temp1,0);
        std::vector<double> filter = f.GetFilter();
        ASSERT_EQ(filtersError.size(), filter.size());
}

TEST(FiltersTest, FindMapForPrevLayer) {
    Filters f;
    f.SetStride(1);
    f.SetHeight(3);
    f.SetWidth(3);
    f.FillFilter();
    std::vector<double> errorMap = { 1.0, 1.0, 1.0, 1.0 };
    std::vector<double> prevError;
    try{
        prevError = f.FindMapForPrevLayer(errorMap);
    }
    catch(std::string *msg){
        std::cout << msg;
    }

    std::vector<double> expected = { 16 , 1.0 };

    ASSERT_EQ(prevError.size(), expected.size());
}

TEST(FiltersTest, Padding) {
    Filters f;
    f.SetStride(1);
    f.SetHeight(3);
    f.SetWidth(3);
    f.FillFilter();
    std::vector<double> errorMap = { 4 , 1.0 };

    std::vector<double> prevError = f.Padding(errorMap);

    std::vector<double> expected = { 8 , 1.0 };

    ASSERT_EQ(errorMap.size(), expected.size());
}


