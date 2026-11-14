#include "../src/filters.h"
#include "../src/maps.h"
#include <gtest/gtest.h>
#include <cmath>


TEST(FilterClassTests, FilterSlideSize) {
    Filters f;
    std::vector<double> bigMat(16,2.0);

    std::vector<double> smallMat(4,4.0);

    f.SetStride(1);

    std::vector<double> res = f.Slide(bigMat, smallMat);

    int size = ((std::sqrt(bigMat.size()) - std::sqrt(smallMat.size())) / f.GetStride()) + 1;

    std::vector<double> target(size * size,32.0);

    ASSERT_EQ(res.size(), target.size());


    for (size_t i = 0; i < target.size(); ++i) {
        EXPECT_DOUBLE_EQ(res[i], target[i]) << "Mismatch at index " << i;
    }
}

TEST(FilterClassTests, FilterSlideValues) {
    Filters f;
    std::vector<double> bigMat(16,2.0);

    std::vector<double> smallMat(4,4.0);

    f.SetStride(1);

    std::vector<double> res = f.Slide(bigMat, smallMat);

    int size = ((std::sqrt(bigMat.size()) - std::sqrt(smallMat.size())) / f.GetStride()) + 1;

    std::vector<double> target(size * size,32.0);


    for (size_t i = 0; i < target.size(); ++i) {
        EXPECT_DOUBLE_EQ(res[i], target[i]) << "Mismatch at index " << i;
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

    std::vector<double> temp = f.Slide(map, f.GetFilter());

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

    std::vector<double> temp = f.Slide(map, f.GetFilter());

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

    std::vector<double> temp = f.Slide(map, f.GetFilter());

    temp =  m.ApplyActivateFunction(temp);

    temp = f.Slide(temp,f.GetFilter());

    temp =  m.ApplyActivateFunction(temp);

    std::vector<double> res =  m.Pooling(temp);

    ASSERT_EQ(res.size(), 1);
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




