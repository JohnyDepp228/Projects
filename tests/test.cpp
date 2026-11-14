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




