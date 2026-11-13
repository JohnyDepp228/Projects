#include "../src/filters.h"
#include <gtest/gtest.h>
#include <cmath>


TEST(FilterClassTests, FilterSlide) {
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


