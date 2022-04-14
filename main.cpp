#pragma GCC target ("avx2")
#pragma GCC optimization ("O3")

#include "TimeTests.h"

int main() {
    TimeTests::random_binomial();
    return 0;
}
