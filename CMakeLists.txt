cmake_minimum_required(VERSION 3.15)
project(dembedder)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pthread -D_GLIBCXX_USE_CXX11_ABI=0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -pthread -fopenmp -O3")


find_package(Boost REQUIRED graph unit_test_framework)
message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
message(STATUS "Boost_VERSION: ${Boost_VERSION}")

include_directories($ENV{ARGH_HOME})
include_directories(${Boost_INCLUDE_DIRS})
add_executable(dembedder main.cpp src/FiniteMetricSpace.cpp src/FiniteMetricSpace.h src/real.h src/MetricGraph.cpp src/MetricGraph.h src/Utils.cpp src/Utils.h src/Algorithms.cpp src/Algorithms.h src/Tree.cpp src/Tree.h src/Random.cpp src/Random.h src/Args.cpp src/Args.h src/VecBinder.cpp src/VecBinder.h src/Geometry.cpp src/Geometry.h)
add_executable(dembedder-test test/TestMetricGraph.cpp src/MetricGraph.cpp src/MetricGraph.h src/Utils.cpp src/Utils.h src/Algorithms.cpp src/Algorithms.h  src/Tree.cpp src/Tree.h src/Random.cpp src/Random.h test/TestUtils.cpp test/TestTree.cpp src/VecBinder.cpp src/VecBinder.h test/TestVecBinder.cpp test/TestGeometry.cpp src/Geometry.cpp src/Geometry.h)
target_link_libraries(dembedder-test LINK_PUBLIC ${Boost_LIBRARIES})