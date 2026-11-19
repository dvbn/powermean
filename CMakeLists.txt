cmake_minimum_required(VERSION 3.18)
project(powermean LANGUAGES CXX)

option(POWERMEAN_BUILD_IOLS "Build the optional compiled no-FE IOLS kernel" ON)
if(POWERMEAN_BUILD_IOLS)
  add_subdirectory(src/powermean/_native/iols)
endif()

option(POWERMEAN_BUILD_BASIS "Build the optional exact FE basis accelerator" OFF)
if(POWERMEAN_BUILD_BASIS)
  add_subdirectory(src/powermean/_native/basis)
endif()
