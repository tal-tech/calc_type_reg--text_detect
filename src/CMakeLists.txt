cmake_minimum_required(VERSION 3.10)

project(CalcTypeReg)
set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/package")
#set(PACKVERSION 2.0.0.1)
######################################################################
#全局编译标志
######################################################################

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

set(CUDA_cublas_device_LIBRARY /usr/local/cuda/lib64)
######################################################################
#全局链接和包含
######################################################################

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/3rdParty)
include_directories(${CMAKE_SOURCE_DIR}/3rdParty/lib)
include_directories(${CMAKE_SOURCE_DIR}/3rdParty/opencv/include)
include_directories(${CMAKE_SOURCE_DIR}/ai_model/include)
include_directories(${CMAKE_SOURCE_DIR}/ai_model/include/darknet)

link_directories(${CMAKE_SOURCE_DIR}/3rdParty/eureka/lib)
link_directories(${CMAKE_SOURCE_DIR}/3rdParty/opencv/lib)
link_directories(${CMAKE_SOURCE_DIR}/ai_model/lib)

find_package(Boost REQUIRED)
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif()

#################################################################L#####
#全局编译输出指定
######################################################################

#IF(CMAKE_BUILD_TYPE MATCHES Debug)
set(CMAKE_VERBOSE_MAKEFILE ON)
#ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)

#string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/${BUILD_TYPE})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/${BUILD_TYPE})



#编译对象
######################################################################
add_executable(CalcTypeReg
    demo.cpp
    ${DIR_COMMON_SRCS}
)

if(Boost_FOUND)
    target_link_libraries(CalcTypeReg ${Boost_LIBRARIES})
endif()

target_include_directories(CalcTypeReg PRIVATE
    ${PROJECT_SOURCE_DIR}/3rdParty/eureka)

target_link_libraries(CalcTypeReg
        jsoncpp
        crypto
        boost_system
        opencv_core
        opencv_imgcodecs
        jieti
        cls_num_rec
    )

######################################################################
#程序发布设置
######################################################################

install(DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ DESTINATION bin)
install(DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/ DESTINATION lib)
install(DIRECTORY 3rdParty/librdkafka/lib DESTINATION .)
install(DIRECTORY 3rdParty/eureka/lib DESTINATION .)
install(DIRECTORY config/ DESTINATION bin/config)
install(DIRECTORY 3rdParty/opencv/lib DESTINATION .)
install(DIRECTORY ai_model/lib DESTINATION .)
install(DIRECTORY ai_model/model DESTINATION bin)
