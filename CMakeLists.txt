#---------------------------------------------------------------------------------------------------
# CMake Build Configuration for Ray Tracing
#---------------------------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.10.0...3.27.0 )

project ( RayTracing LANGUAGES CXX )

# Set to C++17
set ( CMAKE_CXX_STANDARD          17 )
set ( CMAKE_CXX_STANDARD_REQUIRED ON )
set ( CMAKE_CXX_EXTENSIONS        ON )

# Source

set ( CPURENDERER
  src/cpuRenderer.cpp
  src/stb_image.h
  src/aabb.h
  src/camera.h
  src/color.h
  src/constant_medium.h
  src/hittable.h
  src/hittable_list.h
  src/interval.h
  src/material.h
  src/onb.h
  src/pdf.h
  src/perlin.h
  src/quad.h
  src/ray.h
  src/stb_image_utils.h
  src/utils.h
  src/sphere.h
  src/texture.h
  src/vec3.h
)

include_directories(src)

# Specific compiler flags below. We're not going to add options for all possible compilers, but if
# you're new to CMake (like we are), the following may be a helpful example if you're using a
# different compiler or want to set different compiler options.

message (STATUS "Compiler ID: " ${CMAKE_CXX_COMPILER_ID})
message (STATUS "Release flags: " ${CMAKE_CXX_FLAGS_RELEASE})
message (STATUS "Debug flags: " ${CMAKE_CXX_FLAGS_DEBUG})

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # /wd #### - Disable warning
    # /we #### - treat warning as error
    add_compile_options("/W4")      # Enable level-4 warnings
    add_compile_options("/we 4265") # Class has virtual functions, but its non-trivial destructor is not virtual
    add_compile_options("/we 5204") # Class has virtual functions, but its trivial destructor is not virtual
    add_compile_options("/wd 4100") # unreferenced formal parameter
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wnon-virtual-dtor) # Class has virtual functions, but its destructor is not virtual
    add_compile_options(-Wreorder) # Data member will be initialized after [other] data member
    add_compile_options(-Wmaybe-uninitialized) # Variable improperly initialized
    add_compile_options(-Wunused-variable) # Variable is defined but unused
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wnon-virtual-dtor) # Class has virtual functions, but its destructor is not virtual
    add_compile_options(-Wreorder) # Data member will be initialized after [other] data member
    add_compile_options(-Wsometimes-uninitialized) # Variable improperly initialized
    add_compile_options(-Wunused-variable) # Variable is defined but unused
endif()

# Executables
add_executable(cpuRayTracer ${CPURENDERER})

find_package(OpenMP)
target_link_libraries(cpuRayTracer PUBLIC OpenMP::OpenMP_CXX)