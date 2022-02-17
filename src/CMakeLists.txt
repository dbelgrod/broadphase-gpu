if(STQ_WITH_CUDA)
  set(STQ_CUDA_SOURCES
    # src/queue.cu
    # src/queue.h
    # src/para.h
    # src/para.cu
    # src/Type.hpp
    # src/body.h
    # src/body.cu
    # src/root_finder.h
    # src/root_finder.cu
    src/aabb.cuh
    src/aabb.cu
    src/collision.cu
    src/collision.cuh
    src/simulation.cu
    src/simulation.cuh
    src/groundtruth.cu
    src/groundtruth.cuh
    src/sweep.cu
    src/sweep.cuh
    # src/klee.cu
    # src/klee.cuh
    src/util.cu
    src/util.cuh
    src/queue.cu
    src/queue.cuh
    # src/object.cuh
    src/io.cuh
    src/io.cu
    src/timer.cuh

    PARENT_SCOPE
  )
endif()

################################################################################
# Subfolders
################################################################################

if(STQ_WITH_CPU)
  # add_subdirectory(cpu)
  set(STQ_CPU_SOURCES
    src/cpu/aabb.cpp
    src/cpu/aabb.hpp
    src/cpu/io.cpp
    src/cpu/io.hpp
    src/cpu/sweep.cpp
    src/cpu/sweep.hpp

    PARENT_SCOPE
  )
endif()