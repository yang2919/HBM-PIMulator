add_library(ramulator-memorysystem OBJECT)

target_sources(
  ramulator-memorysystem PRIVATE
  bh_memory_system.h
  memory_system.h

  impl/HBM_PIM_system.cpp

)

target_link_libraries(
  ramulator
  PRIVATE
  ramulator-memorysystem
)