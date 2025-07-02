add_library(ramulator-controller OBJECT)

target_sources(
  ramulator-controller PRIVATE
  bh_controller.h 
  bh_scheduler.h 
  controller.h 
  scheduler.h 
  plugin.h
  refresh.h
  rowpolicy.h

  impl/HBM_PIM_controller.cpp
  
  impl/scheduler/bh_scheduler.cpp
  impl/scheduler/blocking_scheduler.cpp
  impl/scheduler/generic_scheduler.cpp
  impl/scheduler/bliss_scheduler.cpp
  impl/scheduler/prac_scheduler.cpp

  impl/refresh/all_bank_refresh.cpp
  
  impl/rowpolicy/basic_rowpolicies.cpp
)

target_link_libraries(
  ramulator
  PRIVATE
  ramulator-controller
)