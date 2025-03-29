cmake_minimum_required(VERSION 3.8)
project(yolo_human_track)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

install(PROGRAMS
  yolo_human_track/yolo_human_track.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME yolo_human_track
)
install(PROGRAMS
  yolo_human_track/pose_transformer.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME pose_transformer
)

install(DIRECTORY
  launch
  params
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
