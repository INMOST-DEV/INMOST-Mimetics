project(MFDES)
set(SOURCE main.cpp)

add_executable(MFDES ${SOURCE})

target_link_libraries(MFDES ${INMOST_LIBRARIES})

install(TARGETS MFDES EXPORT inmost-targets RUNTIME DESTINATION bin)
