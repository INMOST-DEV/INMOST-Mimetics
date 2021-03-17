project(ADMFD)

add_executable(MFDDIFF diffusion.cpp)
add_executable(NMFDDIFF diffusion_nonlinear.cpp)
add_executable(MFDELAST elastic.cpp)

target_link_libraries(MFDDIFF ${INMOST_LIBRARIES})
target_link_libraries(NMFDDIFF ${INMOST_LIBRARIES})
target_link_libraries(MFDELAST ${INMOST_LIBRARIES})

install(TARGETS MFDDIFF EXPORT inmost-targets RUNTIME DESTINATION bin)
install(TARGETS NMFDDIFF EXPORT inmost-targets RUNTIME DESTINATION bin)
install(TARGETS MFDELAST EXPORT inmost-targets RUNTIME DESTINATION bin)
