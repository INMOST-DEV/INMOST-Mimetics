project(ADMFD)

add_executable(MFDDIFF diffusion.cpp)
add_executable(NMFDDIFF diffusion_nonlinear.cpp)
add_executable(MFDELAST elastic.cpp)
list(APPEND EXECS
	MFDDIFF
	NMFDDIFF
	MFDELAST)

foreach(EXEC ${EXECS})
	target_link_libraries(${EXEC} ${INMOST_LIBRARIES})
	install(TARGETS ${EXEC} EXPORT inmost-targets RUNTIME DESTINATION bin)
endforeach(EXEC ${EXECS})
