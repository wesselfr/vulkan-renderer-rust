glslc --version
glslc %~dp0/triangle.vert -o %~dp0/vert.spv
glslc %~dp0/triangle.frag -o %~dp0/frag.spv
pause