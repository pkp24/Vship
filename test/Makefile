build:
	hipcc basic_metric_math_test.cpp -std=c++17 -I "../include" --offload-arch=native -Wno-unused-result -Wno-ignored-attributes $(shell pkg-config --libs ffms2 zimg) -o basic_metric_math_test

check: build
	./basic_metric_math_test

buildcuda:
	nvcc -x cu basic_metric_math_test.cpp -std=c++17 -I "../include" -arch=native $(subst -pthread,-Xcompiler="-pthread",$(shell pkg-config --libs ffms2 zimg)) -o basic_metric_math_test

checkcuda: buildcuda
	./basic_metric_math_test