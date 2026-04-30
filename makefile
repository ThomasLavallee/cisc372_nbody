FLAGS= -DDEBUG
LIBS= -lm
ALWAYS_REBUILD=makefile

nbody: nbody.o compute.o kernels.o
	nvcc $(FLAGS) $^ -o $@ $(LIBS)
nbody.o: nbody.cu planets.h config.h vector.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -c $< 
compute.o: compute.cu config.h vector.h kernels.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -c $< 
kernels.o: kernels.cu kernels.h config.h vector.h
	nvcc $(FLAGS) -c kernels.cu -o kernels.o 

clean:
	rm -f *.o nbody 
