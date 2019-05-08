all: main

main: 
	gcc -o nn.out main.c nn.c dataLoader.c -lm -fopenmp

