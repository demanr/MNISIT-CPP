main: main.o layer.o neuron.o net.o
	g++ -std=c++17 main.o layer.o neuron.o net.o -o main

main.o: main.cpp layer.h neuron.h net.h
	g++ -std=c++17 -c main.cpp -o main.o

net: net.o layer.o neuron.o
	g++ -std=c++17 net.o layer.o neuron.o -o net

net.o: net.cpp net.h
	g++ -std=c++17 -c net.cpp -o net.o

layer: layer.o neuron.o
	g++ -std=c++17 layer.o neuron.o -o layer

layer.o: layer.cpp layer.h
	g++ -std=c++17 -c layer.cpp -o layer.o

neuron: neuron.o
	g++ -std=c++17 neuron.o -o neuron

neuron.o: neuron.cpp neuron.h
	g++ -std=c++17 -c neuron.cpp -o neuron.o