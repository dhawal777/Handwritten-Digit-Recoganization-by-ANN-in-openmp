#include "dataLoader.h"
#include "config.h"

typedef struct Node_{
    double bias;
    double output;
    double backPropValue;
    int numberOfWeights;
    double* weights;
} Node;

typedef struct Layer_{
    int numberOfNodes;
    Node* nodes;
} Layer;

typedef struct Network_{
    Layer inputLayer;
    Layer hiddenLayer;
    Layer outputLayer;
} Network;

void initNetwork(Network* network);
void trainNetwork(Network* network);
void testNetwork(Network *network);
