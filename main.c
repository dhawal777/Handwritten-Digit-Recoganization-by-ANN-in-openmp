#include "nn.h"
#include <omp.h>
#include<time.h>
int main()
{
	// srand( static_cast<unsigned int>(time(NULL)));
    srand( (unsigned int) time(NULL) );
	float start,stop;
	start = omp_get_wtime();
    Network network;
    initNetwork(&network);

    testNetwork(&network);
    // #pragma omp parallel for
    for(int i=0; i<TRAINING_EPOCHS; ++i){
        printf("Training epoch %i/%i\n", i + 1, TRAINING_EPOCHS);
        trainNetwork(&network);
        testNetwork(&network);
    }
    stop = omp_get_wtime();
    float elapse=stop-start;
    printf("\nTime Elapsed=%f",elapse );
    return 0;
}
