#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define FILE_NAME_1_INDEX 1
#define FILE_NAME_2_INDEX 2
#define ITEMS_COUNT_INDEX 3
#define PARAMS_COUNT_INDEX 4

int main(int argc, char** argv)
{
	int params_count;
	unsigned int items_count;
	char* file_name1;
	char* file_name2;

	items_count = atoi(argv[ITEMS_COUNT_INDEX]);
	params_count = atoi(argv[PARAMS_COUNT_INDEX]);
	
	file_name1 = argv[FILE_NAME_1_INDEX];
	file_name2 = argv[FILE_NAME_2_INDEX];

	if(items_count < 1 || params_count < 1)
	{
		printf("{number of items} < 1 or {number of parameters} < 1");
		return -1;
	}

	FILE* file1 = fopen(file_name1, "rb");
	FILE* file2 = fopen(file_name2, "w");

	int tempInt;
	float* tempFloat = (float*)malloc(params_count * sizeof(float));

    for(int i = 0; i < items_count; i++)
    {
    	fread(&tempInt, sizeof(int), 1, file1);

    	//printf("%d ", tempInt);

    	fprintf(file2, "%d ", tempInt);

    	fread(tempFloat, sizeof(float), params_count, file1);

        for(int j = 0; j < params_count; j++)
        {
        	//printf("%.2f ", tempFloat[j]);
        	fprintf(file2, "%.2f ", tempFloat[j]);
        }

        //printf("\n");
        fprintf(file2, "\n");
    }

    fclose(file1);
    fclose(file2);


	return 0;
}