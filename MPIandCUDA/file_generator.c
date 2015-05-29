#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define FILE_NAME_INDEX 1
#define ITEMS_COUNT_INDEX 2
#define PARAMS_COUNT_INDEX 3

int main(int argc, char** argv)
{
	int params_count;
	int items_count;
	char* file_name;

	if(argc < 4)
	{
		printf("Parameters must be {file name} {number of itmes} {number of parameters}\n");
		return -1;
	}

	items_count = atoi(argv[ITEMS_COUNT_INDEX]);
	params_count = atoi(argv[PARAMS_COUNT_INDEX]);
	file_name = argv[FILE_NAME_INDEX];

	if(items_count < 1 || params_count < 1)
	{
		printf("{number of items} < 1 or {number of parameters} < 1");
		return -1;
	}

	srand(time(NULL));

	float items[items_count * params_count];

	FILE* file = fopen(filename, "w");

    for(int i = 0; i < itemsCount; i++)
    {
        fprintf(file, "%d ", clustersIds[i]);

        for(int j = 0; j < paramsCount; j++)
        {
            fprintf(file, "%f ", items[i * paramsCount + j]);
        }

        fprintf(file,"\n");
    }

    fclose(file);


    FILE* file = fopen(file_name, "wb");

    fwrite(items, sizeof(float), items_count * params_count, file);

    fclose(file);


	return 0;
}