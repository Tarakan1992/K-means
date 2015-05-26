#include "item.h"
#include "time.h"
#include "clustering.h"

#define get_time_period(start, end) ((double)(end-start))/CLOCKS_PER_SEC
#define FILE_NAME_INDEX  1
#define ITEMS_COUNT_INDEX  2
#define PARAMS_COUNT_INDEX  3
#define clusters_count_INDEX 4

int main(int argc, char *argv[])
{
    char *file_name;
    int params_count, items_count, clusters_count;
    time_t start, end;

    if(argc < 5)
    {
        printf("\nINVALID INPUT PARAMETERS\n");
        return -1;
    }

    items_count = atoi(argv[ITEMS_COUNT_INDEX]);
    params_count = atoi(argv[PARAMS_COUNT_INDEX]);
    clusters_count = atoi(argv[clusters_count_INDEX]);
    file_name = argv[FILE_NAME_INDEX];

    if(items_count < 1 || params_count < 1 || clusters_count < 1)
    {
        printf("INVALID INPUT PARAMETERS\n");
        return -1;
    }

    if(clusters_count >= items_count)
    {
        printf("NUMBER OF CLUSTERS SHOULD BE NO MORE THAN THE NUMBER OF ITEMS\n");
        return -1;
    }
    
    start = clock();


    //-------------------------------------------------------------------
    //PROGRAMM EXECUTION
    //-------------------------------------------------------------------

    cluster _cluster = load_cluster(file_name, items_count, params_count, clusters_count);
    
    //output_cluster(_cluster);
    clustering(_cluster);

    save_cluster("output.txt", _cluster);

    //-------------------------------------------------------------------
    end = clock();
    printf("Execution time: %f ms \n", get_time_period(start, end));

    return 0;
}