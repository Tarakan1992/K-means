#ifndef __CLUSTER_H
#define __CLUSTER_H

#include "item.h"

typedef struct 
{
    item* items;
    int items_count;
    int params_count;
    int clusters_count;
} cluster;

//output items list to console
void output_cluster(cluster c)
{
    printf("\n---------------------------------------------\n");
    printf("Number of clusters: %d", c.clusters_count);
    printf("\n---------------------------------------------\n");
    output_items(c.items, c.items_count, c.params_count);
}

//loaded items list from file
cluster load_cluster(char* file_name, int items_count, int params_count, int clusters_count)
{
    cluster _cluster;

    _cluster.clusters_count = clusters_count;
    _cluster.params_count = params_count;
    _cluster.items_count = items_count;

    _cluster.items = alloc_items(items_count, params_count);

    FILE* _file = fopen(file_name, "r");

    load_items(_file, _cluster.items, _cluster.items_count, _cluster.params_count);

    fclose(_file);

    return _cluster;
}

void save_cluster(char* file_name, cluster c)
{
    FILE* _file = fopen(file_name, "w");

    save_items(_file, c.items, c.items_count, c.params_count);

    fclose(_file);

}

#endif // __CLUSTER_H
