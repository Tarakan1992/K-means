#ifndef __ITEM_H
#define __ITEM_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct {
    double* parameters;
    int cluster_id;
} item;

//allocated memory for items list
item* alloc_items(int items_count, int params_count)
{
    int i;
    item* _items;

    _items = malloc(sizeof(item) * items_count);

    for(i = 0; i < items_count; i++)
    {
        _items[i].parameters = malloc(sizeof(double) * params_count);
    }

    return _items;
}

//loaded item from file
void load_item(FILE* file, double* ms, int params_count)
{
    int i;

    for(i = 0; i < params_count; i++)
    {
        fscanf(file, "%lf", &ms[i]);
    }
}

//load items list from file
void load_items(FILE* file, item* items, int items_count, int params_count)
{
    int i; 

    for(i = 0; i < items_count; i++)
    {
        load_item(file, items[i].parameters, params_count);
        items[i].cluster_id = 0;
    }
}

//output item
void output_item(item o_item, int params_count)
{
    int i;
    printf("Cluster Id: %d\n", o_item.cluster_id);
    printf("Parameters:");
    for(i = 0; i < params_count; i++)
    {
        printf(" %f",o_item.parameters[i]);
    }
}

//output items list
void output_items(item* items, int items_count, int params_count)
{
    int i;

    for(i = 0; i < items_count; i++)
    {
        printf("#.%d\n", i + 1);
        output_item(items[i], params_count);
        printf("\n---------------------------------------------\n");
    }
}

//copy item
void copy_item(item* source, item* destination, int params_count)
{
    int i;

    for(i = 0; i < params_count; i++)
    {
        destination->parameters[i] = source->parameters[i];
    }

    destination->cluster_id = source->cluster_id;
}

//copy items list
void copy_items(item* destination, item* source, int items_count, int params_count)
{
    int i;

    for(i = 0; i < items_count; i++)
    {
        copy_item(&source[i], &destination[i], params_count);
    }
}

//return clone item list
item* clone_items(item* source, int items_count, int params_count)
{
    item* _result = alloc_items(items_count, params_count);

    copy_items(_result, source, items_count, params_count);

    return _result;
}

//return true if item1 equal item2
int params_is_equal(item item1, item item2, int params_count)
{
    int i;

    for(i = 0; i < params_count; i++)
    {
        if(abs(item1.parameters[i] - item2.parameters[i]) > 0.00001)
        {
            return 0;
        }
    }

    return 1;
}

//return true if items1 equal items2
int items_is_equal(item* items1, item* items2, int items_count, int params_count)
{
    if(items1 == NULL || items2 == NULL)
    {
        return 0;
    }

    int i;

    for(i = 0; i < items_count; i++)
    {
        if(!params_is_equal(items1[i], items2[i], params_count))
        {
            return 0;
        }
    }

    return 1;
}

void save_item(FILE* file, item item1, int params_count)
{
    int i;

    fprintf(file, "%d ", item1.cluster_id);

    for(i = 0; i < params_count; i++)
    {
        fprintf(file, "%lf ", item1.parameters[i]);
    }
}

void save_items(FILE* file, item* items, int items_count, int params_count)
{
    int i;

    for(i = 0; i < items_count; i++)
    {
        save_item(file, items[i], params_count);
        fprintf(file, "\n");
    }
}

//evclid distances
double evclid_distance(item item1, item item2, int count)
{
    int i;

    double _result = 0;

    for(i = 0; i < count; i++)
    {
        double difference = item1.parameters[i] - item2.parameters[i];
        _result += difference * difference;
    }

    return _result;
}


#endif // __ITEM_H
