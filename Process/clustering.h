#ifndef __CLUSTERING_H
#define __CLUSTERING_H


#include "item.h"
#include "cluster.h"


//get default clustering centers
item* init_clustering_centers(cluster c)
{
	int i;

	item* centers = clone_items(c.items, c.clusters_count, c.params_count);

	for(i = 0; i < c.clusters_count; i++)
	{
		centers[i].cluster_id = i;
	}

	return centers;
}

int get_item_cluster_id(item item1, item* centers, int clusters_count, int params_count)
{
	int i;

	double min_e_distance = evclid_distance(item1, centers[0], params_count);
	int cluster_id = 0;

	for(i = 1; i < clusters_count; i++)
	{
		int new_e_distance = evclid_distance(item1, centers[i], params_count);
		if(min_e_distance > new_e_distance)
		{
			min_e_distance = new_e_distance;
			cluster_id = i;
		}
	}

	return cluster_id;
}


void update_items_cluster_id(cluster c, item* centers)
{
	int i;

	for(i = 0; i < c.items_count; i++)
	{
		c.items[i].cluster_id = get_item_cluster_id(c.items[i], centers, c.clusters_count, c.params_count);
	}
}

item* new_cluster_centers(cluster c)
{
	int i, j, k;
	item* new_centers = alloc_items(c.items_count, c.params_count);

	for(i = 0; i < c.clusters_count; i++)
	{
		int count = 0;

		new_centers[i].cluster_id = i;
		
		for(j = 0; j < c.items_count; j++)
		{
			if(c.items[j].cluster_id == i)
			{
				count++;
				for(k = 0; k < c.params_count; k++)
				{
					new_centers[i].parameters[k] += c.items[j].parameters[k];
				}
			}
		}

		for(k = 0; k < c.params_count; k++)
		{
			new_centers[i].parameters[k] /= count;
		}
	}

	return new_centers;
}

//clustering method
void clustering(cluster c)
{
	item* centers = init_clustering_centers(c);
	item* previous_centers = alloc_items(c.clusters_count, c.params_count);

	output_items(centers, c.clusters_count, c.params_count);

	while(!items_is_equal(centers, previous_centers, c.clusters_count, c.params_count))
	{
		copy_items(previous_centers, centers, c.clusters_count, c.params_count);

		update_items_cluster_id(c, centers);
		
		free(centers);

		centers = new_cluster_centers(c);
	}
}


#endif // __CLUSTERING_H
