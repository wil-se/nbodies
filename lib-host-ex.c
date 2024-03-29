#include <stdio.h>
#include "lib-host-common.h"
#include "lib-host-ex.h"
#include <stdlib.h>
#include <math.h>

#define G 6.67e-11
#define dt 10

void compute_ex_forces(){
	set_new_memory();	
	set_new_vectors();
	for(int i=0; i<n; i++){
		new_x[i] = x[i];
		new_y[i] = y[i];
		new_z[i] = z[i];
		new_sx[i] = sx[i];
		new_sy[i] = sy[i];
		new_sz[i] = sz[i];
	}	
	
	for(int j=0; j<n; j++){
		for(int k=0; k<n; k++){
			if(j != k){
				compute_ex_force(j, k);
			}
		}
	}
	set_vectors();
	free_new_memory();
}


void compute_ex_frc(int body1, int body2){
	long double part_force[3];
	long double force[3];
	long double distance;
	long double cubic_distance;
	long double mass_product;
	
	force[0] = 0;
	force[1] = 0;
	force[2] = 0;

	part_force[0] = x[body1] - x[body2];
	part_force[1] = y[body1] - y[body2];
	part_force[2] = z[body1] - z[body2];

	distance = sqrt(pow(part_force[0],2) + pow(part_force[1],2) + pow(part_force[2],2));

	cubic_distance = pow(distance, 3);
	mass_product = mass[body1]*mass[body2];

	force[0] = part_force[0]*(G*mass_product/cubic_distance);
	force[1] = part_force[1]*(G*mass_product/cubic_distance);
	force[2] = part_force[2]*(G*mass_product/cubic_distance);

	new_x[body1] += sx[body1]*dt + (force[0]/mass[body1])*dt*dt*0.5;
	new_y[body1] += sy[body1]*dt + (force[1]/mass[body1])*dt*dt*0.5;
	new_z[body1] += sz[body1]*dt + (force[2]/mass[body1])*dt*dt*0.5;

	new_sx[body1] += (force[0]/mass[body1])*dt;
	new_sy[body1] += (force[1]/mass[body1])*dt;
	new_sz[body1] += (force[2]/mass[body1])*dt;
}

// forza applicata al corpo 2 esercitata dal corpo 1
void compute_ex_force(int body2, int body1){
	long double acc[3] = {0, 0, 0};
	long double force[3] = {0, 0, 0};
	long double distance[3] = {x[body2] - x[body1], y[body2] - y[body1], z[body2] - z[body1]};
	long double unit_vector[3] = {distance[0]/fabs(distance[0]), distance[1]/fabs(distance[1]), distance[2]/fabs(distance[2])};	

	force[0] = -G*((mass[body1]*mass[body2]/pow(distance[0], 2)))*unit_vector[0];
	force[1] = -G*((mass[body1]*mass[body2]/pow(distance[1], 2)))*unit_vector[1];
	force[2] = -G*((mass[body1]*mass[body2]/pow(distance[2], 2)))*unit_vector[2];
	
	acc[0] = force[0]/mass[body2];
	acc[1] = force[1]/mass[body2];
	acc[2] = force[2]/mass[body2];
	
	new_x[body1] += sx[body2]*dt + (acc[0])*dt*dt*0.5;
	new_y[body1] += sy[body2]*dt + (acc[1])*dt*dt*0.5;
	new_z[body1] += sz[body2]*dt + (acc[2])*dt*dt*0.5;
		
	long double new_acc[3] = {0, 0, 0};
	long double new_force[3] = {0, 0, 0};
	long double new_distance[3] = {new_x[body2] - x[body1], new_y[body2] - y[body1], new_z[body2] - z[body1]};
	long double new_unit_vector[3] = {new_distance[0]/fabs(new_distance[0]), new_distance[1]/fabs(new_distance[1]), new_distance[2]/fabs(new_distance[2])};
	
	new_force[0] = -G*((mass[body1]*mass[body2]/pow(new_distance[0], 2)))*new_unit_vector[0];
	new_force[1] = -G*((mass[body1]*mass[body2]/pow(new_distance[1], 2)))*new_unit_vector[1];
	new_force[2] = -G*((mass[body1]*mass[body2]/pow(new_distance[2], 2)))*new_unit_vector[2];

	new_acc[0] = new_force[0]/mass[body2];
	new_acc[1] = new_force[1]/mass[body2];
	new_acc[2] = new_force[2]/mass[body2];
	
	new_sx[body2] += 0.5*(acc[0] + new_acc[0])*dt;
	new_sy[body2] += 0.5*(acc[1] + new_acc[1])*dt;
	new_sz[body2] += 0.5*(acc[2] + new_acc[2])*dt;
}








