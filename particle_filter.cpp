/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	// std[]: GPS measurement uncertainty [x [m], y [m], theta [rad]]

	std::default_random_engine gen;

	// create a normal (Gaussian) distribution for x, y, theta
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);

	// Set the number of particles
	num_particles = 3000;

	// iterate through the particles to set the values
	for (unsigned int i=0; i<num_particles; ++i){
		Particle particle;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1.0;
		particles.push_back(particle);
		weights.push_back(1.0);
	}

	// Set initialized to be true
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	double new_x;
	double new_y;
	double new_theta;
	
	std::default_random_engine gen;

	for (unsigned int i=0; i<num_particles; ++i) {
		double old_x = particles[i].x;
		double old_y = particles[i].y;
		double old_theta = particles[i].theta;

		// Update
		
		if (fabs(yaw_rate) < 0.001) { // when yaw_rate is zero
			new_theta = old_theta;
			new_x = old_x + velocity * delta_t * cos(old_theta);
			new_y = old_y + velocity * delta_t * sin(old_theta);
		}
		else {
			new_theta = old_theta + yaw_rate * delta_t;
			new_x = old_x + velocity/yaw_rate * (sin(new_theta) - sin(old_theta));
			new_y = old_y + velocity/yaw_rate * (cos(old_theta) - cos(new_theta));
		}

		// create a normal (Gaussian) distribution for x, y, theta
		std::normal_distribution<double> dist_x(new_x, std_pos[0]);
		std::normal_distribution<double> dist_y(new_y, std_pos[1]);
		std::normal_distribution<double> dist_theta(new_theta, std_pos[2]);

		// update particles
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
		// particles[i].x = new_x;
		// particles[i].y = new_y;
		// particles[i].theta = new_theta;
	} 
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	std::vector<LandmarkObs> new_observations; 
	while (predicted.size() > 0) {
		int predicted_id = predicted[0].id;
		double predicted_x = predicted[0].x;
		double predicted_y = predicted[0].y;

		double min = 2;
		bool find = false;

		// define observation
		LandmarkObs obs;

		for (unsigned int i=0; i<observations.size(); ++i) {
			double obs_x = observations[i].x;
			double obs_y = observations[i].y;

			// calculate the distance between predicted and observation landmarks
			double dist = sqrt((predicted_x-obs_x)*(predicted_x-obs_x)
				+ (predicted_y-obs_y)*(predicted_y-obs_y));
			if (dist < min) {
				obs.id = predicted_id;
				obs.x = obs_x;
				obs.y = obs_y;
				min = dist;
				find = true;
			}
		}
		if (find) {
			new_observations.push_back(obs);
		}

		// erase the first item from predicted
		predicted.erase(predicted.begin());
	}
	observations = new_observations;

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
	
	// set parameters
	double sigma_x = std_landmark[0];
	double sigma_y = std_landmark[1];

	// iterate through particles to update weights
	for (unsigned int i=0; i<num_particles; ++i) {
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;

		// initialize predicted vector
		std::vector<LandmarkObs> predicted; 

		// iterate through lanmarks to find the correct landmarks within the range
		for (unsigned int j=0; j<map_landmarks.landmark_list.size(); ++j) {
			int landmark_id = map_landmarks.landmark_list[j].id_i;
			double landmark_x = map_landmarks.landmark_list[j].x_f;
			double landmark_y = map_landmarks.landmark_list[j].y_f;

			// if the landmark is within range of sensor_range then append it to predicted vector
			if (sqrt((p_x-landmark_x)*(p_x-landmark_x) + (p_y-landmark_y)*(p_y-landmark_y))
				<= sensor_range) {

				LandmarkObs obs;
				obs.id = landmark_id;
				obs.x = landmark_x;
				obs.y = landmark_y;
				predicted.push_back(obs);
			}
		}

		// initialize observations vector
		std::vector<LandmarkObs> new_observations; 

		// make new observation vector to take account the rotation and translation
		for (unsigned int j=0; j<observations.size(); ++j) {
			double o_x = observations[j].x;
			double o_y = observations[j].y;

			LandmarkObs obs;
			obs.x = o_x*cos(p_theta)+o_y*sin(p_theta)+p_x;
			obs.y = o_x*sin(p_theta)+o_y*cos(p_theta)+p_y;
			new_observations.push_back(obs);
		}

		// associate landmark ids to new_observations
		dataAssociation(predicted, new_observations);

		// calculate weight values
		double prob = 1.0;
		double rho;

		bool find = false;

		for (int j=0; j<new_observations.size(); ++j) {
			int obs_id = new_observations[j].id;
			double x = new_observations[j].x;
			double y = new_observations[j].y;

			double ang = atan2(y-p_y, x-p_x);
			rho = 0.3 * sin(ang*0.5);

			for (int k=0; k<predicted.size(); ++k) {
				int pred_id = predicted[k].id;
				if (obs_id == pred_id) {
					double mu_x = predicted[k].x;
					double mu_y = predicted[k].y;

					// use a multi-variate Gaussian distribution
					prob *= 0.5/(M_PI*sigma_x*sigma_y*sqrt(1-rho*rho))
						* exp(-0.5/(1-rho*rho)
						*((x-mu_x)*(x-mu_x)/(sigma_x*sigma_x)
						+ (y-mu_y)*(y-mu_y)/(sigma_y*sigma_y)
						- 2*rho*(x-mu_x)*(y-mu_y)/(sigma_x*sigma_y)));
					find = true;
				}
			}
		}

		if (find==false) {
			prob = 0.0;			
		}

		// update the weights to particles
		particles[i].weight = prob;
		weights[i] = prob;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::default_random_engine generator;
	std::discrete_distribution<double> distribution(weights.begin(), weights.end());

	// declare the empty particles vector
	std::vector<Particle> new_particles;

	for (unsigned int i = 0 ; i < num_particles; ++i) {
		int ind = distribution(generator);
		new_particles.push_back(particles[ind]);
	}

	// update particles
	particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
