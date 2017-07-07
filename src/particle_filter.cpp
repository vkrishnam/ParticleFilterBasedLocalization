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
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <random> // Need this for sampling from distributions

#include "particle_filter.h"



using namespace std;

inline double dist_calc(double x1, double y1, double x2, double y2) {
    return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}
/*
 * * Map measurements from vehicle coordinates to map coordinates system.
 * * We use the following set of equations:
 * * X_map = X_p + (X_o*cos(theta)) - (Y_o*sin(theta))
 * * Y_map = Y_p + (X_o*sin(theta)) + (Y_o*cos(theta))
 * * where,
 * * (X_p, Y_p, theta) is the particle state.
 * * (X_o, Y_o) is the landmark measurement from vehicle coordinates.
 * * (X_map, Y_map) is the transformed map coordinates.
 * */
inline vector<LandmarkObs> vehicle_to_map_coordinates(vector<LandmarkObs> obs, Particle p_i) {
    vector<LandmarkObs> obs_mc;
    double X_p = p_i.x;
    double Y_p = p_i.y;
    double theta = p_i.theta;

    for (int j=0; j<obs.size(); j++) {
        LandmarkObs l;
        double X_o = obs[j].x;
        double Y_o = obs[j].y;

        l.x = X_p + (X_o * cos(theta)) - (Y_o * sin(theta));
        l.y = Y_p + (X_o * sin(theta)) + (Y_o * cos(theta));
        l.id = obs[j].id;

        obs_mc.push_back(l);
    }

    return obs_mc;
}



void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 1000;
    default_random_engine gen;
    // This line creates a normal (Gaussian) distribution for x
    normal_distribution<double> dist_x(x, std[0]);
    // Create normal distributions for y and psi
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_psi(theta, std[2]);
    for (int i = 0; i < num_particles; ++i) {
        double sample_x, sample_y, sample_psi;
        Particle p = Particle();

        sample_x = dist_x(gen);
        sample_y = dist_y(gen);
        sample_psi = dist_psi(gen);

        p.id = i;
        p.x  = sample_x;
        p.y  = sample_y;
        p.theta = sample_psi;
        p.weight = 1.0;

        weights.push_back(p.weight);
        particles.push_back(p);
    }
    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    default_random_engine gen;
    // This line creates a normal (Gaussian) distribution for x
    normal_distribution<double> dist_x(0, std_pos[0]);
    // Create normal distributions for y and psi
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_psi(0, std_pos[2]);

    for (int i = 0; i < num_particles; ++i) {
        double sample_x, sample_y, sample_psi;
        double temp, thetaf;

        sample_x = particles[i].x;
        sample_y = particles[i].y;
        sample_psi = particles[i].theta;

        temp = (velocity/yaw_rate);
        thetaf = sample_psi + (yaw_rate * delta_t);

        if(fabs(yaw_rate) < 0.001) {
            sample_x += velocity*delta_t*cos(sample_psi);
            sample_y += velocity*delta_t*sin(sample_psi);
        }
        else {
            sample_x += temp*(sin(thetaf)-sin(sample_psi));
            sample_y += temp*(cos(sample_psi)-cos(thetaf));
        }

        sample_x += dist_x(gen);
        sample_y += dist_y(gen);
        sample_psi = thetaf+dist_psi(gen);

        particles[i].x = sample_x;
        particles[i].y = sample_y;
        particles[i].theta = sample_psi;

    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
    int len = observations.size();
    for (int j=0; j<len; j++) {
        LandmarkObs obs = observations[j];
        double x0 = obs.x;
        double y0 = obs.y;
        double dist, min_dist;

        min_dist = dist_calc(x0, y0, predicted[0].x, predicted[0].y);
        int best_id = predicted[0].id;

        for (int k=1; k<predicted.size(); k++) {
            LandmarkObs pred = predicted[k];
            double x1 = pred.x;
            double y1 = pred.y;
            dist = dist_calc(x0, y0, x1, y1);
            if (dist < min_dist) {
                min_dist = dist;
                best_id = pred.id;
            }
        }
        observations[j].id = best_id;
    }
}

inline std::vector<LandmarkObs> data_Association(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.
    std::vector<LandmarkObs> lmarks_asc;

    int len = observations.size();
    for (int j=0; j<len; j++) {
        LandmarkObs obs = observations[j];
        double x0 = obs.x;
        double y0 = obs.y;
        double dist, min_dist;

        min_dist = dist_calc(x0, y0, predicted[0].x, predicted[0].y);
        int best_id = 0;

        for (int k=1; k<predicted.size(); k++) {
            LandmarkObs pred = predicted[k];
            double x1 = pred.x;
            double y1 = pred.y;
            dist = dist_calc(x0, y0, x1, y1);
            if (dist < min_dist) {
                min_dist = dist;
                //best_id = pred.id;
                best_id = k;
            }
        }

        lmarks_asc.push_back(predicted[best_id]);
    }

    return lmarks_asc;
}

/*
 * Calculate new weight based on Gaussian distribution.
 * We assume the observation measurements are independent.
 */
inline double calc_wt(std::vector<LandmarkObs> obs_mc, std::vector<LandmarkObs> lmarks_asc, double std_landmark[]) {
    double sigma2_x = std_landmark[0] * std_landmark[0];
    double sigma2_y = std_landmark[1] * std_landmark[1];
    double weight = 1.0;
    double scale = 1.0/(sqrt(2*M_PI*sigma2_x*sigma2_y));

    if (obs_mc.size() != lmarks_asc.size()) {
        cout << "Mismatch in size between obs and associated landmarks" << endl;
        return 0.0;
    }

    for (int j=0; j<obs_mc.size(); j++) {
        double p0_x = obs_mc[j].x;
        double p0_y = obs_mc[j].y;

        double p1_x = lmarks_asc[j].x;
        double p1_y = lmarks_asc[j].y;
        double d = (-0.5) * (((1/sigma2_x)*(p0_x-p1_x)*(p0_x-p1_x)) + ((1/sigma2_y)*(p0_y-p1_y)*(p0_y-p1_y)));

        weight *= scale * exp(d);
    }

    return weight;
}

/*
 * Choose map landmarks within sensor range of particle.
 */
inline vector<LandmarkObs> choose_map_landmarks_within_range(vector<Map::single_landmark_s> map_lms, Particle p_i, double sensor_range) {
    vector<LandmarkObs> lmarks_range;
    //vector<Map::single_landmark_s> map_lms = map_landmarks.landmark_list;
    double x0 = p_i.x;
    double y0 = p_i.y;

   for(int j=0; j<map_lms.size(); j++) {
        double x1 = map_lms[j].x_f;
        double y1 = map_lms[j].y_f;
        int id = map_lms[j].id_i;

        double r = dist_calc(x0, y0, x1, y1);
        if (r < sensor_range) {
            LandmarkObs lo;
            lo.x = x1;
            lo.y = y1;
            lo.id = id;

            lmarks_range.push_back(lo);
        }
    }

    return lmarks_range;
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
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
   /*
     * Following steps need to be done to get the updated weights.
     * For each particle (p[i]) :
     * (a) Choose map landmarks within sensor range of particle.
     * (b) Map measurements from vehicle coordinates to map coordinates system.
     * (c) Associate each transformed observation point to a map landmark point.
     * (d) Calculate new weight based on Gaussian distribution.
     */
    for (int i=0; i<num_particles; i++) {

        Particle p_i = particles[i];

        /*
         * Choose map landmarks within sensor range of particle.
         */
        vector<Map::single_landmark_s> map_lms = map_landmarks.landmark_list;
        vector<LandmarkObs> lmarks_range = choose_map_landmarks_within_range(map_lms, p_i, sensor_range);

        /*
         * Map measurements from vehicle coordinates to map coordinates system.
         */
        vector<LandmarkObs> obs_mc = vehicle_to_map_coordinates(observations, particles[i]);

        /*
         * Associate each transformed observation point to a map landmark point.
         * We use the "nearest neighbour" rule to perform association.
         */
        std::vector<LandmarkObs> lmarks_asc = data_Association(lmarks_range, obs_mc);
        /*
         * Calculate new weight based on Gaussian distribution.
         * We assume the observation measurements are independent.
         */
        double weight = calc_wt(obs_mc, lmarks_asc, std_landmark);

        particles[i].weight = weight;
        weights[i] = weight;
    }
    
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<int> dbn_wts(weights.begin(), weights.end());

    std::vector<Particle> new_particles;

    for (int i=0; i<num_particles; i++) {
        Particle new_particle = particles[dbn_wts(gen)];
        new_particles.push_back(new_particle);
    }
    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
