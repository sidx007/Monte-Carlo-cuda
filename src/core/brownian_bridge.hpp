#pragma once
#include <vector>
#include <cmath>

struct BBNode {
    int   left;    
    int   right;   
    int   mid;     
    double w_l;    
    double w_r;    
    double std;    
};

std::vector<BBNode> build_brownian_bridge(int m, double dt);

void apply_brownian_bridge(const double* z,   
                            const std::vector<BBNode>& bridge,
                            double* W,         
                            int m);

void simulate_path_bb(const double* z,   
                       const std::vector<BBNode>& bridge,
                       double S0, double r, double v,
                       double dt, int m,
                       double* S_path);
