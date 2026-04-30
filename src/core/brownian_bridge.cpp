#include "brownian_bridge.hpp"
#include <queue>
#include <cstring>
#include <vector>

std::vector<BBNode> build_brownian_bridge(int m, double dt) {
    std::vector<BBNode> nodes;
    nodes.reserve(m);

    if (m <= 0) return nodes;

    BBNode endpoint;
    endpoint.left = 0;
    endpoint.right = 0;
    endpoint.mid = m;
    endpoint.w_l = 0.0;
    endpoint.w_r = 0.0;
    endpoint.std = std::sqrt(static_cast<double>(m) * dt);
    nodes.push_back(endpoint);

    struct Interval { int l, r; };
    std::queue<Interval> q;
    q.push({0, m});

    while (!q.empty()) {
        auto [l, r] = q.front(); q.pop();
        if (r - l <= 1) continue;
        int mid = (l + r + 1) / 2;

        double t_l   = l  * dt;
        double t_mid = mid * dt;
        double t_r   = r  * dt;

        double span = t_r - t_l;
        BBNode node;
        node.left  = l;
        node.right = r;
        node.mid   = mid;
        node.w_l   = (t_r - t_mid) / span;
        node.w_r   = (t_mid - t_l) / span;
        node.std   = std::sqrt((t_mid - t_l) * (t_r - t_mid) / span);
        nodes.push_back(node);

        if (mid - l > 1) q.push({l, mid});
        if (r - mid > 1) q.push({mid, r});
    }
    return nodes;
}

void apply_brownian_bridge(const double* z,
                            const std::vector<BBNode>& bridge,
                            double* W, int m) {
    std::memset(W, 0, (m + 1) * sizeof(double));

    for (int k = 0; k < static_cast<int>(bridge.size()); ++k) {
        const BBNode& b = bridge[k];
        W[b.mid] = b.w_l * W[b.left] + b.w_r * W[b.right] + b.std * z[k];
    }
}

void simulate_path_bb(const double* z,   
                       const std::vector<BBNode>& bridge,
                       double S0, double r, double v,
                       double dt, int m,
                       double* S_path)   
{
    std::vector<double> W(m + 1, 0.0);
    apply_brownian_bridge(z, bridge, W.data(), m);

    double drift = (r - 0.5 * v * v);
    S_path[0] = S0;
    for (int i = 1; i <= m; ++i) {
        double dW = W[i] - W[i - 1];
        S_path[i] = S_path[i - 1] * std::exp(drift * dt + v * dW);
    }
}
