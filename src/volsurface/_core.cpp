// Andersen QE simulation of Heston, as a fused C++ loop.
//
// The Python implementation walks the whole path population one time step at a
// time, which costs a dozen or so passes over memory per step and allocates a
// temporary for each. This kernel instead walks one path from start to finish
// with the state living in registers, so the arithmetic is the same and the
// memory traffic nearly disappears.
//
// It is deliberately driven by the same array of uniforms as the Python code,
// two per time step. That keeps antithetic sampling and Sobol working
// unchanged, and it makes the two implementations comparable path by path
// rather than only in distribution.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double PSI_C = 1.5;

// Inverse standard normal CDF: Acklam's rational approximation refined by one
// Halley step, which takes it to roughly machine precision and keeps this
// kernel numerically interchangeable with scipy's ndtri.
double ndtri(double p) {
    static const double a[6] = {-3.969683028665376e+01, 2.209460984245205e+02,
                                -2.759285104469687e+02, 1.383577518672690e+02,
                                -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[5] = {-5.447609879822406e+01, 1.615858368580409e+02,
                                -1.556989798598866e+02, 6.680131188771972e+01,
                                -1.328068155288572e+01};
    static const double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01,
                                -2.400758277161838e+00, -2.549732539343734e+00,
                                4.374664141464968e+00, 2.938163982698783e+00};
    static const double d[4] = {7.784695709041462e-03, 3.224671290700398e-01,
                                2.445134137142996e+00, 3.754408661907416e+00};
    const double p_low = 0.02425, p_high = 1.0 - p_low;

    if (p <= 0.0) return -std::numeric_limits<double>::infinity();
    if (p >= 1.0) return std::numeric_limits<double>::infinity();

    double x;
    if (p < p_low) {
        const double q = std::sqrt(-2.0 * std::log(p));
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if (p <= p_high) {
        const double q = p - 0.5, rr = q * q;
        x = (((((a[0] * rr + a[1]) * rr + a[2]) * rr + a[3]) * rr + a[4]) * rr + a[5]) * q /
            (((((b[0] * rr + b[1]) * rr + b[2]) * rr + b[3]) * rr + b[4]) * rr + 1.0);
    } else {
        const double q = std::sqrt(-2.0 * std::log(1.0 - p));
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
             ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }

    // Halley refinement on Phi(x) - p
    const double e = 0.5 * std::erfc(-x / std::sqrt(2.0)) - p;
    const double u = e * std::sqrt(2.0 * M_PI) * std::exp(0.5 * x * x);
    return x - u / (1.0 + 0.5 * x * u);
}

}  // namespace

py::array_t<double> simulate_qe(
    py::array_t<double, py::array::c_style | py::array::forcecast> uniforms,
    double S0, double T, double kappa, double theta, double xi, double rho,
    double v0, double r, double q) {
    const auto buf = uniforms.request();
    if (buf.ndim != 2) throw std::invalid_argument("uniforms must be two dimensional");
    const py::ssize_t n_paths = buf.shape[0];
    const py::ssize_t dim = buf.shape[1];
    if (dim == 0 || dim % 2 != 0)
        throw std::invalid_argument("uniforms must have two columns per time step");
    const py::ssize_t n_steps = dim / 2;
    if (xi <= 0.0 || kappa <= 0.0) throw std::invalid_argument("kappa and xi must be positive");

    const double* U = static_cast<const double*>(buf.ptr);
    auto out = py::array_t<double>(n_paths);
    double* ST = static_cast<double*>(out.request().ptr);

    const double dt = T / static_cast<double>(n_steps);
    const double g1 = 0.5, g2 = 0.5;  // central discretisation
    const double K1 = g1 * dt * (kappa * rho / xi - 0.5) - rho / xi;
    const double K2 = g2 * dt * (kappa * rho / xi - 0.5) + rho / xi;
    const double K3 = g1 * dt * (1.0 - rho * rho);
    const double K4 = g2 * dt * (1.0 - rho * rho);
    const double A = K2 + 0.5 * K4;
    const double decay = std::exp(-kappa * dt);
    const double xi2 = xi * xi;
    const double drift = (r - q) * dt;
    const double log_S0 = std::log(S0);

    // Paths are independent, so the population splits cleanly across cores.
    // Exceptions cannot cross a thread boundary into Python, so a failed
    // martingale correction raises a flag that is checked once the workers join.
    std::atomic<bool> bad_correction{false};

    auto run_range = [&](py::ssize_t begin, py::ssize_t end) {
        for (py::ssize_t i = begin; i < end; ++i) {
            const double* u = U + i * dim;
            double v = v0;
            double x = log_S0;

            for (py::ssize_t s = 0; s < n_steps; ++s) {
                const double u_v = u[2 * s];
                const double u_s = u[2 * s + 1];

                const double m = theta + (v - theta) * decay;
                const double s2 = (v * xi2 * decay / kappa) * (1.0 - decay) +
                                  (theta * xi2 / (2.0 * kappa)) * (1.0 - decay) * (1.0 - decay);
                const double psi = (m > 0.0) ? s2 / (m * m)
                                             : std::numeric_limits<double>::infinity();

                double v_next, k0;
                if (psi <= PSI_C) {
                    const double inv_psi = 1.0 / psi;
                    const double root = std::sqrt(2.0 * inv_psi) *
                                        std::sqrt(std::fmax(2.0 * inv_psi - 1.0, 0.0));
                    const double b2 = 2.0 * inv_psi - 1.0 + root;
                    const double a = m / (1.0 + b2);
                    if (A * a >= 0.5) {
                        bad_correction.store(true, std::memory_order_relaxed);
                        return;
                    }
                    const double z = std::sqrt(b2) + ndtri(u_v);
                    v_next = a * z * z;
                    k0 = -(A * a * b2) / (1.0 - 2.0 * A * a) + 0.5 * std::log(1.0 - 2.0 * A * a);
                } else {
                    const double pz = (psi - 1.0) / (psi + 1.0);
                    const double beta = (1.0 - pz) / std::fmax(m, 1e-300);
                    if (A >= beta) {
                        bad_correction.store(true, std::memory_order_relaxed);
                        return;
                    }
                    v_next = (u_v <= pz)
                                 ? 0.0
                                 : std::log(std::fmax((1.0 - pz) / (1.0 - u_v), 1e-300)) / beta;
                    k0 = -std::log(pz + (1.0 - pz) * beta / (beta - A));
                }

                k0 -= K1 * v + 0.5 * K3 * v;
                const double var = K3 * v + K4 * v_next;
                x += drift + k0 + K1 * v + K2 * v_next +
                     std::sqrt(var > 0.0 ? var : 0.0) * ndtri(u_s);
                v = v_next;
            }
            ST[i] = std::exp(x);
        }
    };

    {
        py::gil_scoped_release release;  // pure C++ from here, no Python objects touched
        unsigned n_threads = std::thread::hardware_concurrency();
        if (n_threads == 0) n_threads = 1;
        // one thread is plenty below this size, and spawning costs more than it saves
        const py::ssize_t min_chunk = 4096;
        n_threads = static_cast<unsigned>(
            std::max<py::ssize_t>(1, std::min<py::ssize_t>(n_threads, n_paths / min_chunk)));

        if (n_threads == 1) {
            run_range(0, n_paths);
        } else {
            std::vector<std::thread> workers;
            workers.reserve(n_threads);
            const py::ssize_t chunk = (n_paths + n_threads - 1) / n_threads;
            for (unsigned t = 0; t < n_threads; ++t) {
                const py::ssize_t begin = static_cast<py::ssize_t>(t) * chunk;
                const py::ssize_t end = std::min(begin + chunk, n_paths);
                if (begin < end) workers.emplace_back(run_range, begin, end);
            }
            for (auto& w : workers) w.join();
        }
    }

    if (bad_correction.load(std::memory_order_relaxed))
        throw std::runtime_error("martingale correction undefined: reduce the step size");
    return out;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "Fused C++ kernel for the Andersen QE simulation of Heston.";
    m.def("simulate_qe", &simulate_qe,
          py::arg("uniforms"), py::arg("S0"), py::arg("T"), py::arg("kappa"),
          py::arg("theta"), py::arg("xi"), py::arg("rho"), py::arg("v0"),
          py::arg("r") = 0.0, py::arg("q") = 0.0,
          "Terminal spots from the QE scheme, driven by the given uniforms.");
}
