/* Author: True Price <jtprice at cs.unc.edu>
 *
 * To compile:
 *   g++ -Wall -std=c++11 -O3 -fPIC -c sfsmodule.cc -I/usr/local/include/eigen3
 *   g++ -O3 -shared sfsmodule.o -o sfs.so
 *
 * To compile on BASS (with gcc 4.4.7):
 *   g++ -Wall -std=c++0x -O3 -fPIC -c sfsmodule.cc
 *     -I$HOME/anaconda/include/python2.7/
 *     -I$HOME/anaconda/lib/python2.7/site-packages/numpy/core/include/numpy
 *     -I$HOME/local/include/eigen3
 *   g++ -O3 -shared sfsmodule.o -o sfs.so
 *
 * For more information on compiling C libraries for Python, see:
 * http://docs.python.org/extending/extending.html
 */

/*#include <cstddef>
#ifndef __SFS_LIBRARY__
#define __SFS_LIBRARY__

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define nullptr NULL
#define 	M_1_PI   0.31830988618379067154	

#include <cmath>

#include <Python.h>
#include <numpy/arrayobject.h>

#include "util.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <math.h>
#include <memory> // for unique_ptr*/

//#include <cstddef>

#ifndef __SFS_LIBRARY__
#define __SFS_LIBRARY__

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define nullptr NULL
#define 	M_1_PI   0.31830988618379067154	

//#include <cmath>
#include <Python.h>
#include <numpy/arrayobject.h>

#include "util.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <math.h>
#include <memory> // for unique_ptr

//#include <chrono>

//
// typedefs and enums
//

// to easily change data types (e.g. double to float), if desired
typedef double sfs_t;

enum ReflectanceModelType {
  LAMBERTIAN_MODEL,
  OREN_NAYAR_MODEL,
  PHONG_MODEL,
  POWER_MODEL,
  COOK_TORRANCE_MODEL
};

//
// global constants
//

const size_t MAX_NUM_SFS_ITER = 2000;

const size_t MAX_NUM_NEWTON_ITER = 20;
const sfs_t NEWTON_TOL = 1e-6; // tolerance for Newton iterations

const sfs_t SPECULAR_THRESH = 1.00; // somewhat hand-wavey threshold for
                                    // identifying specular pixels

// for numerically approximating sigma values for the Hamiltonian
#define NDOTL_STEP 0.001
#define INV_NDOTL_STEP 1000.
#define NUM_DH_LUT_BINS 1000
//const sfs_t NDOTL_STEP = 0.001;
//const sfs_t INV_NDOTL_STEP = 1. / NDOTL_STEP;
//const size_t NUM_SIGMA_LUT_BINS = size_t(INV_NDOTL_STEP);

// Approximate maximum value of d(ndotl)/dp to use
//
// The maximum depends on x and y, so what we do is fit a parabola to the
// maximum values at each x and y, to approximate the sigma
//
// this value was numerically estimated by taking (x, y) \in (-2,2)x(-2,2) in
// steps of size 0.1, then estimating d(ndotl)/dp for p \in (-1,1)x(-1,1) with
// steps of size 0.001; taking the derivatives of d(ndotl)/dp w.r.t p and q
// yields q = -y / (x^2 + y^2 + 1)
//
// symmetry allows us to assume the same maximum for d(ndotl)/dq, flipping x/y
//
// Python code:
/*
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

p = np.linspace(-1, 1, 2001)
psq = p * p

max_deriv = np.empty((41, 41))
pmax = np.empty((41, 41))
for j, xi in enumerate(np.linspace(-2, 2, 41)):
  for i, yi in enumerate(np.linspace(-2, 2, 41)):
    xsq_ysq_1 = xi * xi + yi * yi + 1.
    q = -yi / xsq_ysq_1
    cross_term = xi * p + yi * q + 1.
    temp_term = 1. / (psq + q * q + cross_term * cross_term)
    temp_term *= np.sqrt(temp_term)
    dndotl_dp = np.abs(
      1. / np.sqrt(xsq_ysq_1) * temp_term * (p + cross_term * xi))
    idx_max = np.argmax(dndotl_dp)
    pmax[i,j] = p[idx_max]
    max_deriv[i,j] = dndotl_dp[idx_max]

# fit a parabola to the values
# the minimum is at (0, 0), so no need to model any offset
def f(data, a, b, c):
  return a * data[0] * data[0] + b * data[1] * data[1] + c

x, y = np.meshgrid(np.linspace(-2, 2, 41), np.linspace(-2, 2, 41))
a, b, c = curve_fit(f, np.vstack((x.ravel(), y.ravel())), max_deriv.ravel())[0]
print a, b, c

# plotting the fit as a sanity check
plt.figure()
plt.imshow(max_deriv)
plt.colorbar()
plt.figure()
plt.imshow(a * x * x + b * y * y + c) # never less than the observed values
plt.colorbar()
plt.show()
*/
const sfs_t SIGMA_A = 0.39798809209727204;
const sfs_t SIGMA_B = 0.14348878921048919;
const sfs_t SIGMA_C = 0.38311673769356436;

//
// classes
//

// base class for reflectance models
struct ReflectanceModelBase {
  ReflectanceModelBase(const sfs_t _falloff) : falloff(_falloff) {}

  inline sfs_t compute_hamiltonian(const sfs_t ndotl) const {
    return _compute_hamiltonian(ndotl);
    //return std::max(_compute_hamiltonian(ndotl), 0.);
    //return std::min(_compute_hamiltonian(ndotl), 1.);
    //return std::max(std::min(_compute_hamiltonian(ndotl), 1.), 0.);
  }

  virtual sfs_t _compute_hamiltonian(const sfs_t ndotl) const = 0;

  // this provides a numerical approximation of the maximum change in the
  // Hamiltonian per change in cos(theta)
  // we create a lookup table for each value of ndotl, where the value is
  //   max(dh/dp) for p \in [0, ndotl]
  virtual inline void estimate_sigmas() {
    // initialize for ndotl = 0
    sfs_t ndotl = NDOTL_STEP;
    sfs_t H_prev = 0., H = compute_hamiltonian(ndotl);
    dH_lut[0] = H * INV_NDOTL_STEP;

    // cumulative LUT; the maximum is the maximum derivative of any current
    // value
    for (size_t i = 1; i < NUM_DH_LUT_BINS; ++i) {
      ndotl += NDOTL_STEP;
      sfs_t H_next = compute_hamiltonian(ndotl);
      dH_lut[i] = std::max(std::abs(H_next - H_prev) * INV_NDOTL_STEP * 0.5,
                           dH_lut[i - 1]);
      H_prev = H;
      H = H_next;
    }
  }

  sfs_t falloff;  // light falloff parameter (m in r^-m, where m > 0)

  Eigen::Matrix<sfs_t, NUM_DH_LUT_BINS, 1> dH_lut;
};

// Lambertian model
struct LambertianModel : public ReflectanceModelBase {
  LambertianModel(const sfs_t _alpha, const sfs_t _falloff)
      : ReflectanceModelBase(_falloff), alpha(_alpha) {}

  virtual sfs_t _compute_hamiltonian(const sfs_t ndotl) const {
    return alpha * ndotl;
  }

  sfs_t alpha; // H = alpha * cos(theta)
};

// Oren-Nayar model
struct OrenNayarModel : public ReflectanceModelBase {
  OrenNayarModel(const sfs_t _alpha, const sfs_t _beta, const sfs_t _falloff)
      : ReflectanceModelBase(_falloff), alpha(_alpha), beta(_beta) {}

  virtual sfs_t _compute_hamiltonian(const sfs_t ndotl) const {
    return alpha * ndotl + beta * (1. - ndotl * ndotl);
  }

  sfs_t alpha, beta; // cos, sin^2 coefficients
};

// Phong model
struct PhongModel : public ReflectanceModelBase {
  PhongModel(const sfs_t _alpha, const sfs_t _beta, const sfs_t _eta,
             const sfs_t _falloff)
      : ReflectanceModelBase(_falloff), alpha(_alpha), beta(_beta), eta(_eta) {}

  virtual sfs_t _compute_hamiltonian(const sfs_t ndotl) const {
    return alpha * ndotl + beta * std::pow(ndotl, eta);
  }

  sfs_t alpha, beta, eta; // H = alpha * cos(theta) + beta * cos(theta)^eta
};

// Powers-of-cosine model
struct PowerModel : public ReflectanceModelBase {
  PowerModel(const Eigen::Array<sfs_t, Eigen::Dynamic, 1> _cos_coeffs,
             const Eigen::Array<sfs_t, Eigen::Dynamic, 1> _sin_coeffs,
             const sfs_t _falloff)
      : ReflectanceModelBase(_falloff), cos_coeffs(_cos_coeffs),
        sin_coeffs(_sin_coeffs) {}

  virtual sfs_t _compute_hamiltonian(const sfs_t ndotl) const {
    const auto sin_term = std::sqrt((1. - ndotl) * 0.5); //=sin(arccos(ndotl)/2)
    const auto coeffs = cos_coeffs + sin_term * sin_coeffs;

    sfs_t H = 0., ndotlk = ndotl;
    for (size_t k = 0; k < (size_t)coeffs.rows(); ++k) {
      H += coeffs[k] * ndotlk;
      ndotlk *= ndotl;
    }

    return std::max(H, 0.);
  }

  sfs_t ambient_coeff;
  Eigen::Array<sfs_t, Eigen::Dynamic, 1> cos_coeffs, sin_coeffs;
};

// Cook-Torrance reflectance model
struct CookTorranceModel : public ReflectanceModelBase {
  CookTorranceModel(const sfs_t _d, const sfs_t _s, const sfs_t _inv_m_sq,
                    const sfs_t _falloff)
      : ReflectanceModelBase(_falloff), d(_d), s(_s), inv_m_sq(_inv_m_sq) {}

  virtual sfs_t _compute_hamiltonian(const sfs_t ndotl) const {
    const auto ndotlsq = ndotl * ndotl;
    const auto G = std::min(1., 2. * ndotlsq); // geometric attenuation term

    // note: tan(arccos(x)) = sqrt(1 - x^2) / x
    return d * ndotl +
           s * inv_m_sq * G * M_1_PI / (ndotlsq * ndotlsq * ndotl) *
               std::exp((1. - 1. / ndotlsq) * inv_m_sq);
  }

  sfs_t d; // scaling of diffuse term (size/intensity of light source)
  sfs_t s; // scaling of specular term (size/intensity of light source) times
           // Fresnel term [(n-1)/(n+1)]^2, where n is the index of refraction
  sfs_t inv_m_sq; // 1 / m^2; (MSE of slope of facets)
};

// a convenience struct for holding all the arrays/values needed for SFS
struct SFSData {
  SFSData(PyObject *py_L, PyObject *py_Lhat, PyObject *py_v, PyObject *py_x,
          PyObject *py_y, PyObject *py_z_est, PyObject *py_inv_sqrt_xsq_ysq_1,
          PyObject *py_wxfw, PyObject *py_wxbk, PyObject *py_wyfw,
          PyObject *py_wybk, PyObject *py_lambdas, const sfs_t _fx,
          const sfs_t _fy,PyObject *py_vmask)
      : L(PyArrayMap<sfs_t>(py_L)),
        Lhat(PyArrayMap<sfs_t>(py_Lhat)),
        v(PyArrayMap<sfs_t>(py_v)),
        x(PyArrayMap<sfs_t>(py_x)),
        y(PyArrayMap<sfs_t>(py_y)),
        z_est(PyArrayMap<sfs_t>(py_z_est)),
        inv_sqrt_xsq_ysq_1(PyArrayMap<sfs_t>(py_inv_sqrt_xsq_ysq_1)),
        wxfw(PyArrayMap<sfs_t>(py_wxfw)),
        wxbk(PyArrayMap<sfs_t>(py_wxbk)),
        wyfw(PyArrayMap<sfs_t>(py_wyfw)),
        wybk(PyArrayMap<sfs_t>(py_wybk)),
        lambdas(PyArrayMap<sfs_t>(py_lambdas)),
		vmask(PyArrayMap<sfs_t>(py_vmask)),
        fx(_fx), fy(_fy) {
           h = Lhat.rows();
           w = Lhat.cols();
        }
		

  // need to use a delegating constructor for calculating h, w, etc.
  /*
  SFSData(PyObject *py_L, PyObject *py_Lhat, PyObject *py_v, PyObject *py_x,
          PyObject *py_y, PyObject *py_z_est, PyObject *py_inv_sqrt_xsq_ysq_1,
          PyObject *py_wxfw, PyObject *py_wxbk, PyObject *py_wyfw,
          PyObject *py_wybk, PyObject *py_lambdas, const sfs_t _fx,
          const sfs_t _fy)
      : SFSData(PyArrayMap<sfs_t>(py_L), PyArrayMap<sfs_t>(py_Lhat),
                PyArrayMap<sfs_t>(py_v), PyArrayMap<sfs_t>(py_x),
                PyArrayMap<sfs_t>(py_y), PyArrayMap<sfs_t>(py_z_est),
                PyArrayMap<sfs_t>(py_inv_sqrt_xsq_ysq_1),
                PyArrayMap<sfs_t>(py_wxfw), PyArrayMap<sfs_t>(py_wxbk),
                PyArrayMap<sfs_t>(py_wyfw), PyArrayMap<sfs_t>(py_wybk),
                PyArrayMap<sfs_t>(py_lambdas), _fx, _fy) {}
  */

  // data arrays
  const PyArrayMap<sfs_t> L;
  const PyArrayMap<sfs_t> Lhat;
  PyArrayMap<sfs_t> v; // this is our output array
  const PyArrayMap<sfs_t> x;
  const PyArrayMap<sfs_t> y;
  const PyArrayMap<sfs_t> z_est;
  const PyArrayMap<sfs_t> inv_sqrt_xsq_ysq_1;
  const PyArrayMap<sfs_t> wxfw, wxbk, wyfw, wybk;
  const PyArrayMap<sfs_t> lambdas;
  const PyArrayMap<sfs_t> vmask;

  // pre-computed image-weighted derivatives
  // Eigen::Array<sfs_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  //    wxfw, wxbk, wyfw, wybk;

  // model properties
  const sfs_t fx, fy; // camera focal lengths (== step sizes)
  size_t h, w;  // height, width

/*
private:
  // delegating constructor
  SFSData(const PyArrayMap<sfs_t> _L, const PyArrayMap<sfs_t> _Lhat,
          PyArrayMap<sfs_t> _v, const PyArrayMap<sfs_t> _x,
          const PyArrayMap<sfs_t> _y, const PyArrayMap<sfs_t> _z_est,
          const PyArrayMap<sfs_t> _inv_sqrt_xsq_ysq_1,
          const PyArrayMap<sfs_t> _wxfw, const PyArrayMap<sfs_t> _wxbk,
          const PyArrayMap<sfs_t> _wyfw, const PyArrayMap<sfs_t> _wybk,
          const PyArrayMap<sfs_t> _lambdas, const sfs_t _fx, const sfs_t _fy)
      : L(_L), Lhat(_Lhat), v(_v), x(_x), y(_y), z_est(_z_est),
        inv_sqrt_xsq_ysq_1(_inv_sqrt_xsq_ysq_1), wxfw(_wxfw), wxbk(_wxbk),
        wyfw(_wyfw), wybk(_wybk), lambdas(_lambdas), fx(_fx), fy(_fy),
        h(Lhat.rows()), w(Lhat.cols()) {}
*/
};

//
// C++ dedicated functions
//

void step(SFSData &sfs_data, ReflectanceModelBase &model, int sx, int sy) {
  size_t istart, istop, jstart, jstop;

  if (sx == -1) {
    jstart = sfs_data.w - 2;
    jstop = 0; // exclusive
  } else {
    jstart = 1;
    jstop = sfs_data.w - 1;
  }

  if (sy == -1) {
    istart = sfs_data.h - 2;
    istop = 0;
  } else {
    istart = 1;
    istop = sfs_data.h - 1;
  }

  for (size_t i = istart; i != istop; i += sy) {
    for (size_t j = jstart; j != jstop; j += sx) {

	 
	  /*if (sfs_data.vmask(i,j)!=0){
		  continue;
      }
	  else{*/
		  // estimate derivatives
		  const sfs_t vxfw = sfs_data.v(i, j + 1) - sfs_data.v(i, j);
		  const sfs_t vyfw = sfs_data.v(i + 1, j) - sfs_data.v(i, j);
		  const sfs_t vxbk = sfs_data.v(i, j) - sfs_data.v(i, j - 1);
		  const sfs_t vybk = sfs_data.v(i, j) - sfs_data.v(i - 1, j);

		  const sfs_t vx = sfs_data.fx * (sfs_data.wxfw(i, j) * vxfw +
										  sfs_data.wxbk(i, j) * vxbk);
		  const sfs_t vy = sfs_data.fy * (sfs_data.wyfw(i, j) * vyfw +
										  sfs_data.wybk(i, j) * vybk);

		  // compute cos(theta) = n dot l
		  const sfs_t x = sfs_data.x(i, j), y = sfs_data.y(i, j);
		  const sfs_t inv_sqrt_xsq_ysq_1 = sfs_data.inv_sqrt_xsq_ysq_1(i, j);

		  sfs_t cross_term = x * vx + y * vy + 1.;
		  const sfs_t ndotl =
			  inv_sqrt_xsq_ysq_1 /
			  std::sqrt(vx * vx + vy * vy + cross_term * cross_term);

		  // now, compute the maximum for dH/d(ndotl); the maximum value of ndotl
		  // occurs (for fixed p) when
		  //   q = -y / (x^2 + y^2 + 1), or equivalently,
		  //   ndotl_max = [(x^2+y^2+1)(p^2(x^2+y^2+1) + 2px + 1) / (y^2+1)]^(1/2)
		  // we then search for the largest value of dH/d(ndotl) from 0 to this
		  // maximum

		  const sfs_t xsq = x * x, ysq = y * y;
		  const sfs_t xsq_ysq_1 = xsq + ysq + 1.;
		  const sfs_t inv_xsq_1 = 1. / (xsq + 1.);
		  const sfs_t inv_ysq_1 = 1. / (ysq + 1.);

		  // maximum possible values for ndotl given p/q +/-
		  const sfs_t ndotl_max_x =
			  inv_sqrt_xsq_ysq_1 /
			  std::sqrt(std::min(
				  (vxfw * vxfw * xsq_ysq_1 + 2. * vxfw * x + 1.) * inv_ysq_1,
				  (vxbk * vxbk * xsq_ysq_1 + 2. * vxbk * x + 1.) * inv_ysq_1));
		  const sfs_t ndotl_max_y =
			  inv_sqrt_xsq_ysq_1 /
			  std::sqrt(std::min(
				  (vyfw * vyfw * xsq_ysq_1 + 2. * vyfw * y + 1.) * inv_xsq_1,
				  (vybk * vybk * xsq_ysq_1 + 2. * vybk * y + 1.) * inv_xsq_1));

		  // compute Hamiltonian and sigmas,
		  // with scaling by 1 / (L * sqrt(x^2 + y^2 + 1)^m)
		  const sfs_t H = sfs_data.Lhat(i, j) * model.compute_hamiltonian(ndotl);
		  const sfs_t sigma_x =
			  sfs_data.Lhat(i, j) * (SIGMA_A * xsq + SIGMA_B * ysq + SIGMA_C) *
			  model.dH_lut[(size_t)(ndotl_max_x * (NUM_DH_LUT_BINS - 1))];
		  const sfs_t sigma_y =
			  sfs_data.Lhat(i, j) * (SIGMA_A * ysq + SIGMA_B * xsq + SIGMA_C) *
			  model.dH_lut[(size_t)(ndotl_max_y * (NUM_DH_LUT_BINS - 1))];

		  const auto vxavg =
			  sfs_data.fx * (sfs_data.wxfw(i, j) * sfs_data.v(i, j + 1) +
							 sfs_data.wxbk(i, j) * sfs_data.v(i, j - 1));
		  const auto vyavg =
			  sfs_data.fy * (sfs_data.wyfw(i, j) * sfs_data.v(i + 1, j) +
							 sfs_data.wybk(i, j) * sfs_data.v(i - 1, j));

		  const sfs_t a = 1. / (sigma_x * sfs_data.fx + sigma_y * sfs_data.fy);
		  //const sfs_t b = a * H;
		  //const sfs_t c = a * (sigma_x * vxavg + sigma_y * vyavg);

		  // Newton iteration
		  sfs_t vnew = sfs_data.v(i, j);
		  //if (sfs_data.L(i, j) < SPECULAR_THRESH) {
			const sfs_t lambda = sfs_data.lambdas(i, j);
			const auto b =
				a * ((H + lambda * sfs_data.z_est(i, j)) / (1. + lambda) +
					 (sigma_x * vxavg + sigma_y * vyavg));

			bool converged = false;
			for (size_t n = 0; !converged && n < MAX_NUM_NEWTON_ITER; ++n) {
			  sfs_t exp_mv = std::exp(model.falloff * vnew);

			  sfs_t G = vnew + a * exp_mv - b;
			  sfs_t g = 1. + a * model.falloff * exp_mv;

			  //sfs_t G = vnew + a * exp_mv - c;
			  //sfs_t g = 1. + a * model.falloff * exp_mv;
			  //exp_mv *= sfs_data.Lhat(i, j);
			  //if (exp_mv < H) {
			  //  G -= a * exp_mv;
			  //  g -= a * model.falloff * exp_mv;
			  //} else {
			  //  G -= b;
			  //}

			  vnew -= G / g;
			  converged = (G > -NEWTON_TOL && G < NEWTON_TOL);
			}
		//  } else { // specular pixel; assign to the average
		//    vnew = a * (sigma_x * vxavg + sigma_y * vyavg);
		//  }

		  sfs_data.v(i, j) = std::min(sfs_data.v(i, j), vnew);
	  //}
    }
  }
}

//
// module functions
//

// global object reference that we initialize with py_initialize()
// this is somewhat a hack for not wanting to fix bad design
std::unique_ptr<SFSData> g_sfs_data;
std::unique_ptr<ReflectanceModelBase> g_reflectance_model;

extern "C" {

static PyObject *py_initialize(PyObject *self, PyObject *args) {
  PyObject *py_L, *py_Lhat, *py_v, *py_x, *py_y, *py_z_est,
      *py_inv_sqrt_xsq_ysq_1, *py_wxfw, *py_wxbk, *py_wyfw, *py_wybk,
      *py_lambdas;
  sfs_t fx, fy;
  PyObject *py_vmask;

  if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOddO", &py_L, &py_Lhat, &py_v, &py_x,
                        &py_y, &py_z_est, &py_inv_sqrt_xsq_ysq_1, &py_wxfw,
                        &py_wxbk, &py_wyfw, &py_wybk, &py_lambdas, &fx, &fy,&py_vmask)) {
    return NULL;
  }

  g_sfs_data.reset(new SFSData(py_L, py_Lhat, py_v, py_x, py_y, py_z_est,
                               py_inv_sqrt_xsq_ysq_1, py_wxfw, py_wxbk, py_wyfw,
                               py_wybk, py_lambdas, fx, fy,py_vmask));

  Py_RETURN_NONE;
}

static PyObject *py_set_model(PyObject *self, PyObject *args) {
  ReflectanceModelType model_type;
  PyObject *py_model_params;
  sfs_t falloff;

  if (!PyArg_ParseTuple(args, "iOd", &model_type, &py_model_params, &falloff)) {
    return NULL;
  }

  // input model parameters are simply a NumPy array
  PyArrayMapVector<sfs_t>::type model_params(py_model_params);

  ReflectanceModelBase *reflectance_model = NULL;

  // TODO: probably should add some error-checking on model parameters, here
  switch (model_type) {
    case LAMBERTIAN_MODEL: {
      reflectance_model = new LambertianModel(model_params[0], falloff);
      break;
    }

    case OREN_NAYAR_MODEL: {
      reflectance_model =
          new OrenNayarModel(model_params[0], model_params[1], falloff);
      break;
    }

    case PHONG_MODEL: {
      reflectance_model = new PhongModel(model_params[0], model_params[1],
                                         model_params[2], falloff);
      break;
    }

    case POWER_MODEL: {
      const size_t num_coeff = model_params.rows() / 2;
      reflectance_model =
          new PowerModel(model_params.head(num_coeff),
                         model_params.segment(num_coeff, num_coeff), falloff);
      break;
    }

    case COOK_TORRANCE_MODEL: {
      reflectance_model = new CookTorranceModel(
          model_params[0], model_params[1], model_params[2], falloff);
      break;
    }

    default: {
      PyErr_SetString(PyExc_RuntimeError, "Invalid reflectance model type.");
      return NULL;
    }
  }

  reflectance_model->estimate_sigmas();
  g_reflectance_model.reset(reflectance_model);

  Py_RETURN_NONE;
}



static PyObject *py_step(PyObject *self, PyObject *args) {

 // for (size_t i = 0; i != (*g_sfs_data).h; i += 1) {
 //   for (size_t j = 0; j != (*g_sfs_data).w; j += 1) {
	//		//Get current value of vmask
	//		
	//	if ((*g_sfs_data).vmask(i,j) != 0){
	//		(*g_sfs_data).v(i,j)=log((*g_sfs_data).vmask(i,j));
	//	}
	//}
 // }
  step(*g_sfs_data, *g_reflectance_model, -1, -1);
  step(*g_sfs_data, *g_reflectance_model, -1, 1);
  step(*g_sfs_data, *g_reflectance_model, 1, -1);
  step(*g_sfs_data, *g_reflectance_model, 1, 1);

  Py_RETURN_NONE;
}

//
// module administration
//

static PyMethodDef sfsMethods[] = {
    {"initialize", py_initialize, METH_VARARGS, "Run Shape from Shading."},
    {"set_model", py_set_model, METH_VARARGS, "Set reflectance model to use."},
    {"step", py_step, METH_NOARGS,
     "Perform one SFS sweep (in all four directions)."},
    {NULL, NULL, 0, NULL}};

PyMODINIT_FUNC initsfs(void) {
  PyObject *module = Py_InitModule("sfs", sfsMethods);
  
  // give access to each of the reflectance model types
  PyModule_AddIntMacro(module, LAMBERTIAN_MODEL);
  PyModule_AddIntMacro(module, OREN_NAYAR_MODEL);
  PyModule_AddIntMacro(module, PHONG_MODEL);
  PyModule_AddIntMacro(module, POWER_MODEL);
  PyModule_AddIntMacro(module, COOK_TORRANCE_MODEL);

  import_array(); /* for using numpy array objects */
}
}

#endif // __SFS_LIBRARY__

