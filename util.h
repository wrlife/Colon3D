/* Author: True Price <jtprice at cs.unc.edu>
 *
 * Header file for mapping python arrays to C++ STL and Eigen objects
 *
 * When compiling, be sure to include: -I/usr/local/include/eigen3
 *   (or whatever you need to do to reference unsupported/Eigen>
 */

#ifndef __PYTHON_TO_CPP_UTIL_H__
#define __PYTHON_TO_CPP_UTIL_H__

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

//#include <cmath>
#include <Python.h>
#include <numpy/arrayobject.h>

#include <Eigen/Core>
//#include <unsupported/Eigen/CXX11/Tensor>

#include <vector>

template <typename T>
inline std::vector<T> PyIter_to_vector(PyObject *iterable) {
  PyObject *iterator = PyObject_GetIter(iterable);
  if (iterator == NULL || !PyIter_Check(iterator)) {
    if (iterator != NULL) {
      Py_DECREF(iterator);
    }
    throw(std::exception("Object does not support iteration."));
  }

  std::vector<T> out;
  out.reserve(PyObject_Length(iterable));

  PyObject *elem;

  while (elem = PyIter_Next(iterator)) {
    out.push_back(T(elem));
    Py_DECREF(elem);
  }

  Py_DECREF(iterator);

  if (PyErr_Occurred()) {
    throw(std::exception(
        "Unspecified error while converting Python object to std::vector."));
  }

  return out;
}

//
// Map 1- or 2-D NumPy arrays to Eigen matrices
//

// map a numpy array to an eigen matrix
template <typename T, int R = Eigen::Dynamic, int C = Eigen::Dynamic,
          int Options = Eigen::RowMajor>
class PyArrayMap : public Eigen::Map<Eigen::Array<T, R, C, Options>> {
public:
  // Empty and Zeros act like their numpy counterparts; numpy_type must be
  // specified in order to properly create the underlying numpy array
  static PyArrayMap Empty(int numpy_type, int nrows = -1, int ncols = -1);
  static PyArrayMap Zeros(int numpy_type, int nrows = -1, int ncols = -1);

  PyArrayMap() : Eigen::Map<Eigen::Array<T, R, C, Options>>(NULL, 0, 0),
      arr_(NULL) {}

  PyArrayMap(PyObject *arr)
      : Eigen::Map<Eigen::Array<T, R, C, Options>>(
            (T *)PyArray_DATA((PyArrayObject *)arr), calculate_num_rows(arr),
            calculate_num_cols(arr)),
        arr_(arr) {
    this->eval(); // force update of values (TODO: why is this necessary?)
    Py_XINCREF(arr_);
  }

  // copy constructor
  PyArrayMap(const PyArrayMap &other)
      : Eigen::Map<Eigen::Array<T, R, C, Options>>(other) {
    arr_ = other.arr_;
    Py_XINCREF(arr_);
  }

  /*
  // assignment operator
  PyArrayMap &operator=(const PyArrayMap &other) {
    Eigen::Map<Eigen::Array<T, R, C, Options>>::operator=(other);
    //Eigen::Map<Eigen::Array<T, R, C, Options>>::_set(other);
    //Py_XDECREF(arr_);
    // *this = PyArrayMap(other.arr_);

    Py_XDECREF(arr_);
    arr_ = other.arr_;
    Py_XINCREF(arr_);

    return *this;
  }
  */

  ~PyArrayMap() {
    if (Py_IsInitialized()) { // this avoids segfaults on program exit
      Py_XDECREF(arr_);
    }
  }

  // This function is pretty unsafe, but we'll just trust that you don't edit
  // the memory properties of the returned matrix
  PyObject *get_py_array() const { return arr_; }

private:
  static inline PyArrayMap<T, R, C, Options> create_numpy_array_(
      int numpy_type, int nrows, int ncols,
      PyObject *(*numpy_creation_func)(int, npy_intp *, PyArray_Descr *, int));

  static inline int calculate_num_rows(PyObject *arr) {
    // row-major vectors have only one row
    if (Options == Eigen::RowMajor && PyArray_NDIM((PyArrayObject *)arr) == 1) {
      return 1;
    }

    // otherwise, the array is either column-major or MxN
    return (int)PyArray_DIM((PyArrayObject *)arr, 0);
  }

  static inline int calculate_num_cols(PyObject *arr) {
    // columns-major vectors have only one column
    if (Options == Eigen::ColMajor && PyArray_NDIM((PyArrayObject *)arr) == 1) {
      return 1;
    }

    // otherwise, the array is either row-major or MxN
    return (PyArray_NDIM((PyArrayObject *)arr) == 1)
               ? (int)PyArray_DIM((PyArrayObject *)arr, 0)
               : (int)PyArray_DIM((PyArrayObject *)arr, 1);
  }

  PyObject *arr_;
};

template <typename T, int R, int C, int Options>
inline PyArrayMap<T, R, C, Options>
PyArrayMap<T, R, C, Options>::create_numpy_array_(
    int numpy_type, int nrows, int ncols,
    PyObject *(*numpy_creation_func)(int, npy_intp *, PyArray_Descr *, int)) {
  npy_intp dims[2] = {R, C};

  if (R == Eigen::Dynamic) {
    if (nrows < 0) {
      throw(std::exception("Number of rows not specified."));
    }

    dims[0] = nrows;

    if (C == Eigen::Dynamic) {
      if (ncols < 0) {
        throw(std::exception("Number of columns not specified."));
      }

      dims[1] = ncols;
    }
  } else if (C == Eigen::Dynamic) {
    if (ncols >= 0) {
      dims[1] = ncols;
    } else if (nrows >= 0) {
      dims[1] = nrows;
    } else {
      throw(std::exception("Number of columns not specified."));
    }
  }

  PyArray_Descr *descr = PyArray_DescrFromType(numpy_type);
  return PyArrayMap<T, R, C, Options>(
      (*numpy_creation_func)(2, dims, descr, false));
}

template <typename T, int R, int C, int Options>
PyArrayMap<T, R, C, Options>
PyArrayMap<T, R, C, Options>::Empty(int numpy_type, int nrows, int ncols) {
  return create_numpy_array_(numpy_type, nrows, ncols, &PyArray_Empty);
}

template <typename T, int R, int C, int Options>
PyArrayMap<T, R, C, Options>
PyArrayMap<T, R, C, Options>::Zeros(int numpy_type, int nrows, int ncols) {
  return create_numpy_array_(numpy_type, nrows, ncols, &PyArray_Zeros);
}

// some c++0x hacking to get around not having template aliases

template <typename T, int C>
struct PyArrayMapVarRows {
  typedef PyArrayMap<T, Eigen::Dynamic, C> type;
};

template <typename T, int R>
struct PyArrayMapVarCols {
  typedef PyArrayMap<T, R, Eigen::Dynamic> type;
};

template <typename T, int R = Eigen::Dynamic>
struct PyArrayMapVector {
  typedef PyArrayMap<T, R, 1, Eigen::ColMajor> type;
};

template <typename T, int C = Eigen::Dynamic>
struct PyArrayMapRowVector {
  typedef PyArrayMap<T, 1, C> type;
};

//
// Mapping of 3D NumPy arrays (i.e., images) to Eigen Tensor objects
// NOTE: this code is pretty thoroughly untested
//

/*
// Eigen's tensor support is still a work in progress at this point in time, so
// we need to define our own specialized functions, here

// TODO: figure out how to subclass Tensor, as it's missing functionality I'd
//       either like to have (such as fill()) or would like to add (e.g. at())
template <typename T>
using Image = Eigen::Tensor<T, 3, Eigen::RowMajor>;
template <typename T>
using ImageElem = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstImageElem = Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>;

template <typename T>
class ImageMap : public Eigen::TensorMap<Image<T>> {
public:
  ImageMap(PyObject *im)
      : height(PyArray_DIM((PyArrayObject *)im, 0)),
        width(PyArray_DIM((PyArrayObject *)im, 1)),
        depth(PyArray_DIM((PyArrayObject *)im, 2)),
        Eigen::TensorMap<Image<T>>((double *)PyArray_DATA((PyArrayObject *)im),
                                   (int)PyArray_DIM((PyArrayObject *)im, 0),
                                   (int)PyArray_DIM((PyArrayObject *)im, 1),
                                   (int)PyArray_DIM((PyArrayObject *)im, 2)) {}

  inline ImageElem<T> at(const int i, const int j) {
    return ImageElem<T>(&(*this)(i, j, 0), depth, 1);
  }

  inline ConstImageElem<T> at(const int i, const int j) const {
    return ConstImageElem<T>(&(*this)(i, j, 0), depth, 1);
  }

  template <typename OtherDerived>
  ImageMap<T> &operator=(const OtherDerived &other) {
    (*this).Eigen::TensorMap<Image<T>>::operator=(other);
    return *this;
  }

  const int height, width, depth;
};
*/

#endif // __PYTHON_TO_CPP_UTIL_H__

