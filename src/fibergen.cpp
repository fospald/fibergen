/**

\brief fibergen



*/

// http://de.mathworks.com/company/newsletters/articles/the-watershed-transform-strategies-for-image-segmentation.html


//#define TEST_MERGE_ISSUE		// test openmp issue due to kernel bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=65589
//#define TEST_DIST_EVAL		// report distance evaluations for fiber generator 

/*
NOTES:
- Pass array from Pyton to C++: http://www.shocksolution.com/python-basics-tutorials-and-examples/boostpython-numpy-example/
- http://scalability.org/?p=5770
- https://software.intel.com/sites/products/documentation/studio/composer/en-us/2011Update/compiler_c/intref_cls/common/intref_avx2_mm256_i32gather_pd.htm
- http://stackoverflow.com/questions/3596622/negative-nan-is-not-a-nan

TODO:
- use https://bitbucket.org/account/user/VisParGroup/projects/UTANS for CT images
- use Frangi ITK filter (probably itk::Hessian3DToVesselnessMeasureImageFilter) for CT images (as in Herrmann paper: Methods for fibre orientation analysis of X-ray tomography images of steel fibre reinforced concrete (SFRC))
- add and check error tolerances for boundary conditions
- adaptive time stepping and Newton step relaxation on non-convergence 
- try http://numba.pydata.org/
- try https://code.google.com/p/blaze-lib/
- check applyDeltaStaggered
- calcStress const
- use correct mixing rule (s. Milton 9.3)
- swap sigma and epsilon (save vtk, get_field function) in dual scheme (viscosity mode)
- G0OperatorFourierStaggered is 4 times the cost of a FFT
- closestFiber is very expensive
- fast phi method not periodic (bug)
- test different convergence checks (what makes sense for basic and cg schemes? ask Matti!)
- Angular Gaussian in 2D
- make use of tr(epsilon)=0
- clear all fibers action
- general output directory
- different materials
- strain energy check
- write escapecodes only if isatty(FILE)
- correct material constants/equations for 2d
- place_fiber periodic
- implement some checks/results from "David A. Jack and Douglas E. Smith: Elastic Properties of Short-fiber Polymer Composites, Derivation and Demonstration of Analytical Forms for Expectation and Variance from Orientation Tensors", Journal of Composite Materials 2008 42: 277

multigrid improvements:
- solve vector poisson eqation instead of 3 scalar equations
- blocking of smoother 2x2x2 blocks will reduce cache misses
- do not check residuals just run a fixed number of vcycles
- or compute residual within the smoother loop
- combine last smoothing with restriction operation
*/

//#define USE_MANY_FFT

#include <Python.h>
#include <fftw3.h>
#ifdef OPENMP_ENABLED
	#include <omp.h>
#endif
#include <unistd.h>
#include <stdint.h>

#ifdef INTRINSICS_ENABLED
	#include <immintrin.h>
#endif

#ifdef IACA_ENABLED
	#include <iacaMarks.h>
#else
	#define IACA_START
	#define IACA_END
#endif

#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/tee.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/asio.hpp>

#include <boost/format.hpp>
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include <boost/optional/optional.hpp>
#include <boost/ref.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/preprocessor/stringize.hpp>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/assignment.hpp>
#include <boost/numeric/ublas/storage.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp> 
#include <boost/numeric/conversion/bounds.hpp>

#include <boost/numeric/bindings/traits/ublas_vector2.hpp>
#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/numeric/bindings/lapack/gesvd.hpp>
#include <boost/numeric/bindings/lapack/geev.hpp>
#include <boost/numeric/bindings/lapack/gesv.hpp>
#include <boost/numeric/bindings/lapack/syev.hpp>
#include <boost/numeric/bindings/lapack/sysv.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#include <boost/accumulators/statistics/variance.hpp>

#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>

#include <boost/exception/diagnostic_information.hpp>
#include <boost/throw_exception.hpp>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/posix_time/posix_time_io.hpp>
#include <boost/detail/endian.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <boost/math/special_functions/ellint_rj.hpp>
#include <boost/math/special_functions/ellint_rf.hpp>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

#include <boost/program_options.hpp>
#include <boost/limits.hpp>
#include <boost/random.hpp>
#include <boost/thread.hpp>

#define NPY_NO_DEPRECATED_API 7
//#include <numpy/noprefix.h>
#include <numpy/npy_3kcompat.h>

#define PNG_SKIP_SETJMP_CHECK
#define png_infopp_NULL (png_infopp)NULL
#define int_p_NULL (int*)NULL
#if BOOST_VERSION >= 106800
#include <boost/gil.hpp>
#include <boost/gil/extension/io/png.hpp>
#include <boost/gil/io/write_view.hpp>
#else
#include <boost/gil/gil_all.hpp>
#include <boost/gil/extension/io/png_dynamic_io.hpp>
#endif

#include <boost/python.hpp>
#include <boost/python/raw_function.hpp>
#if BOOST_VERSION >= 106500
#include <boost/python/numpy.hpp>
#endif

#undef ITK_ENABLED

#ifdef ITK_ENABLED
#include <itkImage.h>
#include "itkBinaryThinningImageFilter3D.h"
#endif

#include <csignal>
#include <stdexcept>
#include <execinfo.h>
#include <cxxabi.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <limits>
#include <list>
#include <algorithm>

namespace ptree = boost::property_tree;
namespace ublas = boost::numeric::ublas;
namespace gil = boost::gil;
namespace acc = boost::accumulators;
namespace pt = boost::posix_time;
namespace po = boost::program_options;
namespace lapack = boost::numeric::bindings::lapack;
namespace py = boost::python;

// http://www.cplusplus.com/forum/unices/36461/
#define _DEFAULT_TEXT	"\033[0m"
#define _BOLD_TEXT	"\033[1m"
#define _RED_TEXT	"\033[0;31m"
#define _GREEN_TEXT	"\033[0;32m"
#define _YELLOW_TEXT	"\033[1;33m"
#define _BLUE_TEXT	"\033[0;34m"
#define _WHITE_TEXT	"\033[0;97m"
#define _INVERSE_COLORS	"\033[7m"
#define _CLEAR_EOL	"\033[K"

#define DEFAULT_TEXT	TTYOnly(_DEFAULT_TEXT)
#define BOLD_TEXT	TTYOnly(_BOLD_TEXT)
#define RED_TEXT	TTYOnly(_RED_TEXT)
#define GREEN_TEXT	TTYOnly(_GREEN_TEXT)
#define YELLOW_TEXT	TTYOnly(_YELLOW_TEXT)
#define BLUE_TEXT	TTYOnly(_BLUE_TEXT)
#define WHITE_TEXT	TTYOnly(_WHITE_TEXT)
#define INVERSE_COLORS	TTYOnly(_INVERSE_COLORS)
#define CLEAR_EOL	TTYOnly(_CLEAR_EOL)


#define PRECISION Logger::instance().precision()

#ifdef DEBUG
	#define DEBP(x) LOG_COUT << x << std::endl
#else
	#define DEBP(x)
#endif

#define STD_INFINITY(T) std::numeric_limits<T>::infinity()


#define BEGIN_TRIPLE_LOOP(var, nx, ny, nz, nzp) \
	for (std::size_t var ## _i = 0; var ## _i < (nx); var ## _i++) { \
		for (std::size_t var ## _j = 0; var ## _j < (ny); var ## _j++) { \
			std::size_t var = var ## _i*(ny)*(nzp) + var ## _j*(nzp); \
			for (std::size_t var ## _k = 0; var ## _k < (nz); var ## _k++) {
#define END_TRIPLE_LOOP(var) var++; }}}


#ifdef __GNUC__
	#define noinline __attribute__ ((noinline))
#else
	#define noinline
#endif


#if 0
class TTYOnly
{
	const char* _code;
public:
	TTYOnly(const char* code) : _code(code) { }
	friend std::ostream& operator<<(std::ostream& os, const TTYOnly& tto);
};

std::ostream& operator<<(std::ostream& os, const TTYOnly& tto)
{
//	if (isatty(fileno(stdout)) && isatty(fileno(stderr))) os << tto._code;
	return os;
}
#else
	#define TTYOnly(x) (x)
#endif


//! Class for logging
class Logger
{
protected:
	typedef boost::iostreams::tee_device<std::ostream, std::ofstream> Tee;
	typedef boost::iostreams::stream<Tee> TeeStream;

	std::size_t _indent;	// current output indentation
	boost::shared_ptr<Tee> _tee;
	boost::shared_ptr<TeeStream> _tee_stream;
	boost::shared_ptr<std::ofstream> _tee_file;
	boost::shared_ptr<std::ostream> _stream;

	static boost::shared_ptr<Logger> _instance;

	void cleanup()
	{
		_stream.reset();
		_tee_stream.reset();
		_tee.reset();
		_tee_file.reset();
	}

public:
	Logger() : _indent(0)
	{
		_stream.reset(new std::ostream(std::cout.rdbuf()));
	}

	Logger(const std::string& tee_filename) : _indent(0)
	{
		setTeeFilename(tee_filename);
	}

	~Logger()
	{
		cleanup();
	}

	void setTeeFilename(const std::string& tee_filename)
	{
		cleanup();
		_tee_file.reset(new std::ofstream(tee_filename.c_str()));
		_tee.reset(new Tee(std::cout, *_tee_file));
		_tee_stream.reset(new TeeStream(*_tee));
		_stream.reset(new std::ostream(_tee_stream->rdbuf()));
	}

	//! Increase indent
	void incIndent() { _indent++; }

	//! Decrease indent
	void decIndent() { _indent = (size_t) std::max(((int)_indent)-1, 0); }

	//! Write indent to stream
	void indent(std::ostream& stream) const
	{
		std::string space(2*_indent, ' ');
		stream.write(space.c_str(), space.size()*sizeof(char));
	}

	std::streamsize precision() const
	{
		return _stream->precision();
	}

	void flush()
	{
		_stream->flush();
	}

	//! Return standard output stream
	std::ostream& cout()
	{
		std::ostream& stream = *_stream;
		stream << DEFAULT_TEXT;
		//stream << _indent;
		indent(stream);
		return stream;
	}

	//! Return error stream
	std::ostream& cerr() const
	{
		std::ostream& stream = std::cerr;
		// indent(stream);
		stream << RED_TEXT << "ERROR: ";
		return stream;
	}

	//! Return warning stream
	std::ostream& cwarn() const
	{
		std::ostream& stream = std::cerr;
		// indent(stream);
		stream << YELLOW_TEXT << "WARNING: ";
		return stream;
	}

	//! Return static instance
	static Logger& instance()
	{
		if (!_instance) {
			_instance.reset(new Logger());
		}

		return *_instance;
	}
};

// Static logger instance
boost::shared_ptr<Logger> Logger::_instance;

// Shortcut macros for cout and cerr
#define LOG_COUT Logger::instance().cout()
#define LOG_CERR Logger::instance().cerr()
#define LOG_CWARN Logger::instance().cwarn()

// Static exception object
boost::shared_ptr<std::exception> _except;


#ifndef OPENMP_ENABLED
void omp_set_nested(bool n) { }
void omp_set_dynamic(bool d) { }
void omp_set_num_threads(int n) {
	if (n > 1) LOG_CWARN << "OpenMP is disabled, only running with 1 thread!" << std::endl;
}
int omp_get_thread_num() { return 0; }
int omp_get_num_threads() { return 1; }
int omp_get_max_threads() { return 1; }
#endif


//! Set current exception message and print message to cerr
void set_exception(const std::string& msg)
{
	#pragma omp critical
	{
		// ignore multiple exceptions
		if (!_except) {
			_except.reset(new std::runtime_error(msg));
			LOG_CERR << "Exception set: " << msg << std::endl;
		}
	}
}

//! Print backtrace of current function calls
inline void print_stacktrace(std::ostream& stream)
{
	// print stack trace
	void *trace_elems[32];
	int trace_elem_count = backtrace(trace_elems, sizeof(trace_elems)/sizeof(void*));
	char **symbol_list = backtrace_symbols(trace_elems, trace_elem_count);
	std::size_t funcnamesize = 265;
	char* funcname = (char*) malloc(funcnamesize);

	stream << "Stack trace:" << std::endl;

	// iterate over the returned symbol lines. skip the first, it is the
	// address of this function.
	int c = 0;
	for (int i = trace_elem_count-1; i > 1; i--)
	{
		char *begin_name = 0, *begin_offset = 0, *end_offset = 0;

		// find parentheses and +address offset surrounding the mangled name:
		// ./module(function+0x15c) [0x8048a6d]
		for (char *p = symbol_list[i]; *p; ++p)
		{
			if (*p == '(')
				begin_name = p;
			else if (*p == '+')
				begin_offset = p;
			else if (*p == ')' && begin_offset) {
				end_offset = p;
				break;
			}
		}

		if (begin_name && begin_offset && end_offset && begin_name < begin_offset)
		{
			*begin_name++ = '\0';
			*begin_offset++ = '\0';
			*end_offset = '\0';

			// mangled name is now in [begin_name, begin_offset) and caller
			// offset in [begin_offset, end_offset). now apply
			// __cxa_demangle():

			int status;
			char* ret = abi::__cxa_demangle(begin_name, funcname, &funcnamesize, &status);
			if (status == 0) {
				funcname = ret; // use possibly realloc()-ed string
				stream << (boost::format("%02d. %s: " _WHITE_TEXT "%s+%s") % c % symbol_list[i] % funcname % begin_offset).str() << std::endl;
			}
			else {
				// demangling failed. Output function name as a C function with
				// no arguments.
				stream << (boost::format("%02d. %s: %s()+%s") % c % symbol_list[i] % begin_name % begin_offset).str() << std::endl;
			}
		}
		else
		{
			// couldn't parse the line? print the whole line.
			stream << (boost::format("%02d. " _WHITE_TEXT "%s") % c % symbol_list[i]).str() << std::endl;
		}

		c++;
	}

	free(symbol_list);
	free(funcname);
}



// Static empty ptree object
static ptree::ptree empty_ptree;


//! Open file for output, truncate file if already exists
inline void open_file(std::ofstream& fs, const std::string& filename)
{
	fs.open(filename.c_str(), std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);

	if (fs.fail()) {
		BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("error opening '%s': %s") % filename % strerror(errno)).str()));
	}
}


//! Class with math operations for Voigt's notation
class Voigt
{
public:
	//! Identity matrix for dimension dim
	//! \param dim the dimension (3, 6 or 9)
	//! \return matrix
	template <typename T>
	static inline ublas::matrix<T> Id4(std::size_t dim) 
	{
		if (dim == 9 || dim == 3) {
			return ublas::identity_matrix<T>(dim);
		}

		ublas::matrix<T> Id = ublas::identity_matrix<T>(6);
		Id(3,3) = 0.5;
		Id(4,4) = 0.5;
		Id(5,5) = 0.5;
		return Id;
	}

	//! Outer product of rank 2 identity matrices as Voigt matrix
	//! \param dim the dimension (3, 6 or 9)
	//! \return matrix
	template <typename T>
	static inline ublas::matrix<T> II4(std::size_t dim) 
	{
		ublas::matrix<T> II = ublas::zero_matrix<T>(dim);
		II(0,0) = 1.0; II(0,1) = 1.0; II(0,2) = 1.0;
		II(1,0) = 1.0; II(1,1) = 1.0; II(1,2) = 1.0;
		II(2,0) = 1.0; II(2,1) = 1.0; II(2,2) = 1.0;
		return II;
	}

	//! 2-norm of a vector
	//! \param v the vector
	template <typename T>
	static inline T norm_2(const ublas::vector<T>& v)
	{
		if (v.size() == 9 || v.size() == 3) {
			return std::sqrt(ublas::inner_prod(v, v));
		}

		return std::sqrt(ublas::inner_prod(v, v) + v(3)*v(3) + v(4)*v(4) + v(5)*v(5));
	}

	//! Dyadic product of Voigt vectors
	//! \param a first operand
	//! \param b second operand
	//! \return result scalar
	template <typename T>
	static inline T dyad(const ublas::vector<T>& a, const ublas::vector<T>& b)
	{
		if (a.size() == 9 || a.size() == 3) {
			return ublas::inner_prod(a, b);
		}

		ublas::vector<T> ac(a);
		ac(3) *= 2;
		ac(4) *= 2;
		ac(5) *= 2;

		return ublas::inner_prod(ac, b);
	}

	//! Dyadic product of Voigt matrix and Voigt vector
	//! \param M matrix
	//! \param v vector
	//! \return result vector
	template <typename T>
	static inline ublas::vector<T> dyad4(const ublas::matrix<T>& M, const ublas::vector<T>& v)
	{
		if (v.size() == 9 || v.size() == 3) {
			return ublas::prod(M, v);
		}

		ublas::vector<T> vc(v);
		vc(3) *= 2;
		vc(4) *= 2;
		vc(5) *= 2;

		return ublas::prod(M, vc);
	}

	//! Dyadic product of Voigt matrices
	//! \param A first operand
	//! \param B second operand
	//! \return result matrix
	template <typename T>
	static inline ublas::matrix<T> dyad4(const ublas::matrix<T>& A, const ublas::matrix<T>& B)
	{
		std::size_t dim = A.size1();

		if (dim == 9 || dim == 3) {
			return ublas::prod(A, B);
		}

		ublas::matrix<T> C(dim, dim);

		for (std::size_t i = 0; i < dim; i++) {
			ublas::column(C, i) = Voigt::dyad4(A, ublas::vector<T>(ublas::column(B, i)));
		}

		return C;
	}
};


//! Compute a orthonormal vector to v
//! \param v vector
//! \return orthonormal vector
template <typename T, int DIM>
inline ublas::c_vector<T, DIM> orthonormal_vector(const ublas::c_vector<T, DIM>& v)
{
	int i_max = 0, i_min = 0;

	for (int i = 0; i < DIM; i++) {
		if (fabs(v[i]) < fabs(v[i_min])) i_min = i;
		if (fabs(v[i]) > fabs(v[i_max])) i_max = i;
	}

	if (i_min == i_max) i_min = (i_max+1) % DIM;

	ublas::c_vector<T, DIM> x = v;
	x[i_min] = -v[i_max];
	x[i_max] = v[i_min];

	x = x - ublas::inner_prod(x, v)*x;
	x = x/ublas::norm_2(x);
	return x;
}


//! Compute cross product between two vectors
//! \param lhs first vector
//! \param rhs second vector
//! \return cross product vector
template <class V1, class V2>
inline boost::numeric::ublas::vector<typename boost::numeric::ublas::promote_traits<typename V1::value_type, typename V2::value_type>::promote_type>
cross_prod(const V1& lhs, const V2& rhs)
{
    BOOST_UBLAS_CHECK(lhs.size() == 3, boost::numeric::ublas::external_logic());
    BOOST_UBLAS_CHECK(rhs.size() == 3, boost::numeric::ublas::external_logic());

    typedef typename boost::numeric::ublas::promote_traits<typename V1::value_type, typename V2::value_type>::promote_type promote_type;

    boost::numeric::ublas::vector<promote_type> temporary(3);

    temporary(0) = lhs(1) * rhs(2) - lhs(2) * rhs(1);
    temporary(1) = lhs(2) * rhs(0) - lhs(0) * rhs(2);
    temporary(2) = lhs(0) * rhs(1) - lhs(1) * rhs(0);

    return temporary;
}


//! Remove common leading whitespace from (multi-line) string
//! \param s string to dedent
//! \return dedented string
std::string dedent(const std::string& s)
{
	std::vector<std::string> lines;
	std::string indent;
	bool have_indent = false;

	boost::split(lines, s, boost::is_any_of("\n"));

	for (std::size_t i = 0; i < lines.size(); i++)
	{
		boost::algorithm::trim_right_if(lines[i], boost::is_any_of(" \t\v\f\r"));

		if (lines[i].size() == 0) continue;

		if (!have_indent)
		{
			std::string trimmed_line = boost::algorithm::trim_left_copy_if(lines[i], boost::is_any_of(" \t\v\f\r"));
			indent = lines[i].substr(0, lines[i].size() - trimmed_line.size());
			have_indent = true;
			continue;
		}
		
		for (std::size_t j = 0; j < std::max(lines[i].size(), indent.size()); j++) {
			if (lines[i][j] != indent[j]) {
				indent = indent.substr(0, j);
				break;
			}
		}
	}

	for (std::size_t i = 0; i < lines.size(); i++) {
		if (lines[i].size() == 0) continue;
		lines[i] = lines[i].substr(indent.size());
	}

	return boost::join(lines, "\n");
}


//! Class for evaluating python code and managing local variables accross evaluations
class PY
{
protected:
	static boost::shared_ptr<PY> _instance;

	py::object main_module, main_namespace;
	py::dict locals;
	bool enabled;

public:
	PY()
	{
		enabled = false;
	}

#if 0
	~PY()
	{
		LOG_COUT << "~PY" << std::endl;
	}
#endif

	//! Execute python code
	//! \param code the python code
	//! \return result of executing the code
	py::object exec(const std::string& code)
	{
		if (enabled) {
			std::string c = dedent(code);
			return py::exec(c.c_str(), main_namespace, locals);
		}

		return py::object();
	}

	//! Evaluate python expression as type T
	//! \param expr the expression string
	//! \return result converted to type T
	template <class T>
	T eval(const std::string& expr)
	{
		if (enabled) {
			std::string e = expr;
			boost::trim(e);
			py::object result = py::eval(e.c_str(), main_namespace, locals);
			T ret = py::extract<T>(result);
			return ret;
		}

		return boost::lexical_cast<T>(expr);
	}

	//! Get string from ptree and eval as python expression to type T
	//! \param pt the ptree
	//! \param prop the path of the property
	//! \param default_value the default value, if the property does not exists
	//! \return the poperty value
	template <class T>
	T get(const ptree::ptree& pt, const std::string& prop, T default_value)
	{
		boost::optional<std::string> value = pt.get_optional<std::string>(prop);

		if (!value) {
			return default_value;
		}

		return eval<T>(*value);
	}

	//! Get string from ptree and eval as python expression to type T
	//! \param pt the ptree
	//! \param prop the path of the property
	//! \return the poperty value
	//! \exception if the property does not exists
	template <class T>
	T get(const ptree::ptree& pt, const std::string& prop)
	{
		boost::optional<std::string> value = pt.get_optional<std::string>(prop);

		if (!value) {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Undefined property: %s!") % prop).str()));
		}

		return eval<T>(*value);
	}

	//! Enable/Disable python evaluation
	//! If python evaluation is disabled strings are cast to the requested type by a lexical cast
	void set_enabled(bool enabled)
	{
		if (this->enabled == enabled) return;
		this->enabled = enabled;
		
		if (enabled)
		{
			// init main namespace
			main_module = py::import("__main__");
			main_namespace = main_module.attr("__dict__");
			py::exec("from math import *", main_namespace);
		}
	}

	//! Clear all local variables
	void clear_locals()
	{
		this->locals = py::dict();
	}

	//! Add a local variable
	//! \param key the variable name
	//! \param value the variable value
	void add_local(const std::string& key, const py::object& value)
	{
		this->locals[key] = value;
	}

	//! Return static instance of class
	static PY& instance()
	{
		if (!_instance) {
			_instance.reset(new PY());
		}

		return *_instance;
	}

	//! Release static instance of class
	static void release()
	{
		_instance.reset();
	}
};

// Static instance of PY class
boost::shared_ptr<PY> PY::_instance;


//! Shortcut for getting a property from a ptree with python evaluation
//! \param pt the ptree
//! \param prop the property path
//! \return the property
template <class T>
T pt_get(const ptree::ptree& pt, const std::string& prop)
{
	return PY::instance().get<T>(pt, prop);
}

//! Shortcut for getting a property from a ptree with python evaluation and default value
//! \param pt the ptree
//! \param prop the property path
//! \param default_value the default value, if property was not found
//! \return the property
template <class T>
T pt_get(const ptree::ptree& pt, const std::string& prop, T default_value)
{
	return PY::instance().get<T>(pt, prop, default_value);
}

//! Shortcut for getting a property from a ptree as std::string
//! \param pt the ptree
//! \param prop the property path
//! \return the property
template <>
std::string pt_get(const ptree::ptree& pt, const std::string& prop)
{
	boost::optional<std::string> value = pt.get_optional<std::string>(prop);

	if (!value) {
		BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Undefined property: %s!") % prop).str()));
	}

	return *value;
}

//! Shortcut for getting a property from a ptree as std::string with default value
//! \param pt the ptree
//! \param prop the property path
//! \param default_value the default value, if property was not found
//! \return the property
template <>
std::string pt_get(const ptree::ptree& pt, const std::string& prop, std::string default_value)
{
	boost::optional<std::string> value = pt.get_optional<std::string>(prop);

	if (!value) {
		return default_value;
	}

	return *value;
}

//! Format a vector expression as string (for console output)
//! \param v the vector expression
//! \param indent enable line indentation using the current logger indent
//! \return the formatted text
template <class VE>
std::string format(const ublas::vector_expression<VE>& v, bool indent = false)
{
	typedef typename VE::value_type T;
	std::size_t N = v().size();
	std::stringstream ss;

	if (indent) {
		Logger::instance().incIndent();
		Logger::instance().indent(ss);
	}

	ss << "( ";
	for (std::size_t i = 0; i < N; i++) {
		if (i > 0) ss << ", ";
		ss << BLUE_TEXT
			<< std::setiosflags(std::ios::scientific | std::ios::showpoint)
			<< std::setprecision(PRECISION) << std::setw(PRECISION+7) << std::right
			<< v()(i) << DEFAULT_TEXT;
	}
	ss << " )";

	if (indent) {
		Logger::instance().decIndent();
	}

	return ss.str();
}


//! Format a matrix expression as string (for console output), lines are indented using the current logger indent
//! \param m the matrix expression
//! \return the formatted text
template <class ME>
std::string format(const ublas::matrix_expression<ME>& m)
{
	typedef typename ME::value_type T;
	std::size_t N = m().size1();
	std::size_t M = m().size2();

	T small = boost::numeric::bounds<T>::smallest();
	T large = boost::numeric::bounds<T>::highest();
	int logmax = std::log10(small), logmin = -logmax;

	for (std::size_t i = 0; i < N; i++) {
		for (std::size_t j = 0; j < M; j++) {
			int lg = std::log10(small + std::min(std::fabs(m()(i, j)), large));
			logmax = std::max(logmax, lg);
			logmin = std::min(logmin, lg);
		}
	}

	std::stringstream ss;

	Logger::instance().incIndent();
	for (std::size_t i = 0; i < N; i++) {
		ss << std::endl;
		Logger::instance().indent(ss);
		for (std::size_t j = 0; j < M; j++) {
			if (j > 0) ss << " ";
			int lg = std::log10(small + std::min(std::fabs(m()(i, j)), large));
			int color = 255 - std::min(logmax - lg, 10)*2;
			ss << "\x1b[38;5;" << color << "m" << TTYOnly(i == j ? _INVERSE_COLORS : "")
				<< std::setiosflags(std::ios::scientific | std::ios::showpoint)
				<< std::setprecision(PRECISION) << std::setw(PRECISION+7) << std::right
				<< m()(i, j) << DEFAULT_TEXT;
		}
	}
	Logger::instance().decIndent();

	//std::cout << "'" << ss.str() << "'" << std::endl;

	return ss.str();
}


//! Format a 3d tensor as string (for console output)
//! \param data pointer to the data, adressed as data[i*ny*nzp + j*nzp + k] for (i,j,k)-th element
//! \param nx number of elements in x (index i)
//! \param ny number of elements in y (index j)
//! \param nz number of elements in z (index k)
//! \param nzp number of elements in z including padding
//! \return the formatted text
template <typename T>
std::string format(const T* data, std::size_t nx, std::size_t ny, std::size_t nz, std::size_t nzp)
{
	ublas::matrix<T> A(ny, nx);
	std::string s;

	for (std::size_t kk = 0; kk < nz; kk++) {
		if (kk > 0) s += "\n";
		for (std::size_t jj = 0; jj < ny; jj++) {
			for (std::size_t ii = 0; ii < nx; ii++) {
				A(jj, ii) = data[ii*ny*nzp + jj*nzp + kk];
			}
		}
		s += format(A) + "\n";
	}

	return s;
}


//! Compute square of 2-norm of vector
//! \param v the vector
//! \return the norm
template <typename T>
inline T norm_2_sqr(const ublas::vector<T>& v)
{
	return ublas::inner_prod(v, v);
}


//! Set components of a vector
//! \param v the vector
//! \param x0 the value for v[0]
//! \param x1 the value for v[1]
//! \param x2 the value for v[2]
template <class V, typename T>
inline void set_vector(V& v, T x0, T x1, T x2)
{
	if (v.size() >= 1) v[0] = x0;
	if (v.size() >= 2) v[1] = x1;
	if (v.size() >= 3) v[2] = x2;
}

//! Set components of a vector
//! \param attr ptree with settings
//! \param v the vector
//! \param name0 settings name for component 0
//! \param name1 settings name for component 1
//! \param name2 settings name for component 2
//! \param def0 default value for component 0
//! \param def1 default value for component 1
//! \param def2 default value for component 2
template <class V, typename T>
inline void read_vector(const ptree::ptree& attr, V& v, const char* name0, const char* name1, const char* name2, T def0, T def1, T def2)
{
	if (v.size() >= 1) v[0] = pt_get<T>(attr, name0, def0);
	if (v.size() >= 2) v[1] = pt_get<T>(attr, name1, def1);
	if (v.size() >= 3) v[2] = pt_get<T>(attr, name2, def2);
}

//! Set components of a matrix
//! \param attr ptree with settings
//! \param m the matrix
//! \param prefix prefix for the component names "prefix%d%d"
//! \param symmetric set to true for symmetric matrix
template <typename T>
inline void read_matrix(const ptree::ptree& attr, ublas::matrix<T>& m, const std::string& prefix, bool symmetric)
{
	for (std::size_t i = 0; i < m.size1(); i++) {
		for (std::size_t j = 0; j < m.size2(); j++) {
			std::string name = (((boost::format("%s%d%d") % prefix) % (i+1)) % (j+1)).str();
			boost::optional< const ptree::ptree& > a = attr.get_child_optional(name);
#if 0
			if (!a && i == j) {
				name = (((boost::format("%s%d") % prefix) % (i+1))).str();
				a = attr.get_child_optional(name);
			}
#endif
			if (a) {
				m(i,j) = pt_get<T>(attr, name, m(i,j));
				if (symmetric) m(j,i) = m(i,j);
			}
		}
	}
}

//! Set components of a Voigt vector
//! \param attr ptree with settings
//! \param v the vector
//! \param prefix prefix for the component names "prefix%d"
template <typename T>
inline void read_voigt_vector(const ptree::ptree& attr, ublas::vector<T>& v, const std::string& prefix)
{
	const std::size_t voigt_indices[9] = {11, 22, 33, 23, 13, 12, 32, 31, 21};

	for (std::size_t i = 0; i < 3; i++) {
		v(i) = pt_get<T>(attr, ((boost::format("%s%d") % prefix) % (i+1)).str(), v(i));
	}

	for (std::size_t i = 0; i < v.size(); i++) {
		v(i) = pt_get<T>(attr, ((boost::format("%s%d") % prefix) % voigt_indices[i]).str(), v(i));
	}
}


//! Matrix inversion routine. Uses gesv in uBLAS to invert a matrix.
//! \param input input matrix
//! \param inverse inverse ouput matrix
template<typename T, int DIM>
void InvertMatrix(const ublas::c_matrix<T,DIM,DIM>& input, ublas::c_matrix<T,DIM,DIM>& inverse)
{
#if 1
	ublas::c_matrix<T,DIM,DIM> icopy(input);
	inverse.assign(ublas::identity_matrix<T>(input.size1()));
	int res = lapack::gesv(icopy, inverse);
	if (res != 0) {
		BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Matrix inversion failed for matrix:\n%s!") % format(input)).str()));
	}
#else
	typedef ublas::permutation_matrix<std::size_t> pmatrix;

	// create a working copy of the input
	ublas::c_matrix<T,DIM,DIM> A(input);

	// create a permutation matrix for the LU-factorization
	pmatrix pm(A.size1());

	// perform LU-factorization
	int res = ublas::lu_factorize(A, pm);
	if (res != 0) {
		BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("LU factorization failed for matrix:\n%s!") % format(input)).str()));
	}

	// create identity matrix of "inverse"
	inverse.assign(ublas::identity_matrix<T>(A.size1()));

	// backsubstitute to get the inverse
	ublas::lu_substitute(A, pm, inverse);
#endif
}


template <typename T, int DIM>
inline T halfspace_box_cut_volume_old(const ublas::c_vector<T, DIM>& x, const ublas::c_vector<T, DIM>& n, const ublas::c_vector<T, DIM>& x0, T dx, T dy, T dz)
{
	T eps = std::numeric_limits<T>::epsilon()*(dx+dy+dz);

	// compute intersection volume of cube (x0, dx, dy, dz) and halfspace (x, n)

	ublas::c_vector<T, DIM> x1, x2, x3;

	x1[0] = x0[0] + dx;
	x1[1] = x0[1];
	x1[2] = x0[2] + dz;

	x2[0] = x0[0] + dx;
	x2[1] = x0[1] + dy;
	x2[2] = x0[2];

	x3[0] = x0[0];
	x3[1] = x0[1] + dy;
	x3[2] = x0[2] + dz;

	const ublas::c_vector<T, DIM>* px[4];
	px[0] = &x0;
	px[1] = &x1;
	px[2] = &x2;
	px[3] = &x3;

	// list of edges
	// e0: x0 + t*ex
	// e1: x0 + t*ey
	// e2: x0 + t*ez
	//
	// e3: x1 - t*ex
	// e4: x1 + t*ey
	// e5: x1 - t*ez
	//
	// e6: x2 - t*ex
	// e7: x2 - t*ey
	// e8: x2 + t*ez
	//
	// e9: x3 + t*ex
	// ea: x3 - t*ey
	// eb: x3 - t*ez

	static T signs[12] = {1,1,1, -1,1,-1, -1,-1,1, 1,-1,-1};

	T dx0n = ublas::inner_prod(x - x0, n);
	T dx1n = ublas::inner_prod(x - x1, n);
	T dx2n = ublas::inner_prod(x - x2, n);
	T dx3n = ublas::inner_prod(x - x3, n);

	// compute intersection parameters
	T t[12];
	for (int k = 0; k < 3; k++) {
		if (n[k] != 0) {
			t[0 + k] = signs[0 + k]*dx0n/n[k];
			t[3 + k] = signs[3 + k]*dx1n/n[k];
			t[6 + k] = signs[6 + k]*dx2n/n[k];
			t[9 + k] = signs[9 + k]*dx3n/n[k];
		}
		else {
			t[0 + k] = t[3 + k] = t[6 + k] = t[9 + k] = -1;
		}
	}

	// check for intersections
	bool i[12];
	for (int k = 0; k < 4; k++) {
		i[3*k + 0] = (t[3*k + 0] >= 0 && t[3*k + 0] <= dx);
		i[3*k + 1] = (t[3*k + 1] >= 0 && t[3*k + 1] <= dy);
		i[3*k + 2] = (t[3*k + 2] >= 0 && t[3*k + 2] <= dz);
	}

	// get first intersection point
	ublas::c_vector<T, DIM> pi0, pi1, pi2;
	int pi0_index = -1;
	for (int k = 0; k < 12; k++) {
		if (i[k]) {
			pi0 = *px[k/3];
			pi0[k%3] += signs[k]*t[k];
			pi0_index = k;
			break;
		}
	}

#if 0
	#define DEBUG_HSBCV(x) LOG_COUT << x << std::endl;

	DEBUG_HSBCV("x " << format(x));
	DEBUG_HSBCV("n " << format(n));
	DEBUG_HSBCV("x0 " << format(x0));
	for (int k = 0; k < 12; k++) {
		DEBUG_HSBCV("intersection " << k << ": " << (i[k] ? 1 : 0) << " t=" << t[k]);
	}
#else
	#define DEBUG_HSBCV(x)
#endif

	T I;

	// perform volume integration by surface integration (Gauss)
	if (pi0_index < 0) {
		// no intersection found
		I = (dx0n > 0) ? (dx*dy*dz) : 0;
		DEBUG_HSBCV("no intersection" << I);
		return I;
	}

	// V = int_V 1 dx = int_S z*n_z ds
	// only the facets with n_z != 0 are important
	// we shift the orign of the cube to to x1 so only one surface integral is left

	// edges for each facet:
	// f0: 0 5 3 2
	// f1: 6 8 9 11
	// f2: 5 4 8 7
	// f3: 2 1 11 10
	// f4: 0 7 6 1  (n_z = -1)
	// f5: 3 10 9 4 (n_z = +1)

	if (i[0] && i[6]) {
		I = 0.5*(t[0] + dx - t[6])*dy*dz;
		if (n[0] < 0) I = dx*dy*dz - I;
		DEBUG_HSBCV("case 1 " << I);
	}
	else if (i[1] && i[7]) {
		I = 0.5*(t[1] + dy - t[7])*dx*dz;
		if (n[1] < 0) I = dx*dy*dz - I;
		DEBUG_HSBCV("case 2 " << I);
	}
	else if (i[0] && i[1]) {
		I = 0.5*t[0]*t[1]*dz;
		if (n[0] < 0) I = dx*dy*dz - I;
		DEBUG_HSBCV("case 3 " << I);
	}
	else if (i[1] && i[6]) {
		I = 0.5*(dy-t[1])*(dx-t[6])*dz;
		if (n[0] < 0) I = dx*dy*dz - I;
		DEBUG_HSBCV("case 4 " << I);
	}
	else if (i[6] && i[7]) {
		I = 0.5*t[6]*t[7]*dz;
		if (n[0] > 0) I = dx*dy*dz - I;
		DEBUG_HSBCV("case 5 " << I);
	}
	else if (i[7] && i[0]) {
		I = 0.5*(dy-t[7])*(dx-t[0])*dz;
		if (n[0] > 0) I = dx*dy*dz - I;
		DEBUG_HSBCV("case 6 " << I);
	}
	else {
		I = (n[2] > 0) ? (dx*dy*dz) : 0;
		DEBUG_HSBCV("case 7 " << I);
	}

	static int facets[6][4] = {
		{0, 5, 3, 2},
		{6, 8, 9, 11},
		{5, 4, 8, 7},
		{2, 1, 11, 10},
		{0, 7, 6, 1},
		{3, 10, 9, 4},
	};

	for (int k = 0; k < 6; k++) {
		bool ok = false;
		int j;
		for (j = 0; j < 4; j++) {
			int e = facets[k][j];
			if (i[e] && e != pi0_index) {
				pi1 = *px[e/3];
				pi1[e%3] += signs[e]*t[e];
				if (ublas::norm_2(pi1 - pi0) > eps) {
					break;
				}
			}
		}
		for (j++; j < 4; j++) {
			int e = facets[k][j];
			if (i[e] && e != pi0_index) {
				pi2 = *px[e/3];
				pi2[e%3] += signs[e]*t[e];
				if (ublas::norm_2(pi2 - pi0) > eps && ublas::norm_2(pi2 - pi1) > eps) {
					ok = true;
					break;
				}
			}
		}
		if (ok) {
			// compute triangle contribution to surface integral
			T dI = 0.5*ublas::norm_2(cross_prod(pi1-pi0, pi2-pi0))*n[2]*((pi0[2]+pi1[2]+pi2[2])/3.0 - x1[2]);
			DEBUG_HSBCV("dI: " << dI << " " << format(pi0) << " " << format(pi1) << " " << format(pi2));
			I += dI;
		}
	}

	return I;
}


//! Volume of cut between a box and a halfspace
//! \param x point in plane
//! \param n plane normal
//! \param x0 vertex of box
//! \param dx box dimensions x
//! \param dy box dimensions y
//! \param dz box dimensions z
template <typename T, int DIM>
inline T halfspace_box_cut_volume(const ublas::c_vector<T, DIM>& x, const ublas::c_vector<T, DIM>& n, const ublas::c_vector<T, DIM>& x0, T dx, T dy, T dz)
{
#if 0
	#define DEBUG_HSBCV(x) LOG_COUT << x << std::endl;

	DEBUG_HSBCV("### called");
	DEBUG_HSBCV("x " << format(x));
	DEBUG_HSBCV("n " << format(n));
	DEBUG_HSBCV("x0 " << format(x0));
#else
	#define DEBUG_HSBCV(x)
#endif

	// vertices:
	// 0: x0
	// 1: x0 + dx
	// 2: x0 + dy
	// 3: x0 + dz
	// 4: x0 + dx + dy
	// 5: x0 + dy + dz
	// 6: x0 + dx + dz
	// 7: x0 + dx + dy + dz
	ublas::c_vector<T, DIM> vertices[8];
	vertices[0] = x0;
	vertices[1] = x0 + dx*ublas::unit_vector<T>(DIM, 0);
	vertices[2] = x0 + dy*ublas::unit_vector<T>(DIM, 1);
	vertices[3] = x0 + dz*ublas::unit_vector<T>(DIM, 2);
	vertices[4] = vertices[1] + dy*ublas::unit_vector<T>(DIM, 1);
	vertices[5] = vertices[2] + dz*ublas::unit_vector<T>(DIM, 2);
	vertices[6] = vertices[3] + dx*ublas::unit_vector<T>(DIM, 0);
	vertices[7] = vertices[6] + dy*ublas::unit_vector<T>(DIM, 1);

	// vertex pairs for edges
	static int edges[12][2] = {
		{0, 1}, {2, 4}, {3, 6}, {5, 7}, // x
		{0, 2}, {1, 4}, {3, 5}, {6, 7}, // y
		{0, 3}, {1, 6}, {2, 5}, {4, 7}, // z
	};

	// edge quadruples for faces
	static int faces[6][4] = {
		{8, 6, -10, -4},
		{9, 7, -11, -5},
		{0, 9, -2, -8},
		{1, 11, -3, -10},
		{0, 5, -1, -4},
		{2, 7, -3, -6},
	};

	//static int face_normal_indices[6] = {0, 0, 1, 1, 2, 2};
	static int face_normal_signs[6] = {-1, 1, -1, 1, -1, 1};

	// determine which vertices are inside of halfspace
	bool inside[8];
	int num_inside = 0;
	for (int i = 0; i < 8; i++) {
		inside[i] = ublas::inner_prod(vertices[i] - x, n) < 0;
		num_inside += inside[i];
		DEBUG_HSBCV("inside " << i << ": " << (inside[i] ? 1 : 0));
	}

	// determine edge intersections relative to x0 in edge direction
	// no more than 6 edge intersections are possible
	T intersection_distance[6];
	int intersection_edge[12];
	int num_intersections = 0;
	int any_intersection_edge = -1;
	for (int i = 0; i < 12; i++) {
		if (inside[edges[i][0]] + inside[edges[i][1]] == 1) {
			// edge has intersection
			// the edge direction is the i/4-th unit vector (as we sorted the edge list this way)
			// the intersection parameter is determined by
			// dot(vertices[edges[i][0]] + t*ublas::unit_vector<T>(DIM, i/4) - x, n) = 0
			intersection_distance[num_intersections] = ublas::inner_prod(x - vertices[edges[i][0]], n)/n[i/4];
			intersection_edge[i] = num_intersections;
			DEBUG_HSBCV("intersection " << i << ": " << intersection_distance[num_intersections]);
			any_intersection_edge = i;
			num_intersections++;
		}
		else {
			intersection_edge[i] = -1;
		}
	}

	if (num_intersections == 0) {
		// no intersection means all or nothing
		DEBUG_HSBCV("num_intersections == 0");
		DEBUG_HSBCV("V = " << (inside[0] ? (dx*dy*dz) : 0));
		return inside[0] ? (dx*dy*dz) : 0;
	}

	static int crossp_indices[3][2] = {
		{1, 2},
		{2, 0},
		{0, 1},
	};

	// get one intersection point
	ublas::c_vector<T, DIM> xi = vertices[edges[any_intersection_edge][0]] +
		intersection_distance[intersection_edge[any_intersection_edge]]*
		ublas::unit_vector<T>(DIM, any_intersection_edge/4);

	// decide if we flip from inside to outside volume computation
	bool flip = (num_inside > 4);
	DEBUG_HSBCV("flip: " << (flip ? 1 : 0));

	// integrate over all faces
	// the volume is calculated by Gauss divergence theorem, using f(x) = x - xi
	// where xi is any intersection point
	// int_V div f = 3 |V| = int_S dot(f,n) dS = sum_facets F dot(f(some point of facet F), facet normal)*|facet area|
	ublas::c_vector<T, DIM> points[5];
	T V = 0;
	for (int f = 0; f < 6; f++)
	{
		DEBUG_HSBCV("face: " << f);

		int ni = f>>1;		// facet normal index
		int num_points = 0;	// number of points

		for (int i = 0; i < 4; i++) {
			// add all inside and intersection points as boundary points for the integration area
			// the points will be ordered in circular manner
			int e = faces[f][i];
			int i1 = 0;
			int i2 = 1;
			// do we need to reverse the edge vertex order?
			if (e < 0) {
				e = -e;
				i1 = 1;
				i2 = 0;
			}
			if (num_points == 0 && inside[edges[e][i1]] ^ flip) {
				// vertex 0 inside
				// the first vertex of an edge is only used for the first point to avoid repeating
				// (first vertex of first edge = last vertex of last edge)
				points[num_points] = vertices[edges[e][i1]];
				DEBUG_HSBCV("point 1: " << points[num_points]);
				if (points[0][ni] == xi[ni]) break;
				num_points++;
			}
			if (intersection_edge[e] >= 0) {
				// edge has intersection
				points[num_points] = vertices[edges[e][0]] + intersection_distance[intersection_edge[e]]*ublas::unit_vector<T>(DIM, e/4);
				DEBUG_HSBCV("point 2: " << points[num_points]);
				if (num_points == 0 && points[0][ni] == xi[ni]) break;
				num_points++;
			}
			if (i < 3 && inside[edges[e][i2]] ^ flip) {
				// vertex 1 inside
				// the last vertex of an edge is not used for the last edge to avoid repeating
				// (first vertex of first edge = last vertex of last edge)
				points[num_points] = vertices[edges[e][i2]];
				DEBUG_HSBCV("point 3: " << points[num_points]);
				if (num_points == 0 && points[0][ni] == xi[ni]) break;
				num_points++;
			}
		}

		if (num_points < 3) {
			DEBUG_HSBCV("num_points < 3");
			// primitve has no area
			continue;
		}

		int i1 = crossp_indices[ni][0];
		int i2 = crossp_indices[ni][1];

		// compute area
		T area = 0;
		for (int i = 2; i < num_points; i++) {
			// ublas::norm_2(cross_prod(points[i-1] - points[0], points[i] - points[0]))
			T da = std::abs((points[i-1][i1] - points[0][i1])*(points[i][i2] - points[0][i2]) -
				(points[i-1][i2] - points[0][i2])*(points[i][i1] - points[0][i1]));
			area += da;
			DEBUG_HSBCV("da " << da);
		}

		T d = points[0][ni] - xi[ni];	// normal distance to xi
		V += face_normal_signs[f]*d*area;
		DEBUG_HSBCV("dV " << (face_normal_signs[f]*d*area));
	}

	// need to divide by 6, 0.5 for the facet area and 1/3 for the div theorem
	V *= (1.0/6.0);

	if (flip) {
		// compute inside volume from outside volume
		V = dx*dy*dz - V;
	}

	DEBUG_HSBCV("V: " << V);
	return V;
}

//! A progress bar for console output
template <typename T>
class ProgressBar
{
public:

	//! Create progress bar with maximum value and a number of update steps
	//! \param max the maximum value for the progress parameter
	//! \param steps number of update steps (the number of times the progress text is updated)
	ProgressBar(T max = 100, T steps = 100)
		: _max(max), _dp(100/steps), _p(0), _p_old(-1)
	{
	}

	//! Increment the progress by one
	//! \return true if you should print the progress message
	bool update()
	{
		return update(_p + 1);
	}

	//! Update the progress to value p
	//! \return true if you should print the progress message
	bool update(T p)
	{
		_p = std::min(std::max(p, (T)0), _max);
		return (std::abs(_p - _p_old) > _dp) || complete();
	}

	//! Returns true if the progress is complete
	bool complete()
	{
		return (_p >= _max);
	}

	//! Returns text for the end of the progress message, i.e.
	//! cout << pb.message() << "saving..." << pb.end();
	const char* end()
	{
		Logger::instance().flush();

		if (complete()) {
			return _DEFAULT_TEXT _CLEAR_EOL "\n";
		}
		return _DEFAULT_TEXT _CLEAR_EOL "\r";
	}

	//! Returns the current progress message as stream to cout
	std::ostream& message()
	{
		T percent = _p/_max*100;
		_p_old = _p;
		std::ostream& stream = LOG_COUT;
		stream << (complete() ? GREEN_TEXT : YELLOW_TEXT) << (boost::format("%.2f%% complete: ") % percent);
		return stream;
	}

protected:
	T _max, _dp, _p, _p_old;
};



//! Class for measuring elasped time between construction and destruction and console output of timings
class Timer
{
protected:
	class Info {
	public:
		Info() : calls(0), total_duration() { }
		std::size_t calls;
		pt::time_duration total_duration;
	};

	typedef std::map<std::string, boost::shared_ptr<Info> > InfoMap;
	static InfoMap info_map;
	
	std::string _text;
	bool _print, _log;
	pt::ptime _t0;
	boost::shared_ptr<Info> _info;

public:
	Timer() : _print(false)
	{
		start();
	}

	//! Constructor. The timer is started automatically.
	//! \param text text for display in the console, when timer starts/finishes and the statistics
	//! \param print enable console output 
	//! \param log enable logging for statistics of function calls etc. 
	Timer(const std::string& text, bool print = true, bool log = true) : _text(text), _print(print), _log(log)
	{
#ifdef DEBUG
		_print = true;
#endif

		if (_print) {
			LOG_COUT << BOLD_TEXT << "Begin " << _text << std::endl;
			Logger::instance().incIndent();
		}
		start();

		if (_log) {
			InfoMap::iterator item = Timer::info_map.find(text);
			if (item != Timer::info_map.end()) {
				_info = item->second;
			}
			else {
				_info.reset(new Info());
				Timer::info_map.insert(InfoMap::value_type(text, _info));
			}
		}
	}

	//! Start the timer
	void start()
	{
		_t0 = pt::microsec_clock::universal_time();
	}
	
	//! Return current elaspsed time
	pt::time_duration duration()
	{
		pt::ptime t = pt::microsec_clock::universal_time();
		return (t - _t0);
	}

	//! Return current elasped time in seconds
	double seconds()
	{
		return duration().total_milliseconds()*1e-3;
	}

	//! Return current elasped time in seconds
	operator double() { return seconds(); }

	~Timer()
	{
		pt::time_duration dur = duration();

		if (_print) {
			Logger::instance().decIndent();
			LOG_COUT << BOLD_TEXT << "Finished " << _text << " in " << (dur.total_milliseconds()*1e-3) << "s" << std::endl;
		}

		if (_info) {
			_info->calls ++;
			_info->total_duration += dur;
		}
	}

	//! Print statistics
	static void print_stats();

	//! Clear statistics
	static void reset_stats();
};


Timer::InfoMap Timer::info_map;


void Timer::print_stats()
{
	std::size_t title_width = 8;
	std::size_t calls_width = 5;
	std::size_t time_width = 9;
	std::size_t time_per_call_width = 9;
	std::size_t relative_width = 9;
	pt::time_duration total;

	for (Timer::InfoMap::iterator i = Timer::info_map.begin(); i != Timer::info_map.end(); ++i) {
		Timer::Info& info = *(i->second);
		if (info.calls == 0) continue;
		title_width = std::max(title_width, i->first.length());
		total += info.total_duration;
		calls_width = std::max(calls_width, 1+(std::size_t)std::log10((double)info.calls));
	}

	double sec_total = total.total_milliseconds()*1e-3;

	LOG_COUT << (boost::format("%s|%s|%s|%s|%s\n")
		% boost::io::group(std::setw(title_width), "Function")
		% boost::io::group(std::setw(calls_width), "Calls")
		% boost::io::group(std::setw(time_width), "Time")
		% boost::io::group(std::setw(time_per_call_width), "Time/Call")
		% boost::io::group(std::setw(relative_width), "Relative")
	).str();

	LOG_COUT << std::string(title_width, '=')
		+ "+" + std::string(calls_width, '=')
		+ "+" + std::string(time_width, '=')
		+ "+" + std::string(time_per_call_width, '=')
		+ "+" + std::string(relative_width, '=')
		+ "\n";

	for (Timer::InfoMap::iterator i = Timer::info_map.begin(); i != Timer::info_map.end(); ++i) {
		Timer::Info& info = *(i->second);
		if (info.calls == 0) continue;
		double sec = info.total_duration.total_milliseconds()*1e-3;
		LOG_COUT << (boost::format("%s|%d|%g|%g|%g\n")
			% boost::io::group(std::setw(title_width), i->first)
			% boost::io::group(std::setw(calls_width), info.calls)
			% boost::io::group(std::setw(time_width), std::setprecision(4), sec)
			% boost::io::group(std::setw(time_per_call_width), std::setprecision(4), sec/info.calls)
			% boost::io::group(std::setw(relative_width), std::setprecision(4), sec/sec_total*100.0)
		).str();
	}

	LOG_COUT << std::string(title_width, '=')
		+ "+" + std::string(calls_width, '=')
		+ "+" + std::string(time_width, '=')
		+ "+" + std::string(time_per_call_width, '=')
		+ "+" + std::string(relative_width, '=')
		+ "\n";
	
	LOG_COUT << (boost::format("%s|%s|%4g|%s|%4g\n")
		% boost::io::group(std::setw(title_width), "total")
		% boost::io::group(std::setw(calls_width), "-")
		% boost::io::group(std::setw(time_width), std::setprecision(4), sec_total)
		% boost::io::group(std::setw(time_per_call_width), "-")
		% boost::io::group(std::setw(relative_width), std::setprecision(4), 100.0)
	).str();
}


void Timer::reset_stats()
{
	info_map.clear();
}


//! Class for reading a tetrahedron mesh from a ASCII VTK file
template <typename T>
class TetVTKReader
{
public:
	//! Read data from file
	//! \param filename the VTK filename
	//! \param points output of the vertices
	//! \param tets output of vector of 4-tupes describing the vertex indices of each tetrahedron
	void read(const std::string& filename, std::vector< ublas::c_vector<T,3> >& points, std::vector< ublas::c_vector<std::size_t,4> >& tets)
	{
		/*
			# vtk DataFile Version 3.0
			vtk output
			ASCII
			DATASET UNSTRUCTURED_GRID
			POINTS 7 float
			0.5 0 0 -0.5 0.5 0 -0.5 0.25 0.433013 
			-0.5 -0.25 0.433013 -0.5 -0.5 6.12323e-17 -0.5 -0.25 -0.433013 
			-0.5 0.25 -0.433013 
			CELLS 10 40
			3 4 3 5 
			3 5 3 6 
			3 3 2 6 
			3 1 6 2 
			3 0 1 2 
			3 0 2 3 
			3 0 3 4 
			3 0 4 5 
			3 0 5 6 
			3 0 6 1 

			CELL_TYPES 10
			5
			5
			5
			5
			5
			5
			5
			5
			5
			5
		*/

		std::ifstream infile(filename.c_str());
		std::string line, tok;

		while (std::getline(infile, line))
		{
			boost::algorithm::to_lower(line);

			if (line == "ascii") {
				break;
			}
			else if (line == "binary") {
				BOOST_THROW_EXCEPTION(std::runtime_error("binary vtk not implemented"));
			}
		}

	
		while (std::getline(infile, line))
		{
			boost::algorithm::to_lower(line);

			if (boost::starts_with(line, "points")) {
				break;
			}
		}

		std::size_t num_points;
		{
			std::stringstream ss(line);
			ss >> tok >> num_points >> tok;
		}
	
		for (std::size_t i = 0; i < num_points; i++) {
			ublas::c_vector<T,3> point;
			infile >> point[0] >> point[1] >> point[2];
			points.push_back(point);
		}

		while (std::getline(infile, line))
		{
			boost::algorithm::to_lower(line);

			if (boost::starts_with(line, "cells")) {
				break;
			}
		}

		std::size_t num_cells;
		std::size_t num_point_ids;
		{
			std::stringstream ss(line);
			ss >> tok >> num_cells >> num_point_ids;
		}

		std::vector<bool> read_cells(num_cells);

		for (std::size_t i = 0; i < num_cells; i++) {
			ublas::c_vector<std::size_t,4> cell;
			std::size_t nvalues;
			infile >> nvalues;
			if (nvalues != 4) {
				for (std::size_t k = 0; k < nvalues; k++) {
					T value;
					infile >> value;
				}
				read_cells[i] = false;
				continue;
			}
			infile >> cell[0] >> cell[1] >> cell[2] >> cell[3];
			tets.push_back(cell);
			read_cells[i] = true;
		}

		while (std::getline(infile, line))
		{
			boost::algorithm::to_lower(line);

			if (boost::starts_with(line, "cell_types")) {
				break;
			}
		}

		for (std::size_t i = 0; i < read_cells.size(); i++) {
			std::size_t cell_type;
			infile >> cell_type;
			if (!read_cells[i]) continue;
			if (cell_type != 10) {
				BOOST_THROW_EXCEPTION(std::runtime_error("vtk file does not contain tetrahedrons"));
			}
		}
	}
};



//! Class for reading a tetrahedron mesh from a Dolfin/FEniCS XLM mesh file
template <typename T>
class TetDolfinXMLReader
{
public:
	void read(const std::string& filename, std::vector< ublas::c_vector<T,3> >& points, std::vector< ublas::c_vector<std::size_t,4> >& tets)
	{
		ptree::ptree xml_root;
		read_xml(filename, xml_root, ptree::xml_parser::trim_whitespace);

		const ptree::ptree& dolfin = xml_root.get_child("dolfin", empty_ptree);
		const ptree::ptree& mesh = dolfin.get_child("mesh", empty_ptree);
		const ptree::ptree& vertices = mesh.get_child("vertices", empty_ptree);
		const ptree::ptree& cells = mesh.get_child("cells", empty_ptree);

		BOOST_FOREACH(const ptree::ptree::value_type &v, vertices)
		{
			if (v.first != "vertex") continue;

			const ptree::ptree& attr = v.second.get_child("<xmlattr>", empty_ptree);
			ublas::c_vector<T,3> point;
			point[0] = pt_get<T>(attr, "x", 0.0);
			point[1] = pt_get<T>(attr, "y", 0.0);
			point[2] = pt_get<T>(attr, "z", 0.0);
			
			//LOG_COUT << "point: " << format(point) << std::endl;
			points.push_back(point);
		}

		BOOST_FOREACH(const ptree::ptree::value_type &v, cells)
		{
			if (v.first != "tetrahedron") continue;

			const ptree::ptree& attr = v.second.get_child("<xmlattr>", empty_ptree);
			ublas::c_vector<std::size_t,4> tet;
			tet[0] = pt_get<std::size_t>(attr, "v0", 0);
			tet[1] = pt_get<std::size_t>(attr, "v1", 0);
			tet[2] = pt_get<std::size_t>(attr, "v2", 0);
			tet[3] = pt_get<std::size_t>(attr, "v3", 0);

			//LOG_COUT << "tet: " << format(tet) << std::endl;
			tets.push_back(tet);
		}
	}
};


//! Class for reading a triangle surface mesh from a STL file
template <typename T>
class STLReader
{
public:
	typedef struct {
		ublas::c_vector<T,3> n;
		ublas::c_vector<T,3> v[3];
	} Facet;

	void read(const std::string& filename, std::vector< STLReader<T>::Facet >& facets)
	{
		std::ifstream infile(filename.c_str());
		std::string line, tok;

		std::getline(infile, line);

		if (boost::starts_with(line, "solid"))
		{
			// ascii format
			/*
				solid name
				facet normal n1 n2 n3
					outer loop
						vertex p1x p1y p1z
						vertex p2x p2y p2z
						vertex p3x p3y p3z
					endloop
				endfacet
				endsolid name
			*/

			Facet f;

			infile.ignore(1000, 'm');
			while (infile >> tok >> f.n[0] >> f.n[1] >> f.n[2])
			{
				std::getline(infile, line); // outer loop
				for (int i = 0; i < 3; i++) {
					infile.ignore(100, 'v');
					infile >> tok >> f.v[i][0] >> f.v[i][1] >> f.v[i][2];
				}
				
				infile.ignore(1000, 'm');

#if 0
				LOG_COUT << "facet normal " << f.n[0] << " " << f.n[1] << " " << f.n[2] << std::endl;
				for (int i = 0; i < 3; i++) {
					LOG_COUT << "facet vertex " << i << " " << f.v[i][0] << " " << f.v[i][1] << " " << f.v[i][2] << std::endl;
				}
#endif

				facets.push_back(f);
			}
		}
		else
		{
			// binary format
			/* 
				UINT8[80]         -   Dateikopf (Header)
				UINT32            -   Anzahl der Dreiecke
				foreach triangle
				   REAL32[3]       -    Normalenvektor
				   REAL32[3]       -    Vertex 1
				   REAL32[3]       -    Vertex 2
				   REAL32[3]       -    Vertex 3
				   UINT16          -    Attribute byte count
				end
			*/
			BOOST_THROW_EXCEPTION(std::runtime_error("binary stl not implemented"));
		}
	}
};

//! Standard normal (mu=0, sigma=1) distributed number generator
template<typename T>
class RandomNormal01
{
	// random number generator stuff
	typedef boost::normal_distribution<T> NumberDistribution; 
	typedef boost::mt19937 RandomNumberGenerator; 
	NumberDistribution _distribution; 
	RandomNumberGenerator _generator; 
	boost::variate_generator<RandomNumberGenerator&, NumberDistribution> _rnd; 
	static RandomNormal01<T> _instance;

public:
	
	// constructor
	RandomNormal01() :
		_distribution(0, 1), _generator(), _rnd(_generator, _distribution)
	{
	}
	
	//! change random seed
	void seed(int s) {
		// http://stackoverflow.com/questions/4778797/setting-seed-boostrandom
		_rnd.engine().seed(s);
		_rnd.distribution().reset();
	}
	
	//! return random number
	T rnd() { return _rnd(); }
	
	//! return static instance
	static RandomNormal01& instance() { return _instance; }	
};

template<typename T>
RandomNormal01<T> RandomNormal01<T>::_instance;


//! Uniform number generator in [0,1]
template<typename T>
class RandomUniform01
{
	// random number generator stuff
	typedef boost::uniform_real<T> NumberDistribution; 
	typedef boost::mt19937 RandomNumberGenerator; 
	NumberDistribution _distribution; 
	RandomNumberGenerator _generator; 
	boost::variate_generator<RandomNumberGenerator&, NumberDistribution> _rnd; 
	static RandomUniform01<T> _instance;

public:
	
	// constructor
	RandomUniform01() :
		_distribution(0, 1), _generator(), _rnd(_generator, _distribution)
	{
	}
	
	//! change random seed
	void seed(int s) {
		// http://stackoverflow.com/questions/4778797/setting-seed-boostrandom
		_rnd.engine().seed(s);
		_rnd.distribution().reset();
	}
	
	//! return random number
	T rnd() { return _rnd(); }
	
	//! return static instance
	static RandomUniform01& instance() { return _instance; }	
};

template<typename T>
RandomUniform01<T> RandomUniform01<T>::_instance;


// TODO: some of the constants/tolerances in this routine are not appropiate for 32bit floats!
template<typename T, int DIM>
void stable_A(ublas::c_vector<T, DIM>& a)
{
	T eps = 1e-8;

	if (std::abs(a[0]-a[1]) < eps) {
		if (a[2] > 0.5) {
			a[0] += eps;
			a[2] -= eps;
		}   
		else {
			a[0] -= eps;
			a[2] += eps;
		}   
	}   

	if (std::abs(a[0]-a[2]) < eps) {
		if (a[1] > 0.5) {
			a[0] += eps;
			a[1] -= eps;
		}   
		else {
			a[0] -= eps;
			a[1] += eps;
		}   
	}   

	if (std::abs(a[1]-a[2]) < eps) {
		if (a[0] > 0.5) {
			a[1] += eps;
			a[0] -= eps;
		}   
		else {
			a[1] -= eps;
			a[0] += eps;
		}   
	}   
}



//! Moments of angular central Gaussian distribution from distribution parameters
//! \param b input ACG parameters
//! \param a output moments
template<typename T, int DIM>
void A_from_B(const ublas::c_vector<T, DIM>& b, ublas::c_vector<T, DIM>& a)
{
        a[0] = (1.0/3.0)*boost::math::ellint_rj(b[0], b[1], b[2], b[0]);
        a[1] = (1.0/3.0)*boost::math::ellint_rj(b[0], b[1], b[2], b[1]);
        a[2] = 1.0-a[0]-a[1];
}


//! Inversion of moments of angular central Gaussian distribution using fixed point method
//! \param a input moments
//! \param b output ACG parameters
//! \param tol tolerance
//! \param max_iter maximum number of iterations
//! \param step step size
//! \return number of required iterations
template<typename T, int DIM>
std::size_t compute_B_from_A_fixedpoint(const ublas::c_vector<T, DIM>& a, ublas::c_vector<T, DIM>& b, T tol, std::size_t max_iter = 1000000, T step = 0.5)
{
	ublas::c_vector<T, DIM> r;
	ublas::c_vector<T, DIM> r_old;
	ublas::c_vector<T, DIM> b_old;
	T res_old = STD_INFINITY(T);
	T eps = std::numeric_limits<T>::epsilon();
	RandomUniform01<T> rnd;

	// init random seed
	rnd.seed(0);

	for (std::size_t iter = 0 ;; iter++)
	{
		// keep b positive
		b[0] = std::max(eps, b[0]);
		b[1] = std::max(eps, b[1]);
		b[2] = std::max(eps, b[2]);

		// fix determinant of B to 1
		T p = b[0]*b[1]*b[2];
		b /= std::pow(p, 1/(T)3);

		// compute residual
		T i0 = boost::math::ellint_rd(b[1], b[2], b[0])/(T)3;
		T i1 = boost::math::ellint_rd(b[0], b[2], b[1])/(T)3;
		T i2 = boost::math::ellint_rd(b[1], b[0], b[2])/(T)3;
		T is = i0 + i1 + i2;
		r[0] = i0/is - a[0];
		r[1] = i1/is - a[1];
		r[2] = i2/is - a[2];

		T res = ublas::norm_2(r);

		if (res < tol) {
			return iter;
		}

		if (iter > max_iter) {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Carlson RD elliptic integral inversion failed, residual = %g") % res).str()));
		}

		// accept residual reducing steps only
		if (res > res_old) {
			b = b_old;
			r = r_old;
		}
		else {
			res_old = res;
			r_old = r;
			b_old = b;
		}

		//LOG_COUT << "Carlson RD residual = " << res << " tol = " << tol << std::endl;

		// perform fixed point iteration
#if 1
		b += (rnd.rnd()*step)*r;
#else
		ublas::c_vector<T, DIM> s;
		T ss = 0;
		for (int i = 0; i < DIM; i++) {
			s(i) = std::exp(step*r(i));
			ss += s(i);
		}
		for (int i = 0; i < DIM; i++) {
			r(i) *= s(i)/ss;
		}
#endif
	}
}


//! evaluation of the derivative of the function R_D
// NOTE: there are probably some erroneous equations in here
template<typename T>
T RD_derivative(T x, T y, T z, T r, T s, T t)
{
	T value = 0;

	// internal tolerance deciding when things get close
	T tol = std::pow(std::numeric_limits<T>::epsilon(), 2/(T)3);

	if (std::abs(x-z)>tol) {
		value = (r-t)/(x-z);
	}
	else if (std::abs(x-y)>tol) // almost transversely isotropic
	{
		T x0=(x+z)/2;
		T eps=(x-z)/2;
		T I;

		if (x0>=y) {
			I=2/std::sqrt(x0-y)*std::acos(std::sqrt(y/x0));
		}
		else {
			I=2/std::sqrt(y-x0)*boost::math::acosh(std::sqrt(y/x0));  // probably equation wrong?
		}

		T I3=0;
		T I5=0;
		T sqrty=std::sqrt(y);
		T xn=1;

		for (size_t n = 1; n < 5; n++)
		{
			xn=xn*x0;
			I=(2*n-1)/(2*n*(x0-y))*I-sqrty/(n*xn*(x0-y));

			if (n==2) {
				I3=I;
			}
			else if (n==4) {
				I5=I;
				value=(0.5*I3+0.75*I5*eps*eps);	// probably sign wrong
			}
		}
	}
	else // almost isotropic
	{
		T c1=z-1;
		T c2=x-1;
		T c3=y-1;

		value=1/10.0-3/28.0*c1-3/28.0*c2-1/28.0*c3+5/48.0*c1*c1+1/8.0*c1*c2+1/24.0*c1*c3+5/48.0*c2*c2+1/24.0*c2*c3+1/48.0*c3*c3;
		value=value-35/352.0*c1*c1*c1-45/352.0*c1*c1*c2-15/352.0*c1*c1*c3-45/352.0*c1*c2*c2-9/176.0*c1*c2*c3-9/352.0*c1*c3*c3-35/352.0*c2*c2*c2-15/352.0*c2*c2*c3-9/352.0*c2*c3*c3-5/352.0*c3*c3*c3;
		value=2*value;
	}

	return value;
}

//! Inversion of moments of angular central Gaussian distribution using fixed point method
//! \param ac input moments
//! \param b output ACG parameters
//! \param tol tolerance
//! \return number of required iterations
// TODO: some of the constants/tolerances in this routine are not appropiate for 32bit floats!
template<typename T, int DIM>
std::size_t compute_B_from_A_fixedpoint_II(const ublas::c_vector<T, DIM>& ac, ublas::c_vector<T, DIM>& b, T tol = 1e-10)
{
	T eps = 1e-32;
	std::size_t maxiter = 64;

	ublas::c_vector<T, DIM> a_log_a, e, ab, da;
	ublas::c_vector<T, DIM> a = ac;
	stable_A<T, DIM>(a);

	for (std::size_t i = 0; i < DIM; i++) {
		a[i] += eps;
		a_log_a[i] = a[i]*std::log(a[i]);
		e[i] = 1.65;
	}

	std::size_t iter = 0;

	for (; iter < maxiter; iter++)
	{
		T b_scale = 1.0;

		for (std::size_t i = 0; i < DIM; i++) {
			b[i] = ((T)1.0)/(std::pow(a[i], e[i]) + eps);
			b_scale *= b[i];
		}

		b *= std::pow(b_scale, (T)(-1.0/3.0));

		A_from_B<T, DIM>(b, ab);
		da = a - ab;
		T err = 0;
		for (std::size_t i = 0; i < DIM; i++) {
			err = std::max(std::abs(da[i]), err);
		}

		if (err <= tol) {
			break;
		}

		for (std::size_t i = 0; i < DIM; i++) {
			e[i] += std::min(std::max(e[i]*da[i]/a_log_a[i], (T)-1.0), (T)1.0);
			e[i]  = std::min(std::max(e[i], (T)-64.0), (T)64.0);
		}
	};

	return iter;
}

//! Inversion of moments of angular central Gaussian distribution using Newton method
//! \param a input moments
//! \param b output ACG parameters
//! \param tol tolerance
//! \param max_iter maximum number of iterations
//! \return number of required iterations
template<typename T, int DIM>
std::size_t compute_B_from_A_newton(const ublas::c_vector<T, DIM>& a, ublas::c_vector<T, DIM>& b, T tol, std::size_t max_iter = 1000000)
{
	size_t perm[6][3] = {{0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,1,0}, {2,0,1}};
	size_t p = 0;

	// find permutation of a such that permuted a is ascending
	for (size_t i = 0; i < 6; i++) {
		if (a[perm[i][0]] <= a[perm[i][1]] && a[perm[i][1]] <= a[perm[i][2]]) {
			p = i;
			break;
		}
	}

	T lambda_1 = a[perm[p][0]];
	T lambda_2 = a[perm[p][1]];

	T b_1 = 1;
	T b_2 = 1;
	T b_3 = 1;

	std::size_t iter = 0;

	for (iter = 0 ;; iter++)
	{
		// evaluation of the function    
		T f_1 = boost::math::ellint_rd(b_2, b_3, b_1)/(T)3;
		T f_2 = boost::math::ellint_rd(b_3, b_1, b_2)/(T)3;
		T f_3 = std::max((T)0, 1-f_1-f_2);

		// stopping criterion        
		T res = std::max(std::abs(f_1-lambda_1), std::abs(f_2-lambda_2));

		// LOG_COUT << "f_1 " << f_1 << " f_2 " << f_2 << " res " << res << std::endl;
		// LOG_COUT << "b_1 " << b_1 << " b_2 " << b_2 << " lambda_1 " << lambda_1 << " lambda_2 " << lambda_2 << std::endl;

		if (res < tol) {
			break;
		}
		if (iter > max_iter) {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Carlson RD elliptic integral inversion failed, residual = %g") % res).str()));
		}

		T p = RD_derivative(b_2,b_3,b_1,f_2,f_3,f_1);
		T q = RD_derivative(b_3,b_2,b_1,f_3,f_2,f_1);
		T r = RD_derivative(b_2,b_1,b_3,f_2,f_1,f_3);

		T a = -0.5*(1/b_1 + p + q*(1+1/(b_1*b_1*b_2)));
		T b = 0.5*(p - q/(b_1*b_2*b_2));
		T c = 0.5*(p - r/(b_1*b_1*b_2));
		T d = -0.5*(1/b_2 + p + r*(1+1/(b_1*b_2*b_2)));

		// inversion of the derivative    

		T det = a*d-b*c;

		T e = a;
		a = d/det;
		d = e/det;
		b = -b;
		c = -c;

		T y_1 = lambda_1-f_1;
		T y_2 = lambda_2-f_2;

		T Deltab_1 = a*y_1+b*y_2;
		T Deltab_2 = c*y_1+d*y_2;

		b_1 = std::max(tol, b_1 + Deltab_1);
		b_2 = std::max(tol, b_2 + Deltab_2);
		b_3 = std::max(tol, 1/(b_1*b_2));
	}

	b(perm[p][0]) = b_1;
	b(perm[p][1]) = b_2;
	b(perm[p][2]) = b_3;
	return iter;
}

//! Inversion of moments of angular central Gaussian distribution (default method stub)
//! \param a input moments
//! \param b output ACG parameters
//! \param tol tolerance
//! \return number of required iterations
template<typename T, int DIM>
std::size_t compute_B_from_A(const ublas::c_vector<T, DIM>& a, ublas::c_vector<T, DIM>& b, T tol)
{
	return compute_B_from_A_fixedpoint_II<T, DIM>(a, b, tol);
}



//! Abstract base class for fiber distributions
template<typename T, int DIM>
class DiscreteDistribution
{
protected:
	T _weight;

public:
	DiscreteDistribution() : _weight(1) { }
	
	//! see https://stackoverflow.com/questions/827196/virtual-default-destructors-in-c/827205
	virtual ~DiscreteDistribution() { }

	//! draw a random sample
	//! \param x output of the sample
	//! \param index index of the sample (i.e. for list distributions)
	virtual void drawSample(ublas::c_vector<T, DIM>& x, std::size_t index) = 0;

	//! read settings from ptree
	virtual void readSettings(const ptree::ptree& pt)
	{
		const ptree::ptree& attr = pt.get_child("<xmlattr>", empty_ptree);

		_weight = pt_get<T>(attr, "weight", (T)1);
	}
	
	//! Distribution weight in combination with other Distributions
	virtual T weight() const { return _weight; }
};


//! Dirac distribution
template<typename T, int DIM>
class DiracDistribution : public DiscreteDistribution<T, DIM>
{
public:
	DiracDistribution() { }
	DiracDistribution(const ublas::c_vector<T, DIM>& x) : _x(x) { }
	
	void drawSample(ublas::c_vector<T, DIM>& x, std::size_t index)
	{
		x = _x;
	}
	
	void readSettings(const ptree::ptree& pt)
	{
		DiscreteDistribution<T, DIM>::readSettings(pt);

		const ptree::ptree& attr = pt.get_child("<xmlattr>", empty_ptree);
		const char* names[3] = {"x", "y", "z"};

		for (std::size_t i = 0; i < DIM; i++) {
			_x(i) = pt_get<T>(attr, names[i], (T)0);
		}
	}

protected:
	ublas::c_vector<T, DIM> _x;
};


//! Normal distribution
template<typename T, int DIM>
class NormalDistribution : public DiscreteDistribution<T, DIM>
{
};


//! Normal distribution (on sphere)
template<typename T>
class NormalDistribution<T, 3> : public DiscreteDistribution<T, 3>
{
public:
	NormalDistribution() : _sigma(1) { }
	
	void drawSample(ublas::c_vector<T, 3>& x, std::size_t index)
	{
		T theta = ((T)2*M_PI)*RandomUniform01<T>::instance().rnd();
		ublas::c_vector<T, 3> v = _u*cos(theta) + _w*sin(theta);
		T rnd = RandomUniform01<T>::instance().rnd();
		T phi = atan(boost::math::erfc_inv(2*rnd)*_sigma*sqrt(2));
		x = _x*cos(phi) + v*sin(phi);
	}
	
	void readSettings(const ptree::ptree& pt)
	{
		DiscreteDistribution<T, 3>::readSettings(pt);

		const ptree::ptree& attr = pt.get_child("<xmlattr>", empty_ptree);

		_sigma = pt_get<T>(attr, "sigma", (T)1);
		_x(0) = pt_get<T>(attr, "x", (T)0);
		_x(1) = pt_get<T>(attr, "y", (T)0);
		_x(2) = pt_get<T>(attr, "z", (T)0);
		_x /= ublas::norm_2(_x);
		calcOrtho();
	}
	
protected:
	T _sigma;
	ublas::c_vector<T, 3> _x;
	ublas::c_vector<T, 3> _u;
	ublas::c_vector<T, 3> _w;
	
	void calcOrtho()
	{
		_u(0) = -_x(0)*_x(1);
		_u(1) = _x(0) + _x(2);
		_u(2) = -_x(2)*_x(1);
		_u /= ublas::norm_2(_u);
		_w(0) = _x(1)*_u(2) - _x(2)*_u(1);
		_w(1) = _x(2)*_u(0) - _x(0)*_u(2);
		_w(2) = _x(0)*_u(1) - _x(1)*_u(0);
		_w /= ublas::norm_2(_w);
	}
};


//! Normal distribution specialized for DIM = 2 (circle)
template<typename T>
class NormalDistribution<T, 2> : public DiscreteDistribution<T, 2>
{
public:
	NormalDistribution() : _sigma(1) { }
	
	void drawSample(ublas::c_vector<T, 2>& x, std::size_t index)
	{
		T mu = atan2(_x(1), _x(0));
		T rnd = RandomUniform01<T>::instance().rnd();
		T phi = mu + atan(boost::math::erfc_inv(2*rnd)*_sigma*sqrt(2));
		x(0) = cos(phi);
		x(1) = sin(phi);
	}
	
	void readSettings(const ptree::ptree& pt)
	{
		DiscreteDistribution<T, 2>::readSettings(pt);

		const ptree::ptree& attr = pt.get_child("<xmlattr>", empty_ptree);

		_sigma = pt_get<T>(attr, "sigma", (T)1);
		_x(0) = pt_get<T>(attr, "x", (T)0);
		_x(1) = pt_get<T>(attr, "y", (T)0);
	}
	
protected:
	T _sigma;
	ublas::c_vector<T, 2> _x;
};


//! Normal distribution specialized for DIM = 1
template<typename T>
class NormalDistribution<T, 1> : public DiscreteDistribution<T, 1>
{
public:
	NormalDistribution() : _mu(0), _sigma(1) { }
	
	void drawSample(ublas::c_vector<T, 1>& x, std::size_t index)
	{
		T rnd = RandomNormal01<T>::instance().rnd();
		x(0) = _sigma*rnd + _mu;
	}
	
	void readSettings(const ptree::ptree& pt)
	{
		DiscreteDistribution<T, 1>::readSettings(pt);

		const ptree::ptree& attr = pt.get_child("<xmlattr>", empty_ptree);

		_sigma = pt_get<T>(attr, "sigma", (T)1);
		_mu = pt_get<T>(attr, "mu", (T)1);
	}
	
protected:
	T _mu, _sigma;
};


//! Uniform distribution
template<typename T, int DIM>
class UniformDistribution : public DiscreteDistribution<T, DIM>
{
};


//! Uniform distribution (on sphere)
template<typename T>
class UniformDistribution<T, 3> : public DiscreteDistribution<T, 3>
{
public:
	UniformDistribution() { }
	
	void drawSample(ublas::c_vector<T, 3>& x, std::size_t index)
	{
		T theta = ((T)2*M_PI)*RandomUniform01<T>::instance().rnd();
		T u = RandomUniform01<T>::instance().rnd();
		x(0) = sqrt(1-u*u)*cos(theta);
		x(1) = sqrt(1-u*u)*sin(theta);
		x(2) = u;
	}
};


//! Uniform distribution specialized for DIM = 2 (circle)
template<typename T>
class UniformDistribution<T, 2> : public DiscreteDistribution<T, 2>
{
public:
	UniformDistribution()  { }
	
	void drawSample(ublas::c_vector<T, 2>& x, std::size_t index)
	{
		T theta = ((T)2*M_PI)*RandomUniform01<T>::instance().rnd();
		x(0) = cos(theta);
		x(1) = sin(theta);
	}
};


//! Uniform distribution specialized for DIM = 1 (interval)
template<typename T>
class UniformDistribution<T, 1> : public DiscreteDistribution<T, 1>
{
public:
	UniformDistribution()  { }
	
	void drawSample(ublas::c_vector<T, 1>& x, std::size_t index)
	{
		x(0) = _a + (_b - _a)*RandomUniform01<T>::instance().rnd();
	}

	void readSettings(const ptree::ptree& pt)
	{
		DiscreteDistribution<T, 1>::readSettings(pt);

		const ptree::ptree& attr = pt.get_child("<xmlattr>", empty_ptree);

		_a = pt_get<T>(attr, "a", (T)0);
		_b = pt_get<T>(attr, "b", (T)1);
	}
	
protected:
	T _a, _b;
};



//! Angular central Gaussian distribution
template<typename T, int DIM>
class AngularCentralGaussianDistribution : public DiscreteDistribution<T, DIM>
{
public:
	AngularCentralGaussianDistribution() {
		if (DIM != 3) {
			BOOST_THROW_EXCEPTION(std::runtime_error("ACG only implemented for DIM=3"));
		}
	}

	void drawSample(ublas::c_vector<T, DIM>& x, std::size_t index)
	{
		BOOST_THROW_EXCEPTION(std::runtime_error("not implemented"));
	}
};


//! Angular central Gaussian distribution (on sphere)
template<typename T>
class AngularCentralGaussianDistribution<T, 3> : public DiscreteDistribution<T, 3>
{
public:
	AngularCentralGaussianDistribution() {
	}
	
	void drawSample(ublas::c_vector<T, 3>& x, std::size_t index)
	{
		// draw standard normal distributed values
		// and scale values by singular values
		for (int i = 0; i < 3; i++) {
			x(i) = _bi(i)*RandomNormal01<T>::instance().rnd();
		}

		// transform vector into basis of C
		x = ublas::prod(_U, x);

		// normalize vector
		T norm_x = ublas::norm_2(x);
		if (norm_x == 0) {
			drawSample(x, index);
			return;
		}
		x /= norm_x;
	}
	
	void readSettings(const ptree::ptree& pt)
	{
		DiscreteDistribution<T, 3>::readSettings(pt);

		const ptree::ptree& attr = pt.get_child("<xmlattr>", empty_ptree);

		// read covariance matrix
		_A(0,0) = pt_get<T>(attr, "axx", (T)1/(T)3);
		_A(1,1) = pt_get<T>(attr, "ayy", (T)1/(T)3);
		_A(2,2) = pt_get<T>(attr, "azz", (T)1/(T)3);
		_A(1,0) = _A(0,1) = pt_get<T>(attr, "axy", (T)0);
		_A(2,0) = _A(0,2) = pt_get<T>(attr, "axz", (T)0);
		_A(2,1) = _A(1,2) = pt_get<T>(attr, "ayz", (T)0);

		// check Cauchy-Schwartz bounds
		const T eps = 10*std::numeric_limits<T>::epsilon();
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				T Aij2_bound = _A(i,i)*_A(j,j);
				T Aij2 = _A(i,j)*_A(i,j);
				if ((Aij2 - Aij2_bound) > eps) {
					BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("covariance matrix entry %d,%d exceeds upper bound of %g by more than %g") % i % j % sqrt(Aij2_bound) % eps).str()));
				}
			}
		}

		// normalize A
		T trace = 0;
		for (int i = 0; i < 3; i++) {
			trace += _A(i,i);
		}
		_A /= trace;

		// distribution covariance
		compute_B_from_A();
	}
	
protected:
	// second order FO moment
	ublas::c_matrix<T, 3, 3> _A;
	// matrix of singular vectors
	ublas::c_matrix<T, 3, 3> _U;
	// singular values of A
	ublas::c_vector<T, 3> _a;
	// singular values of B
	ublas::c_vector<T, 3> _b;
	// square root of singular values of B^{-1}
	ublas::c_vector<T, 3> _bi;

	// compute distribution covariance matrix B from A
	noinline void compute_B_from_A()
	{
		Timer __t("angular central gaussian initialization");

		// compute SVD of A
		ublas::c_matrix<T, 3, 3> VT;
		{
			ublas::c_matrix<T, 3, 3> A = _A;
//			lapack::gesvd(A, _a, _U, VT);
			lapack::gesvd(A, _a, VT, _U);
			
			// renormalize a
			T as = _a[0] + _a[1] + _a[2];
			_a /= as;
		}

		LOG_COUT << "Second order moment A:\n" << format(_A) << std::endl;
		LOG_COUT << "Singular vectors of A:\n" << format(_U) << std::endl;
		LOG_COUT << "Singular values of A:\n" << format(_a, true) << std::endl;

		// compute b from a
		T tol = std::pow(std::numeric_limits<T>::epsilon(), 2/(T)3);
		_b[0] = _b[1] = _b[2] = (T)1;
		std::size_t iter = ::compute_B_from_A<T, 3>(_a, _b, tol);
		LOG_COUT << GREEN_TEXT << "Carlson RD elliptic integral inversion finished in " << iter << " iterations (tol = " << tol << ")" << std::endl;

		// compute square root of singular values
		for (int i = 0; i < 3; i++) {
			_bi(i) = 1/sqrt(_b(i));
		}
	
		// print covariance 	
		ublas::c_matrix<T, 3, 3> Binv = ublas::zero_matrix<T>(3);
		Binv(0,0) = 1/_b(0);
		Binv(1,1) = 1/_b(1);
		Binv(2,2) = 1/_b(2);
		Binv = ublas::prod(_U, Binv);
		Binv = ublas::prod(Binv, VT);
		
		LOG_COUT << "Distribution covariance B^{-1}:\n" << format(Binv) << std::endl;
		LOG_COUT << "Singular values of B:\n" << format(_b, true) << std::endl;
	}
};


//! Distribution given by a list
template<typename T, int DIM>
class ListDistribution : public DiscreteDistribution<T, DIM>
{
public:
	ListDistribution() { }

	void drawSample(ublas::c_vector<T, DIM>& x, std::size_t index)
	{
#if 0
		if (index >= _list.size()) {
			BOOST_THROW_EXCEPTION(std::runtime_error("ListDistribution: index out of range!"));
		}
#else
		if (_list.size() == 0) {
			BOOST_THROW_EXCEPTION(std::runtime_error("ListDistribution: index out of range!"));
		}
#endif

		x = _list[index % _list.size()];
	}

	void readSettings(const ptree::ptree& pt)
	{
		BOOST_FOREACH(const ptree::ptree::value_type &v, pt)
		{
			boost::shared_ptr< DiscreteDistribution<T, DIM> > pdf;

			if (v.first == "vec") {
				ublas::c_vector<T, DIM> x;
				const ptree::ptree& attr = v.second.get_child("<xmlattr>", empty_ptree);
				const char* names[3] = {"x", "y", "z"};
				for (std::size_t i = 0; i < DIM; i++) {
					x(i) = pt_get<T>(attr, names[i], (T)0);
				}
				_list.push_back(x);
			}
			else {
				BOOST_THROW_EXCEPTION(std::runtime_error("ListDistribution: unknown item!"));
			}
		}
	}

protected:
	std::vector< ublas::c_vector<T, DIM> > _list;
};


//! A weighted sum of probability density functions
template<typename T, int DIM>
class CompositeDistribution : public DiscreteDistribution<T, DIM>
{
protected:
	typedef std::vector< boost::shared_ptr< DiscreteDistribution<T, DIM> > > pdf_ptr_list;
	pdf_ptr_list _pdfs;	// list of Distributions
	T _sum_weights;

public:
	CompositeDistribution() : _sum_weights(0) { }
	
	void drawSample(ublas::c_vector<T, DIM>& x, std::size_t index)
	{
		if (_pdfs.size() == 0) {
			BOOST_THROW_EXCEPTION(std::runtime_error("CompositeDistribution: no pdf to draw sample from!"));
		}
		
		T p = _sum_weights*RandomUniform01<T>::instance().rnd();
		T sum = 0;

		for (typename pdf_ptr_list::const_iterator i = _pdfs.begin(); i != _pdfs.end(); i++) {
			sum += (*i)->weight();
			if (p <= sum) {
				(*i)->drawSample(x, index);
				return;
			}
		}

		(*_pdfs.rbegin())->drawSample(x, index);
	}
	
	void readSettings(const ptree::ptree& pt)
	{
		DiscreteDistribution<T, DIM>::readSettings(pt);

		//const ptree::ptree& attr = pt.get_child("<xmlattr>", empty_ptree);

		_sum_weights = 0;
		
		BOOST_FOREACH(const ptree::ptree::value_type &v, pt)
		{
			boost::shared_ptr< DiscreteDistribution<T, DIM> > pdf;

			if (v.first == "dirac") {
				pdf.reset(new DiracDistribution<T, DIM>());
			}
			else if (v.first == "normal") {
				pdf.reset(new NormalDistribution<T, DIM>());
			}
			else if (v.first == "acg") {
				pdf.reset(new AngularCentralGaussianDistribution<T, DIM>());
			}
			else if (v.first == "list") {
				pdf.reset(new ListDistribution<T, DIM>());
			}
			else if (v.first == "uniform") {
				pdf.reset(new UniformDistribution<T, DIM>());
			}
			else if (v.first == "composite") {
				pdf.reset(new CompositeDistribution<T, DIM>());
			}
			else if (v.first == "<xmlcomment>") {
				continue;
			}
			else {
				BOOST_THROW_EXCEPTION(std::runtime_error("CompositeDistribution: unknown pdf!"));
			}

			pdf->readSettings(v.second);

			_pdfs.push_back(pdf);
			_sum_weights += pdf->weight();
		}
	}
};


//! Base class for a bounding box interface
template <typename T, int DIM>
class IBoundingBox
{
public:
	//! see https://stackoverflow.com/questions/827196/virtual-default-destructors-in-c/827205
	virtual ~IBoundingBox() { }

	//! radius of the bounding box
	virtual T bbRadius() const = 0;
	
	//! center of the bounding box
	virtual const ublas::c_vector<T, DIM>& bbCenter() const = 0;
	
	//! check for intersection (i.e. minimum distance between bb and this bounding box is less or equal than tol)
	inline bool bbIntersects(const IBoundingBox<T, DIM>& bb, T tol = 0) const
	{
		return (this->bbDistanceMin(bb) <= tol);
	}
	
	//! return center distance to point
	inline T bbDistance(const ublas::c_vector<T, DIM>& p) const
	{
		return ublas::norm_2(p - this->bbCenter());
	}

	//! return maximum distance to point
	//! note: if the distance is less than zero the point is inside the bounding box
	inline T bbDistanceMax(const ublas::c_vector<T, DIM>& p) const
	{
		return (this->bbDistance(p) + this->bbRadius());
	}

	//! return minimum distance to point
	//! note: if the distance is less than zero the point is inside the bounding box
	inline T bbDistanceMin(const ublas::c_vector<T, DIM>& p) const
	{
		return (this->bbDistance(p) - this->bbRadius());
	}

	//! return minimum distance between bounding boxes
	//! note: the distance is correct for positive values only
	//! if the distance is less than zero, it indicates overlapping, but the distance has then no other meaning
	inline T bbDistanceMin(const IBoundingBox<T, DIM>& bb) const
	{
		return (this->bbDistanceMin(bb.bbCenter()) - bb.bbRadius());
	}
};


//! Base class for all fiber (geometric) objects
template <typename T, int DIM>
class Fiber : public IBoundingBox<T, DIM>
{
public:
	//! see https://stackoverflow.com/questions/827196/virtual-default-destructors-in-c/827205
	virtual ~Fiber() { }

	//! Returns the gradient of the distance at the point p in g
	virtual void distanceGrad(const ublas::c_vector<T, DIM>& p, ublas::c_vector<T, DIM>& g) const = 0;

	//! Returns the signed minimum distance between this fiber
	//! and the point p and the point of minimum distance x
	virtual T distanceTo(const ublas::c_vector<T, DIM>& p, ublas::c_vector<T, DIM>& x) const = 0;

	//! Returns the signed minimum distance between this fiber
	//! and another fiber object and the points of minimum distance x and xf
	virtual T distanceTo(const Fiber<T, DIM>& fiber, ublas::c_vector<T, DIM>& x, ublas::c_vector<T, DIM>& xf) const = 0;
	
	//! Returns the signed minimum distance between this fiber
	//! and the plane defined by the point p and normal n
	virtual T distanceToPlane(const ublas::c_vector<T, DIM>& p, const ublas::c_vector<T, DIM>& n) const = 0;

	//! Returns true if p inside the fiber	
	virtual bool inside(const ublas::c_vector<T, DIM>& p) const = 0;

	//! clone fiber
	virtual boost::shared_ptr< Fiber<T, DIM> > clone() const = 0;
	
	//! translate fiber
	virtual void translate(const ublas::c_vector<T, DIM>& dx) = 0;

	//! maximal curvature of geometry
	virtual T curvature() const = 0;

	//! fiber volume
	virtual T volume() const = 0;

	//! fiber orientation
	virtual const ublas::c_vector<T, DIM>& orientation() const = 0;

	//! set the material id
	inline void set_material(std::size_t id) const {
		const_cast< Fiber<T,DIM>* >(this)->_material = id;
		const_cast< Fiber<T,DIM>* >(this)->_material_bits = 1 << id;
	}

	//! get the material id
	inline std::size_t material() const {
		return _material;
	}

	//! get the material id bits
	inline std::size_t material_bits() const {
		return _material_bits;
	}

	//! set the id
	inline void set_id(std::size_t id) const {
		const_cast< Fiber<T,DIM>* >(this)->_id = id;
	}

	//! get the id
	inline std::size_t id() const {
		return _id;
	}

	//! set the parent fiber (in case of ghost fibers)
	inline void set_parent(Fiber<T, DIM>* parent) const {
		const_cast< Fiber<T,DIM>* >(this)->_parent = boost::shared_ptr< Fiber<T, DIM> >(parent, boost::serialization::null_deleter());
	}

	//! get the parent fiber (in case of ghost fibers)
	inline boost::shared_ptr< Fiber<T, DIM> > parent() const {
		return _parent;
	}
	
	//! write fiber type and parameters to txt file
	virtual void writeData(std::ofstream& fs) const { }

protected:
	std::size_t _id;	// the unique identifierer of the fiber
	std::size_t _material;	// the material identifierer of the fiber
	std::size_t _material_bits;	// 1 << _material
	boost::shared_ptr< Fiber<T, DIM> > _parent;
};


#ifdef TEST_DIST_EVAL
// TODO: remove this / make more elegant
static int g_dist_evals = 0;
#endif


//! Class for efficiently computing fiber distances and intersections
template <typename T, int DIM>
class FiberCluster : public IBoundingBox<T, DIM>
{
public:
	typedef std::vector< boost::shared_ptr< const Fiber<T, DIM> > > fiber_ptr_list;
	typedef std::vector< boost::shared_ptr< FiberCluster<T, DIM> > > cluster_ptr_list;

protected:
	ublas::c_vector<T, DIM> _c;	// center of bounding box
	T _B;				// ball size of bounding box
	std::size_t _mcs;		// maximum cluster size
	std::size_t _fiberCount;	// number of fibers incl. of sub-clusters

	fiber_ptr_list _fibers;		// list of managed fibers
	cluster_ptr_list _clusters;	// list of managed sub-clusters
	
public:
	
	//! Constructor
	//! \param fiber defines the initial cluster size and center by a fiber
	//! \param mcs defines the maximum number of fibers in this cluster, if the fiber number
	//! is larger than mcs, fibers are transfered into sub-clusters to increase performance
	FiberCluster(const boost::shared_ptr< const Fiber<T, DIM> >& fiber, std::size_t mcs = 8)
	{
		if (!fiber) {
			BOOST_THROW_EXCEPTION(std::runtime_error("FiberCluster: fiber is null"));
		}
		
		if (mcs < 1) {
			BOOST_THROW_EXCEPTION(std::runtime_error("FiberCluster: maximum cluster size must be at least 1"));
		}
		
		_mcs = mcs;
		_c = fiber->bbCenter();
		_B = fiber->bbRadius();
		_fibers.reserve(mcs + 1);
		_fibers.push_back(fiber);
		_fiberCount = 1;
	}

	//! Test if point is inside some of the objects of specified materials
	//! \param p point
	//! \param mat material bits (i.e. 1<<material_id)
	virtual bool inside(const ublas::c_vector<T, DIM>& p, int mat) const
	{
		// check if point is close enough to bounding box
		if (this->bbDistanceMin(p) > 0) {
			return false;
		}

		// now check for matching fibers
		for (typename fiber_ptr_list::const_iterator i = _fibers.begin(); i != _fibers.end(); i++) {
			//if (mat >= 0 && (*i)->material() != (std::size_t)mat) continue;
			if (((*i)->material_bits() & (std::size_t)mat) == 0) continue;
			if ((*i)->bbDistanceMin(p) <= 0) {
				if ((*i)->inside(p)) return true;
			}
		}

		for (typename cluster_ptr_list::const_iterator i = _clusters.begin(); i != _clusters.end(); i++) {
			if ((*i)->inside(p, mat)) return true;
		}

		return false;
	}

	//! returns the signed minimum distance between any fiber and the point p within an approximate radius of r (actual radius might be larger)
	//! further returns the point of minimum distance x as well as the corresponding fiber
	//! if there is no such fiber within radius r, returns r and fiber will be set to NULL
	virtual T closestFiber(const ublas::c_vector<T, DIM>& p, T r, int mat, ublas::c_vector<T, DIM>& x, boost::shared_ptr< const Fiber<T, DIM> >& fiber) const
	{
		DEBP("p = " << format(p));
		DEBP("bbCenter() = " << format(this->bbCenter()));
		DEBP("bbRadius() = " << this->bbRadius());
		DEBP("bbDistanceMin(p) = " << this->bbDistanceMin(p));
		DEBP("r = " << r);

		// check if point is close enough to bounding box
		if (this->bbDistanceMin(p) > r) {
			fiber.reset();
			return r;
		}
		
		// now shrink the radius to the minimum of the maximum bounding box distance
		for (typename fiber_ptr_list::const_iterator i = _fibers.begin(); i != _fibers.end(); i++) {
			//if (mat >= 0 && (*i)->material() != (std::size_t)mat) continue;
			//if (((*i)->material_bits() & (std::size_t)mat) == 0) continue;
			r = std::min(r, (*i)->bbDistanceMax(p));
		}
		for (typename cluster_ptr_list::const_iterator i = _clusters.begin(); i != _clusters.end(); i++) {
			r = std::min(r, (*i)->bbDistanceMax(p));
		}

		DEBP("r = " << r);

		T dMin = r*1.001;
		ublas::c_vector<T, DIM> xMin(DIM);
		boost::shared_ptr< const Fiber<T, DIM> > fiberMin;

#if 1
		// perform quick initial guess if fiber is set
		// unset the fiber if failed to avoid checking a second time
		// FIXME: this actually does not check if fiber is part of the cluster
		if (fiber) {
			if (fiber->material_bits() & (std::size_t)mat) {
			//if (mat < 0 || fiber->material() == (std::size_t)mat) {
				T d = fiber->distanceTo(p, xMin);
				if (d <= dMin) {
					dMin = d;
					fiberMin = fiber;
				}
				else {
					fiber.reset();
				}
			}
			else {
				fiber.reset();
			}
		}
#endif

		// now check for matching fibers
		for (typename fiber_ptr_list::const_iterator i = _fibers.begin(); i != _fibers.end(); i++) {
			if (((*i)->material_bits() & (std::size_t)mat) == 0) continue;
			//if (mat >= 0 && (*i)->material() != (std::size_t)mat) continue;
			if ((*i)->bbDistanceMin(p) <= dMin) {
				T d = (*i)->distanceTo(p, x);
				if (d < dMin) {
					fiberMin = *i;
					dMin = d;
					xMin = x;
				}
			}
		}

#if 0
		T rp = r;
		#pragma omp parallel for firstprivate(rp) schedule (static)
		for (size_t i = 0; i < _clusters.size(); i++)
		{
			ublas::c_vector<T, DIM> x;
			boost::shared_ptr<const Fiber<T, DIM> > fiber;
			T d = _clusters[i]->closestFiber(p, rp, mat, x, fiber);

			#pragma omp critical
			{
				if (fiber && (d < dMin)) {
					fiberMin = fiber;
					dMin = d;
					xMin = x;
				}
				r = std::min(r, d);
				rp = r;
			}
		}
#else
		for (typename cluster_ptr_list::const_iterator i = _clusters.begin(); i != _clusters.end(); i++) {
			T d = (*i)->closestFiber(p, dMin, mat, x, fiber);
			if (fiber && (d < dMin)) {
				fiberMin = fiber;
				dMin = d;
				xMin = x;
			}
		}
#endif

		// return minimum values
		fiber = fiberMin;
		x = xMin;
		return dMin;		
	}

	//! Data structure holding fiber distance information
	typedef struct {
		ublas::c_vector<T, DIM> x;	// closest point of fiber
		boost::shared_ptr< const Fiber<T, DIM> > fiber;	// the fiber
		T d;	// distance to x
	} ClosestFiberInfo;

	//! returns all closest fibers to the point p within an approximate radius of r (actual radius might be larger)
	virtual void closestFibers(const ublas::c_vector<T, DIM>& p, T r, int mat, std::vector<ClosestFiberInfo>& info_list) const
	{
		// check if point is close enough to bounding box
		if (this->bbDistanceMin(p) > r) {
			return;
		}
		
		ClosestFiberInfo info;

		// now check for matching fibers
		for (typename fiber_ptr_list::const_iterator i = _fibers.begin(); i != _fibers.end(); i++) {
			if (((*i)->material_bits() & (std::size_t)mat) == 0) continue;
			//if (mat >= 0 && (*i)->material() != (std::size_t)mat) continue;
			if ((*i)->bbDistanceMin(p) <= r) {
				info.d = (*i)->distanceTo(p, info.x);
				if (info.d <= r) {
					info.fiber = *i;
					info_list.push_back(info);
				}
			}
		}

		for (typename cluster_ptr_list::const_iterator i = _clusters.begin(); i != _clusters.end(); i++) {
			(*i)->closestFibers(p, r, mat, info_list);
		}
	}

	//! check for point intersection (i.e. minimum distance between point and fibers in this cluster is less than tol)
	//! in case of intersection the intersected fiber and intersection point xf are set, as well as the distance to the fiber d
	bool intersects(const ublas::c_vector<T, DIM>& p, T tol, int mat, boost::shared_ptr< const Fiber<T, DIM> >& fiber, ublas::c_vector<T, DIM>& xf, T& d)
	{
#if 0
		// perform quick intersection guess if fiber is set
		// unset the fiber if failed to avoid checking a second time
		// FIXME: this actually does not check if fiber is part of the cluster
		if (fiber) {
			//if (mat < 0 || fiber->material() == (std::size_t)mat) {
			if (fiber->material_bits() & (std::size_t)mat) {
				d = fiber->distanceTo(p, xf);
				if (d <= tol) {
					return true;
				}
			}
			fiber.reset();
		}
#endif

		// check if bounding boxes intersect
		if (this->bbDistanceMin(p) > tol) {
			return false;
		}
		
		if (_fibers.size() > 0)
		{
			// now check every fiber
			for (typename fiber_ptr_list::const_iterator i = _fibers.begin(); i != _fibers.end(); i++) {
				if (((*i)->material_bits() & (std::size_t)mat) == 0) continue;
				//if (mat >= 0 && (*i)->material() != (std::size_t)mat) continue;
				if ((*i)->bbDistanceMin(p) <= tol) {
					// now need to perform exact distance check
					d = (*i)->distanceTo(p, xf);
					if (d <= tol) {
						fiber = (*i);
						return true;
					}
				}
			}
		}

		// now check every sub-cluster
#if 0
		bool ret = false;
		#pragma omp parallel for schedule (static)
		for (size_t i = 0; i < _clusters.size(); i++)
		{
			boost::shared_ptr< const Fiber<T, DIM> > _fiber;
			ublas::c_vector<T, DIM> _xf;
			T _d;

			if (!ret && _clusters[i]->intersects(p, tol, mat, _fiber, _xf, _d))
			{
				#pragma omp critical
				{
					ret = true;
					fiber = _fiber;
					xf = _xf;
					d = _d;
				}
			}
		}

		return ret;
#else
		for (typename cluster_ptr_list::const_iterator i = _clusters.begin(); i != _clusters.end(); i++) {
			if ((*i)->intersects(p, tol, mat, fiber, xf, d)) {
				return true;
			}
		}

		return false;
#endif
	}
	
	//! check for fiber intersection (i.e. minimum distance between fiber and fibers in this cluster is less than tol)
	//! in case of intersection the intersected fiber and the minimum points x and xf are set, as well as theier distance d
	bool intersects(const Fiber<T, DIM>& fiber, T tol, int mat, boost::shared_ptr< const Fiber<T, DIM> >& fiberi, ublas::c_vector<T, DIM>& x, ublas::c_vector<T, DIM>& xf, T& d)
	{
		// check if bounding boxes intersect
		if (!this->bbIntersects(fiber, tol)) {
			return false;
		}
		
		if (_fibers.size() > 0)
		{
			// now check every fiber
			for (typename fiber_ptr_list::const_iterator i = _fibers.begin(); i != _fibers.end(); i++) {
				if (((*i)->material_bits() & (std::size_t)mat) == 0) continue;
				//if (mat >= 0 && (*i)->material() != (std::size_t)mat) continue;
				if ((*i)->bbIntersects(fiber, tol)) {
					// now need to perform exact distance check
					d = (*i)->distanceTo(fiber, x, xf);
					if (d <= tol) {
						fiberi = (*i);
						return true;
					}
				}
			}
		}

		// now check every sub-cluster
#if 0
		bool ret = false;
		#pragma omp parallel for schedule (static)
		for (size_t i = 0; i < _clusters.size(); i++)
		{
			boost::shared_ptr< const Fiber<T, DIM> > _fiberi;
			ublas::c_vector<T, DIM> _x;
			ublas::c_vector<T, DIM> _xf;
			T _d;

			if (!ret && _clusters[i]->intersects(fiber, tol, mat, _fiberi, _x, _xf, _d))
			{
				#pragma omp critical
				{
					ret = true;
					fiberi = _fiberi;
					x = _x;
					xf = _xf;
					d = _d;
				}
			}
		}

		return ret;
#else
		for (typename cluster_ptr_list::const_iterator i = _clusters.begin(); i != _clusters.end(); i++) {
			if ((*i)->intersects(fiber, tol, mat, fiberi, x, xf, d)) {
				return true;
			}
		}

		return false;
#endif
	}
	
	//! calculate new bounding box radius and center, if we would add fiber to cluster
	//! \param fiber the fiber
	//! \param cnew new bounding box center
	//! \return new bounding box radius
	T calcNewBB(const Fiber<T, DIM>& fiber, ublas::c_vector<T, DIM>* cnew = NULL)
	{
		// calculate new bounding box
		T d = this->bbDistanceMin(fiber);
		T minR = std::min(_B, fiber.bbRadius());
		T maxR = std::max(_B, fiber.bbRadius());
		
		if (d + 2*minR <= 0)
		{
			// the larger object contains the smaller object
			
			if (_B < fiber.bbRadius()) {
				// the larger object is the fiber
				if (cnew != NULL) *cnew = fiber.bbCenter();
				return fiber.bbRadius();
			}
			else {
				// the larger object is already the cluster
				// no change necessary
				if (cnew != NULL) *cnew = _c;
				return _B;
			}
		}
		else
		{
			// compute new center and radius
			
			if (cnew != NULL) {
				ublas::c_vector<T, DIM> a = fiber.bbCenter() - _c;
				*cnew = (_c + fiber.bbCenter() + a*((fiber.bbRadius() - _B)/std::max(ublas::norm_2(a), std::numeric_limits<T>::epsilon())))/2;
			}
			
			return (d/2 + minR + maxR);
		}
	}
	
	//! distance measure between cluster and fiber
	T distanceMeasure(const Fiber<T, DIM>& fiber)
	{
#if 1
		// penalize change of BB volume (better)
		T r = calcNewBB(fiber);
		T r0 = bbRadius();
		return (r*r*r - r0*r0*r0);
#else
		// penalize BB distance
		return this->bbDistanceMin(fiber);
#endif
	}
	
	//! add fiber to cluster
	void add(const boost::shared_ptr< const Fiber<T, DIM> >& fiber)
	{
		if (!fiber) {
			BOOST_THROW_EXCEPTION(std::runtime_error("FiberCluster::add: fiber is null"));
		}
		
		// calculate new bounding box
		_B = calcNewBB(*fiber, &_c);
		
		// check if we exceed maximum cluster size, after adding the fiber
		if (_fibers.size() >= _mcs) {
			this->performSubClustering();
		}
		
		if (_clusters.size() > 0)
		{
			typename cluster_ptr_list::const_iterator iMin = _clusters.begin();
			T dMin = (*iMin)->distanceMeasure(*fiber);
			
			// add fiber to closest sub-cluster
			for (typename cluster_ptr_list::const_iterator i = _clusters.begin()++; i != _clusters.end(); i++)
			{
				T d = (*i)->distanceMeasure(*fiber);
				if (d < dMin) {
					iMin = i;
					dMin = d;
				}
			}

			(*iMin)->add(fiber);
		}
		else
		{
			// add fiber to list
			_fibers.push_back(fiber);
		}

		_fiberCount ++;
	}

	//! transfer all fibers into sub-clusters
	void performSubClustering()
	{
		for (typename fiber_ptr_list::const_iterator i = _fibers.begin(); i != _fibers.end(); i++) {
			_clusters.push_back(boost::shared_ptr< FiberCluster<T, DIM> >(new FiberCluster(*i, _mcs)));
		}

		// release some storage
		_fibers.clear();
	}

	//! return number of fibers in this cluster (including all nested clusters)
	inline std::size_t fiberCount()
	{
		return _fiberCount;
	}
	
	// bounding box interface methods
	
	virtual T bbRadius() const
	{
		return _B;
	}
	
	virtual const ublas::c_vector<T, DIM>& bbCenter() const
	{
		return _c;
	}

	inline const fiber_ptr_list& fibers() const {
		return _fibers;
	}

	inline const cluster_ptr_list& clusters() const {
		return _clusters;
	}

	void writeData(std::ofstream& fs) const
	{
		for (typename fiber_ptr_list::const_iterator i = _fibers.begin(); i != _fibers.end(); i++) {
			(*i)->writeData(fs);
		}

		for (typename cluster_ptr_list::const_iterator i = _clusters.begin()++; i != _clusters.end(); i++)
		{
			(*i)->writeData(fs);
		}
	}
};


//! Cylindrical fiber with spherical end-caps
template <typename T, int DIM>
class CylindricalFiber : public Fiber<T, DIM>
{
protected:
	ublas::c_vector<T, DIM> _a;	// axis direction of cylinder ((c2 - c1) normed)
	ublas::c_vector<T, DIM> _r;	// vector orthogonal to _a with length _R
	ublas::c_vector<T, DIM> _c1;	// base point 1 of cylinder
	ublas::c_vector<T, DIM> _c;	// center of cylinder
	ublas::c_vector<T, DIM> _c2;	// base point 2 of cylinder
	T _R;	// radius of cylinder
	T _L;   // length of cylinder
	T _B;   // ball radius of bounding box

public:

	//! construct from center c, orientation a, length L and radius R
	CylindricalFiber(const ublas::c_vector<T, DIM>& c, const ublas::c_vector<T, DIM>& a, T L, T R)
	{
		T norm_a = ublas::norm_2(a);
		
		if (norm_a != 0) {
			_a = a / norm_a;
		}
		else if (L != 0) {
			BOOST_THROW_EXCEPTION(std::runtime_error("CylindricalFiber: given nonzero fiber length without orientation vector!"));
		}
		else {
			_a *= 0;
		}
		
		_c  = c;
		_L  = std::abs(L);
		_R  = std::abs(R);
		_c1 = _c - (_L/2)*_a;
		_c2 = _c + (_L/2)*_a;
		_B  = std::sqrt(_L*_L/4 + _R*_R);
		_r = orthonormal_vector<T,DIM>(_a)*_R;
	}

	virtual void distanceGrad(const ublas::c_vector<T, DIM>& p, ublas::c_vector<T, DIM>& g) const
	{
		ublas::c_vector<T, DIM> pc1 = p - _c1;
		
		// calculate length of projection of p-c1 onto a
		T t = ublas::inner_prod(pc1, _a);
		
		// compute radial distance
		ublas::c_vector<T, DIM> r = pc1 - t*_a;
		ublas::c_matrix<T, DIM, DIM> aa = ublas::outer_prod(_a, _a);
		ublas::c_matrix<T, DIM, DIM> drdp = ublas::identity_matrix<T>(DIM) - aa;

		// get norm of r
		T norm_r = ublas::norm_2(r);
	
		// remember if point is inside cylinder
		bool inside = (norm_r <= _R && 0 <= t && t <= _L);
	
		// bound t to [0, L]
		T dtdt = ((0 <= t && t <= _L) ? 1 : 0);
		t = std::min(std::max((T)0, t), _L);
		
		ublas::c_vector<T, DIM> x;
		ublas::c_matrix<T, DIM, DIM> dxdp;

		if (norm_r > std::max((T)0, std::max(_R - t, _R - _L + t))) {
			if (norm_r < std::numeric_limits<T>::epsilon()*_R) { // TODO: choose different criteria?
				//x = (t < 0.5*_L) ? _c1 : _c2;
				//dxdp = 0*aa;
				x = _c1 + t*_a + _r;
				dxdp = dtdt*aa;
			}
			else {
				x = _c1 + t*_a + _R*(r/norm_r);
				dxdp = dtdt*aa;
			}
		}
		else if (t < 0.5*_L) {
			x = _c1 + r;
			dxdp = drdp;
		}
		else {
			x = _c1 + _L*_a + r;
			dxdp = drdp;
		}

		// compute distance
		T scale = (inside ? -1 : 1);

		g[0] = ((p[0] - x[0])*(1 - dxdp(0,0)) + (p[1] - x[1])*(-dxdp(1,0))    + (p[2] - x[2])*(-dxdp(2,0)))*scale;
		g[1] = ((p[0] - x[0])*(-dxdp(0,1))    + (p[1] - x[1])*(1 - dxdp(1,1)) + (p[2] - x[2])*(-dxdp(2,1)))*scale;
		g[2] = ((p[0] - x[0])*(-dxdp(0,2))    + (p[1] - x[1])*(-dxdp(1,2))    + (p[2] - x[2])*(1 - dxdp(2,2)))*scale;

		T norm_g = ublas::norm_2(g);

		if (norm_g < std::sqrt(std::numeric_limits<T>::epsilon())) {
			g = ((t < 0.5*_L) ? -1 : 1) * _a;
		}
		else {
			g /= norm_g;
		}

		//LOG_COUT << "grad at " << format(p) << " g=" << format(g) << " x=" << format(x) << std::endl;
		//LOG_COUT << "dxdp " << format(dxdp) << std::endl;
	}

	virtual T distanceTo(const ublas::c_vector<T, DIM>& p, ublas::c_vector<T, DIM>& x) const
	{
		ublas::c_vector<T, DIM> pc1 = p - _c1;
		
		// calculate length of projection of p-c1 onto a
		T t = ublas::inner_prod(pc1, _a);
		
		// compute radial distance
		ublas::c_vector<T, DIM> r = pc1 - t*_a;
		
		// get norm of r
		T norm_r = ublas::norm_2(r);
	
		// remember if point is inside cylinder
		bool inside = (norm_r <= _R && 0 <= t && t <= _L);
	
		// bound t to [0, L]
		t = std::min(std::max((T)0, t), _L);
		
		if (norm_r > std::max((T)0, std::max(_R - t, _R - _L + t))) {
			if (norm_r < std::numeric_limits<T>::epsilon()*_R) { // TODO: choose different criteria?
				//x = (t < 0.5*_L) ? _c1 : _c2;
				x = _c1 + t*_a + _r;
			}
			else {
				x = _c1 + t*_a + _R*(r/norm_r);
			}
		}
		else if (t < 0.5*_L) {
			x = _c1 + r;
		}
		else {
			x = _c1 + _L*_a + r;
		}

		// compute distance
		T d = ublas::norm_2(p - x) * (inside ? -1 : 1);

#ifdef TEST_DIST_EVAL
		#pragma omp atomic
		g_dist_evals ++;
#endif

		// return distance
		return d;
	}

	virtual T distanceTo(const Fiber<T, DIM>& fiber, ublas::c_vector<T, DIM>& x, ublas::c_vector<T, DIM>& xf) const
	{
		// FIXME: this currently returns the distance to a capsule not a cylinder!

#if 0
		const HalfSpaceFiber<T, DIM>* hs = dynamic_cast<const HalfSpaceFiber<T, DIM>*>(&fiber);

		if (hs != NULL) {
			return hs->
		}
#endif
		const CylindricalFiber<T, DIM>* cf = dynamic_cast<const CylindricalFiber<T, DIM>*>(&fiber);

		if (cf == NULL) {
			BOOST_THROW_EXCEPTION(std::runtime_error("CylindricalFiber::distanceTo: can compute minimum distance to another CylindricalFiber only!"));
		}

		// the minimum distance between two smooth convex objects is characterized,
		// by the fact that both normals at the minimum points are parallel.

		// check if c1 is a minimum point

		T d1 = cf->distanceTo(_c1, xf);
		ublas::c_vector<T, DIM> dx1 = xf - _c1;

		if (ublas::inner_prod(dx1, _a) <= 0) {
			// c1 is a minimum point
			x = _c1 + dx1 * (_R / std::max(ublas::norm_2(dx1), std::numeric_limits<T>::epsilon()*_R));
			return (d1 - _R);
		}

		// check if c2 is a minimum point

		T d2 = cf->distanceTo(_c2, xf);
		ublas::c_vector<T, DIM> dx2 = xf - _c2;

		if (ublas::inner_prod(dx2, _a) >= 0) {
			// c2 is a minimum point
			x = _c2 + dx2 * (_R / std::max(ublas::norm_2(dx2), std::numeric_limits<T>::epsilon()*_R));
			return (d2 - _R);
		}

		// the minimum point is determined by
		// fiber->distanceTo(_c1 + t*_a, xf), (_c1 + t*_a - xf)*_a = 0
		// for t in [0,1]
		// which gives the equation for t
		// f1 - t + min(max(0, f2 + t*aa), cf->_L)*aa = 0

		T f1 = ublas::inner_prod(cf->_c1 - _c1, _a);
		T f2 = ublas::inner_prod(_c1 - cf->_c1, cf->_a);
		T aa = ublas::inner_prod(cf->_a, _a);
		T t;

		if (aa == 0) {
			t = f1;
		}
		else {
			// rewrite equation as
			// min(max((f1 - t)/aa, (f1 - t)/aa + f2 + t*aa), (f1 - t)/aa + cf->_L) = 0
			// this gives 3 choices for t

			// T t1 = f1;
			// T t2 = f1 + aa*cf->_L;
			// T t3 = (f1 + aa*f2) / (1 - aa*aa);

			// t1 is valid if
			// (f1 - t)/aa + f2 + t*aa <= 0 and (f1 - t)/aa + cf->_L >= 0
			// i.e. f2 + aa*f1 <= 0 and cf->_L >= 0

			T D = f2 + aa*f1;

			if (D <= 0) {
				t = f1;
			}

			// t2 is valid if
			// max(-cf->_L, -cf->_L + f2 + (f1 + aa*cf->_L)*aa) >= 0
			// i.e. f2 + aa*f1 >= cf->_L*(1 - aa*aa)

			else if (D >= cf->_L*(1 - aa*aa)) {
				t = f1 + aa*cf->_L;
				// this corresponds to "xf = cf->_c2"
			}

			// else it must be t3

			else {
				t = (f1 + aa*f2) / std::max(1 - aa*aa, std::numeric_limits<T>::epsilon());
			}
		}

		ublas::c_vector<T, DIM> p = _c1 + t*_a;
		T d = cf->distanceTo(p, xf);
		ublas::c_vector<T, DIM> dx = xf - p;
		x = p + dx * (_R / std::max(ublas::norm_2(dx), std::numeric_limits<T>::epsilon()*_R));
		return (d - _R);
	}

	virtual T distanceToPlane(const ublas::c_vector<T, DIM>& p, const ublas::c_vector<T, DIM>& n) const
	{
		// FIXME: this is still the code for the capsule...

		T norm_n = ublas::norm_2(n);
		T dist_c1 = ublas::inner_prod(_c1 + _R*_a - p, n) / norm_n;
		T dist_c2 = ublas::inner_prod(_c2 - _R*_a - p, n) / norm_n;
		
		if (dist_c1*dist_c2 <= 0) return 0;
		if (std::fabs(dist_c1) <= _R) return 0;
		if (std::fabs(dist_c2) <= _R) return 0;
		
		if (dist_c1 < 0) {
			return std::max(dist_c1, dist_c2) + _R;
		}
		
		return std::min(dist_c1, dist_c2) - _R;
	}

	virtual bool inside(const ublas::c_vector<T, DIM>& p) const
	{
		BOOST_THROW_EXCEPTION(std::runtime_error("not implemented"));
		return false;
	}
	
	virtual boost::shared_ptr< Fiber<T, DIM> > clone() const
	{
		boost::shared_ptr< Fiber<T, DIM> > fiber(new CylindricalFiber(_c, _a, _L, _R));
		fiber->set_id(this->id());
		fiber->set_material(this->material());
		fiber->set_parent(const_cast< Fiber<T, DIM>* const >(reinterpret_cast<const Fiber<T, DIM>* const>(this)));
		return fiber;
	}
	
	inline virtual void translate(const ublas::c_vector<T, DIM>& dx)
	{
		_c  += dx;
		_c1 += dx;
		_c2 += dx;
	}
	
	inline virtual T curvature() const
	{
		return 1/_R;
	}
	
	inline virtual T volume() const
	{
		if (DIM == 3) {
			return M_PI*_R*_R*_L;
		}

		return 2*_R*_L;
	}
	
	inline virtual const ublas::c_vector<T, DIM>& orientation() const
	{
		return _a;
	}

	inline virtual const ublas::c_vector<T, DIM>& center() const
	{
		return _c;
	}

	inline virtual T length() const
	{
		return _L;
	}
	
	inline virtual T radius() const
	{
		return _R;
	}
	
	// bounding box interface methods
	
	inline virtual T bbRadius() const
	{
		return _B;
	}
	
	inline virtual const ublas::c_vector<T, DIM>& bbCenter() const
	{
		return _c;
	}
};


//! Tetrahedron fiber (i.e. a single tetrahedron)
template <typename T, int DIM>
class TetrahedronFiber : public Fiber<T, DIM>
{
protected:
	ublas::c_vector<T, DIM> _p[4];
	ublas::c_vector<T, DIM> _n[4];
	ublas::c_vector<T, DIM> _a[6];
	ublas::c_matrix<T, 6, 6> _M;
	ublas::c_matrix<T, 3, 3> _Minv;
	ublas::c_vector<T, DIM> _c;
	T _B;

public:

	// construct from points
	TetrahedronFiber(const ublas::c_vector<T, DIM> p[4])
	{
		_p[0] = p[0];
		_p[1] = p[1];
		_p[2] = p[2];
		_p[3] = p[3];
		
		_c = (_p[0] + _p[1] + _p[2] + _p[3])/4;
		
		_B = 0;
		for (std::size_t i = 0; i < 4; i++) {
			_B  = std::max(ublas::norm_2(_p[i] - _c), _B);
		}
		
		// make sure tetrahedron has positive volume
		for (std::size_t i = 0; ; i++) {
			_a[0] = _p[1] - _p[0];
			_a[1] = _p[2] - _p[0];
			_a[2] = _p[3] - _p[0];
			T vol = ublas::inner_prod(cross_prod(_a[0], _a[1]), _a[2]);
			if (i == 1 || vol >= 0) {
				if (std::abs(vol) < std::numeric_limits<T>::epsilon()*(ublas::norm_2(_a[0]) + ublas::norm_2(_a[1]) + ublas::norm_2(_a[2]))) {
					BOOST_THROW_EXCEPTION(std::runtime_error("flat tetrahedron!"));
				}
				break;
			}
			std::swap(_p[0], _p[1]);
			// LOG_COUT << "swapping tet of volume " << vol << std::endl;
		}

		_a[3] = _p[2] - _p[1];
		_a[4] = _p[3] - _p[1];
		_a[5] = _p[3] - _p[2];

		for (std::size_t i = 0; i < 6; i++) {
			for (std::size_t j = i; j < 6; j++) {
				_M(i,j) = _M(j, i) = ublas::inner_prod(_a[i], _a[j]);
			}
		}

		ublas::c_matrix<T, 3, 3> M;
		for (std::size_t i = 0; i < 3; i++) {
			for (std::size_t j = i; j < 3; j++) {
				M(i,j) = M(j, i) = _M(i, j);
			}
		}

#if 0
		for (std::size_t i = 0; i < 4; i++) {
			LOG_COUT << "tetfiber point " << i << ": " << format(_p[i]) << std::endl;
		}
#endif

		InvertMatrix<T,3>(M, _Minv);

		_n[0] = cross_prod(_a[1], _a[0]);
		_n[1] = cross_prod(_a[0], _a[2]);
		_n[2] = cross_prod(_a[2], _a[1]);
		_n[3] = cross_prod(_a[3], _a[4]);

		for (std::size_t i = 0; i < 4; i++) {
			_n[i] /= ublas::norm_2(_n[i]);
		}
	}

	T distanceToTriangle(const ublas::c_vector<T, DIM>& p, ublas::c_vector<T, DIM>& x, std::size_t p0, std::size_t p1, std::size_t p2, std::size_t a0, std::size_t a1, std::size_t a2, T a2_sign, std::size_t n) const
	{
		ublas::c_vector<T, DIM> dp = p - _p[p0];

		// project p to plane
		T d = ublas::inner_prod(dp, _n[n]);
		x = p - d*_n[n];
		dp = x - _p[p0];

		ublas::c_matrix<T, 2, 2> M, Minv;

		M(0,0) = _M(a0, a0);
		M(1,1) = _M(a1, a1);
		M(1,0) = M(0,1) = _M(a0, a1);

		// TODO: use faster code for inversion
		InvertMatrix<T,2>(M, Minv);

		T b[2];
		T t[2];

		b[0] = ublas::inner_prod(dp, _a[a0]);
		b[1] = ublas::inner_prod(dp, _a[a1]);
		t[0] = Minv(0,0)*b[0] + Minv(0,1)*b[1];
		t[1] = Minv(1,0)*b[0] + Minv(1,1)*b[1];

		ublas::c_vector<T, DIM> z[2];
		std::size_t nz = 0;

		if (t[0] <= 0) {
			T t = ublas::inner_prod(dp, _a[a1])/_M(a1, a1);
			t = std::min(std::max(t, (T)0), (T)1);
			z[nz++] = _p[p0] + t*_a[a1];
		}
		if (t[1] <= 0) {
			T t = ublas::inner_prod(dp, _a[a0])/_M(a0, a0);
			t = std::min(std::max(t, (T)0), (T)1);
			z[nz++] = _p[p0] + t*_a[a0];
		}
		if (t[0]+t[1] >= 1) {
			T t = a2_sign*ublas::inner_prod(x - _p[p2], _a[a2])/_M(a2, a2);
			t = std::min(std::max(t, (T)0), (T)1);
			z[nz++] = _p[p2] + (t*a2_sign)*_a[a2];
		}

		if (nz > 0)
		{
			T norm_g[2];
			for (std::size_t i = 0; i < nz; i++) {
				norm_g[i] = ublas::norm_2(p - z[i]);
			}

			std::size_t i_min = 0;
			for (std::size_t i = 1; i < nz; i++) {
				if (norm_g[i] < norm_g[i_min]) i_min = i;
			}

			x = z[i_min];
		}

		return ublas::norm_2(p - x);
	}
	
	T distanceToTriangleGrad(const ublas::c_vector<T, DIM>& p, ublas::c_vector<T, DIM>& g, std::size_t p0, std::size_t p1, std::size_t p2, std::size_t a0, std::size_t a1, std::size_t a2, T a2_sign, std::size_t n) const
	{
		ublas::c_vector<T, DIM> dp = p - _p[p0];

		// project p to plane
		T d = ublas::inner_prod(dp, _n[n]);
		ublas::c_vector<T, DIM> x = p - d*_n[n];
		dp = x - _p[p0];

		ublas::c_matrix<T, 2, 2> M, Minv;

		M(0,0) = _M(a0, a0);
		M(1,1) = _M(a1, a1);
		M(1,0) = M(0,1) = _M(a0, a1);

		// TODO: use faster code for inversion
		InvertMatrix<T,2>(M, Minv);

		T b[2];
		T t[2];

		b[0] = ublas::inner_prod(dp, _a[a0]);
		b[1] = ublas::inner_prod(dp, _a[a1]);
		t[0] = Minv(0,0)*b[0] + Minv(0,1)*b[1];
		t[1] = Minv(1,0)*b[0] + Minv(1,1)*b[1];

		ublas::c_vector<T, DIM> z[2];
		std::size_t nz = 0;

		if (t[0] <= 0) {
			T t = ublas::inner_prod(dp, _a[a1])/_M(a1, a1);
			t = std::min(std::max(t, (T)0), (T)1);
			z[nz++] = _p[p0] + t*_a[a1];
		}
		if (t[1] <= 0) {
			T t = ublas::inner_prod(dp, _a[a0])/_M(a0, a0);
			t = std::min(std::max(t, (T)0), (T)1);
			z[nz++] = _p[p0] + t*_a[a0];
		}
		if (t[0]+t[1] >= 1) {
			T t = a2_sign*ublas::inner_prod(x - _p[p2], _a[a2])/_M(a2, a2);
			t = std::min(std::max(t, (T)0), (T)1);
			z[nz++] = _p[p2] + (t*a2_sign)*_a[a2];
		}

		if (nz == 0) {
			// inside
			g = _n[n];
			return 0;
		}

		T norm_g[2];
		for (std::size_t i = 0; i < nz; i++) {
			norm_g[i] = ublas::norm_2(p - z[i]);
		}

		std::size_t i_min = 0;
		for (std::size_t i = 1; i < nz; i++) {
			if (norm_g[i] < norm_g[i_min]) i_min = i;
		}

		if (norm_g[i_min] == 0) {
			g = _n[n];
		}
		else {
			g = (p - z[i_min]) / norm_g[i_min];
			if (ublas::inner_prod(g, _n[n]) < 0) {
				g = -g;
			}
		}

		return norm_g[i_min];
	}

	virtual void distanceGrad(const ublas::c_vector<T, DIM>& p, ublas::c_vector<T, DIM>& g) const
	{
		// determine barycentric coordinates
		// p = p0 + a0*t0 + a1*t1 + a2*t2

		ublas::c_vector<T, DIM> dp = p - _p[0];

		T b[3];
		T t[3];

		for (std::size_t i = 0; i < 3; i++) {
			b[i] = ublas::inner_prod(dp, _a[i]);
		}
		for (std::size_t i = 0; i < 3; i++) {
			t[i] = _Minv(i,0)*b[0] + _Minv(i,1)*b[1] + _Minv(i,2)*b[2];
		}
		
		ublas::c_vector<T, DIM> z[3];
		T dz[3];
		std::size_t nz = 0;

		if (t[0] <= 0) {
			dz[nz] = distanceToTriangleGrad(p, z[nz], 0, 3, 2, 2, 1, 5, 1, 2); nz++;
		}

		if (t[1] <= 0) {
			dz[nz] = distanceToTriangleGrad(p, z[nz], 0, 1, 3, 0, 2, 4, -1, 1); nz++;
		}

		if (t[2] <= 0) {
			dz[nz] = distanceToTriangleGrad(p, z[nz], 0, 2, 1, 1, 0, 3, 1, 0); nz++;
		}

		if (t[0]+t[1]+t[2] >= 1) {
			dz[nz] = distanceToTriangleGrad(p, z[nz], 1, 2, 3, 3, 4, 5, -1, 3); nz++;
		}

		if (nz > 0)
		{
			std::size_t i_min = 0;
			for (std::size_t i = 1; i < nz; i++) {
				if (dz[i] < dz[i_min]) i_min = i;
			}

			g = z[i_min];
			return;
		}

		// inside
		
		T d[4];
		for (std::size_t i = 0; i < 4; i++) {
			d[i] = ublas::inner_prod(p - _p[i], _n[i]);
		}
		
		std::size_t i_max = 0;
		for (std::size_t i = 1; i < 4; i++) {
			if (d[i] > d[i_max]) i_max = i;
		}

		g = _n[i_max];
	}

	virtual T distanceTo(const ublas::c_vector<T, DIM>& p, ublas::c_vector<T, DIM>& x) const
	{
		// determine barycentric coordinates
		// p = p0 + a0*t0 + a1*t1 + a2*t2

		ublas::c_vector<T, DIM> dp = p - _p[0];

		T b[3];
		T t[3];

		for (std::size_t i = 0; i < 3; i++) {
			b[i] = ublas::inner_prod(dp, _a[i]);
		}
		for (std::size_t i = 0; i < 3; i++) {
			t[i] = _Minv(i,0)*b[0] + _Minv(i,1)*b[1] + _Minv(i,2)*b[2];
		}
		
		ublas::c_vector<T, DIM> z[3];
		T dz[3];
		std::size_t nz = 0;

		if (t[0] <= 0) {
			dz[nz] = distanceToTriangle(p, z[nz], 0, 3, 2, 2, 1, 5, 1, 2); nz++;
		}

		if (t[1] <= 0) {
			dz[nz] = distanceToTriangle(p, z[nz], 0, 1, 3, 0, 2, 4, -1, 1); nz++;
		}

		if (t[2] <= 0) {
			dz[nz] = distanceToTriangle(p, z[nz], 0, 2, 1, 1, 0, 3, 1, 0); nz++;
		}

		if (t[0]+t[1]+t[2] >= 1) {
			dz[nz] = distanceToTriangle(p, z[nz], 1, 2, 3, 3, 4, 5, -1, 3); nz++;
		}

		if (nz > 0)
		{
			std::size_t i_min = 0;
			for (std::size_t i = 1; i < nz; i++) {
				if (dz[i] < dz[i_min]) i_min = i;
			}

			x = z[i_min];
			return dz[i_min];
		}

		// inside
		
		T d[4];
		for (std::size_t i = 0; i < 4; i++) {
			d[i] = ublas::inner_prod(p - _p[i], _n[i]);
		}
		
		std::size_t i_max = 0;
		for (std::size_t i = 1; i < 4; i++) {
			if (d[i] > d[i_max]) i_max = i;
		}

		x = p - d[i_max]*_n[i_max];
		return d[i_max];
	}

	virtual T distanceTo(const Fiber<T, DIM>& fiber, ublas::c_vector<T, DIM>& x, ublas::c_vector<T, DIM>& xf) const
	{
		BOOST_THROW_EXCEPTION(std::runtime_error("TetrahedronFiber::distanceTo(fiber): not implemented!"));
		return 0;
	}

	virtual T distanceToPlane(const ublas::c_vector<T, DIM>& p, const ublas::c_vector<T, DIM>& n) const
	{
		BOOST_THROW_EXCEPTION(std::runtime_error("TetrahedronFiber::distanceToPlane(): not implemented!"));
		return 0;
	}

	virtual bool inside(const ublas::c_vector<T, DIM>& p) const
	{
		// determine barycentric coordinates
		// p = p0 + a0*t0 + a1*t1 + a2*t2

		ublas::c_vector<T, DIM> dp = p - _p[0];

		T b[3];
		T t[3];

		b[0] = ublas::inner_prod(dp, _a[0]);
		b[1] = ublas::inner_prod(dp, _a[1]);
		b[2] = ublas::inner_prod(dp, _a[2]);
		t[0] = _Minv(0,0)*b[0] + _Minv(0,1)*b[1] + _Minv(0,2)*b[2];
		t[1] = _Minv(1,0)*b[0] + _Minv(1,1)*b[1] + _Minv(1,2)*b[2];
		t[2] = _Minv(2,0)*b[0] + _Minv(2,1)*b[1] + _Minv(2,2)*b[2];

		return (t[0] >= 0) && (t[1] >= 0) && (t[2] >= 0) && (t[0]+t[1]+t[2] <= 1);
	}
	
	virtual boost::shared_ptr< Fiber<T, DIM> > clone() const
	{
		boost::shared_ptr< Fiber<T, DIM> > fiber(new TetrahedronFiber(_p));
		fiber->set_id(this->id());
		fiber->set_material(this->material());
		fiber->set_parent(const_cast< Fiber<T, DIM>* const >(reinterpret_cast<const Fiber<T, DIM>* const>(this)));
		return fiber;
	}
	
	inline virtual void translate(const ublas::c_vector<T, DIM>& dx)
	{
		_c    += dx;
		_p[0] += dx;
		_p[1] += dx;
		_p[2] += dx;
	}
	
	inline virtual T curvature() const
	{
		// TODO: the curvature at the edges is actually infinite and this causes trouble at integration for those cases
		return 0;
	}
	
	inline virtual T volume() const
	{
		return 0;
	}
	
	inline virtual const ublas::c_vector<T, DIM>& orientation() const
	{
		return _n[0];
	}

	inline virtual const ublas::c_vector<T, DIM>& center() const
	{
		return _c;
	}

	// bounding box interface methods
	
	inline virtual T bbRadius() const
	{
		return _B;
	}
	
	inline virtual const ublas::c_vector<T, DIM>& bbCenter() const
	{
		return _c;
	}
};


//! Triangle fiber (i.e. a single triangle)
template <typename T, int DIM>
class TriangleFiber : public Fiber<T, DIM>
{
protected:
	ublas::c_vector<T, DIM> _p[3];
	ublas::c_vector<T, DIM> _n;
	ublas::c_vector<T, DIM> _a[3];
	ublas::c_matrix<T, 3, 3> _M;
	ublas::c_matrix<T, 2, 2> _Minv;
	ublas::c_vector<T, DIM> _c;
	T _B;

public:

	// construct from points
	TriangleFiber(const ublas::c_vector<T, DIM> p[3])
	{
		_p[0] = p[0];
		_p[1] = p[1];
		_p[2] = p[2];
		
		_c = (_p[0] + _p[1] + _p[2])/3;
		
		_B = 0;
		for (std::size_t i = 0; i < 3; i++) {
			_B  = std::max(ublas::norm_2(_p[i] - _c), _B);
		}
		
		_a[0] = _p[1] - _p[0];
		_a[1] = _p[2] - _p[0];
		_a[2] = _p[1] - _p[2];
	
		ublas::c_vector<T, DIM> a01 = cross_prod(_a[1], _a[0]);
		T surf = ublas::norm_2(a01);

		// make sure triangle has positive surface
		if (std::abs(surf) < std::numeric_limits<T>::epsilon()*(ublas::norm_2(_a[0]) + ublas::norm_2(_a[1]))) {
			BOOST_THROW_EXCEPTION(std::runtime_error("collapsed triangle!"));
		}

		for (std::size_t i = 0; i < 3; i++) {
			for (std::size_t j = i; j < 3; j++) {
				_M(i,j) = _M(j, i) = ublas::inner_prod(_a[i], _a[j]);
			}
		}

		ublas::c_matrix<T, 2, 2> M;
		for (std::size_t i = 0; i < 2; i++) {
			for (std::size_t j = i; j < 2; j++) {
				M(i,j) = M(j, i) = _M(i, j);
			}
		}

#if 0
		for (std::size_t i = 0; i < 3; i++) {
			LOG_COUT << "trifiber point " << i << ": " << format(_p[i]) << std::endl;
		}
#endif

		InvertMatrix<T,2>(M, _Minv);

		_n = a01 / surf;
	}

	virtual void distanceGrad(const ublas::c_vector<T, DIM>& p, ublas::c_vector<T, DIM>& g) const
	{
		ublas::c_vector<T, DIM> dp = p - _p[0];

		// project p to plane
		T d = ublas::inner_prod(dp, _n);
		ublas::c_vector<T, DIM> x = p - d*_n;
		dp = x - _p[0];

		T b[2];
		T t[2];

		b[0] = ublas::inner_prod(dp, _a[0]);
		b[1] = ublas::inner_prod(dp, _a[1]);
		t[0] = _Minv(0,0)*b[0] + _Minv(0,1)*b[1];
		t[1] = _Minv(1,0)*b[0] + _Minv(1,1)*b[1];

		ublas::c_vector<T, DIM> z[2];
		std::size_t nz = 0;

		if (t[0] <= 0) {
			T t = ublas::inner_prod(dp, _a[1])/_M(1, 1);
			t = std::min(std::max(t, (T)0), (T)1);
			z[nz++] = _p[0] + t*_a[1];
		}
		if (t[1] <= 0) {
			T t = ublas::inner_prod(dp, _a[0])/_M(0, 0);
			t = std::min(std::max(t, (T)0), (T)1);
			z[nz++] = _p[0] + t*_a[0];
		}
		if (t[0]+t[1] >= 1) {
			T t = ublas::inner_prod(x - _p[2], _a[2])/_M(2, 2);
			t = std::min(std::max(t, (T)0), (T)1);
			z[nz++] = _p[2] + t*_a[2];
		}

		if (nz == 0) {
			// inside
			// g = (d < 0) ? -_n : _n;
			g = _n;
			return;
		}

		T norm_g[2];
		for (std::size_t i = 0; i < nz; i++) {
			norm_g[i] = ublas::norm_2(p - z[i]);
		}

		std::size_t i_min = 0;
		for (std::size_t i = 1; i < nz; i++) {
			if (norm_g[i] < norm_g[i_min]) i_min = i;
		}

		if (norm_g[i_min] == 0) {
			g = _n;
		}
		else {
			g = (p - z[i_min]) / norm_g[i_min];
		}
	}

	virtual T distanceTo(const ublas::c_vector<T, DIM>& p, ublas::c_vector<T, DIM>& x) const
	{
		ublas::c_vector<T, DIM> dp = p - _p[0];

		// project p to plane
		T d = ublas::inner_prod(dp, _n);
		x = p - d*_n;
		dp = x - _p[0];

		T b[2];
		T t[2];

		b[0] = ublas::inner_prod(dp, _a[0]);
		b[1] = ublas::inner_prod(dp, _a[1]);
		t[0] = _Minv(0,0)*b[0] + _Minv(0,1)*b[1];
		t[1] = _Minv(1,0)*b[0] + _Minv(1,1)*b[1];

		ublas::c_vector<T, DIM> z[2];
		std::size_t nz = 0;

		if (t[0] <= 0) {
			T t = ublas::inner_prod(dp, _a[1])/_M(1, 1);
			t = std::min(std::max(t, (T)0), (T)1);
			z[nz++] = _p[0] + t*_a[1];
		}
		if (t[1] <= 0) {
			T t = ublas::inner_prod(dp, _a[0])/_M(0, 0);
			t = std::min(std::max(t, (T)0), (T)1);
			z[nz++] = _p[0] + t*_a[0];
		}
		if (t[0]+t[1] >= 1) {
			T t = ublas::inner_prod(x - _p[2], _a[2])/_M(2, 2);
			t = std::min(std::max(t, (T)0), (T)1);
			z[nz++] = _p[2] + t*_a[2];
		}

		if (nz > 0)
		{
			T norm_g[2];
			for (std::size_t i = 0; i < nz; i++) {
				norm_g[i] = ublas::norm_2(p - z[i]);
			}

			std::size_t i_min = 0;
			for (std::size_t i = 1; i < nz; i++) {
				if (norm_g[i] < norm_g[i_min]) i_min = i;
			}

			x = z[i_min];
		}

		return ublas::norm_2(p - x);
	}


	virtual T distanceTo(const Fiber<T, DIM>& fiber, ublas::c_vector<T, DIM>& x, ublas::c_vector<T, DIM>& xf) const
	{
		BOOST_THROW_EXCEPTION(std::runtime_error("TriangleFiber::distanceTo(fiber): not implemented!"));
		return 0;
	}

	virtual T distanceToPlane(const ublas::c_vector<T, DIM>& p, const ublas::c_vector<T, DIM>& n) const
	{
		BOOST_THROW_EXCEPTION(std::runtime_error("TriangleFiber::distanceToPlane(): not implemented!"));
		return 0;
	}
	
	virtual bool inside(const ublas::c_vector<T, DIM>& p) const
	{
		BOOST_THROW_EXCEPTION(std::runtime_error("not implemented"));
		return false;
	}
	
	virtual boost::shared_ptr< Fiber<T, DIM> > clone() const
	{
		boost::shared_ptr< Fiber<T, DIM> > fiber(new TriangleFiber(_p));
		fiber->set_id(this->id());
		fiber->set_material(this->material());
		fiber->set_parent(const_cast< Fiber<T, DIM>* const >(reinterpret_cast<const Fiber<T, DIM>* const>(this)));
		return fiber;
	}
	
	inline virtual void translate(const ublas::c_vector<T, DIM>& dx)
	{
		_c    += dx;
		_p[0] += dx;
		_p[1] += dx;
		_p[2] += dx;
	}
	
	inline virtual T curvature() const
	{
		// TODO: the curvature at the edges is actually infinite and this causes trouble at integration for those cases
		return 0;
	}

	inline virtual T volume() const
	{
		return 0;
	}
	
	inline virtual const ublas::c_vector<T, DIM>& orientation() const
	{
		return _n;
	}

	inline virtual const ublas::c_vector<T, DIM>& center() const
	{
		return _c;
	}

	// bounding box interface methods
	
	inline virtual T bbRadius() const
	{
		return _B;
	}
	
	inline virtual const ublas::c_vector<T, DIM>& bbCenter() const
	{
		return _c;
	}
};


//! Base class for tetrahedron meshes
template <typename T, int DIM>
class TetFiberBase : public Fiber<T, DIM>
{
protected:
	boost::shared_ptr< FiberCluster<T, DIM> > _cluster;
	boost::shared_ptr< FiberCluster<T, DIM> > _surface_cluster;
	ublas::c_vector<T, DIM> _a;
	bool _fill_volume;

public:
	//! construct from points
	//! \param start first index in tets
	//! \param end last index in tets
	//! \param fill fill volume of tetrahedrons (otherwise only a surface mesh)
	//! \param points the vertices
	//! \param tets the tetrahedrons (4 points each)
	void init(std::size_t start, std::size_t end, bool fill,
		std::vector< ublas::c_vector<T,3> >& points,
		std::vector< ublas::c_vector<std::size_t,4> >& tets)
	{
		_fill_volume = fill;
		
		_a = ublas::zero_vector<T>(DIM);
		_a[0] = 1;

		std::vector< ublas::c_vector<std::size_t,3> > tris;
		std::vector< std::string > tri_ids;
		std::map< std::string, int > tri_use;
		
		// init tetrahedron cluster

		for (std::size_t i = start; i < std::min(tets.size(), end); i++)
		{
			ublas::c_vector<T, DIM> p[4];

			for (int k = 0; k < 4; k++) {
				for (int j = 0; j < DIM; j++) {
					p[k][j] = points[tets[i][k]][j];
				}
			}
			
			// make sure tetrahedron has positive volume
			ublas::c_vector<T, DIM> a0 = p[1] - p[0];
			ublas::c_vector<T, DIM> a1 = p[2] - p[0];
			ublas::c_vector<T, DIM> a2 = p[3] - p[0];
			T vol = ublas::inner_prod(cross_prod(a0, a1), a2);
			if (vol < 0) {
				// flip tets with negative volume
				std::swap(p[0], p[1]);
				std::swap(tets[i][0], tets[i][1]);
			}

			try
			{
				boost::shared_ptr< const Fiber<T, DIM> > fiber;
				fiber.reset(new TetrahedronFiber<T, DIM>(p));
				fiber->set_material(0);

				if (_cluster) {
					_cluster->add(fiber);
				}
				else {
					_cluster.reset(new FiberCluster<T, DIM>(fiber));
				}
			}
			catch(...) {
				// ignore tetrahedra of zero volume
				continue;
			}

			// create list of triangles
			ublas::c_vector<std::size_t,3> tri, tri0;
			std::string tri_id;

			#define TRI_PUSH(i0, i1, i2) \
				tri[0] = tets[i][i0]; tri[1] = tets[i][i1]; tri[2] = tets[i][i2]; tri0 = tri; \
				if (tri[0] > tri[1]) { std::swap(tri[0], tri[1]); } \
				if (tri[1] > tri[2]) { std::swap(tri[1], tri[2]); } \
				if (tri[0] > tri[1]) { std::swap(tri[0], tri[1]); } \
				if (tri[0] >= tri[1] || tri[1] >= tri[2]) throw "problem"; \
				tri_id = (boost::format("%d,%d,%d") % tri[0] % tri[1] % tri[2]).str(); \
				if (tri_use.count(tri_id) > 0) { \
					tri_use[tri_id]++; \
				} \
				else { \
					tri_use.insert(std::pair<std::string, int>(tri_id, 1)); \
					tris.push_back(tri0); \
					tri_ids.push_back(tri_id); \
				}
			TRI_PUSH(0, 1, 2);
			TRI_PUSH(0, 3, 1);
			TRI_PUSH(1, 3, 2);
			TRI_PUSH(0, 2, 3);
			#undef TRI_PUSH
		}

		// create surface triangle cluster
		for (std::size_t i = 0; i < tris.size(); i++)
		{
			ublas::c_vector<std::size_t,3>& tri = tris[i];
			std::string tri_id = tri_ids[i];

			if (tri_use[tri_id] >= 2) {
				continue;
			}
		
			ublas::c_vector<T, DIM> p[3];

			for (int k = 0; k < 3; k++) {
				for (int j = 0; j < DIM; j++) {
					p[k][j] = points[tri[k]][j];
				}
			}
			
			try
			{
				boost::shared_ptr< const Fiber<T, DIM> > fiber;
				fiber.reset(new TriangleFiber<T, DIM>(p));
				fiber->set_material(0);

				if (_surface_cluster) {
					_surface_cluster->add(fiber);
				}
				else {
					_surface_cluster.reset(new FiberCluster<T, DIM>(fiber));
				}
			}
			catch(...) {
				// ignore triangle of zero surface
			}
		}

		if (!_cluster || !_surface_cluster) {
			BOOST_THROW_EXCEPTION(std::runtime_error("TetVTKFiber file does not contain any valid geometry!"));
		}
	}

	virtual void distanceGrad(const ublas::c_vector<T, DIM>& p, ublas::c_vector<T, DIM>& g) const
	{
		boost::shared_ptr< const Fiber<T, DIM> > fiber;

		if (_fill_volume) {
			_surface_cluster->closestFiber(p, STD_INFINITY(T), -1, g, fiber);
		}
		else {
			_cluster->closestFiber(p, STD_INFINITY(T), -1, g, fiber);

		}
		fiber->distanceGrad(p, g);
	}

	virtual T distanceTo(const ublas::c_vector<T, DIM>& p, ublas::c_vector<T, DIM>& x) const
	{
		boost::shared_ptr< const Fiber<T, DIM> > fiber;
		T d;

#if 1
		if (_fill_volume) {
			d = _surface_cluster->closestFiber(p, STD_INFINITY(T), -1, x, fiber);
			if (_cluster->inside(p, -1)) {
				d = -d;
			}
		}
		else {
			d = _cluster->closestFiber(p, STD_INFINITY(T), -1, x, fiber);
		}
#else
		d = _cluster->closestFiber(p, STD_INFINITY(T), -1, x, fiber);

		if (_fill_volume && d <= 0) {
			fiber.reset();
			d = -_surface_cluster->closestFiber(p, STD_INFINITY(T), -1, x, fiber);
		}
#endif

		return d;
	}

	virtual T distanceTo(const Fiber<T, DIM>& fiber, ublas::c_vector<T, DIM>& x, ublas::c_vector<T, DIM>& xf) const
	{
		BOOST_THROW_EXCEPTION(std::runtime_error("TetVTKFiber::distanceTo(fiber): not implemented!"));
		return 0;
	}

	virtual T distanceToPlane(const ublas::c_vector<T, DIM>& p, const ublas::c_vector<T, DIM>& n) const
	{
		BOOST_THROW_EXCEPTION(std::runtime_error("TetVTKFiber::distanceToPlane(): not implemented!"));
		return 0;
	}
	
	virtual bool inside(const ublas::c_vector<T, DIM>& p) const
	{
		BOOST_THROW_EXCEPTION(std::runtime_error("not implemented"));
		return false;
	}
	
	virtual boost::shared_ptr< Fiber<T, DIM> > clone() const
	{
		BOOST_THROW_EXCEPTION(std::runtime_error("TetVTKFiber::clone(): not implemented!"));
		boost::shared_ptr< Fiber<T, DIM> > fiber;
		fiber->set_parent(const_cast< Fiber<T, DIM>* const >(reinterpret_cast<const Fiber<T, DIM>* const>(this)));
		return fiber;
	}
	
	inline virtual void translate(const ublas::c_vector<T, DIM>& dx)
	{
		BOOST_THROW_EXCEPTION(std::runtime_error("TetVTKFiber::translate(): not implemented!"));
	}
	
	inline virtual T curvature() const
	{
		// TODO: the curvature at the edges is actually infinite and this causes trouble at integration for those cases
		return 0;
	}

	inline virtual T volume() const
	{
		return 0;
	}
	
	inline virtual const ublas::c_vector<T, DIM>& orientation() const
	{
		return _a;
	}

	inline virtual const ublas::c_vector<T, DIM>& center() const
	{
		return this->bbCenter();
	}

	// bounding box interface methods
	
	inline virtual T bbRadius() const
	{
		return _cluster->bbRadius();
	}
	
	inline virtual const ublas::c_vector<T, DIM>& bbCenter() const
	{
		return _cluster->bbCenter();
	}
};


//! Tetrahedron meshes from ASCII VTK file
template <typename T, int DIM>
class TetVTKFiber : public TetFiberBase<T, DIM>
{
public:
	//! Constructor
	//! \param filename VTK filename
	//! \param start first index of tetrahedrons to use
	//! \param end last index of tetrahedrons to use
	//! \param fill fill volume of tetrahedrons (otherwise only a surface mesh)
	//! \param a rotation matrix for points
	//! \param t translation vector for points
	TetVTKFiber(const std::string& filename, std::size_t start, std::size_t end, bool fill,
		ublas::c_matrix<T,3,3> a, ublas::c_vector<T,3> t)
	{
		TetVTKReader<T> reader;
		std::vector< ublas::c_vector<T,3> > points;
		std::vector< ublas::c_vector<std::size_t,4> > tets;

		reader.read(filename, points, tets);

		for (std::size_t k = 0; k < points.size(); k++) {
			points[k] = ublas::prod(a, points[k]) + t;
		}

		this->init(start, end, fill, points, tets);
	}
};


//! Tetrahedron meshes from Dolfin XML file
template <typename T, int DIM>
class TetDolfinXMLFiber : public TetFiberBase<T, DIM>
{
public:
	//! Constructor
	//! \param filename Dolfin XML filename
	//! \param start first index of tetrahedrons to use
	//! \param end last index of tetrahedrons to use
	//! \param fill fill volume of tetrahedrons (otherwise only a surface mesh)
	//! \param a rotation matrix for points
	//! \param t translation vector for points
	TetDolfinXMLFiber(const std::string& filename, std::size_t start, std::size_t end, bool fill,
		ublas::c_matrix<T,3,3> a, ublas::c_vector<T,3> t)
	{
		TetDolfinXMLReader<T> reader;
		std::vector< ublas::c_vector<T,3> > points;
		std::vector< ublas::c_vector<std::size_t,4> > tets;

		reader.read(filename, points, tets);

		for (std::size_t k = 0; k < points.size(); k++) {
			points[k] = ublas::prod(a, points[k]) + t;
		}

		this->init(start, end, fill, points, tets);
	}
};


//! Tetrahedron meshes from STL file
template <typename T, int DIM>
class STLFiber : public Fiber<T, DIM>
{
protected:
	boost::shared_ptr< FiberCluster<T, DIM> > _cluster;
	ublas::c_vector<T, DIM> _a;
	bool _fill_volume;
public:

	//! Constructor
	//! \param filename STL filename
	//! \param start first index of tetrahedrons to use
	//! \param end last index of tetrahedrons to use
	//! \param fill fill volume of tetrahedrons (otherwise only a surface mesh)
	//! \param a rotation matrix for points
	//! \param t translation vector for points
	STLFiber(const std::string& filename, std::size_t start, std::size_t end, bool fill,
		ublas::c_matrix<T,3,3> a, ublas::c_vector<T,3> t)
	{
		_fill_volume = fill;

		_a = ublas::zero_vector<T>(DIM);
		_a[0] = 1;

		STLReader<T> reader;
		std::vector< typename STLReader<T>::Facet > facets;
		reader.read(filename, facets);
		
		for (std::size_t i = start; i < std::min(facets.size(), end); i++)
		{
			boost::shared_ptr< const Fiber<T, DIM> > fiber;
			ublas::c_vector<T, DIM> p[3];

			for (int k = 0; k < 3; k++) {
				for (int j = 0; j < DIM; j++) {
					p[k][j] = facets[i].v[k][j];
				}

				p[k] = ublas::prod(a, p[k]) + t;
			}

			// check normal
			if (ublas::inner_prod(cross_prod(p[2]-p[0], p[1]-p[0]), facets[i].n) < 0) {
				std::swap(p[1], p[2]); 
			}

			try
			{
				fiber.reset(new TriangleFiber<T, DIM>(p));
				fiber->set_material(0);

				if (_cluster) {
					_cluster->add(fiber);
				}
				else {
					_cluster.reset(new FiberCluster<T, DIM>(fiber));
				}
			}
			catch(...) {}
		}
	}

	virtual void distanceGrad(const ublas::c_vector<T, DIM>& p, ublas::c_vector<T, DIM>& g) const
	{
		boost::shared_ptr< const Fiber<T, DIM> > fiber;

		_cluster->closestFiber(p, STD_INFINITY(T), -1, g, fiber);
		fiber->distanceGrad(p, g);
	}

	virtual T distanceTo(const ublas::c_vector<T, DIM>& p, ublas::c_vector<T, DIM>& x) const
	{
		boost::shared_ptr< const Fiber<T, DIM> > fiber;

		T d = _cluster->closestFiber(p, STD_INFINITY(T), -1, x, fiber);

		if (_fill_volume) {
			return ((ublas::inner_prod(p - x, fiber->orientation()) < 0) ? -1.0 : 1.0)*d;
		}

		return d;
	}

	virtual T distanceTo(const Fiber<T, DIM>& fiber, ublas::c_vector<T, DIM>& x, ublas::c_vector<T, DIM>& xf) const
	{
		BOOST_THROW_EXCEPTION(std::runtime_error("STLFiber::distanceTo(fiber): not implemented!"));
		return 0;
	}

	virtual T distanceToPlane(const ublas::c_vector<T, DIM>& p, const ublas::c_vector<T, DIM>& n) const
	{
		BOOST_THROW_EXCEPTION(std::runtime_error("STLFiber::distanceToPlane(): not implemented!"));
		return 0;
	}
	
	virtual bool inside(const ublas::c_vector<T, DIM>& p) const
	{
		BOOST_THROW_EXCEPTION(std::runtime_error("not implemented"));
		return false;
	}
	
	virtual boost::shared_ptr< Fiber<T, DIM> > clone() const
	{
		BOOST_THROW_EXCEPTION(std::runtime_error("STLFiber::clone(): not implemented!"));
		//boost::shared_ptr< Fiber<T, DIM> > fiber(new STLFiber("", 0, 0, _fill_volume));
		//fiber->set_id(this->id());
		//fiber->set_material(this->material());
		//fiber->set_parent(const_cast< Fiber<T, DIM>* const >(reinterpret_cast<const Fiber<T, DIM>* const>(this)));
		//return fiber;
	}
	
	inline virtual void translate(const ublas::c_vector<T, DIM>& dx)
	{
		BOOST_THROW_EXCEPTION(std::runtime_error("STLFiber::translate(): not implemented!"));
	}
	
	inline virtual T curvature() const
	{
		// TODO: the curvature at the edges is actually infinite and this causes trouble at integration for those cases
		return 0;
	}

	inline virtual T volume() const
	{
		return 0;
	}
	
	inline virtual const ublas::c_vector<T, DIM>& orientation() const
	{
		return _a;
	}

	inline virtual const ublas::c_vector<T, DIM>& center() const
	{
		return this->bbCenter();
	}

	// bounding box interface methods
	
	inline virtual T bbRadius() const
	{
		return _cluster->bbRadius();
	}
	
	inline virtual const ublas::c_vector<T, DIM>& bbCenter() const
	{
		return _cluster->bbCenter();
	}
};


//! Single point fiber
template <typename T, int DIM>
class PointFiber : public Fiber<T, DIM>
{
protected:
	ublas::c_vector<T, DIM> _p;

public:

	//! Constructor
	//! \param p point
	PointFiber(const ublas::c_vector<T, DIM>& p)
	{
		_p  = p;
	}

	virtual void distanceGrad(const ublas::c_vector<T, DIM>& p, ublas::c_vector<T, DIM>& g) const
	{
		BOOST_THROW_EXCEPTION(std::runtime_error("not implemented"));
	}

	virtual T distanceTo(const ublas::c_vector<T, DIM>& p, ublas::c_vector<T, DIM>& x) const
	{
		x = _p;
		return ublas::norm_2(p - _p);
	}

	virtual T distanceTo(const Fiber<T, DIM>& fiber, ublas::c_vector<T, DIM>& x, ublas::c_vector<T, DIM>& xf) const
	{
#if 0
		const HalfSpaceFiber<T, DIM>* hs = dynamic_cast<const HalfSpaceFiber<T, DIM>*>(&fiber);

		if (hs != NULL) {
			return hs->
		}
#endif
		const PointFiber<T, DIM>* pf = dynamic_cast<const PointFiber<T, DIM>*>(&fiber);

		if (pf == NULL) {
			BOOST_THROW_EXCEPTION(std::runtime_error("PointFiber::distanceTo: can compute minimum distance to another PointFiber only!"));
		}

		x = _p;
		return pf->distanceTo(_p, xf);
	}

	virtual T distanceToPlane(const ublas::c_vector<T, DIM>& p, const ublas::c_vector<T, DIM>& n) const
	{
		T norm_n = ublas::norm_2(n);
		return ublas::inner_prod(_p - p, n) / norm_n;
	}
	
	virtual bool inside(const ublas::c_vector<T, DIM>& p) const
	{
		BOOST_THROW_EXCEPTION(std::runtime_error("not implemented"));
		return false;
	}
	
	virtual boost::shared_ptr< Fiber<T, DIM> > clone() const
	{
		boost::shared_ptr< Fiber<T, DIM> > fiber(new PointFiber(_p));
		fiber->set_id(this->id());
		fiber->set_material(this->material());
		fiber->set_parent(const_cast< Fiber<T, DIM>* const >(reinterpret_cast<const Fiber<T, DIM>* const>(this)));
		return fiber;
	}
	
	inline virtual void translate(const ublas::c_vector<T, DIM>& dx)
	{
		_p += dx;
	}
	
	inline virtual T curvature() const
	{
		// TODO: the curvature is actually infinite
		return 0;
	}

	inline virtual T volume() const
	{
		return 0;
	}
	
	inline virtual const ublas::c_vector<T, DIM>& center() const
	{
		return _p;
	}

	inline virtual const ublas::c_vector<T, DIM>& orientation() const
	{
		BOOST_THROW_EXCEPTION(std::runtime_error("not implemented"));
	}

	inline virtual T bbRadius() const
	{
		return 1e-9;
	}
	
	inline virtual const ublas::c_vector<T, DIM>& bbCenter() const
	{
		return _p;
	}

	virtual void writeData(std::ofstream& fs) const
	{
		fs << this->id() << "\t" << _p[0] << "\t" << _p[1] << "\t" << _p[2] << "\t" << (this->parent().get() == this ? 0 : 1) << std::endl;
	}
};



//! Cylindrical fiber with spherical end-caps
template <typename T, int DIM>
class CapsuleFiber : public Fiber<T, DIM>
{
protected:
	ublas::c_vector<T, DIM> _a;	// axis direction of cylinder ((c2 - c1) normed)
	ublas::c_vector<T, DIM> _r;	// vector orthogonal to _a with length _R
	ublas::c_vector<T, DIM> _c1;	// base point 1 of cylinder
	ublas::c_vector<T, DIM> _c;	// center of cylinder
	ublas::c_vector<T, DIM> _c2;	// base point 2 of cylinder
	T _R;	// radius of cylinder
	T _L0;  // total length
	T _L;   // length of cylinder
	T _B;   // ball radius of bounding box

public:

	//! Construct from center c, orientation a, length L and radius R
	CapsuleFiber(const ublas::c_vector<T, DIM>& c, const ublas::c_vector<T, DIM>& a, T L0, T R)
	{
		_L0  = std::abs(L0);
		_R  = std::abs(R);
		_L  = std::max(0.0, _L0 - (4.0/3.0)*_R);
		
		T norm_a = ublas::norm_2(a);
		if (norm_a != 0) {
			_a = a / norm_a;
		}
		else if (_L != 0) {
			BOOST_THROW_EXCEPTION(std::runtime_error("CapsuleFiber: given nonzero fiber length without orientation vector!"));
		}
		else {
			_a *= 0;
		}
		
		_c  = c;
		_c1 = _c - (_L/2)*_a;
		_c2 = _c + (_L/2)*_a;
		_B  = _L/2 + _R;
		_r = orthonormal_vector<T,DIM>(_a)*_R;
	}

	virtual void distanceGrad(const ublas::c_vector<T, DIM>& p, ublas::c_vector<T, DIM>& g) const
	{
		// calculate length of projection of p-c1 onto a
		T t = ublas::inner_prod(p - _c1, _a);
		
		// bound t to [0, L]
		t = std::min(std::max((T)0, t), _L);

		// compute distance
		g = p - _c1 - t*_a;
		T norm_g = ublas::norm_2(g);

		if (norm_g < std::sqrt(std::numeric_limits<T>::epsilon())) {
			g = ((t < 0.5*_L) ? -1 : 1) * _a;
		}
		else {
			g /= norm_g;
		}
	}

	virtual T distanceTo(const ublas::c_vector<T, DIM>& p, ublas::c_vector<T, DIM>& x) const
	{
		// calculate length of projection of p-c1 onto a
		T t = ublas::inner_prod(p - _c1, _a);
		
		// bound t to [0, L]
		t = std::min(std::max((T)0, t), _L);

		// compute minimum point on axis
		x = _c1 + t*_a;

		// compute distance
		T d = ublas::norm_2(p - x);

		if (d < std::numeric_limits<T>::epsilon()*_R) {	// TODO: use different criteria?
			// handle special case if point is on axis of cylinder
			//x = (t < 0.5*_L) ? ublas::c_vector<T, DIM>(_c1 - _R*_a) : ublas::c_vector<T, DIM>(_c2 + _R*_a);
			x += _r;
		}
		else {
			// translate minimum point to surface
			x += (p - x)*(_R / d);
		}

		// translate minimum point to surface and correct distance
		d -= _R;

		DEBP("p = " << format(p) << " c1 = " << format(_c1) << " a = " << format(_a) << " d = " << d);

#ifdef TEST_DIST_EVAL
		#pragma omp atomic
		g_dist_evals ++;
#endif

		// return distance
		return d;
	}

	virtual T distanceTo(const Fiber<T, DIM>& fiber, ublas::c_vector<T, DIM>& x, ublas::c_vector<T, DIM>& xf) const
	{
#if 0
		const HalfSpaceFiber<T, DIM>* hs = dynamic_cast<const HalfSpaceFiber<T, DIM>*>(&fiber);

		if (hs != NULL) {
			return hs->
		}
#endif
		const CapsuleFiber<T, DIM>* cf = dynamic_cast<const CapsuleFiber<T, DIM>*>(&fiber);

		if (cf == NULL) {
			BOOST_THROW_EXCEPTION(std::runtime_error("CapsuleFiber::distanceTo: can compute minimum distance to another CapsuleFiber only!"));
		}

		// the minimum distance between two smooth convex objects is characterized,
		// by the fact that both normals at the minimum points are parallel.

		// check if c1 is a minimum point

		T d1 = cf->distanceTo(_c1, xf);
		ublas::c_vector<T, DIM> dx1 = xf - _c1;

		if (ublas::inner_prod(dx1, _a) <= 0) {
			// c1 is a minimum point
			x = _c1 + dx1 * (_R / std::max(ublas::norm_2(dx1), std::numeric_limits<T>::epsilon()*_R));
			return (d1 - _R);
		}

		// check if c2 is a minimum point

		T d2 = cf->distanceTo(_c2, xf);
		ublas::c_vector<T, DIM> dx2 = xf - _c2;

		if (ublas::inner_prod(dx2, _a) >= 0) {
			// c2 is a minimum point
			x = _c2 + dx2 * (_R / std::max(ublas::norm_2(dx2), std::numeric_limits<T>::epsilon()*_R));
			return (d2 - _R);
		}

		// the minimum point is determined by
		// fiber->distanceTo(_c1 + t*_a, xf), (_c1 + t*_a - xf)*_a = 0
		// for t in [0,1]
		// which gives the equation for t
		// f1 - t + min(max(0, f2 + t*aa), cf->_L)*aa = 0

		T f1 = ublas::inner_prod(cf->_c1 - _c1, _a);
		T f2 = ublas::inner_prod(_c1 - cf->_c1, cf->_a);
		T aa = ublas::inner_prod(cf->_a, _a);
		T t;

		if (aa == 0) {
			t = f1;
		}
		else {
			// rewrite equation as
			// min(max((f1 - t)/aa, (f1 - t)/aa + f2 + t*aa), (f1 - t)/aa + cf->_L) = 0
			// this gives 3 choices for t

			// T t1 = f1;
			// T t2 = f1 + aa*cf->_L;
			// T t3 = (f1 + aa*f2) / (1 - aa*aa);

			// t1 is valid if
			// (f1 - t)/aa + f2 + t*aa <= 0 and (f1 - t)/aa + cf->_L >= 0
			// i.e. f2 + aa*f1 <= 0 and cf->_L >= 0

			T D = f2 + aa*f1;

			if (D <= 0) {
				t = f1;
			}

			// t2 is valid if
			// max(-cf->_L, -cf->_L + f2 + (f1 + aa*cf->_L)*aa) >= 0
			// i.e. f2 + aa*f1 >= cf->_L*(1 - aa*aa)

			else if (D >= cf->_L*(1 - aa*aa)) {
				t = f1 + aa*cf->_L;
				// this corresponds to "xf = cf->_c2"
			}

			// else it must be t3

			else {
				t = (f1 + aa*f2) / std::max(1 - aa*aa, std::numeric_limits<T>::epsilon());
			}
		}

		ublas::c_vector<T, DIM> p = _c1 + t*_a;
		T d = cf->distanceTo(p, xf);
		ublas::c_vector<T, DIM> dx = xf - p;
		x = p + dx * (_R / std::max(ublas::norm_2(dx), std::numeric_limits<T>::epsilon()*_R));
		return (d - _R);
	}

	virtual T distanceToPlane(const ublas::c_vector<T, DIM>& p, const ublas::c_vector<T, DIM>& n) const
	{
		T norm_n = ublas::norm_2(n);
		T dist_c1 = ublas::inner_prod(_c1 - p, n) / norm_n;
		T dist_c2 = ublas::inner_prod(_c2 - p, n) / norm_n;
		
		if (dist_c1*dist_c2 <= 0) return 0;
		if (std::fabs(dist_c1) <= _R) return 0;
		if (std::fabs(dist_c2) <= _R) return 0;
		
		if (dist_c1 < 0) {
			return std::max(dist_c1, dist_c2) + _R;
		}
		
		return std::min(dist_c1, dist_c2) - _R;
	}
	
	virtual bool inside(const ublas::c_vector<T, DIM>& p) const
	{
		BOOST_THROW_EXCEPTION(std::runtime_error("not implemented"));
		return false;
	}
	
	virtual boost::shared_ptr< Fiber<T, DIM> > clone() const
	{
		boost::shared_ptr< Fiber<T, DIM> > fiber(new CapsuleFiber(_c, _a, _L0, _R));
		fiber->set_id(this->id());
		fiber->set_material(this->material());
		fiber->set_parent(const_cast< Fiber<T, DIM>* const >(reinterpret_cast<const Fiber<T, DIM>* const>(this)));
		return fiber;
	}
	
	inline virtual void translate(const ublas::c_vector<T, DIM>& dx)
	{
		_c  += dx;
		_c1 += dx;
		_c2 += dx;
	}
	
	inline virtual T curvature() const
	{
		return 1/_R;
	}

	inline virtual T volume() const
	{
		if (DIM == 3) {
			return M_PI*_R*_R*(_L + 4.0/3.0*_R);
		}

		return (M_PI*_R*_R + 2*_R*_L);
	}
	
	inline virtual const ublas::c_vector<T, DIM>& orientation() const
	{
		return _a;
	}

	inline virtual const ublas::c_vector<T, DIM>& center() const
	{
		return _c;
	}

	inline virtual T length() const
	{
		return _L0;
	}
	
	inline virtual T radius() const
	{
		return _R;
	}
	
	// bounding box interface methods
	
	inline virtual T bbRadius() const
	{
		return _B;
	}
	
	inline virtual const ublas::c_vector<T, DIM>& bbCenter() const
	{
		return _c;
	}

	virtual void writeData(std::ofstream& fs) const
	{
		fs	<< this->id() << "\t"
			<< _c[0] << "\t" << _c[1] << "\t" << _c[2] << "\t" 
			<< _a[0] << "\t" << _a[1] << "\t" << _a[2] << "\t" 
			<< _R << "\t" << _L0 << "\t" << (this->parent().get() == this ? 0 : 1) << std::endl;
	}
};


#if 1
//! Half space fiber (points with negative distance to plane are inside)
template <typename T, int DIM>
class HalfSpaceFiber : public Fiber<T, DIM>
{
protected:
	ublas::c_vector<T, DIM> _p;	// point of plane
	ublas::c_vector<T, DIM> _n;	// normal of plane

public:

	//! Construct from point p, normal n
	HalfSpaceFiber(const ublas::c_vector<T, DIM>& p, const ublas::c_vector<T, DIM>& n)
	{
		T norm_n = ublas::norm_2(n);
		
		if (norm_n != 0) {
			_n = n / norm_n;
		}
		else {
			BOOST_THROW_EXCEPTION(std::runtime_error("HalfSpaceFiber: given zero normal vector!"));
		}
		
		_p  = p;
	}

	virtual void distanceGrad(const ublas::c_vector<T, DIM>& p, ublas::c_vector<T, DIM>& g) const
	{
		g = _n;
	}

	virtual T distanceTo(const ublas::c_vector<T, DIM>& p, ublas::c_vector<T, DIM>& x) const
	{
		T d = ublas::inner_prod(p - _p, _n);
		x = p - _n*d;
		return d;
	}

	virtual T distanceTo(const Fiber<T, DIM>& fiber, ublas::c_vector<T, DIM>& x, ublas::c_vector<T, DIM>& xf) const
	{
		BOOST_THROW_EXCEPTION(std::runtime_error("not implemented"));
		/*
		const HalfSpaceFiber<T, DIM>* hs = dynamic_cast<const HalfSpaceFiber<T, DIM>*>(&fiber);

		if (hs != NULL) {
			T D = ublas::inner_prod(_n, n);
		}
		*/

		return fiber.distanceTo(fiber, xf, x);
	}

	virtual T distanceToPlane(const ublas::c_vector<T, DIM>& p, const ublas::c_vector<T, DIM>& n) const
	{
		T D = ublas::inner_prod(_n, n);

		if (D == 1) {
			// planes are parallel
			return ublas::inner_prod(p - _p, _n);
		}
		
		return 0;
	}
	
	virtual bool inside(const ublas::c_vector<T, DIM>& p) const
	{
		BOOST_THROW_EXCEPTION(std::runtime_error("not implemented"));
		return false;
	}
	
	virtual boost::shared_ptr< Fiber<T, DIM> > clone() const
	{
		boost::shared_ptr< Fiber<T, DIM> > fiber(new HalfSpaceFiber(_p, _n));
		fiber->set_id(this->id());
		fiber->set_material(this->material());
		fiber->set_parent(const_cast< Fiber<T, DIM>* const >(reinterpret_cast<const Fiber<T, DIM>* const>(this)));
		return fiber;
	}
	
	virtual void translate(const ublas::c_vector<T, DIM>& dx)
	{
		_p  += dx;
	}
	
	inline virtual T curvature() const
	{
		return 0;
	}

	virtual T volume() const
	{
		return STD_INFINITY(T);
	}
	
	virtual const ublas::c_vector<T, DIM>& orientation() const
	{
		return _n;
	}

	// bounding box interface methods
	
	virtual T bbRadius() const
	{
		return STD_INFINITY(T);
	}
	
	virtual const ublas::c_vector<T, DIM>& bbCenter() const
	{
		return _p;
	}
};
#endif



//! Geometry writer for paraview Python code
template <typename T, int DIM>
class PVPyWriter
{
protected:
	std::string _filename;
	bool _bbox, _fibers, _clusters;
	std::ofstream _fs;

public:
	PVPyWriter(const std::string& filename, bool bbox = true, bool fibers = true, bool clusters = true) :
		_filename(filename), _bbox(bbox), _fibers(fibers), _clusters(clusters)
	{
		open_file(_fs, filename);
	}
	
	void writeCluster(const FiberCluster<T, DIM>& cluster)
	{
		const typename FiberCluster<T, DIM>::fiber_ptr_list& fibers = cluster.fibers();
		const typename FiberCluster<T, DIM>::cluster_ptr_list& clusters = cluster.clusters();

		for (typename FiberCluster<T, DIM>::fiber_ptr_list::const_iterator i = fibers.begin(); i != fibers.end(); i++) {
			writeFiber(**i);
		}

		for (typename FiberCluster<T, DIM>::cluster_ptr_list::const_iterator i = clusters.begin(); i != clusters.end(); i++) {
			writeCluster(**i);
		}

		writeSphere(cluster.bbCenter(), cluster.bbRadius());
	}

	void writeBox(const ublas::c_vector<T, DIM>& x0, const ublas::c_vector<T, DIM>& dim)
	{
		if (!_bbox) return;

		_fs << "box(" <<
			x0[0] << ", " << x0[1] << ", " << x0[2] << ", " <<
			dim[0] << ", " << dim[1] << ", " << dim[2] << ")\n";
	}

	void writeSphere(const ublas::c_vector<T, DIM>& x0, T R)
	{
		if (!_clusters) return;

		_fs << "sphere(" <<
			x0[0] << ", " << x0[1] << ", " << x0[2] << ", " <<
			R << ")\n";
	}

	void writeFiber(const Fiber<T, DIM>& fiber)
	{
		if (!_fibers) return;

		const CapsuleFiber<T, DIM>* cf = dynamic_cast<const CapsuleFiber<T, DIM>*>(&fiber);

		if (cf == NULL) {
			BOOST_THROW_EXCEPTION(std::runtime_error("CapsuleFiber::distanceTo: can compute minimum distance to another CapsuleFiber only!"));
		}

		const ublas::c_vector<T, DIM>& c = cf->center();
		const ublas::c_vector<T, DIM>& a = cf->orientation();

		_fs << "fiber(" <<
			c[0] << ", " << c[1] << ", " << c[2] << ", " <<
			a[0] << ", " << a[1] << ", " << a[2] << ", " <<
			cf->radius() << ", " << cf->length() << ")\n";
	}
};


//! Class for writing VTK files with structured cube meshes.
template <typename T>
class VTKCubeWriter
{
public:
	
	typedef struct {
		enum e { ASCII, BINARY };
	} WriteModes;
	
	typedef struct {
		enum e { NONE, SCALARS, VECTORS };
	} FieldTypes;
	
	typedef typename WriteModes::e WriteMode;
	typedef typename FieldTypes::e FieldType;

protected:
	std::size_t _nx, _ny, _nz;
	T _sx, _sy, _sz;
	T _x0, _y0, _z0;
	std::string _filename;
	std::size_t _iSlice;
	std::ofstream _fs;
	WriteMode _mode;
	FieldType _type;
	std::string _name;
	std::size_t _numFields;
	std::size_t _numComponents;
	boost::shared_ptr< ProgressBar<T> > _pb;

public:
	// Example VTK-file
	// note: 181440 = 20160*9
	/*
		# vtk DataFile Version 2.0
		fibergen
		ASCII
		DATASET UNSTRUCTURED_GRID
		FIELD FieldData 1
		TIME 1 1 double
		0.059
		POINTS 22995 float
		0.00166667 0.0015625 0.0015625
		...
		CELLS 20160 181440
		8 7 6 5 4 3 2 1 0
		8 ...
		CELL_TYPES 20160
		11
		...
		CELL_DATA 20160
		VECTORS fluidVelocity float
		0.31373 -0.134942 -0.134947
		...
		SCALARS fluidPressure float
		LOOKUP_TABLE default
		6611.42
		...
	*/

	VTKCubeWriter(const std::string& filename, WriteMode mode,
		std::size_t nx, std::size_t ny, std::size_t nz,
		T sx = 1, T sy = 1, T sz = 1,
		T x0 = 0, T y0 = 0, T z0 = 0
	) :
		_nx(nx), _ny(ny), _nz(nz), 
		_sx(sx), _sy(sy), _sz(sz), 
		_x0(x0), _y0(y0), _z0(z0),
		_filename(filename), _iSlice(0),
		_mode(mode), _type(FieldTypes::NONE), _numFields(0)
	{
		open_file(_fs, filename);
	}
	
	~VTKCubeWriter()
	{
		if (_mode == WriteModes::BINARY) {
			_fs << "\n";
		}

		// close file
		_fs.close();
	}

	void writeMesh()
	{
		std::size_t nCells = _nx*_ny*_nz;
		T sx = _sx/(T)_nx;
		T sy = _sy/(T)_ny;
		T sz = _sz/(T)_nz;

		// write header
		_fs << "# vtk DataFile Version 2.0\nfibergen\n";

		if (_mode == WriteModes::ASCII) {
			_fs << "ASCII\n";
		}
		else {
			_fs << "BINARY\n";
		}

#if 1
		// this is more efficient than writing the whole grid, however

		_fs << "DATASET STRUCTURED_POINTS\n";
#ifdef REVERSE_ORDER
		_fs << "DIMENSIONS " << (_nz+1) << " " << (_ny+1) << " " << (_nx+1) << "\n";
		_fs << "ORIGIN " << _z0 << " " << _y0 << " " << _x0 << "\n";
		_fs << "SPACING " << sz << " " << sy << " " << sx << "\n";
#else
		_fs << "DIMENSIONS " << (_nx+1) << " " << (_ny+1) << " " << (_nz+1) << "\n";
		_fs << "ORIGIN " << _x0 << " " << _y0 << " " << _z0 << "\n";
		_fs << "SPACING " << sx << " " << sy << " " << sz << "\n";
#endif

#else
		std::size_t nPoints = (_nx+1)*(_ny+1)*(_nz+1);

		// write header
		_fs << "DATASET UNSTRUCTURED_GRID\nFIELD FieldData 1\nTIME 1 1 double\n0\n";

		// compute number of points and cells
		
		// write points

		_fs << "POINTS " << nPoints << " float\n";

		if (_mode == WriteModes::ASCII)
		{
			for (std::size_t i = 0; i <= _nx; i++) {
				for (std::size_t j = 0; j <= _ny; j++) {
					for (std::size_t k = 0; k <= _nz; k++) {
						_fs << (float)(_x0 + i*sx) << " " << (float)(_y0 + j*sy) << " " << (float)(_z0 + k*sz) << "\n";
					}
				}
			}
		}
		else {
			BOOST_THROW_EXCEPTION(std::runtime_error("binary mode not supported"));
		}

		// write cells

		_fs << "CELLS " << nCells << " " << (nCells*9) << "\n";

		std::size_t mz = (_nz+1);
		std::size_t myz = (_ny+1)*mz;

		if (_mode == WriteModes::ASCII)
		{
			for (std::size_t i = 0; i < _nx; i++) {
				for (std::size_t j = 0; j < _ny; j++) {
					for (std::size_t k = 0; k < _nz; k++) {
						_fs << "8 "
							<< (i*myz + j*mz + k) << " "
							<< (i*myz + j*mz + k+1) << " " 
							<< (i*myz + (j+1)*mz + k) << " " 
							<< (i*myz + (j+1)*mz + k+1) << " " 
							<< ((i+1)*myz + j*mz + k) << " "
							<< ((i+1)*myz + j*mz + k+1) << " " 
							<< ((i+1)*myz + (j+1)*mz + k) << " "
							<< ((i+1)*myz + (j+1)*mz + k+1) << "\n";
					}
				}
			}
		}
		else {
			BOOST_THROW_EXCEPTION(std::runtime_error("binary mode not supported"));
		}
		
		_fs << "CELL_TYPES " << nCells << "\n";

		if (_mode == WriteModes::ASCII)
		{
			for (std::size_t i = 0; i < nCells; i++) {
				_fs << "11\n";
			}
		}
		else {
			BOOST_THROW_EXCEPTION(std::runtime_error("binary mode not supported"));
		}
#endif

		// begin data section
		_fs << "CELL_DATA " << nCells << "\n";
	}
	
	template <typename R>
	void beginWriteField(const std::string& name, FieldType type = FieldTypes::SCALARS)
	{
		if ((_mode == WriteModes::BINARY) && (_numFields > 0)) {
			_fs << "\n";
		}

		if (type == FieldTypes::SCALARS) {
			_fs << "SCALARS";
			_numComponents = 1;
		}
		else if (type == FieldTypes::VECTORS) {
			_fs << "VECTORS";
			_numComponents = 3;
		}
		else {
			BOOST_THROW_EXCEPTION(std::runtime_error("field type not supported"));
		}
		
		_type = type;
		_name = name;

		_fs << " " << name << " ";

		if (::boost::is_same<R, double>::value) {
			_fs << "double";
		}
		else if (::boost::is_same<R, float>::value) {
			_fs << "float";
		}
		else {
			BOOST_THROW_EXCEPTION(std::runtime_error("data type not supported"));
		}

		_fs << "\n";

		if (type == FieldTypes::SCALARS) {
			_fs << "LOOKUP_TABLE default\n";
		}

		_numFields++;
		_iSlice = 0;
		_pb.reset(new ProgressBar<T>());
	}

	inline void endian_swap(float& n)
	{
		BOOST_STATIC_ASSERT(sizeof(float) == 4);

		char *b = reinterpret_cast<char*>(&n);
		std::swap( b[0], b[3] );
		std::swap( b[1], b[2] );
	}

	inline void endian_swap(double& n)
	{
		BOOST_STATIC_ASSERT(sizeof(double) == 8);

		char *b = reinterpret_cast<char*>(&n);
		std::swap( b[0], b[7] );
		std::swap( b[1], b[6] );
		std::swap( b[2], b[5] );
		std::swap( b[3], b[4] );
	}

	void progessMessage(std::size_t nSlices)
	{
		_iSlice++;
		T percent = _iSlice*100/(T)nSlices;

		if (_pb->update(percent)) {
			_pb->message() << "saving " << _name << " field to file " << _filename << _pb->end();
		}
	}

	template <typename R, typename D>
	void writeZYSlice(D* data, std::size_t rowPadding = 0)
	{
		D* adata[1];
		adata[0] = data;
		this->writeZYSlice<R,D>(adata, rowPadding);
	}

	template <typename R, typename D>
	void writeZYSlice(D** data, std::size_t rowPadding = 0)
	{
		progessMessage(_ny);

		if (_mode == WriteModes::ASCII)
		{
			std::size_t kk = 0;
			for (std::size_t j = 0; j < _ny; j++) {
				for (std::size_t k = 0; k < _nz; k++) {
					for (std::size_t c = 0; c < _numComponents; c++) {
						if (c > 0) _fs << " ";
						_fs << ((R)data[c][kk]);
					}
					_fs << "\n";
					kk++;
				}
				kk += rowPadding;
			}
		}
		else
		{
			std::vector<R> buffer(_nz*_numComponents);

			std::size_t kk = 0;
			for (std::size_t j = 0; j < _ny; j++) {
				for (std::size_t k = 0; k < _nz; k++) {
					for (std::size_t c = 0; c < _numComponents; c++) {
						std::size_t w = k*_numComponents + c;
						buffer[w] = data[c][kk];
#ifdef BOOST_LITTLE_ENDIAN
						endian_swap(buffer[w]);
#endif
					}
					kk++;
				}
				_fs.write((const char*) &(buffer[0]), _nz*_numComponents*sizeof(R));
				kk += rowPadding;
			}
		}
	}

	template <typename R, typename D>
	void writeXYSlice(D* data, std::size_t xStride, std::size_t yStride)
	{
		D* adata[1];
		adata[0] = data;
		this->writeXYSlice<R,D>(adata, xStride, yStride);
	}

	template <typename R, typename D>
	void writeXYSlice(D** data, std::size_t xStride, std::size_t yStride)
	{
		progessMessage(_nz);

		if (_mode == WriteModes::ASCII)
		{
			for (std::size_t j = 0; j < _ny; j++) {
				for (std::size_t i = 0; i < _nx; i++) {
					for (std::size_t c = 0; c < _numComponents; c++) {
						if (c > 0) _fs << " ";
						_fs << ((R)data[c][i*xStride + j*yStride]);
					}
					_fs << "\n";
				}
			}
		}
		else
		{
			std::vector<R> buffer(_nx*_numComponents);

			for (std::size_t j = 0; j < _ny; j++) {
				for (std::size_t i = 0; i < _nx; i++) {
					for (std::size_t c = 0; c < _numComponents; c++) {
						std::size_t w = i*_numComponents + c;
						buffer[w] = (R)data[c][i*xStride + j*yStride];
#ifdef BOOST_LITTLE_ENDIAN
						endian_swap(buffer[w]);
#endif
					}
	
				}
				_fs.write((const char*) &(buffer[0]), _nx*_numComponents*sizeof(R));
			}
		}
	}
};


//! Class for generating random fiber distributions within a RVE.
template <typename T, int DIM>
class FiberGenerator
{
protected:
	std::size_t _N;		// number of fibers to generate
	std::size_t _M;		// maximum number of tries to reach N
	T _V;			// fiber volume fraction to generate
	std::size_t _seed;	// random seed
	std::size_t _mcs;	// maximum cluster size (number of fibers)
	T _L;				// fiber length
	T _R;				// fiber radius
	T _dmin;			// minimum distance between fibers
	T _dmax;			// maximum distance between fibers
	ublas::c_vector<T, 3> _dim;	// RVE dimensions
	ublas::c_vector<T, 3> _x0;	// RVE origin
	bool _intersecting;		// allow intersecting structures
	bool _periodic;			// create periodic structures
	bool _periodic_fast;	// use faster algorithm
	bool _periodic_x;
	bool _periodic_y;
	bool _periodic_z;
	bool _planar_x;
	bool _planar_y;
	bool _planar_z;

	std::size_t _material;	// current material id

	std::vector<T> _stats_v;	// volume fraction for each material
	std::size_t _stats_n;	// number of fibers
	std::size_t _stats_i;	// iterations

	std::string _type;	// fiber type

	// generated fiber cluster
	boost::shared_ptr< FiberCluster<T, DIM> > _cluster;
	
	// fiber distribution
	boost::shared_ptr< DiscreteDistribution<T, DIM> > _orientation_distribution;
	boost::shared_ptr< DiscreteDistribution<T, 1> > _length_distribution;
	boost::shared_ptr< DiscreteDistribution<T, 1> > _radius_distribution;

	// FO moments of the generated distribution	
	ublas::c_matrix<T, DIM, DIM> _A2;
	ublas::c_matrix<ublas::c_matrix<T, DIM, DIM>, DIM, DIM> _A4;

public:
	
	typedef struct {
		enum e { DISTANCE, ORIENTATION, NORMALS, FIBER_ID, MATERIAL_ID, FIBER_TRANSLATION };
	} SampleDataTypes;

	typedef typename SampleDataTypes::e SampleDataType;
	
	//! constructor
	FiberGenerator()
	{
		defaultSettings();
	}
	
	//! load default settings
	void defaultSettings()
	{
		_N = std::numeric_limits<std::size_t>::max();
		_V = STD_INFINITY(T);
		_M = 1000000;
		_seed = 0;
		_mcs = 8;
		_L = 0.1;
		_R = 0.01;
		_dmin = 0.0;
		_dmax = STD_INFINITY(T);
		set_vector(_dim, (T)1, (T)1, (T)1);
		set_vector(_x0, (T)0, (T)0, (T)0);
		_planar_x = false;
		_planar_y = false;
		_planar_z = false;
		_periodic = true;
		_periodic_x = true;
		_periodic_y = true;
		_periodic_z = true;
		_periodic_fast = false;
		_intersecting = false;
		_material = 0;
		_stats_v.resize(1);
		_stats_n = 0;
		_stats_i = 0;
		_type = "capsule";
		initMoments();
	}

	void initMoments()
	{
		_A2 = ublas::zero_matrix<T>(DIM);
		for (int i = 0; i < DIM; i++) {
			for (int j = 0; j < DIM; j++) {
				_A4(i,j) = ublas::zero_matrix<T>(DIM);
			}
		}
	}
	
	//! read settings from ptree
	void readSettings(const ptree::ptree& pt)
	{
		_N = pt_get<std::size_t>(pt, "n", _N);
		_V = pt_get<T>(pt, "v", _V);
		_M = pt_get<std::size_t>(pt, "m", _M);
		_seed = pt_get(pt, "seed", _seed);
		_mcs = pt_get(pt, "mcs", _mcs);
		_L = pt_get<T>(pt, "length", _L);
		_R = pt_get<T>(pt, "radius", _R);
		_dmin = pt_get<T>(pt, "dmin", _dmin);
		_dmax = pt_get<T>(pt, "dmax", _dmax);
		read_vector(pt, _dim, "dx", "dy", "dz", _dim(0), _dim(1), _dim(2));
		read_vector(pt, _x0, "x0", "y0", "z0", _x0(0), _x0(1), _x0(2));
		_periodic = pt_get(pt, "periodic", _periodic);
		//_periodic_fast = pt_get(pt, "periodic.<xmlattr>.fast", _periodic_fast);
		_planar_x = pt_get(pt, "planar.<xmlattr>.x", _planar_x);
		_planar_y = pt_get(pt, "planar.<xmlattr>.y", _planar_y);
		_planar_z = pt_get(pt, "planar.<xmlattr>.z", _planar_z);
		_periodic_x = pt_get(pt, "periodic.<xmlattr>.x", _periodic_x) && _periodic && !_planar_x;
		_periodic_y = pt_get(pt, "periodic.<xmlattr>.y", _periodic_y) && _periodic && !_planar_y;
		_periodic_z = pt_get(pt, "periodic.<xmlattr>.z", _periodic_z) && _periodic && !_planar_z;
		_intersecting = pt_get(pt, "intersecting", _intersecting);
		_type = pt_get<std::string>(pt, "type", _type);

		if (_periodic_fast && _dmin != 0) {
			BOOST_THROW_EXCEPTION(std::runtime_error("FiberGenerator: dmin != 0 is incompatible with periodic_fast=True"));
		}
	}
	
	inline T closestFiber(const ublas::c_vector<T, DIM>& p, T r, int mat, ublas::c_vector<T, DIM>& x, boost::shared_ptr< const Fiber<T, DIM> >& fiber) const
	{
		if (!_cluster) {
			BOOST_THROW_EXCEPTION(std::runtime_error("there are no fibers"));
		}

		return _cluster->closestFiber(p, r, mat, x, fiber);
	}

	inline void closestFibers(const ublas::c_vector<T, DIM>& p, T r, int mat, std::vector<typename FiberCluster<T, DIM>::ClosestFiberInfo>& info_list) const
	{
		if (!_cluster) return;
		return _cluster->closestFibers(p, r, mat, info_list);
	}

	// inline T dmin() const { return _intersecting ? -std::numeric_limits<T>::infinity() : _dmin; }
	inline T dmin() const { return _dmin; }
	inline T dmax() const { return _dmax; }

	void setLengthDistribution(const boost::shared_ptr< DiscreteDistribution<T, 1> >& dist)
	{
		_length_distribution = dist;
	}

	void setRadiusDistribution(const boost::shared_ptr< DiscreteDistribution<T, 1> >& dist)
	{
		_radius_distribution = dist;
	}

	void setOrientationDistribution(const boost::shared_ptr< DiscreteDistribution<T, DIM> >& dist)
	{
		_orientation_distribution = dist;
	}

	void selectMaterial(std::size_t id)
	{
		_material = id;
	}

	void addFiber(const boost::shared_ptr< const Fiber<T, DIM> >& fiber)
	{
		if (!_cluster) {
			_cluster.reset(new FiberCluster<T, DIM>(fiber, _mcs));
		}
		else {
			_cluster->add(fiber);
		}

		fiber->set_id(_cluster->fiberCount());
		fiber->set_material(_material);
		fiber->set_parent(const_cast<Fiber<T, DIM> *>(fiber.get()));

		_stats_v.resize(std::max(_stats_v.size(), _material+1));
		_stats_v[_material] += fiber->volume();
		_stats_n += 1;
		_stats_i += 1;

		updateMoments(fiber->orientation());
	}

	void updateMoments(const ublas::c_vector<T, DIM>& a)
	{
		ublas::c_vector<T, DIM> na = a/ublas::norm_2(a);
		ublas::c_matrix<T, DIM, DIM> aa = ublas::outer_prod(na, na);

		_A2 += aa;

		for (int i = 0; i < DIM; i++) {
			for (int j = 0; j < DIM; j++) {
				_A4(i,j) += na(i)*na(j)*aa;
			}
		}
	}

	//! Run the fiber generator
	noinline void run(T V = 0, std::size_t N = 0, std::size_t M = 0, T dmin = -STD_INFINITY(T), T dmax = STD_INFINITY(T), int intersecting = -1, int intersecting_materials = -1)
	{
		Timer __t("generating fiber distribution");

		#if 0
		// clear existing cluster
		_cluster.reset();
		// init FO moments
		initMoments();
		#endif

		FiberCluster<T, DIM>* cluster = _cluster.get();
		bool cluster_created = false;
		ublas::c_vector<T, DIM> x_i;
		ublas::c_vector<T, DIM> xf_i;
		boost::shared_ptr< const Fiber<T, DIM> > fiber;
		boost::shared_ptr< const Fiber<T, DIM> > fiber_i;
		T d_i;
		
		// init random number generator seeds
		RandomUniform01<T>::instance().seed(_seed);
		RandomNormal01<T>::instance().seed(_seed);

		// stuff for periodic fiber generation
		
		ublas::c_vector<T, DIM> zero;
		ublas::c_vector<T, DIM> one;
	
		ublas::c_vector<T, DIM> npx, nnx, tpx, tnx;
		ublas::c_vector<T, DIM> npy, nny, tpy, tny;
		ublas::c_vector<T, DIM> npz, nnz, tpz, tnz;
	
		set_vector(zero, _x0(0), _x0(1), _x0(2));
		set_vector(one, _x0(0)+_dim(0), _x0(1)+_dim(1), _x0(2)+_dim(2));

		set_vector(npx,  (T)1, (T)0, (T)0);
		set_vector(npy,  (T)0, (T)1, (T)0);
		set_vector(npz,  (T)0, (T)0, (T)1);
		set_vector(nnx, (T)-1, (T)0, (T)0);
		set_vector(nny,  (T)0,(T)-1, (T)0);
		set_vector(nnz,  (T)0, (T)0,(T)-1);

		set_vector(tpx,  _dim(0), (T)0, (T)0);
		set_vector(tpy,  (T)0, _dim(1), (T)0);
		set_vector(tpz,  (T)0, (T)0, _dim(2));
		set_vector(tnx, -_dim(0), (T)0, (T)0);
		set_vector(tny,  (T)0,-_dim(1), (T)0);
		set_vector(tnz,  (T)0, (T)0,-_dim(2));

		ublas::c_vector<T, DIM>* check_n[6] = {&npx, &nnx,  &npy, &nny,  &npz, &nnz };
		ublas::c_vector<T, DIM>* check_p[6] = {&one, &zero, &one, &zero, &one, &zero};
		ublas::c_vector<T, DIM>* check_t[6] = {&tnx, &tpx,  &tny, &tpy,  &tnz, &tpz };
		std::size_t inersects[6];
		std::size_t nintersects;
		ublas::c_vector<T, DIM> tr;
		std::vector< boost::shared_ptr< Fiber<T, DIM> > > clones;
		boost::shared_ptr< Fiber<T, DIM> > clone;

		// create non intersecting fiber distribution
		std::size_t i = 0;
		std::size_t n = 0;
		T progress;
		ProgressBar<T> pb;
		T V_RVE = _dim(0)*_dim(1)*((DIM > 2) ? _dim(2) : 1);
		T v = 0;

		if (V <= 0) V = _V;
		if (N <= 0) N = _N;
		if (M <= 0) M = _M;
		if (dmin < 0 && std::isinf(dmin)) dmin = _dmin;
		if (dmax > 0 && std::isinf(dmax)) dmax = _dmax;
		if (intersecting < 0) intersecting = _intersecting;

		// TODO: parallelize fiber generation
		for (;;)
		{
			// calculate and report progress
			progress = 0;
			if (!intersecting) progress = std::max(progress, i/(T)M);
			progress = std::max(progress, n/(T)N);
			progress = std::max(progress, v/(T)V);
			progress = std::min(progress, (T)1);
			progress *= 100;

			//LOG_COUT << "n/(T)N: " << (n/(T)N) << " v/(T)V): " << (v/(T)V) << " i/(T)M: " << (i/(T)M) << std::endl;

			if (pb.update(progress)) {
				pb.message() << "iterations = " << i << ", fibers = " << n << ", volume fraction = " << v << pb.end();
			}

			// check break conditions
			if (pb.complete() || _except) {
				break;
			}

			// create next random fiber
			fiber = randomFiber(n);
			fiber->set_id(_stats_n + n + 1);
			fiber->set_material(_material);
			fiber->set_parent(const_cast< Fiber<T,DIM>* >(fiber.get()));
			i++;

			if (cluster == NULL || (n == 0 && cluster_created)) {
				// init fiber cluster
				cluster = new FiberCluster<T, DIM>(fiber, _mcs);
				_cluster.reset(cluster);
				cluster_created = true;
			}
			else if (!intersecting) {
				bool valid = !cluster->intersects(*fiber, dmin, intersecting_materials, fiber_i, x_i, xf_i, d_i);
				//valid = valid && (d_i <= dmax);
				if (!valid) continue; // fiber intersects existing fiber
			}

			if (_periodic_x || _periodic_y || _periodic_z)
			{
				// add fiber to a periodic grid
				// we need to check if generated fiber violates periodicity in any kind
				// for every intersected sidewall of the 1-cube we need to check if there is an itersection with any other fiber
				// on the opposite side
				
				bool valid = true;
				clones.clear();
				
				if (!_periodic_fast)
				{
					for (int q = (_periodic_x ? -1 : 0); q <= (_periodic_x ? 1 : 0); q++) {
						tr(0) = q*_dim(0);
						for (int p = (_periodic_y ? -1 : 0); p <= (_periodic_y ? 1 : 0); p++) {
							tr(1) = p*_dim(1);
							for (int k = (_periodic_z ? -1 : 0); k <= (_periodic_z ? 1 : 0); k++) {
								tr(2) = k*_dim(2);
								if (q == 0 && p == 0 && k == 0) continue;
								clone = fiber->clone();
								clone->translate(tr);
								if (!intersecting) {
									valid = valid && !cluster->intersects(*clone, dmin, intersecting_materials, fiber_i, x_i, xf_i, d_i);
									//valid = valid && (d_i <= dmax);
									if (!valid) break;
									for (std::size_t h = 0; h < clones.size(); h++) {
										T d = clones[h]->distanceTo(*clone, x_i, xf_i);
										valid = valid && (d >= dmin);
										if (!valid) break;
									}
								}
								if (!valid) break;
								clones.push_back(clone);
							}
							if (!valid) break;
						}
						if (!valid) break;
					}
					
					if (!valid) continue;
				}
				else
				{
					// NOTE: this method does not ensure a correct far-field / periodicity for the distance map,
					// however it generates less fibers, and collision checking is less expensive
				
					nintersects = 0;

					for (std::size_t k = 0; k < (2*DIM); k++)
					{
						T dk = fiber->distanceToPlane(*check_p[k], *check_n[k]);

						if (dk == 0) {
							// intersects box
							clone = fiber->clone();
							clone->translate(*check_t[k]);
							if (!intersecting) {
								valid = valid && !cluster->intersects(*clone, dmin, intersecting_materials, fiber_i, x_i, xf_i, d_i);
								//valid = valid && (d_i <= dmax);
								if (!valid) break;
							}
							inersects[nintersects] = k;
							nintersects++;
							clones.push_back(clone);
						}
					}
				
					if (!valid) continue;
				
					if (nintersects >= 2) {
						// add clone fiber accross the diagonal of the intersection
						clone = fiber->clone();
						clone->translate(*check_t[inersects[0]] + *check_t[inersects[1]]);
						if (!intersecting) {
							valid = valid && !cluster->intersects(*clone, dmin, intersecting_materials, fiber_i, x_i, xf_i, d_i);
							//valid = valid && (d_i <= dmax);
							if (!valid) continue;
						}
						clones.push_back(clone);
					}
				
					if (nintersects == 3) {
						// add clone fibers accross all possible diagonals
						clone = fiber->clone();
						clone->translate(*check_t[inersects[0]] + *check_t[inersects[2]]);
						if (!intersecting) {
							valid = valid && !cluster->intersects(*clone, dmin, intersecting_materials, fiber_i, x_i, xf_i, d_i);
							//valid = valid && (d_i <= dmax);
							if (!valid) continue;
						}
						clones.push_back(clone);
						clone = fiber->clone();
						clone->translate(*check_t[inersects[1]] + *check_t[inersects[2]]);
						if (!intersecting) {
							valid = valid && !cluster->intersects(*clone, dmin, intersecting_materials, fiber_i, x_i, xf_i, d_i);
							//valid = valid && (d_i <= dmax);
							if (!valid) continue;
						}
						clones.push_back(clone);
						clone = fiber->clone();
						clone->translate(*check_t[inersects[0]] + *check_t[inersects[1]] + *check_t[inersects[2]]);
						if (!intersecting) {
							valid = valid && !cluster->intersects(*clone, dmin, intersecting_materials, fiber_i, x_i, xf_i, d_i);
							//valid = valid && (d_i <= dmax);
							if (!valid) continue;
						}
						clones.push_back(clone);
					}
					else if (nintersects == 3) {
						// security check
						BOOST_THROW_EXCEPTION(std::runtime_error("FiberGenerator: fiber intersects with more than 3 planes"));
					}
				}
				
				// add the clones to the cluster	
				for (std::size_t k = 0; k < clones.size(); k++) {
					cluster->add(clones[k]);
				}
			}

			// add fiber to cluster
			if (n > 0 || !cluster_created) cluster->add(fiber);
			n++;

			// update fiber volume
			v += fiber->volume()/V_RVE;

			// update FO moments
			updateMoments(fiber->orientation());
		}

		// LOG_COUT << "Number of fibers = " << n << std::endl;
		// LOG_COUT << "Fiber volume fraction = " << v << std::endl;

		_stats_v.resize(std::max(_stats_v.size(), _material+1));
		_stats_v[_material] += v;
		_stats_n += n;
		_stats_i += i;
	}

	double getVolumeFraction(std::size_t material_id) { return _stats_v[material_id]; }
	std::size_t getStatsN() { return _stats_n; }
	std::size_t getStatsI() { return _stats_i; }
	
	// generate a random fiber
	boost::shared_ptr< const Fiber<T, DIM> > randomFiber(std::size_t index)
	{
		ublas::c_vector<T, DIM> x;
		ublas::c_vector<T, DIM> a;
		boost::shared_ptr< const Fiber<T, DIM> > fiber;
		ublas::c_vector<T, 1> vec1;

		if (!_orientation_distribution) {
			//BOOST_THROW_EXCEPTION(std::runtime_error("FiberGenerator: no fiber distribution set"));
			_orientation_distribution.reset(new UniformDistribution<T, DIM>());
		}

		if (!_length_distribution) {
			vec1[0] = _L;
			_length_distribution.reset(new DiracDistribution<T, 1>(vec1));
		}

		if (!_radius_distribution) {
			vec1[0] = _R;
			_radius_distribution.reset(new DiracDistribution<T, 1>(vec1));
		}

		ublas::c_vector<T, DIM> npx, nnx;
		ublas::c_vector<T, DIM> npy, nny;
		ublas::c_vector<T, DIM> npz, nnz;
		ublas::c_vector<T, DIM> zero, one;
	
		set_vector(zero, _x0(0), _x0(1), _x0(2));
		set_vector(one, _x0(0)+_dim(0), _x0(1)+_dim(1), _x0(2)+_dim(2));

		set_vector(npx,  (T)1, (T)0, (T)0);
		set_vector(npy,  (T)0, (T)1, (T)0);
		set_vector(npz,  (T)0, (T)0, (T)1);
		set_vector(nnx, (T)-1, (T)0, (T)0);
		set_vector(nny,  (T)0,(T)-1, (T)0);
		set_vector(nnz,  (T)0, (T)0,(T)-1);

		ublas::c_vector<T, DIM>* check_n[6] = {&npx, &nnx,  &npy, &nny,  &npz, &nnz };
		ublas::c_vector<T, DIM>* check_p[6] = {&one, &zero, &one, &zero, &one, &zero};

		bool planar[3];
		planar[0] = _planar_x;
		planar[1] = _planar_y;
		planar[2] = _planar_z;

		for (;;)
		{
			// draw orientation vector a from distribution
			_orientation_distribution->drawSample(a, index);

			T norm_a = ublas::norm_2(a);
			if (norm_a == 0) {
				BOOST_THROW_EXCEPTION(std::runtime_error("randomFiber: orientation vector of length zero!"));
			}
			a /= norm_a;

			// draw length and radius
			_length_distribution->drawSample(vec1, index);
			T L = vec1[0];
			_radius_distribution->drawSample(vec1, index);
			T R = vec1[0];

			for (int i = 0; i < DIM; i++) {
				if (planar[i]) {
					x[i] = _x0(i) + 0.5*_dim(i);
				}
				else {
					T m = ((0.5*L + R)*std::abs(a[i]) + std::sqrt(1 - a[i]*a[i])*R)*1.001;
					x[i] = _x0(i) - m + (_dim(i) + 2*m)*RandomUniform01<T>::instance().rnd();
				}
			}

#if 0
			// move center to fiber boundary to create a more realistic distribution near the walls
			// TODO: what about the radial direction? However this makes 2d stuff more difficult
			T t = 2*RandomUniform01<T>::instance().rnd() - 1;
			x += a*(0.5*t*(L + 2*R));

			// NOTE: this was a bad idea, because it changes the distribution near the walls
			// flip sign of a such that angle with other vectors is always <= pi/2
			bool flip = false;
			for (int i = 0; i < DIM; i++) {
				if (a(i) < 0) {
					flip = true;
					break;
				}
				if (a(i) > 0) {
					break;
				}
			}
			if (flip) a = -a;
#endif

			if (_type == "capsule") {
				fiber.reset(new CapsuleFiber<T, DIM>(x, a, L, R));
			}
			else if (_type == "cylinder") {
				fiber.reset(new CylindricalFiber<T, DIM>(x, a, L, R));
			}
			//else if (_type == "halfspace") {
			//	fiber.reset(new HalfSpaceFiber<T, DIM>(x, a));
			//}
			else {
				BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown fiber type '%s'") % _type).str()));
			}

			// check if fiber is completely outside of RVE

			for (int i = 0; i < DIM; i++) {
				if (x(i) < _x0(i) || x(i) > (_x0(i) + _dim(i))) {
					// fiber center is outside of box
					bool intersects = false;
					for (std::size_t k = 0; k < (2*DIM); k++) {
						T dk = fiber->distanceToPlane(*check_p[k], *check_n[k]);
						if (dk == 0) {
							// intersects box
							intersects = true;
							break;
						}
					}
					if (!intersects) {
						// fiber is completely outside of box
						fiber.reset();
					}
					break;
				}
			}

			if (fiber) {
				return fiber;
			}
		}

		return fiber;
	}

	T trace(const ublas::c_matrix<T,DIM,DIM>& A) const
	{
		T trace = 0;
		for (int i = 0; i < DIM; i++) {
			trace += A(i,i);
		}
		return trace;
	}

	// return A2
	ublas::c_matrix<T,DIM,DIM> getA2() const
	{
		return _A2 / trace(_A2);
	}

	//! Return 4-th order orientation moment tensor
	ublas::c_matrix<ublas::c_matrix<T,DIM,DIM>,DIM,DIM> getA4() const
	{
		ublas::c_matrix<ublas::c_matrix<T,DIM,DIM>,DIM,DIM> A4 = _A4;
		ublas::c_matrix<T,DIM,DIM> A2 = ublas::zero_matrix<T>(DIM);

		for (int i = 0; i < DIM; i++) {
			A2 += A4(i,i);
		}

		T scale = 1/trace(A2);
		
		for (int i = 0; i < DIM; i++) {
			for (int j = 0; j < DIM; j++) {
				A4(i,j) *= scale;
			}
		}

		return A4;
	}

	template< typename R >
	void sampleXYSlice(std::size_t z, std::size_t nx, std::size_t ny, std::size_t nz, R* data, int mat = -1, SampleDataType type = SampleDataTypes::DISTANCE,
		std::size_t rowPadding = 0, bool fast = false, bool split = false) const
	{
		ublas::c_vector<T, DIM> a0;
		ublas::c_vector<T, DIM> a1;
		ublas::c_vector<T, DIM> a2;
		
		set_vector(a0, _x0(0), _x0(1), _x0(2) + (z + (T)0.5)*_dim(2)/nz);
		set_vector(a1, _dim(0), (T)0, (T)0);
		set_vector(a2, (T)0, _dim(1), (T)0);
		
		sampleSlice(a0, a1, a2, nx, ny, data, mat, type, rowPadding, fast, split);
	}
	
	template< typename R >
	void sampleZYSlice(std::size_t x, std::size_t nx, std::size_t ny, std::size_t nz, R* data, int mat = -1, SampleDataType type = SampleDataTypes::DISTANCE,
		std::size_t rowPadding = 0, bool fast = false, bool split = false) const
	{
		ublas::c_vector<T, DIM> a0;
		ublas::c_vector<T, DIM> a1;
		ublas::c_vector<T, DIM> a2;
		
		set_vector(a0, _x0(0) + (x + (T)0.5)*_dim(0)/nx, _x0(1), _x0(2));
		set_vector(a1, (T)0, (T)0, _dim(2));
		set_vector(a2, (T)0, _dim(1), (T)0);
		
		sampleSlice(a0, a1, a2, nz, ny, data, mat, type, rowPadding, fast, split);
	}
	
	//! Sample a distance map, the pixel values are calulated as following:
	//! p[i,j] = min_distance_to_fiber(a0 + a1*i/h + a2*j/w)
	//! i=0...h-1, j=0..w-1
	template< typename R >
	void sampleSlice(
		const ublas::c_vector<T, DIM>& a0, const ublas::c_vector<T, DIM>& a1, const ublas::c_vector<T, DIM>& a2,
		std::size_t n1, std::size_t n2, R* data, int mat = -1, SampleDataType type = SampleDataTypes::DISTANCE,
		std::size_t rowPadding = 0, bool fast = false, bool split = false) const
	{
		ublas::c_vector<T, DIM> p;
		ublas::c_vector<T, DIM> p0;
		ublas::c_vector<T, DIM> x;
		std::vector<boost::shared_ptr< const Fiber<T, DIM> > > fiber(n1);
		ublas::c_vector<T, DIM> dx(a1/n1);
		ublas::c_vector<T, DIM> dy(a2/n2);
		std::size_t rowSize = n1 + rowPadding;
		std::size_t dimStride = 1;
		T d;
		
		p0 = a0 + 0.5*dx + 0.5*dy;
		
		if (!_cluster) {
			BOOST_THROW_EXCEPTION(std::runtime_error("there are no fibers for sampling"));
		}

		// switch to accurate method for orientation
		if (type == SampleDataTypes::ORIENTATION || type == SampleDataTypes::NORMALS || type == SampleDataTypes::FIBER_TRANSLATION) {
			fast = false;
			if (!split) {
				rowSize += 2*n1;
			}
			else {
				dimStride = rowSize*n2;
			}
		}

		if (fast)
		{
			// fast method for distance sampling

			if (type == SampleDataTypes::DISTANCE)
			{
				#pragma omp parallel for private(p, x, d) firstprivate(fiber) schedule (static)
				for (std::size_t i = 0; i < n2; i++) {
					R* row = data + i*rowSize;
					p = p0 + dy*i;
					for (std::size_t j = 0; j < n1; j++) {
						*row = _cluster->intersects(p, 0, mat, fiber[j], x, d) ? (R)d : STD_INFINITY(R);
						row++;
						p += dx;
					}
				}
			}
			else if (type == SampleDataTypes::FIBER_ID)
			{
				#pragma omp parallel for private(p, x, d) firstprivate(fiber) schedule (static)
				for (std::size_t i = 0; i < n2; i++) {
					R* row = data + i*rowSize;
					p = p0 + dy*i;
					for (std::size_t j = 0; j < n1; j++) {
						*row = _cluster->intersects(p, 0, mat, fiber[j], x, d) ? (R)fiber[j]->id() : (R)-1;
						row++;
						p += dx;
					}
				}
			}
			else if (type == SampleDataTypes::MATERIAL_ID)
			{
				#pragma omp parallel for private(p, x, d) firstprivate(fiber) schedule (static)
				for (std::size_t i = 0; i < n2; i++) {
					R* row = data + i*rowSize;
					p = p0 + dy*i;
					for (std::size_t j = 0; j < n1; j++) {
						*row = _cluster->intersects(p, 0, mat, fiber[j], x, d) ? (R)fiber[j]->material() : (R)-1;
						row++;
						p += dx;
					}
				}
			}
		}
		else
		{
			// accurate method

			if (type == SampleDataTypes::DISTANCE)
			{
				#pragma omp parallel for private(p, x, d) firstprivate(fiber) schedule (static)
				for (std::size_t i = 0; i < n2; i++) {
					R* row = data + i*rowSize;
					p = p0 + dy*i;
					for (std::size_t j = 0; j < n1; j++) {
						d = _cluster->closestFiber(p, STD_INFINITY(T), mat, x, fiber[j]);
						*row = d;
						row++;
						p += dx;
					}
				}
			}
			else if (type == SampleDataTypes::MATERIAL_ID)
			{
				#pragma omp parallel for private(p, x, d) firstprivate(fiber) schedule (static)
				for (std::size_t i = 0; i < n2; i++) {
					R* row = data + i*rowSize;
					p = p0 + dy*i;
					for (std::size_t j = 0; j < n1; j++) {
						d = _cluster->closestFiber(p, STD_INFINITY(T), mat, x, fiber[j]);
						*row = (d <= 0) ? (R)fiber[j]->material() : (R)-1;
						row++;
						p += dx;
					}
				}
			}
			else if (type == SampleDataTypes::FIBER_ID)
			{
				#pragma omp parallel for private(p, x, d) firstprivate(fiber) schedule (static)
				for (std::size_t i = 0; i < n2; i++) {
					R* row = data + i*rowSize;
					p = p0 + dy*i;
					for (std::size_t j = 0; j < n1; j++) {
						_cluster->closestFiber(p, STD_INFINITY(T), mat, x, fiber[j]);
						*row = (R)fiber[j]->id();
						row++;
						p += dx;
					}
				}
			}
			else if (type == SampleDataTypes::FIBER_TRANSLATION)
			{
				#pragma omp parallel for private(p, x, d) firstprivate(fiber) schedule (static)
				for (std::size_t i = 0; i < n2; i++) {
					R* row = data + i*rowSize;
					p = p0 + dy*i;
					for (std::size_t j = 0; j < n1; j++) {
						_cluster->closestFiber(p, STD_INFINITY(T), mat, x, fiber[j]);
						ublas::vector<T> translation = fiber[j]->bbCenter() - fiber[j]->parent()->bbCenter();
						for (int k = 0; k < DIM; k++) {
							row[k*dimStride] = translation(k);
						}
						for (int k = DIM; k < 3; k++) {
							row[k*dimStride] = 0;
						}
						row += split ? 1 : 3;
						p += dx;
					}
				}
			}
			else if (type == SampleDataTypes::ORIENTATION)
			{
				#pragma omp parallel for private(p, x, d) firstprivate(fiber) schedule (static)
				for (std::size_t i = 0; i < n2; i++) {
					R* row = data + i*rowSize;
					p = p0 + dy*i;
					for (std::size_t j = 0; j < n1; j++) {
						_cluster->closestFiber(p, STD_INFINITY(T), mat, x, fiber[j]);
						const ublas::c_vector<T, DIM>& orientation = fiber[j]->orientation();
						for (int k = 0; k < DIM; k++) {
							row[k*dimStride] = orientation(k);
						}
						for (int k = DIM; k < 3; k++) {
							row[k*dimStride] = 0;
						}
						row += split ? 1 : 3;
						p += dx;
					}
				}
			}
			else if (type == SampleDataTypes::NORMALS)
			{
				#pragma omp parallel for private(p, x, d) firstprivate(fiber) schedule (static)
				for (std::size_t i = 0; i < n2; i++) {
					R* row = data + i*rowSize;
					p = p0 + dy*i;
					for (std::size_t j = 0; j < n1; j++) {
						_cluster->closestFiber(p, STD_INFINITY(T), mat, x, fiber[j]);
						ublas::c_vector<T, DIM> normal;
						fiber[j]->distanceGrad(p, normal);
						for (int k = 0; k < DIM; k++) {
							row[k*dimStride] = normal(k);
						}
						for (int k = DIM; k < 3; k++) {
							row[k*dimStride] = 0;
						}
						row += split ? 1 : 3;
						p += dx;
					}
				}
			}
		}
	}
		
	//! Write a ParaView Python script
	void writePVPy(const std::string& filename, bool bbox = true, bool fibers = true, bool clusters = true) const
	{
		PVPyWriter<T, DIM> pw(filename, bbox, fibers, clusters);
		pw.writeBox(_x0, _dim);
		if (_cluster) pw.writeCluster(*_cluster);
	}
	
	void writeData(const std::string& filename) const
	{
		std::ofstream fs;
		open_file(fs, filename);
		if (!_cluster) return;
		_cluster->writeData(fs);
	}

	//! Write a VTK distance map
	template< typename R >
	void writeVTK(const std::string& filename,
		std::size_t nx, std::size_t ny, std::size_t nz,
		bool fast, bool distance, bool normals, bool orientation, bool fiber_id, bool material_id,
		int mat = -1, bool binary = true) const
	{
		VTKCubeWriter<R> cw(filename, binary ? VTKCubeWriter<R>::WriteModes::BINARY : VTKCubeWriter<R>::WriteModes::ASCII,
			nx, ny, nz, _dim(0), _dim(1), _dim(2), _x0[0], _x0[1], _x0[2]);
	
		cw.writeMesh();

		if (distance)
		{
			cw.template beginWriteField<R>("Distance");

#ifdef REVERSE_ORDER
			std::vector<R> data(ny*nz);
			R* pdata = &(data[0]);

			for (std::size_t i = 0; i < nx; i++) {
				sampleZYSlice(i, nx, ny, nz, pdata, mat, SampleDataTypes::DISTANCE, 0, fast);
				cw.template writeZYSlice<R>(pdata);
			}
#else
			std::vector<R> data(nx*ny);
			R* pdata = &(data[0]);

			for (std::size_t i = 0; i < nz; i++) {
				sampleXYSlice(i, nx, ny, nz, pdata, mat, SampleDataTypes::DISTANCE, 0, fast);
				cw.template writeXYSlice<R>(pdata, 1, nx);
			}
#endif
		}

		if (normals)
		{
			cw.template beginWriteField<R>("Normals", VTKCubeWriter<R>::FieldTypes::VECTORS);

#ifdef REVERSE_ORDER
			std::vector<R> data(ny*nz*3);
			R* pdata = &(data[0]);
			R* pdata_vec[3];
			pdata_vec[0] = pdata + 0*ny*nz;
			pdata_vec[1] = pdata + 1*ny*nz;
			pdata_vec[2] = pdata + 2*ny*nz;

			for (std::size_t i = 0; i < nx; i++) {
				sampleZYSlice(i, nx, ny, nz, pdata, mat, SampleDataTypes::NORMALS, 0, fast, true);
				cw.template writeZYSlice<R>(pdata_vec);
			}
#else
			std::vector<R> data(nx*ny*3);
			R* pdata = &(data[0]);
			R* pdata_vec[3];
			pdata_vec[0] = pdata + 0*nx*ny;
			pdata_vec[1] = pdata + 1*nx*ny;
			pdata_vec[2] = pdata + 2*nx*ny;

			for (std::size_t i = 0; i < nz; i++) {
				sampleXYSlice(i, nx, ny, nz, pdata, mat, SampleDataTypes::NORMALS, 0, fast, true);
				cw.template writeXYSlice<R>(pdata_vec, 1, nx);
			}
#endif
		}

		if (orientation)
		{
			cw.template beginWriteField<R>("Orientation", VTKCubeWriter<R>::FieldTypes::VECTORS);

#ifdef REVERSE_ORDER
			std::vector<R> data(ny*nz*3);
			R* pdata = &(data[0]);
			R* pdata_vec[3];
			pdata_vec[0] = pdata + 0*ny*nz;
			pdata_vec[1] = pdata + 1*ny*nz;
			pdata_vec[2] = pdata + 2*ny*nz;

			for (std::size_t i = 0; i < nx; i++) {
				sampleZYSlice(i, nx, ny, nz, pdata, mat, SampleDataTypes::ORIENTATION, 0, fast, true);
				cw.template writeZYSlice<R>(pdata_vec);
			}
#else
			std::vector<R> data(nx*ny*3);
			R* pdata = &(data[0]);
			R* pdata_vec[3];
			pdata_vec[0] = pdata + 0*nx*ny;
			pdata_vec[1] = pdata + 1*nx*ny;
			pdata_vec[2] = pdata + 2*nx*ny;

			for (std::size_t i = 0; i < nz; i++) {
				sampleXYSlice(i, nx, ny, nz, pdata, mat, SampleDataTypes::ORIENTATION, 0, fast, true);
				cw.template writeXYSlice<R>(pdata_vec, 1, nx);
			}
#endif
		}

		if (material_id)
		{
			cw.template beginWriteField<R>("MaterialID");

#ifdef REVERSE_ORDER
			std::vector<R> data(ny*nz);
			R* pdata = &(data[0]);

			for (std::size_t i = 0; i < nx; i++) {
				sampleZYSlice(i, nx, ny, nz, pdata, -1, SampleDataTypes::MATERIAL_ID, 0, fast);
				cw.template writeZYSlice<R>(pdata);
			}
#else
			std::vector<R> data(nx*ny);
			R* pdata = &(data[0]);

			for (std::size_t i = 0; i < nz; i++) {
				sampleXYSlice(i, nx, ny, nz, pdata, -1, SampleDataTypes::MATERIAL_ID, 0, fast);
				cw.template writeXYSlice<R>(pdata, 1, nx);
			}
#endif
		}

		if (fiber_id)
		{
			cw.template beginWriteField<R>("FiberID");

#ifdef REVERSE_ORDER
			std::vector<R> data(ny*nz);
			R* pdata = &(data[0]);

			for (std::size_t i = 0; i < nx; i++) {
				sampleZYSlice(i, nx, ny, nz, pdata, mat, SampleDataTypes::FIBER_ID, 0, fast);
				cw.template writeZYSlice<R>(pdata);
			}
#else
			std::vector<R> data(nx*ny);
			R* pdata = &(data[0]);

			for (std::size_t i = 0; i < nz; i++) {
				sampleXYSlice(i, nx, ny, nz, pdata, mat, SampleDataTypes::FIBER_ID, 0, fast);
				cw.template writeXYSlice<R>(pdata, 1, nx);
			}
#endif
		}
	}

	//! Sample a distance map, the pixel values are calulated as following:
	//! p[i,j] = min(pow(max(min_distance_to_fiber(a0 + a1*i/h + a2*j/w), 0), exponent)*255*scale, 255)
	//! i=0...h-1, j=0..w-1
	//! the image is written as png to filename
	void writeDistanceMap(const std::string& filename,
		const ublas::c_vector<T, DIM>& a0, const ublas::c_vector<T, DIM>& a1, const ublas::c_vector<T, DIM>& a2,
		std::size_t h, std::size_t w, T offset, T scale, T exponent, bool fast, int mat = -1) const
	{
		// create image of size w x h
		gil::gray8_image_t img(w, h);
		const gil::gray8_view_t& img_view = gil::view(img);

		scale *= 255;

#ifdef TEST_DIST_EVAL
		g_dist_evals = 0;
#endif

		std::vector<T> data(w*h);
		T* pdata = &(data[0]);
		ProgressBar<T> pb;

		sampleSlice(a0, a1, a2, w, h, pdata, mat, SampleDataTypes::DISTANCE, 0, fast);
		
		for (std::size_t i = 0; i < h; i++)
		{
			gil::gray8_view_t::x_iterator row = img_view.row_begin(i);

			for (std::size_t j = 0; j < w; j++) {
				row[j] = (int)std::min(std::pow(std::max(*pdata + offset, (T)0), exponent)*scale, (T)255);
				pdata++;
			}

			// FIXME: does a progress bar make sense here?	
			std::size_t p = (((i+1)*100)/h);
			if (pb.update(p)) {
				pb.message() << "saving file " << filename << pb.end();
			}
		}
		
#ifdef TEST_DIST_EVAL
		LOG_COUT << g_dist_evals << " distance evaluations (" << (g_dist_evals/((double)w*h)) << "/pixel)" << std::endl;
#endif

		// write png to filename
#if BOOST_VERSION >= 106800
		// TODO: does this work
		// gil::write_view(filename, gil::const_view(img));
		LOG_CWARN << "Writing PNG distance map currently not implemented." << std::endl;	
#else
		gil::png_write_view(filename, gil::const_view(img));
#endif
	}
};


//! Base class for 3-dimensional FFTs
template<typename T>
class FFT3Base
{
protected:
	std::size_t _howmany;
	std::size_t _nx, _ny, _nz;
	unsigned _flags;

	void copy_inplace(const void* x, void* y)
	{
		if (x == y) return;

		// TODO: use parallel memcpy?
		memcpy(y, x, _howmany*_nx*_ny*2*(_nz/2+1)*sizeof(T));
	}

public:
	FFT3Base(std::size_t howmany, std::size_t nx, std::size_t ny, std::size_t nz, const std::string& planner_flag) : _howmany(howmany), _nx(nx), _ny(ny), _nz(nz)
	{
		if (planner_flag == "estimate") {
			_flags = FFTW_ESTIMATE;
		}
		else if (planner_flag == "measure") {
			_flags = FFTW_MEASURE;
		}
		else if (planner_flag == "patient") {
			_flags = FFTW_PATIENT;
		}
		else if (planner_flag == "exhaustive") {
			_flags = FFTW_EXHAUSTIVE;
		}
		else if (planner_flag == "wisdom_only") {
			_flags = FFTW_WISDOM_ONLY;
		}
		else {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown FFT planner flag '%s'") % planner_flag).str()));
		}
	}
};


//! Abstract class for performing FFT 
template<typename T>
class FFT3 : public FFT3Base<T>
{
public:
	FFT3(std::size_t nx, std::size_t ny, std::size_t nz, T* data, const std::string& planner_flag);

	//! Perform forward transformation
	void forward(const T* x, std::complex<T>* y);

	//! Perform backward transformation
	void backward(const std::complex<T>* x, T* y);
};


//! FFT specialization for double
template<>
class FFT3<double> : public FFT3Base<double>
{
protected:
	fftw_plan _fplan, _bplan;

public:
	FFT3(std::size_t howmany, std::size_t nx, std::size_t ny, std::size_t nz, double* data, const std::string& planner_flag)
		: FFT3Base<double>(howmany, nx, ny, nz, planner_flag)
	{
		// TODO: test other planner flags http://www.fftw.org/doc/Planner-Flags.html#Planner-Flags
		if (howmany >= 1) {
			int n[3]; n[0] = (int)nx; n[1] = (int)ny; n[2] = (int)nz;
			int nzc = nz/2+1;
			int nzp = 2*nzc;
			_fplan = fftw_plan_many_dft_r2c(3, n, howmany, data, NULL, 1, nx*ny*nzp, reinterpret_cast<fftw_complex*>(data), NULL, 1, nx*ny*nzc, _flags);
			_bplan = fftw_plan_many_dft_c2r(3, n, howmany, reinterpret_cast<fftw_complex*>(data), NULL, 1, nx*ny*nzc, data, NULL, 1, nx*ny*nzp, _flags);
		}
		else {
			_fplan = fftw_plan_dft_r2c_3d(nx, ny, nz, data, reinterpret_cast<fftw_complex*>(data), _flags);
			_bplan = fftw_plan_dft_c2r_3d(nx, ny, nz, reinterpret_cast<fftw_complex*>(data), data, _flags);
		}
	}
	
	~FFT3() {
		fftw_destroy_plan(_fplan);
		fftw_destroy_plan(_bplan);
	}
	
	noinline void forward(const double* x, std::complex<double>* y)
	{
		Timer __t("forward FFT double", false);
		copy_inplace(x, y);
		fftw_execute_dft_r2c(_fplan, reinterpret_cast<double*>(y), reinterpret_cast<fftw_complex*>(y));
	}
	
	noinline void backward(const std::complex<double>* x, double* y)
	{
		Timer __t("backward FFT double", false);
		copy_inplace(x, y);
		fftw_execute_dft_c2r(_bplan, reinterpret_cast<fftw_complex*>(y), y);
	}
};

//! FFT specialization for float
template<>
class FFT3<float> : public FFT3Base<float>
{
protected:
	fftwf_plan _fplan, _bplan;

public:	FFT3(std::size_t howmany, std::size_t nx, std::size_t ny, std::size_t nz, float* data, const std::string& planner_flag)
		: FFT3Base<float>(howmany, nx, ny, nz, planner_flag)
	{
		// TODO: test other planner flags http://www.fftw.org/doc/Planner-Flags.html#Planner-Flags
		if (howmany > 1) {
			int n[3]; n[0] = (int)nx; n[1] = (int)ny; n[2] = (int)nz;
			int nzc = nz/2+1;
			int nzp = 2*nzc;
			_fplan = fftwf_plan_many_dft_r2c(3, n, howmany, data, n, 1, nx*ny*nzp, reinterpret_cast<fftwf_complex*>(data), n, 1, nx*ny*nzc, _flags);
			_bplan = fftwf_plan_many_dft_c2r(3, n, howmany, reinterpret_cast<fftwf_complex*>(data), n, 1, nx*ny*nzc, data, n, 1, nx*ny*nzp, _flags);
		}
		else {
			_fplan = fftwf_plan_dft_r2c_3d(nx, ny, nz, data, reinterpret_cast<fftwf_complex*>(data), _flags);
			_bplan = fftwf_plan_dft_c2r_3d(nx, ny, nz, reinterpret_cast<fftwf_complex*>(data), data, _flags);
		}
	}

	~FFT3() {
		fftwf_destroy_plan(_fplan);
		fftwf_destroy_plan(_bplan);
	}
	
	noinline void forward(const float* x, std::complex<float>* y)
	{
		Timer __t("forward FFT float", false);
		copy_inplace(x, y);
		fftwf_execute_dft_r2c(_fplan, reinterpret_cast<float*>(y), reinterpret_cast<fftwf_complex*>(y));
	}
	
	noinline void backward(const std::complex<float>* x, float* y)
	{
		Timer __t("backward FFT float", false);
		copy_inplace(x, y);
		fftwf_execute_dft_c2r(_bplan, reinterpret_cast<fftwf_complex*>(y), y);
	}
};


//! Class for isotropic material properties and constant conversion
template<typename T, int DIM>
class Material
{
protected:
	typedef struct {
		std::string name1;
		std::string name2;
		T* value1;
		T* value2;
		void (Material<T, DIM>::* calc)(void);
		void set(const std::string& name1, const std::string& name2, T* value1, T* value2,
			void (Material<T, DIM>::* calc)(void))
		{
			this->name1 = name1;
			this->name2 = name2;
			this->value1 = value1;
			this->value2 = value2;
			this->calc = calc;
		}
	} calc_pair;

public:
	T K;		// Bulk modulus [Pa]
	T E;		// Young's modulus [Pa]
	T mu, lambda;	// Lamé parameters [Pa]
	T nu;		// Poisson' ratio [1]
	T phi;		// volume fraction
	T M;		// P-wave modulus
	std::string prefix;	// prefix for names
	std::string postfix;	// postfix for names

	Material() :
		prefix(""), postfix("")
	{}

	Material(std::string prefix, std::string postfix) :
		prefix(prefix), postfix(postfix)
	{}

	// read material definition from ptree
	void readSettings(const ptree::ptree& pt)
	{
		const ptree::ptree& attr = pt.get_child("<xmlattr>", empty_ptree);
		std::vector<calc_pair> cp(10);
		int icalc = -1;

		cp[0].set("K", "E", &K, &E, &Material<T, DIM>::calc_from_K_E);
		cp[1].set("K", "lambda", &K, &lambda, &Material<T, DIM>::calc_from_K_lambda);
		cp[2].set("K", "mu", &K, &mu, &Material<T, DIM>::calc_from_K_mu);
		cp[3].set("K", "nu", &K, &nu, &Material<T, DIM>::calc_from_K_nu);
		cp[4].set("E", "mu", &E, &mu, &Material<T, DIM>::calc_from_E_mu);
		cp[5].set("E", "nu", &E, &nu, &Material<T, DIM>::calc_from_E_nu);
		cp[6].set("lambda", "mu", &lambda, &mu, &Material<T, DIM>::calc_from_lambda_mu);
		cp[7].set("lambda", "nu", &lambda, &nu, &Material<T, DIM>::calc_from_lambda_nu);
		cp[8].set("mu", "nu", &mu, &nu, &Material<T, DIM>::calc_from_mu_nu);
		cp[9].set("mu", "M", &mu, &M, &Material<T, DIM>::calc_from_mu_M);

		for (std::size_t i = 0; i < cp.size(); i++)
		{
			std::size_t c1 = attr.count(prefix + cp[i].name1 + postfix);
			std::size_t c2 = attr.count(prefix + cp[i].name2 + postfix);

			if (c1 == 1 && c2 == 1) {
				icalc = (int) i;
			}
			else if (c1 > 1 || c2 > 1 || (icalc >= 0 && (
				(c1 > 0 && cp[i].name1 != cp[icalc].name1 && cp[i].name1 != cp[icalc].name2) ||
				(c2 > 0 && cp[i].name2 != cp[icalc].name1 && cp[i].name2 != cp[icalc].name2))))
			{
				BOOST_THROW_EXCEPTION(std::runtime_error("Ambiguous material definition"));
			}
		}

		if (icalc < 0) {
			BOOST_THROW_EXCEPTION(std::runtime_error("Incomplete material definition"));
		}

		phi = pt_get<T>(attr, prefix + "phi" + postfix, 0);
		*(cp[icalc].value1) = pt_get<T>(attr, prefix + cp[icalc].name1 + postfix);
		*(cp[icalc].value2) = pt_get<T>(attr, prefix + cp[icalc].name2 + postfix);
		(this->*(cp[icalc].calc))();
	}
	
	void calc_from_K_E()
	{
		lambda= (3*K*(3*K-E))/(9*K-E);
		mu =    (3*K*E)/(9*K-E);
		nu =    (3*K-E)/(6*K);
		M =     (3*K*(3*K+E))/(9*K-E);
	}

	void calc_from_K_lambda()
	{
		E =     (9*K*(K-lambda))/(3*K-lambda);
		mu =    (3*(K-lambda))/(2);
		nu =    (lambda)/(3*K-lambda);
		M =     3*K-2*lambda;
	}

	void calc_from_K_mu()
	{
		E =     (9*K*mu)/(3*K+mu);
		lambda= K-(2*mu)/(3);
		nu =    (3*K-2*mu)/(2*(3*K+mu));
		M =     K+(4*mu)/(3);
	}

	void calc_from_K_nu()
	{
		E =     3*K*(1-2*nu);
		lambda= (3*K*nu)/(1+nu);
		mu =    (3*K*(1-2*nu))/(2*(1+nu));
		M =     (3*K*(1-nu))/(1+nu);
	}

	void calc_from_E_mu()
	{
		K =     (E*mu)/(3*(3*mu-E));
		lambda= (mu*(E-2*mu))/(3*mu-E);
		nu =    (E)/(2*mu)-1;
		M =     (mu*(4*mu-E))/(3*mu-E);
	}

	void calc_from_E_nu()
	{
		K =     (E)/(3*(1-2*nu));
		lambda= (E*nu)/((1+nu)*(1-2*nu));
		mu =    (E)/(2*(1+nu));
		M =     (E*(1-nu))/((1+nu)*(1-2*nu));
	}

	void calc_from_lambda_mu()
	{
		K =     lambda+(2*mu)/(3);
		E =     (mu*(3*lambda+2*mu))/(lambda+mu);
		nu =    (lambda)/(2*(lambda+mu));
		M =     lambda+2*mu;
	}

	void calc_from_lambda_nu()
	{
		K =     (lambda*(1+nu))/(3*nu);
		E =     (lambda*(1+nu)*(1-2*nu))/(nu);
		mu =    (lambda*(1-2*nu))/(2*nu);
		M =     (lambda*(1-nu))/(nu);
	}

	void calc_from_mu_nu()
	{
		K =     (2*mu*(1+nu))/(3*(1-2*nu));
		E =     2*mu*(1+nu);
		lambda= (2*mu*nu)/(1-2*nu);
		M =     (2*mu*(1-nu))/(1-2*nu);
	}

	void calc_from_mu_M()
	{
		K =     M-(4*mu)/(3);
		E =     (mu*(3*M-4*mu))/(M-mu);
		lambda= M-2*mu;
		nu =    (M-2*mu)/(2*M-2*mu);
	}
};


//! Hashin bounds for a isotropic material
template<class T>
class HashinBounds
{
public:
	static void get(T mu1, T lambda1, T phi1, T mu2, T lambda2, T phi2, T& kl, T& mul, T& ku, T& muu)
	{
		T k1 = lambda1 + 2.0/3.0*mu1, k2 = lambda2 + 2.0/3.0*mu2;

/*
		kl = k2 + phi1*(k1-k2)*(k2+4.0/3.0*mu2)/(k2+4.0/3.0*mu2 + phi2*(k1-k2));
		ku = k1 + phi2*(k2-k1)*(k1+4.0/3.0*mu1)/(k1+4.0/3.0*mu1 + phi1*(k2-k1));
		if (ku < kl) std::swap(kl, ku);

		mul = mu2 + phi1*(mu1-mu2)*5*mu2*(k2+4.0/3.0*mu2)/(5*mu2*(k2+4.0/3.0*mu2) + 2*phi2*(mu1-mu2)*(k2+2*mu2));
		muu = mu1 + phi2*(mu2-mu1)*5*mu1*(k1+4.0/3.0*mu1)/(5*mu1*(k1+4.0/3.0*mu1) + 2*phi1*(mu2-mu1)*(k1+2*mu1));
		if (muu < mul) std::swap(mul, muu);
*/

		kl = k2 + phi1*(k1-k2)*(k2+4.0/3.0*mu2)/(k2+4.0/3.0*mu2 + phi2*(k1-k2));
		ku = k1 + phi2*(k2-k1)*(k1+4.0/3.0*mu1)/(k1+4.0/3.0*mu1 + phi1*(k2-k1));
		if (ku < kl) std::swap(kl, ku);

		mul = mu2 + phi1*(mu1-mu2)/(1 + 2*phi2*(mu1-mu2)/(5*mu2) + 4*phi2*(mu1-mu2)/(15*k2+20*mu2));
		muu = mu1 + phi2*(mu2-mu1)/(1 + 2*phi1*(mu2-mu1)/(5*mu1) + 4*phi1*(mu2-mu1)/(15*k1+20*mu1));
		if (muu < mul) std::swap(mul, muu);
	}
};


//! A Multigrid level for the multigrid solver
template<class T>
class MultiGridLevel
{
public:
	std::size_t nx, ny, nz;	// grid size
	T Lx, Ly, Lz;		// RVE dimensions
	T hx, hy, hz;		// 1/(cell dimension)^2
	T hxyz;			// = -2*(hx + hy + hz)
	T* _r;			// the residual
	T* _x;			// the solution
	T* _b;			// the right hand side
	bool free_rxb;		// free r, x, b on object descruction
	std::size_t nyzp;		// data stride for x component
	std::size_t nzp;		// data stride for y component
	std::size_t n;			// nx*ny*nzp
	std::size_t nxyz;		// nx*ny*nz
	std::size_t n_pre_smooth;	// number pre smooth
	std::size_t n_post_smooth;	// number pre smooth
	std::size_t smooth_bs;		// smoother blocksize2*phi2*(mu1-mu2)*(k2+4.0/3.0*mu2+2.0/3.0*mu2)/(5*mu2*(k2+4.0/3.0*mu2))
	T smooth_relax;			// smoothing relaxation factor
	std::string pre_smoother;	// pre smoother "jacobi"=Jacobi, "fgs"=forward Gauss Seidel, "bgs"=backward GS
	std::string post_smoother;	// post smoother (as above)
	std::string coarse_solver;	// coarse grid solver "fft" or "lu"
	std::string prolongation_op;	// prolongation operator "straight_injection" or "full_weighting"
	bool enable_timing;		// enable timing of operations
	bool residual_checking;		// enable residual checking for direct solver

	T alpha;		// coarse grid scaling factor
	T hax, hay, haz;	// hx, hy, hz scaled by alpha
	T haxyz;		// hxyz scaled by alpha

	// NOTE: the value at the cell i,j,k is given by
	// v[row] where v is r, x or b and row = i*nyzp + j*nzp + k

	// pointer to coarser and finer levels
	boost::shared_ptr< MultiGridLevel<T> > coarser_level;
	boost::shared_ptr< MultiGridLevel<T> > finer_level;

	// offsets for computing (periodic) forward and backward finite differences along a specific component
	std::vector<int> ffd_x, ffd_y, ffd_z;
	std::vector<int> bfd_x, bfd_y, bfd_z;

	// FFT object for direct solver
	boost::shared_ptr< FFT3<T> > _fft;

	bool safe_mode;

	MultiGridLevel(std::size_t nx, std::size_t ny, std::size_t nz, std::size_t nzp, T Lx, T Ly, T Lz, bool alloc_rxb) :
		nx(nx), ny(ny), nz(nz), Lx(Lx), Ly(Ly), Lz(Lz), nzp(nzp)
	{
		safe_mode = false;

		nyzp = ny*nzp;
		n = nx*nyzp;
		nxyz = nx*ny*nz;

		if (alloc_rxb) {
			_r = (T*) fftw_malloc(sizeof(T)*n);
			_b = (T*) fftw_malloc(sizeof(T)*n);
			_x = (T*) fftw_malloc(sizeof(T)*n);
			if (_r == NULL || _b == NULL || _x == NULL) {
				BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Memory alloaction of %d bytes failed!") % (sizeof(T)*n)).str()));
			}
		}

		free_rxb = alloc_rxb;
		n_pre_smooth = n_post_smooth = 1;
		smooth_bs = -1;
		smooth_relax = 1.0;
		pre_smoother  = "fgs";
		post_smoother = "bgs";
		coarse_solver = "fft";
		prolongation_op = "full_weighting";
		enable_timing = false;
		residual_checking = false;

		// compute finite difference factors
		hax = hx = nx*nx/(Lx*Lx);
		hay = hy = ny*ny/(Ly*Ly);
		haz = hz = nz*nz/(Lz*Lz);
		haxyz = hxyz = -2*(hx + hy + hz);
		alpha = 1;

		// init forward and backward finite difference offsets
		ffd_x.resize(nx);
		bfd_x.resize(nx);
		for (std::size_t ii = 0; ii < nx; ii++) {
			ffd_x[ii] = (((int)((ii+1)%nx)) - (int)ii)*(int)nyzp;
		}
		for (std::size_t ii = 0; ii < nx; ii++) {
			bfd_x[ii] = -ffd_x[nx - 1 - ii];
		}
		ffd_y.resize(ny);
		bfd_y.resize(ny);
		for (std::size_t jj = 0; jj < ny; jj++) {
			ffd_y[jj] = (((int)((jj+1)%ny)) - (int)jj)*(int)nzp;
		}
		for (std::size_t jj = 0; jj < ny; jj++) {
			bfd_y[jj] = -ffd_y[ny - 1 - jj];
		}
		ffd_z.resize(nz);
		bfd_z.resize(nz);
		for (std::size_t kk = 0; kk < nz; kk++) {
			ffd_z[kk] = (((int)((kk+1)%nz)) - (int)kk);
		}
		for (std::size_t kk = 0; kk < nz; kk++) {
			bfd_z[kk] = -ffd_z[nz - 1 - kk];
		}
	}

	~MultiGridLevel()
	{
		if (free_rxb) {
			fftw_free(_r);
			fftw_free(_x);
			fftw_free(_b);
		}
	}

	//! return the result of the operator A*e_k for the row identified with the voxel (ii,jj,kk)
	//! where k has to be k = ii*nyzp + jj*nzp + kk and e_k is the k-th unit vector
	//! (i.e. returns the k-th diagonal element of A)
	//! diagA is essentially haxyz, however if nx or ny or nz == 1 then not!
	inline T diagA(std::size_t ii, std::size_t jj, std::size_t kk, std::size_t k)
	{
		return	hax*(((int)(ffd_x[ii] == 0)) + ((int)(bfd_x[ii] == 0))) +
			hay*(((int)(ffd_y[jj] == 0)) + ((int)(bfd_y[jj] == 0))) +
			haz*(((int)(ffd_z[kk] == 0)) + ((int)(bfd_z[kk] == 0))) + haxyz;
	}

	//! return the result of the operator A*x for the row identified with the voxel (ii,jj,kk)
	//! where k has to be k = ii*nyzp + jj*nzp + kk
	inline T applyA(const T* x, std::size_t ii, std::size_t jj, std::size_t kk, std::size_t k)
	{
		return	hax*(x[k + ffd_x[ii]] + x[k + bfd_x[ii]]) +
			hay*(x[k + ffd_y[jj]] + x[k + bfd_y[jj]]) +
			haz*(x[k + ffd_z[kk]] + x[k + bfd_z[kk]]) + haxyz*x[k];
	}

	//! compute y = A*x
	//! this is actually never used in performance critical code
	void applyA(const T* x, T* y)
	{
		#pragma omp parallel for schedule (static) collapse(2)
		for (std::size_t ii = 0; ii < nx; ii++) {
			for (std::size_t jj = 0; jj < ny; jj++) {
				std::size_t k = ii*nyzp + jj*nzp;
				for (std::size_t kk = 0; kk < nz; kk++) {
					y[k] = applyA(x, ii, jj, kk, k);
					k++;
				}
			}
		}
	}

	//! compute the residual
	void compute_residual(const T* x, const T* b, T* r)
	{
//		boost::shared_ptr<Timer> t;
//		if (!finer_level) t.reset(new Timer("compute_residual"));

		if (!safe_mode && hax == hay && hay == haz && nx > 2 && ny > 2 && nz > 2)
		{
			// fast version

			const size_t nyzp = this->nyzp;
			const size_t nzp = this->nzp;
			const size_t nx_minus_1 = nx - 1;
			const size_t ny_minus_1 = ny - 1;
			const size_t nz_minus_1 = nz - 1;
			const T minus_ha = -hax;
			const T minus_ha6 = -haxyz;

			#pragma omp parallel for schedule (static) collapse(2)
			for (std::size_t ii = 0; ii < nx; ii++)
			{
				for (std::size_t jj = 0; jj < ny; jj++)
				{
					std::size_t k = ii*nyzp + jj*nzp;

					if (ii == 0 || ii == nx_minus_1 || jj == 0 || jj == ny_minus_1)
					{
						// TODO: we can improve the performance here further, but probably neglectable
						for (std::size_t kk = 0; kk < nz; kk++) {
							r[k] = b[k] - applyA(x, ii, jj, kk, k);
							k++;
						}
					}
					else
					{
						r[k] = b[k] - applyA(x, ii, jj, 0, k);
						k++;
						for (std::size_t kk = 1; kk < nz_minus_1; kk++) {
							r[k] = b[k] + minus_ha*(
									x[k + nyzp] + x[k - nyzp] +
									x[k + nzp]  + x[k - nzp] +
									x[k + 1]    + x[k - 1]) + minus_ha6*x[k]
								;
							k++;
						}
						r[k] = b[k] - applyA(x, ii, jj, nz_minus_1, k);
					}
				}
			}
			
			return;
		}

		// slow fallback version of the code above

		#pragma omp parallel for schedule (static) collapse(2)
		for (std::size_t ii = 0; ii < nx; ii++) {
			for (std::size_t jj = 0; jj < ny; jj++) {
				std::size_t k = ii*nyzp + jj*nzp;
				for (std::size_t kk = 0; kk < nz; kk++) {
					r[k] = b[k] - applyA(x, ii, jj, kk, k);
					k++;
				}
			}
		}
	}

	//! restrict residual to coarser grid
	void restrict_residual(const T* r, T* bc)
	{
		boost::shared_ptr<Timer> t;
		if (enable_timing) t.reset(new Timer("restrict_residual"));

		if (prolongation_op == "straight_injection") {
			restrict_residual_straight_injection(r, bc);
		}
		else if (prolongation_op == "full_weighting") {
			restrict_residual_full_weighting(r, bc);
		}
		else {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown prolongation operator '%s'") % prolongation_op).str()));
		}
	}

	void restrict_residual_straight_injection(const T* r, T* bc)
	{
		const size_t nxc = coarser_level->nx;
		const size_t nyc = coarser_level->ny;
		const size_t nzc = coarser_level->nz;

		const size_t nyzpc = coarser_level->nyzp;
		const size_t nzpc = coarser_level->nzp;

		const size_t divx = nx/(2*nxc);
		const size_t divy = ny/(2*nyc);
		const size_t divz = nz/(2*nzc);

		if (!safe_mode && divx && divy && divz)
		{
			// create copy of class variables to keep them on the stack
			const size_t nyzp_plus_nzp = nyzp + nzp;
			const size_t nyzp_plus_nzp_plus_1 = nyzp_plus_nzp + 1;
			const size_t nyzp = this->nyzp;
			const size_t nyzp_plus_1 = nyzp + 1;
			const size_t nzp = this->nzp;
			const size_t nzp_plus_1 = nzp + 1;

			#pragma omp parallel for schedule (static) collapse(2)
			for (std::size_t ii = 0; ii < nxc; ii++)
			{
				for (std::size_t jj = 0; jj < nyc; jj++)
				{
					std::size_t kc = ii*nyzpc + jj*nzpc;
					std::size_t kf = (ii*nyzp + jj*nzp) << 1;

					for (std::size_t kk = 0; kk < nzc; kk++)
					{
IACA_START
						// this is just the sum over the dx*dy*dz fine grid cells
						// including special cases for 1d and 2d
						bc[kc] = (
							r[kf] +
							r[kf + 1] +
							r[kf + nzp] +
							r[kf + nzp_plus_1] +
							r[kf + nyzp] +
							r[kf + nyzp_plus_1] +
							r[kf + nyzp_plus_nzp] +
							r[kf + nyzp_plus_nzp_plus_1]
						);

						kc ++;
						kf += 2;
IACA_END
					}
				}
			}

			return;
		}

		// slow fallback version of the code above

		const T div = 1.0/((2 - divx)*(2 - divy)*(2 - divz));
		
		#pragma omp parallel for schedule (static) collapse(2)
		for (std::size_t ii = 0; ii < nxc; ii++)
		{
			for (std::size_t jj = 0; jj < nyc; jj++)
			{
				std::size_t kc = ii*nyzpc + jj*nzpc;
				std::size_t kf = ii*nyzp*(divx+1) + jj*nzp*(divy+1);

				for (std::size_t kk = 0; kk < nzc; kk++)
				{
					// this is just the sum over the dx*dy*dz fine grid cells
					// including special cases for 1d and 2d
					bc[kc] = div*(
						r[kf] +
						r[kf + divz] +
						r[kf + divy*nzp] +
						r[kf + divy*nzp + divz] +
						r[kf + divx*nyzp] +
						r[kf + divx*nyzp + divz] +
						r[kf + divx*nyzp + divy*nzp] +
						r[kf + divx*nyzp + divy*nzp + divz]
					);

					kc ++;
					kf += divz+1;
				}
			}
		}
	}

	void restrict_residual_full_weighting(const T* r, T* bc)
	{
		const size_t nxc = coarser_level->nx;
		const size_t nyc = coarser_level->ny;
		const size_t nzc = coarser_level->nz;

		const size_t nyzpc = coarser_level->nyzp;
		const size_t nzpc = coarser_level->nzp;

		const size_t divx = nx/(2*nxc);
		const size_t divy = ny/(2*nyc);
		const size_t divz = nz/(2*nzc);

		const T div = 1.0/((2 - divx)*(2 - divy)*(2 - divz));
		
		#pragma omp parallel for schedule (static) collapse(2)
		for (std::size_t ii = 0; ii < nxc; ii++)
		{
			for (std::size_t jj = 0; jj < nyc; jj++)
			{
				register std::size_t kc = ii*nyzpc + jj*nzpc;
				register std::size_t kf = ii*nyzp*(divx+1) + jj*nzp*(divy+1);

				for (std::size_t kk = 0; kk < nzc; kk++)
				{
					const register std::size_t x_plus = ffd_x[2*ii];
					const register std::size_t x_minus = bfd_x[2*ii];
					const register std::size_t y_plus = ffd_y[2*jj];
					const register std::size_t y_minus = bfd_y[2*jj];
					const register std::size_t z_plus = ffd_z[2*kk];
					const register std::size_t z_minus = bfd_z[2*kk];

					bc[kc] = div*(
						r[kf] + 0.5*(
							r[kf + x_plus] +
							r[kf + x_minus] +
							r[kf + y_plus] +
							r[kf + y_minus] +
							r[kf + z_plus] +
							r[kf + z_minus] 
						) + 0.25*(
							r[kf + x_plus + y_plus] +
							r[kf + x_plus + z_plus] +
							r[kf + y_plus + z_plus] +
							r[kf + x_plus + y_minus] +
							r[kf + x_plus + z_minus] +
							r[kf + y_plus + z_minus] +
							r[kf + x_minus + y_plus] +
							r[kf + x_minus + z_plus] +
							r[kf + y_minus + z_plus] +
							r[kf + x_minus + y_minus] +
							r[kf + x_minus + z_minus] +
							r[kf + y_minus + z_minus]
						) + 0.125*(
							r[kf + x_plus + y_plus + z_plus] +
							r[kf + x_minus + y_plus + z_plus] +
							r[kf + x_plus + y_minus + z_plus] +
							r[kf + x_minus + y_minus + z_plus] +
							r[kf + x_plus + y_plus + z_minus] +
							r[kf + x_minus + y_plus + z_minus] +
							r[kf + x_plus + y_minus + z_minus] +
							r[kf + x_minus + y_minus + z_minus]
						));

					kc ++;
					kf += divz+1;
				}
			}
		}
	}

	//! correct solution x by prolongating the error x from the coarser level
	//! to the finer level and adding it to the current solution x
	//! x = x + P_coarse x_coarse (where x_coarse is the approximate error)
	void correct_solution(const T* xc, T* x)
	{
		boost::shared_ptr<Timer> t;
		if (enable_timing) t.reset(new Timer("correct_solution"));

		if (prolongation_op == "straight_injection") {
			correct_solution_straight_injection(xc, x);
		}
		else if (prolongation_op == "full_weighting") {
			correct_solution_full_weighting(xc, x);
		}
		else {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown prolongation operator '%s'") % prolongation_op).str()));
		}
	}

	void correct_solution_straight_injection(const T* xc, T* x)
	{
		const size_t nyzpc = coarser_level->nyzp;
		const size_t nzpc = coarser_level->nzp;

		const size_t nxc = coarser_level->nx;
		const size_t nyc = coarser_level->ny;
		const size_t nzc = coarser_level->nz;

		const size_t divx = nx/(2*nxc);
		const size_t divy = ny/(2*nyc);
		const size_t divz = nz/(2*nzc);

		if (!safe_mode && divx && divy && divz)
		{
			// create copy of class variables to keep them on the stack
			const size_t nyzp_plus_nzp = nyzp + nzp;
			const size_t nyzp_plus_nzp_plus_1 = nyzp_plus_nzp + 1;
			const size_t nyzp = this->nyzp;
			const size_t nyzp_plus_1 = nyzp + 1;
			const size_t nzp = this->nzp;
			const size_t nzp_plus_1 = nzp + 1;

			#pragma omp parallel for schedule (static) collapse(2)
			for (std::size_t ii = 0; ii < nxc; ii++)
			{
				for (std::size_t jj = 0; jj < nyc; jj++)
				{
					std::size_t kc = ii*nyzpc + jj*nzpc;
					std::size_t kf = (ii*nyzp + jj*nzp) << 1;

					for (std::size_t kk = 0; kk < nzc; kk++)
					{
IACA_START
						const register T a = xc[kc];

						x[kf] += a;
						x[kf + 1] += a;
						x[kf + nzp] += a;
						x[kf + nzp_plus_1] += a;
						x[kf + nyzp] += a;
						x[kf + nyzp_plus_1] += a;
						x[kf + nyzp_plus_nzp] += a;
						x[kf + nyzp_plus_nzp_plus_1] += a;

						kc ++;
						kf += 2;
IACA_END
					}

				}
			}

			return;
		}

		// slow fallback version of the code above

		// if the grid is not divided in some spatial dimensions we need to compensate duplicate additions of terms
		const T div = 1.0/((2 - divx)*(2 - divy)*(2 - divz));

		#pragma omp parallel for schedule (static) collapse(2)
		for (std::size_t ii = 0; ii < nxc; ii++)
		{
			for (std::size_t jj = 0; jj < nyc; jj++)
			{
				std::size_t kc = ii*nyzpc + jj*nzpc;
				std::size_t kf = ii*nyzp*(divx+1) + jj*nzp*(divy+1);

				for (std::size_t kk = 0; kk < nzc; kk++)
				{
					const register T a = div*xc[kc];

					x[kf] += a;
					x[kf + divz] += a;
					x[kf + divy*nzp] += a;
					x[kf + divy*nzp + divz] += a;
					x[kf + divx*nyzp] += a;
					x[kf + divx*nyzp + divz] += a;
					x[kf + divx*nyzp + divy*nzp] += a;
					x[kf + divx*nyzp + divy*nzp + divz] += a;

					kc ++;
					kf += divz+1;
				}
			}
		}
	}

	void correct_solution_full_weighting(const T* xc, T* x)
	{
		const size_t nyzpc = coarser_level->nyzp;
		const size_t nzpc = coarser_level->nzp;

		const size_t nxc = coarser_level->nx;
		const size_t nyc = coarser_level->ny;
		const size_t nzc = coarser_level->nz;

		const size_t divx = nx/(2*nxc);
		const size_t divy = ny/(2*nyc);
		const size_t divz = nz/(2*nzc);

		// if the grid is not divided in some spatial dimensions we need to compensate duplicate additions of terms
		const T div = 1.0/((2 - divx)*(2 - divy)*(2 - divz));

		#pragma omp parallel for schedule (static) collapse(2)
		for (std::size_t ii = 0; ii < nxc; ii++)
		{
			for (std::size_t jj = 0; jj < nyc; jj++)
			{
				register std::size_t kc = ii*nyzpc + jj*nzpc;
				register std::size_t kf = ii*nyzp*(divx+1) + jj*nzp*(divy+1);

				for (std::size_t kk = 0; kk < nzc; kk++)
				{
					const register T div_xc = div*xc[kc];
					const register T a = 0.5*div_xc;
					const register T b = 0.25*div_xc;
					const register T c = 0.125*div_xc;

					const register std::size_t x_plus = ffd_x[2*ii];
					const register std::size_t x_minus = bfd_x[2*ii];
					const register std::size_t y_plus = ffd_y[2*jj];
					const register std::size_t y_minus = bfd_y[2*jj];
					const register std::size_t z_plus = ffd_z[2*kk];
					const register std::size_t z_minus = bfd_z[2*kk];

					x[kf] += div_xc;

					x[kf + x_plus] += a;
					x[kf + x_minus] += a;
					x[kf + y_plus] += a;
					x[kf + y_minus] += a;
					x[kf + z_plus] += a;
					x[kf + z_minus] += a;

					x[kf + x_plus + y_plus] += b;
					x[kf + x_plus + z_plus] += b;
					x[kf + y_plus + z_plus] += b;
					x[kf + x_plus + y_minus] += b;
					x[kf + x_plus + z_minus] += b;
					x[kf + y_plus + z_minus] += b;
					x[kf + x_minus + y_plus] += b;
					x[kf + x_minus + z_plus] += b;
					x[kf + y_minus + z_plus] += b;
					x[kf + x_minus + y_minus] += b;
					x[kf + x_minus + z_minus] += b;
					x[kf + y_minus + z_minus] += b;

					x[kf + x_plus + y_plus + z_plus] += c;
					x[kf + x_minus + y_plus + z_plus] += c;
					x[kf + x_plus + y_minus + z_plus] += c;
					x[kf + x_minus + y_minus + z_plus] += c;
					x[kf + x_plus + y_plus + z_minus] += c;
					x[kf + x_minus + y_plus + z_minus] += c;
					x[kf + x_plus + y_minus + z_minus] += c;
					x[kf + x_minus + y_minus + z_minus] += c;

					kc ++;
					kf += divz+1;
				}
			}
		}
	}

	//! compute scaling constant for coarse grid operator
	T compute_alpha()
	{
		if (!finer_level) return 1;

		const size_t nxf = finer_level->nx;
		const size_t nyf = finer_level->ny;
		const size_t nzf = finer_level->nz;

		const size_t dx = nxf/nx;
		const size_t dy = nyf/ny;
		const size_t dz = nzf/nz;

		return finer_level->alpha*dx*dy*dz;

		// FIXME:
		// This give more or less the same result for uniform refinement,
		// but should give actually better results for non-uniform refinement
		// however in some cases it does not work, have to investigate this
		// see Notes_on_Multigrid_Methods.tex ...

		ublas::c_matrix<T, 3, 3> A[3];	// stencil for Laplacian of fine level
		
		A[0] = A[1] = A[2] = ublas::zero_matrix<T>(3);

		A[1](1,1) = finer_level->haxyz;
		A[0](1,1) = A[2](1,1) = finer_level->hax;
		A[1](0,1) = A[1](2,1) = finer_level->hay;
		A[1](1,0) = A[1](1,2) = finer_level->haz;

		T RAP = 0;

		for (int qi = 0; qi < (int)dx; qi++) {
			for (int qj = 0; qj < (int)dy; qj++) {
				for (int qk = 0; qk < (int)dz; qk++) {
					for (int si = -1; si <= 1; si++) {
						for (int sj = -1; sj <= 1; sj++) {
							for (int sk = -1; sk <= 1; sk++) {
								if ((nx + (qi + si)/(int)dx) % nx != 0) continue;
								if ((ny + (qj + sj)/(int)dy) % ny != 0) continue;
								if ((nz + (qk + sk)/(int)dz) % nz != 0) continue;
								RAP += A[si+1](sj+1,sk+1);
							}
						}
					}
				}
			}
		}

		T alpha = RAP/hxyz;

		return alpha;
	}

	T mean(const T* x) const
	{
		T s = 0;

		#pragma omp parallel for schedule (static) collapse(2) reduction(+:s)
		for (std::size_t ii = 0; ii < nx; ii++)
		{
			for (std::size_t jj = 0; jj < ny; jj++)
			{
				std::size_t kf = ii*nyzp + jj*nzp;

				for (std::size_t kk = 0; kk < nz; kk++)
				{
					s += x[kf + kk];
				}
			}
		}

		return (s/nxyz);
	}

	void print_norm(const std::string& m, const T* x) const
	{
		LOG_COUT << "NORM: " << m << ": " << norm(x) << std::endl;
	}

	void print_mean(const std::string& m, const T* x) const
	{
		return;
		print_norm(m, x);
		LOG_COUT << "MEAN: " << m << ": " << mean(x) << std::endl;
	}

	// pre-smooth x
	void pre_smooth(const T* b, T* x)
	{
		for (std::size_t i = 0; i < n_pre_smooth; i++) {
			smooth(pre_smoother, b, x);
		}
	}

	// post-smooth x
	void post_smooth(const T* b, T* x)
	{
		for (std::size_t i = 0; i < n_post_smooth; i++) {
			smooth(post_smoother, b, x);
		}
	}

	void project_zero(T* x)
	{
		shift_tensor(x, mean(x));
	}

	void shift_tensor(T* x, T s)
	{
		if (s == 0) return;

		#pragma omp parallel for schedule (static)
		for (std::size_t i = 0; i < n; i++) {
			x[i] -= s;
		}
	}

	void smooth(const std::string& smoother, const T* b, T* x)
	{
		boost::shared_ptr<Timer> t;
		if (enable_timing) t.reset(new Timer("smooth"));

		if (smoother == "jacobi") {
			smooth_jacobi(b, x);
		}
		else if (smoother == "fgs") {
			smooth_gauss_seidel(b, x);
		}
		else if (smoother == "bgs") {
			smooth_gauss_seidel_backward(b, x);
		}
		else {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown multigrid smoother '%s'") % smoother).str()));
		}
	}

	void smooth_jacobi(const T* b, T* x)
	{
		compute_residual(x, b, _r);

		// classic Gauss-Seidel
		#pragma omp parallel for schedule (static) collapse(2)
		for (std::size_t ii = 0; ii < nx; ii++) {
			for (std::size_t jj = 0; jj < ny; jj++) {
				std::size_t k = ii*nyzp + jj*nzp;
				for (std::size_t kk = 0; kk < nz; kk++) {
					x[k] += smooth_relax*_r[k]/diagA(ii, jj, kk, k);
					k++;
				}
			}
		}
	}

	//! apply gauss seidel smoother to x
	//! x = x - D^-1 * (b - A*x)
	// FIXME: might not work in parallel
	void smooth_gauss_seidel(const T* b, T* x, bool transpose = false)
	{
/*
		std::size_t bsx = smooth_bs;
		std::size_t bsy = smooth_bs;
		std::size_t bsz = smooth_bs;

		std::size_t nbx = 1 + (nx - 1)/bsx;
		std::size_t nby = 1 + (ny - 1)/bsy;
		std::size_t nbz = 1 + (nz - 1)/bsz;
		
		for (std::size_t r = 0; r <= 1; r++)
		{
			for (std::size_t bx = 0; bx < nbx; bx++)
			{
				std::size_t i0 = bx*bsx, i1 = std::min(i0 + bsx, nx);

				for (std::size_t by = 0; by < nby; by++)
				{
					std::size_t j0 = by*bsy, j1 = std::min(j0 + bsy, ny);

					for (std::size_t bz = 0; bz < nbz; bz++)
					{
						std::size_t k0 = bz*bsz, k1 = std::min(k0 + bsz, nz);

#pragma omp parallel for schedule (static) collapse(2) 
						for (std::size_t ii = i0; ii < i1; ii++) {
							for (std::size_t jj = j0; jj < j1; jj++) {
								std::size_t q = (ii + jj + k0 + r) % 2 + k0;
								std::size_t k = ii*nyzp + jj*nzp + q;
								for (std::size_t kk = q; kk < k1; kk += 2) {
									x[k] += (b[k] - applyA(x, ii, jj, kk, k))/diagA(ii, jj, kk, k);
									k += 2;
								}
							}
						}
					}
				}
			}
		}

		return;
*/

#if 0
		#pragma omp parallel for schedule (static) collapse(2)
		for (std::size_t ii = 0; ii < nx; ii+=2) {
			for (std::size_t jj = 0; jj < ny; jj+=2) {
				std::size_t k = (ii*nyzp + jj*nzp);
				for (std::size_t kk = 0; kk < nz; kk+=2) {
					x[k] += (b[k] - applyA(x, ii, jj, kk, k))/diagA(ii, jj, kk, k);
					x[k+1] += (b[k+1] - applyA(x, ii, jj, kk+1, k+1))/diagA(ii, jj, kk+1, k+1);
					x[k+nyzp] += (b[k+nyzp] - applyA(x, ii+1, jj, kk, k+nyzp))/diagA(ii+1, jj, kk, k+nyzp);
					x[k+1+nyzp] += (b[k+1+nyzp] - applyA(x, ii+1, jj, kk+1, k+1+nyzp))/diagA(ii+1, jj, kk+1, k+1+nyzp);
					x[k+nzp] += (b[k+nzp] - applyA(x, ii, jj+1, kk, k+nzp))/diagA(ii, jj+1, kk, k+nzp);
					x[k+1+nzp] += (b[k+1+nzp] - applyA(x, ii, jj+1, kk+1, k+1+nzp))/diagA(ii, jj+1, kk+1, k+1+nzp);
					x[k+nyzp+nzp] += (b[k+nyzp+nzp] - applyA(x, ii+1, jj+1, kk, k+nyzp+nzp))/diagA(ii+1, jj+1, kk, k+nyzp+nzp);
					x[k+1+nyzp+nzp] += (b[k+1+nyzp+nzp] - applyA(x, ii+1, jj+1, kk+1, k+1+nyzp+nzp))/diagA(ii+1, jj+1, kk+1, k+1+nyzp+nzp);
					k+=2;
				}
			}
		}

		// project to zero mean
		shift_tensor(x, s/nxyz);
		return;

		if (hax == hay && hay == haz && nx > 2 && ny > 2 && nz > 2)
		{
			// fast version

			const size_t nyzp = this->nyzp;
			const size_t nzp = this->nzp;
			const size_t nx_minus_1 = nx - 1;
			const size_t ny_minus_1 = ny - 1;
			const size_t nz_minus_1 = nz - 1;
			const T Dinv = 1.0/haxyz;
			const T minus_Dinv_ha = -Dinv*hax;
			const T minus_Dinv_ha6 = -Dinv*haxyz;
			T* x = this->x;
			T* b = this->b;

//			#pragma omp parallel for schedule (static) collapse(2)
			for (std::size_t ii = 0; ii < nx; ii++)
			{
				for (std::size_t jj = 0; jj < ny; jj++)
				{
					std::size_t k = ii*nyzp + jj*nzp;

					if (ii == 0 || ii == nx_minus_1 || jj == 0 || jj == ny_minus_1)
					{
						// TODO: we can improve the performance here further, but probably neglectable
						for (std::size_t kk = 0; kk < nz; kk++) {
							x[k] += (b[k] - applyA(x, ii, jj, kk, k))/diagA(ii, jj, kk, k);
							k++;
						}
					}
					else
					{
						x[k] += (b[k] - applyA(x, ii, jj, 0, k))/diagA(ii, jj, 0, k);
						k++;
						for (std::size_t kk = 1; kk < nz_minus_1; kk++) {
							x[k] += Dinv*b[k] + minus_Dinv_ha*(
								x[k + nyzp] + x[k - nyzp] +
								x[k + nzp]  + x[k - nzp] +
								x[k + 1]    + x[k - 1]) + minus_Dinv_ha6*x[k]
							;
							k++;
						}
						x[k] += (b[k] - applyA(x, ii, jj, nz_minus_1, k))/diagA(ii, jj, nz_minus_1, k);
					}
				}
			}

			// project to zero mean
//			shift_tensor(x, s/nxyz);
			return;
		}

#endif

		if (!safe_mode && hax == hay && hay == haz && nx > 2 && ny > 2 && nz > 2)
		{
			// fast version

			const size_t nyzp = this->nyzp;
			const size_t nzp = this->nzp;
			const size_t nx_minus_1 = nx - 1;
			const size_t ny_minus_1 = ny - 1;
			const size_t nz_minus_1 = nz - 1;
			const T Dinv = 1.0/haxyz;
			const T minus_Dinv_ha = -Dinv*hax;
			const T minus_Dinv_ha6 = -Dinv*haxyz;
			const std::size_t s = transpose ? 1 : 0;

			for (std::size_t r = s; r <= (1+s); r++)
			{
				#pragma omp parallel for schedule (static) collapse(2)
				for (std::size_t ii = 0; ii < nx; ii++)
				{
					for (std::size_t jj = 0; jj < ny; jj++)
					{
						std::size_t q = (ii + jj + r) % 2;
						std::size_t k = ii*nyzp + jj*nzp + q;

						if (ii == 0 || ii == nx_minus_1 || jj == 0 || jj == ny_minus_1)
						{
							// TODO: we can improve the performance here further, but probably neglectable
							for (std::size_t kk = q; kk < nz; kk += 2) {
								x[k] += (b[k] - applyA(x, ii, jj, kk, k))/diagA(ii, jj, kk, k);
								k += 2;
							}
							continue;
						}

						if (q == 0) {
							x[k] += (b[k] - applyA(x, ii, jj, 0, k))/diagA(ii, jj, 0, k);
							k += 2;
							q += 2;
						}

						for (std::size_t kk = q; kk < nz_minus_1; kk += 2) {
IACA_START
							x[k] += Dinv*b[k] + minus_Dinv_ha*(
								x[k - nyzp] + x[k - nzp]  +
								x[k - 1]    + x[k + 1]    +
								x[k + nzp]  + x[k + nyzp]) + minus_Dinv_ha6*x[k];
							k += 2;
IACA_END
						}

						if ((nz_minus_1 - q) % 2 == 0) {
							x[k] += (b[k] - applyA(x, ii, jj, nz_minus_1, k))/diagA(ii, jj, nz_minus_1, k);
						}
					}
				}


#if 0
				// compute residuals on red/black nodes

				// red-black Gauss-Seidel
				T res[2];
				res[0] = res[1] = 0;

				for (std::size_t w = 0; w <= 1; w++)
				{
					#pragma omp parallel for schedule (static) collapse(2) 
					for (std::size_t ii = 0; ii < nx; ii++) {
						for (std::size_t jj = 0; jj < ny; jj++) {
							std::size_t q = (ii + jj + w) % 2;
							std::size_t k = ii*nyzp + jj*nzp + q;
							for (std::size_t kk = q; kk < nz; kk += 2) {
								res[w] += fabs(b[k] - applyA(x, ii, jj, kk, k));
								k += 2;
							}
						}
					}
				}

				LOG_COUT << "residual 0 " << res[0] << " residual 1 " << res[1] << std::endl;
#endif

			}


			return;
		}

#if 1
		// red-black Gauss-Seidel
		const std::size_t s = transpose ? 1 : 0;

		for (std::size_t r = s; r <= (1+s); r++)
		{
			#pragma omp parallel for schedule (static) collapse(2) 
			for (std::size_t ii = 0; ii < nx; ii++) {
				for (std::size_t jj = 0; jj < ny; jj++) {
					std::size_t q = (ii + jj + r) % 2;
					std::size_t k = ii*nyzp + jj*nzp + q;
					for (std::size_t kk = q; kk < nz; kk += 2) {
						x[k] += (b[k] - applyA(x, ii, jj, kk, k))/diagA(ii, jj, kk, k);
						k += 2;
					}
				}
			}
		}
#else
		// classic Gauss-Seidel

		#pragma omp parallel for schedule (static) collapse(2)
		for (std::size_t ii = 0; ii < nx; ii++) {
			for (std::size_t jj = 0; jj < ny; jj++) {
				std::size_t k = ii*nyzp + jj*nzp;
				for (std::size_t kk = 0; kk < nz; kk++) {
					x[k] += (b[k] - applyA(x, ii, jj, kk, k))/diagA(ii, jj, kk, k);
					k++;
				}
			}
		}
#endif
	}

	//! apply gauss seidel smoother to x
	//! x = x - D^-1 * (b - A*x)
	void smooth_gauss_seidel_backward(const T* b, T* x)
	{
		smooth_gauss_seidel(b, x, true);
	}

	//! set current solution to zero
	void zero(T* x)
	{
#if 1
		// TODO: use parallel memset
		memset(x, 0, n*sizeof(T));
#else
		#pragma omp parallel for schedule (static)
		for (std::size_t i = 0; i < n; i++) {
			x[i] = 0;
		}
#endif
	}

	void solve_direct(const T* b, T* x)
	{
		if (coarse_solver == "fft") {
			solve_direct_fft(b, x);
		}
		else if (coarse_solver == "lu") {
			solve_direct_lu(b, x);
		}
		else {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown solver type '%s'") % coarse_solver).str()));
		}
	}

	//! solve the equation A*x = b directly usig fft
	void solve_direct_fft(const T* b, T* x)
	{
		if (!_fft) {
			_fft.reset(new FFT3<T>(1, nx, ny, nz, x, "measure"));
		}

		std::complex<T>* xc = (std::complex<T>*) x;

		// compute FFT of rhs
		_fft->forward(b, xc);

		// calculate solution in Fourier domain
		const T xi0_0 = 2.0*M_PI/nx, xi1_0 = 2.0*M_PI/ny, xi2_0 = 2.0*M_PI/nz;
		const std::size_t nzc = nz/2 + 1;
		const T c = 2*(T)nxyz;

		#pragma omp parallel for schedule (static)
		for (std::size_t ii = 0; ii < nx; ii++)
		{
			const T xi0 = xi0_0*ii;

			for (std::size_t jj = 0; jj < ny; jj++)
			{
				const T xi1 = xi1_0*jj;

				// calculate current index in complex tensor tau[*]
				std::size_t k = ii*ny*nzc + jj*nzc;

				for (std::size_t kk = 0; kk < nzc; kk++)
				{
					const T xi2 = xi2_0*kk;

					xc[k] /= c*(
						hax*(std::cos(xi0) - (T)1) +
						hay*(std::cos(xi1) - (T)1) +
						haz*(std::cos(xi2) - (T)1)
					);

					k++;
				}
			}
		}

		// set zero component
		xc[0] = (T)0;

		// transform solution to spatial domain
		_fft->backward(xc, x);
	}

	//! solve the equation A*x = b directly usig LU factorization
	void solve_direct_lu(const T* b, T* x)
	{
		// init matrix A
		ublas::matrix<T> A = ublas::zero_matrix<T>(nxyz);
		ublas::vector<T> B(nxyz);
		
		#pragma omp parallel for schedule (static) collapse(2)
		for (std::size_t ii = 0; ii < nx; ii++) {
			for (std::size_t jj = 0; jj < ny; jj++) {
				std::size_t row = (ii*ny + jj)*nz;
				std::size_t k = ii*nyzp + jj*nzp;
				for (std::size_t kk = 0; kk < nz; kk++) {
					A(row, row) += haxyz;
					A(row, row + (ffd_x[ii]/(int)nyzp)*((int)(ny*nz))) += hax;
					A(row, row + (bfd_x[ii]/(int)nyzp)*((int)(ny*nz))) += hax;
					A(row, row + (ffd_y[jj]/(int)nzp)*((int)nz)) += hay;
					A(row, row + (bfd_y[jj]/(int)nzp)*((int)nz)) += hay;
					A(row, row + (ffd_z[kk])) += haz;
					A(row, row + (bfd_z[kk])) += haz;
					B(row) = b[k + kk];
					row++;
				}
			}
		}

		// incorporate zero mean condition
		// TODO: make this symmetric and use Cholesky
		B(0) = (T)0;
		for (std::size_t i = 0; i < nxyz; i++) {
			A(0, i) = (T)1;
		}

		// solve A*x = b
		// TODO: store factorization for repeated use
		ublas::vector<T> X(B);
		ublas::permutation_matrix<size_t> pm(A.size1());
		ublas::lu_factorize(A, pm);
		ublas::lu_substitute(A, pm, X);

		// copy solution back
		#pragma omp parallel for schedule (static) collapse(2)
		for (std::size_t ii = 0; ii < nx; ii++) {
			for (std::size_t jj = 0; jj < ny; jj++) {
				std::size_t row = (ii*ny + jj)*nz;
				std::size_t k = ii*nyzp + jj*nzp;
				for (std::size_t kk = 0; kk < nz; kk++) {
					x[k + kk] = X(row + kk);
				}
			}
		}
	}

	//! perform V-cycle
	void vcycle()
	{
		vcycle(_r, _b, _x);
	}

	//! perform V-cycle
	void vcycle(T* r, const T* b, T* x)
	{
		boost::shared_ptr<Timer> t;
		if (enable_timing) t.reset(new Timer("vcycle"));

		if (!coarser_level)
		{
			// we are on the coarsest level
			// solve the equation directly usig LU factorization
			solve_direct(b, x);
			return;
		}

//		project_zero(x);

		// x = S(x)
		print_mean("before pre_smooth x", x);
		print_mean("before pre_smooth b", b);
		pre_smooth(b, x);
		print_mean("after pre_smooth", x);

		// r = b - A*x
		compute_residual(x, b, r);
		print_mean("after compute_residual", r);

		// restrict residual to coarse grid (b_coarse = R r)
		restrict_residual(r, coarser_level->_b);
		coarser_level->print_mean("after restrict_residual", coarser_level->_r);

		// compute (approximate) solution to Ax = r on coarse grid
		// thus x on the coarse grid contains an approximation of the error
		coarser_level->zero(coarser_level->_x);
		coarser_level->vcycle();

		// correct solution x = x + P_coarse x_coarse
		print_mean("before correct_solution", x);
		correct_solution(coarser_level->_x, x);
		print_mean("after correct_solution", x);

		// perform post smoothing
		// x = S(x)
		post_smooth(b, x);

		// 
//		project_zero(x);
	}

	//! compute x += a*y
	void incTensor(T*x, T a, const T* y)
	{
		#pragma omp parallel for schedule (static)
		for (std::size_t i = 0; i < n; i++) {
			x[i] += a*y[i];
		}
	}
	
	//! compute r = x + a*y
	void xpayTensor(T* r, const T* x, T a, const T* y)
	{
		#pragma omp parallel for schedule (static)
		for (std::size_t i = 0; i < n; i++) {
			r[i] = x[i] + a*y[i];
		}
	}
	
	void copyTensor(const T* a, T* b)
	{
		memcpy(b, a, sizeof(T)*n);
	}

	//! run direct solver
	//! \param r residual
	//! \param b right hand side
	//! \param x solution and initial guess
	//! \param tol stopping tolerance (l2 norm of residual)
	//! \param maxiter maximum number of iterations
	void run_direct(T* r, const T* b, T* x, T tol, std::size_t maxiter)
	{
		T small = boost::numeric::bounds<T>::smallest();
		T res_norm0 = 0;

		// iteration counter
		std::size_t k = 0;

		for (;;)
		{
			if (residual_checking) {
				compute_residual(x, b, r);
				T res_norm = norm(r) + small;
				if (k == 0) res_norm0 = res_norm;
				T rel_res = res_norm / res_norm0;
				LOG_COUT << "residual " << k << ": relative l2-norm: " << rel_res << " absolute: " << res_norm << std::endl;
				if (rel_res <= tol) {
					break;
				}
			}

			if (k >= maxiter) {
				break;
			}

			vcycle(r, b, x);

			k ++;
		}
	}

	//! Run preconditioned CG solver
	//! \param z temporary array
	//! \param d temporary array
	//! \param r residual
	//! \param h predonditioned residual (C*r)
	//! \param b right hand side
	//! \param x solution and initial guess
	//! \param tol stopping tolerance (C-norm of residual)
	//! \param maxiter maximum number of iterations
	void run_pcg(T* z, T* d, T* r, T* h, const T* b, T* x, T tol, std::size_t maxiter)
	{
		// r = b - A*x
		compute_residual(x, b, r);

		// h = C*r (preconditioned residual)
		zero(h);
		vcycle(z, r, h);
		project_zero(h);

		// NOTE: gamma is negative since -C is poitive semi-definite
		T gamma = dot(h, r);

		// d = h
		copyTensor(h, d);

		// norm of h
		T small = boost::numeric::bounds<T>::smallest();
		T res_norm0 = std::sqrt(std::abs(gamma)/nxyz) + small;
		T res_norm = res_norm0;
		T rel_res = 1;
		
		// iteration counter
		std::size_t k = 0;

		// TODO: improve convergence test position
	
		for (;;)
		{
			LOG_COUT << "residual " << k << ": relative C-norm: " << rel_res << " absolute: " << res_norm << std::endl;

			if (rel_res <= tol || k >= maxiter) {
				break;
			}

			// z = A*d
			applyA(d, z);
			
			T alpha = gamma / dot(d, z);

			// x += alpha*d
			incTensor(x, alpha, d);

			// r -= alpha*z
			incTensor(r, -alpha, z);

			// h = C*r (preconditioned residual)
			zero(h);
			vcycle(z, r, h);
			project_zero(h);

			T gamma_new = dot(r, h);
			T beta = gamma_new / gamma;

			// d = h + beta*d
			xpayTensor(d, h, beta, d);

			gamma = gamma_new;
			res_norm = std::sqrt(std::abs(gamma)/nxyz);
			rel_res = res_norm / res_norm0;
			k++;
		}
	}

	//! print the diagonal of A
	void print_diag()
	{
		for (std::size_t ii = 0; ii < nx; ii++) {
			for (std::size_t jj = 0; jj < ny; jj++) {
				std::size_t k = ii*nyzp + jj*nzp;
				for (std::size_t kk = 0; kk < nz; kk++) {
					LOG_COUT << diagA(ii,jj,kk,k) << " ";
					k++;
				}
			}
		}
	}

	T dot(const T* a, const T* b)
	{
		T s = 0;

		#pragma omp parallel for schedule (static) collapse(2) reduction(+:s)
		for (std::size_t ii = 0; ii < nx; ii++) {
			for (std::size_t jj = 0; jj < ny; jj++) {
				std::size_t k = ii*nyzp + jj*nzp;
				for (std::size_t kk = 0; kk < nz; kk++) {
					s += a[k]*b[k];
					k++;
				}
			}
		}

		return s;
	}

	T norm(const T* x) const
	{
		T s = 0;

		#pragma omp parallel for schedule (static) collapse(2) reduction(+:s)
		for (std::size_t ii = 0; ii < nx; ii++) {
			for (std::size_t jj = 0; jj < ny; jj++) {
				std::size_t k = ii*nyzp + jj*nzp;
				for (std::size_t kk = 0; kk < nz; kk++) {
					s += x[k]*x[k];
					k++;
				}
			}
		}

		return std::sqrt(s/nxyz);
	}

	void print_r() { printField("r", _r); }
	void print_b() { printField("b", _b); }
	void print_x() { printField("x", _x); }

	void print()
	{
		print_r();
		print_b();
		print_x();
	}

	inline void printField(const std::string& name, const T* x)
	{
		LOG_COUT << name << ":" << std::endl;
		LOG_COUT << format(x, nx, ny, nz, nzp) << std::endl << std::endl;
	}

	void init_levels(std::size_t nmin = 2)
	{
		std::size_t divx = 2;
		std::size_t divy = 2;
		std::size_t divz = 2;

		std::size_t nxc = nx/divx;
		std::size_t nyc = ny/divy;
		std::size_t nzc = nz/divz;

		if (nxc < nmin || nxc * divx != nx) { nxc = nx; divx = 1; return; }
		if (nyc < nmin || nyc * divy != ny) { nyc = ny; divy = 1; return; }
		if (nzc < nmin || nzc * divz != nz) { nzc = nz; divz = 1; return; }

		std::size_t nxyzc = nxc*nyc*nzc;

		if (nxyzc == nxyz) return;

		std::size_t nzpc = 2*(nzc/2+1);

		coarser_level.reset(new MultiGridLevel<T>(nxc, nyc, nzc, nzpc, Lx, Ly, Lz, true));
		coarser_level->finer_level = boost::shared_ptr< MultiGridLevel<T> >(this, boost::serialization::null_deleter());
		coarser_level->n_pre_smooth = n_pre_smooth;
		coarser_level->n_post_smooth = n_post_smooth;
		coarser_level->pre_smoother = pre_smoother;
		coarser_level->post_smoother = post_smoother;
		coarser_level->smooth_bs = smooth_bs;
		coarser_level->smooth_relax = smooth_relax;
		coarser_level->coarse_solver = coarse_solver;
		coarser_level->prolongation_op = prolongation_op;
		coarser_level->enable_timing = false;
		coarser_level->residual_checking = residual_checking;
		coarser_level->safe_mode = safe_mode;

		// compute scaling for coarse grid operator
		T alpha = coarser_level->compute_alpha();
		coarser_level->alpha = alpha;
		coarser_level->hax = alpha*coarser_level->hx;
		coarser_level->hay = alpha*coarser_level->hy;
		coarser_level->haz = alpha*coarser_level->hz;
		coarser_level->haxyz = alpha*coarser_level->hxyz;

//		LOG_COUT << "n:" << nxc << " alpha: " << alpha << std::endl;

		// do recursive initaliization of coarser levels
		coarser_level->init_levels(nmin);
	}
};


//! Base class for tensors
template<typename T, int DIM>
class Tensor : public ublas::c_vector<T, DIM>
{
public:
	T* E;

	inline Tensor()
	{
		this->E = this->data();

#ifdef NAN_FILL
		T nan = 0/(T)0;
		for (std::size_t i = 0; i < DIM; i++) {
			this->E[i] = nan;
		}
#endif
	}

	inline Tensor(const T* data) {
		this->E = this->data();
		this->copyFrom(data);
	}


	inline operator T*() const { return this->E; }

	//! zero tensor and set one diagonal entry to one
	inline void eye(std::size_t index)
	{
		std::memset(this->E, 0, DIM*sizeof(T));
		(*this)[index] = 1;
	}

	//! return dimension of tensor
	inline int dim() { return DIM; }

	//! copy tensor data from pointer v
	inline void copyFrom(const T* v)
	{
		memcpy(this->E, v, DIM*sizeof(T));
	}

	//! computes the Greens strain tensor E = (F^T*F-I)/2
	inline void greenStrain(const T* F)
	{
		// 0 8 7  0 5 4 
		// 5 1 6  8 1 3
		// 4 3 2  7 6 2
		this->E[0] = 0.5*(F[0]*F[0] + F[8]*F[8] + F[7]*F[7] - 1);
		this->E[1] = 0.5*(F[5]*F[5] + F[1]*F[1] + F[6]*F[6] - 1);
		this->E[2] = 0.5*(F[4]*F[4] + F[3]*F[3] + F[2]*F[2] - 1);
		this->E[3] = 0.5*(F[5]*F[4] + F[1]*F[3] + F[6]*F[2]);
		this->E[4] = 0.5*(F[0]*F[4] + F[8]*F[3] + F[7]*F[2]);
		this->E[5] = 0.5*(F[0]*F[5] + F[8]*F[1] + F[7]*F[6]);
	}

	//! computes the directional derivative dE/dF : W of Greens strain tensor E = (F^T*F-I)/2
	inline void greenStrainDeriv(const T* F, const T* W)
	{
		this->E[0] = 0.5*(W[0]*F[0] + W[8]*F[8] + W[7]*F[7]  +  F[0]*W[0] + F[8]*W[8] + F[7]*W[7]);
		this->E[1] = 0.5*(W[5]*F[5] + W[1]*F[1] + W[6]*F[6]  +  F[5]*W[5] + F[1]*W[1] + F[6]*W[6]);
		this->E[2] = 0.5*(W[4]*F[4] + W[3]*F[3] + W[2]*F[2]  +  F[4]*W[4] + F[3]*W[3] + F[2]*W[2]);
		this->E[3] = 0.5*(W[5]*F[4] + W[1]*F[3] + W[6]*F[2]  +  F[5]*W[4] + F[1]*W[3] + F[6]*W[2]);
		this->E[4] = 0.5*(W[0]*F[4] + W[8]*F[3] + W[7]*F[2]  +  F[0]*W[4] + F[8]*W[3] + F[7]*W[2]);
		this->E[5] = 0.5*(W[0]*F[5] + W[8]*F[1] + W[7]*F[6]  +  F[0]*W[5] + F[8]*W[1] + F[7]*W[6]);
	}

	//! computes the Greens strain tensor E = F^T*F
	inline void rightCauchyGreen(const T* F)
	{
		// 0 8 7  0 5 4 
		// 5 1 6  8 1 3
		// 4 3 2  7 6 2
		this->E[0] = (F[0]*F[0] + F[8]*F[8] + F[7]*F[7]);
		this->E[1] = (F[5]*F[5] + F[1]*F[1] + F[6]*F[6]);
		this->E[2] = (F[4]*F[4] + F[3]*F[3] + F[2]*F[2]);
		this->E[3] = (F[5]*F[4] + F[1]*F[3] + F[6]*F[2]);
		this->E[4] = (F[0]*F[4] + F[8]*F[3] + F[7]*F[2]);
		this->E[5] = (F[0]*F[5] + F[8]*F[1] + F[7]*F[6]);
	}

	//! computes the Greens strain tensor directional derivative dE = dF^T*F + F^T*dF
	inline void rightCauchyGreenDeriv(const T* F, const T* W)
	{
		// 0 8 7  0 5 4 
		// 5 1 6  8 1 3
		// 4 3 2  7 6 2
		this->E[0] = (F[0]*W[0] + F[8]*W[8] + F[7]*W[7]) + (W[0]*F[0] + W[8]*F[8] + W[7]*F[7]);
		this->E[1] = (F[5]*W[5] + F[1]*W[1] + F[6]*W[6]) + (W[5]*F[5] + W[1]*F[1] + W[6]*F[6]);
		this->E[2] = (F[4]*W[4] + F[3]*W[3] + F[2]*W[2]) + (W[4]*F[4] + W[3]*F[3] + W[2]*F[2]);
		this->E[3] = (F[5]*W[4] + F[1]*W[3] + F[6]*W[2]) + (W[5]*F[4] + W[1]*F[3] + W[6]*F[2]);
		this->E[4] = (F[0]*W[4] + F[8]*W[3] + F[7]*W[2]) + (W[0]*F[4] + W[8]*F[3] + W[7]*F[2]);
		this->E[5] = (F[0]*W[5] + F[8]*W[1] + F[7]*W[6]) + (W[0]*F[5] + W[8]*F[1] + W[7]*F[6]);
	}

	//! fill tensor with random values
	inline void random()
	{
		for (std::size_t j = 0; j < DIM; j++) {
			this->E[j] = RandomNormal01<T>::instance().rnd();
		}
	}

	//! set tensor to zero
	inline void zero()
	{
		std::memset(this->E, 0, DIM*sizeof(T));
	}

	//! print tensor data
	inline void print(const char* name)
	{
		LOG_COUT << "tensor " << name << ": " << format(*this) << std::endl;
	}
};


//! Identity tensor base class
template<typename T, int DIM>
class TensorIdentity
{
public:
	static const T Id[DIM*DIM];
};


//! 3x3 identity matrix
template<typename T>
class TensorIdentity<T, 3>
{
public:
	static const T Id[9];
};

template<typename T>
const T TensorIdentity<T, 3>::Id[9] = {
	1, 0, 0,
	0, 1, 0,
	0, 0, 1,
};


//! 6x6 identity matrix
template<typename T>
class TensorIdentity<T, 6>
{
public:
	static const T Id[36];
};

template<typename T>
const T TensorIdentity<T, 6>::Id[36] = {
	1, 0, 0, 0, 0, 0,
	0, 1, 0, 0, 0, 0,
	0, 0, 1, 0, 0, 0,
	0, 0, 0, 1, 0, 0,
	0, 0, 0, 0, 1, 0,
	0, 0, 0, 0, 0, 1,
};


//! 9x9 identity matrix
template<typename T>
class TensorIdentity<T, 9>
{
public:
	static const T Id[81];
};

template<typename T>
const T TensorIdentity<T, 9>::Id[81] = {
	1, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 1, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 1, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 1, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 1, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 1, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 1, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 1, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 1,
};


template<typename T> class Tensor3;

//! Class for 3x3 matrix
// the components are stored as vector: e11, e22, e33, e23, e13, e12, e32, e31, e21
template<typename T>
class Tensor3x3 : public Tensor<T, 9>
{
public:
	static inline T det(const T* A);
	static inline T trace(const T* A);
	static inline T dot(const T* A, const T* B);
	static inline T dotT(const T* A, const T* B);

	inline Tensor3x3() : Tensor<T, 9>() { }
	inline Tensor3x3(const T* data) : Tensor<T, 9>(data) { }
	inline Tensor3x3(const Tensor3x3<T>& t) : Tensor<T, 9>(t.E) { }
	inline Tensor3x3(const ublas::vector<T>& t) : Tensor<T, 9>(&(t[0])) { }
	inline Tensor3x3(const ublas::matrix<T>& m) : Tensor<T, 9>() {
		this->E[0] = m(0,0);
		this->E[1] = m(1,1);
		this->E[2] = m(2,2);
		this->E[3] = m(1,2);
		this->E[4] = m(0,2);
		this->E[5] = m(0,1);
		this->E[6] = m(2,1);
		this->E[7] = m(2,0);
		this->E[8] = m(1,0);
	}

	//! copy to ublas matrix
	inline void copyTo(ublas::matrix<T>& m) {
		m(0,0) = this->E[0];
		m(1,1) = this->E[1];
		m(2,2) = this->E[2];
		m(1,2) = this->E[3];
		m(0,2) = this->E[4];
		m(0,1) = this->E[5];
		m(2,1) = this->E[6];
		m(2,0) = this->E[7];
		m(1,0) = this->E[8];
	}

	//! compute inverse of A
	inline void inv(const T* A)
	{
#if 0
		ublas::c_matrix<T,3,3> Acopy;
		ublas::c_matrix<T,3,3> Ainv;
		Acopy(0,0) = A[0];
		Acopy(1,1) = A[1];
		Acopy(2,2) = A[2];
		Acopy(1,2) = A[3];
		Acopy(0,2) = A[4];
		Acopy(0,1) = A[5];
		Acopy(2,1) = A[6];
		Acopy(2,0) = A[7];
		Acopy(1,0) = A[8];
		int res = lapack::gesv(Acopy, Ainv);
		if (res != 0) {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Matrix inversion failed for matrix:\n%s!") % format(Acopy)).str()));
		}
		this->E[0] = Ainv(0,0);
		this->E[1] = Ainv(1,1);
		this->E[2] = Ainv(2,2);
		this->E[3] = Ainv(1,2);
		this->E[4] = Ainv(0,2);
		this->E[5] = Ainv(0,1);
		this->E[6] = Ainv(2,1);
		this->E[7] = Ainv(2,0);
		this->E[8] = Ainv(1,0);
#else
		T invdet = 1/Tensor3x3<T>::det(A);
		this->E[0] =  (A[1]*A[2]-A[6]*A[3])*invdet;
		this->E[1] =  (A[0]*A[2]-A[4]*A[7])*invdet;
		this->E[2] =  (A[0]*A[1]-A[8]*A[5])*invdet;
		this->E[3] = -(A[0]*A[3]-A[8]*A[4])*invdet;
		this->E[4] =  (A[5]*A[3]-A[4]*A[1])*invdet;
		this->E[5] = -(A[5]*A[2]-A[4]*A[6])*invdet;
		this->E[6] = -(A[0]*A[6]-A[7]*A[5])*invdet;
		this->E[7] =  (A[8]*A[6]-A[7]*A[1])*invdet;
		this->E[8] = -(A[8]*A[2]-A[3]*A[7])*invdet;
#endif
	}

	inline void invDeriv(const T* A, const T* Ainv, const T* dA)
	{
		Tensor3x3<T> X;
		X.mult(Ainv, dA);
		this->mult(X, Ainv);
		(*this) *= -1;
	}

	//! multiply 3x3 by 3x3 transpose
	inline void mult_t(const T* F, const T* S)
	{
		this->E[0] = F[0]*S[0] + F[5]*S[5] + F[4]*S[4];
		this->E[1] = F[8]*S[8] + F[1]*S[1] + F[3]*S[3];
		this->E[2] = F[7]*S[7] + F[6]*S[6] + F[2]*S[2];
		this->E[3] = F[8]*S[7] + F[1]*S[6] + F[3]*S[2];
		this->E[4] = F[0]*S[7] + F[5]*S[6] + F[4]*S[2];
		this->E[5] = F[0]*S[8] + F[5]*S[1] + F[4]*S[3];
		this->E[6] = F[7]*S[8] + F[6]*S[1] + F[2]*S[3];
		this->E[7] = F[7]*S[0] + F[6]*S[5] + F[2]*S[4];
		this->E[8] = F[8]*S[0] + F[1]*S[5] + F[3]*S[4];
	}

	//! multiply 3x3 by symmetric 3x3
	inline void mult_sym(const T* F, const T* S)
	{
		this->E[0] = F[0]*S[0] + F[5]*S[5] + F[4]*S[4];
		this->E[1] = F[8]*S[5] + F[1]*S[1] + F[3]*S[3];
		this->E[2] = F[7]*S[4] + F[6]*S[3] + F[2]*S[2];
		this->E[3] = F[8]*S[4] + F[1]*S[3] + F[3]*S[2];
		this->E[4] = F[0]*S[4] + F[5]*S[3] + F[4]*S[2];
		this->E[5] = F[0]*S[5] + F[5]*S[1] + F[4]*S[3];
		this->E[6] = F[7]*S[5] + F[6]*S[1] + F[2]*S[3];
		this->E[7] = F[7]*S[0] + F[6]*S[5] + F[2]*S[4];
		this->E[8] = F[8]*S[0] + F[1]*S[5] + F[3]*S[4];
	}

	//! multiply symmetric 3x3 by symmetric 3x3
	inline void mult_sym_sym(const T* F, const T* S)
	{
		this->E[0] = F[0]*S[0] + F[5]*S[5] + F[4]*S[4];
		this->E[1] = F[5]*S[5] + F[1]*S[1] + F[3]*S[3];
		this->E[2] = F[4]*S[4] + F[3]*S[3] + F[2]*S[2];
		this->E[3] = F[5]*S[4] + F[1]*S[3] + F[3]*S[2];
		this->E[4] = F[0]*S[4] + F[5]*S[3] + F[4]*S[2];
		this->E[5] = F[0]*S[5] + F[5]*S[1] + F[4]*S[3];
		this->E[6] = F[4]*S[5] + F[3]*S[1] + F[2]*S[3];
		this->E[7] = F[4]*S[0] + F[3]*S[5] + F[2]*S[4];
		this->E[8] = F[5]*S[0] + F[1]*S[5] + F[3]*S[4];
	}

	//! create rotation matrix from n1 to n2 (http://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula)
	//! n1 and n2 must have length 1
	// https://github.com/OpenFOAM/OpenFOAM-2.1.x/blob/master/src/OpenFOAM/primitives/transform/transform.H#L45
	inline void rot(const Tensor3<T>& n1, const Tensor3<T>& n2)
	{
		Tensor3<T> n3;
		n3.cross(n1, n2);
	
		T t, c = n1.dot(n2);

		if (c < 0) {
			t = -1.0/(1 - c);
		}
		else {
			t =  1.0/(1 + c);
		}

		this->E[0] = n2[0]*n1[0] - n1[0]*n2[0] + t*n3[0]*n3[0] + c;
		this->E[1] = n2[1]*n1[1] - n1[1]*n2[1] + t*n3[1]*n3[1] + c;
		this->E[2] = n2[2]*n1[2] - n1[2]*n2[2] + t*n3[2]*n3[2] + c;
		this->E[3] = n2[1]*n1[2] - n1[1]*n2[2] + t*n3[1]*n3[2];
		this->E[4] = n2[0]*n1[2] - n1[0]*n2[2] + t*n3[0]*n3[2];
		this->E[5] = n2[0]*n1[1] - n1[0]*n2[1] + t*n3[0]*n3[1];
		this->E[6] = n2[2]*n1[1] - n1[2]*n2[1] + t*n3[2]*n3[1];
		this->E[7] = n2[2]*n1[0] - n1[2]*n2[0] + t*n3[2]*n3[0];
		this->E[8] = n2[1]*n1[0] - n1[1]*n2[0] + t*n3[1]*n3[0];
	}

	// 0 5 4  0 5 4
	// 8 1 3  8 1 3
	// 7 6 2  7 6 2
	inline void mult(const T* F, const T* S)
	{
		this->E[0] = F[0]*S[0] + F[5]*S[8] + F[4]*S[7];
		this->E[1] = F[8]*S[5] + F[1]*S[1] + F[3]*S[6];
		this->E[2] = F[7]*S[4] + F[6]*S[3] + F[2]*S[2];
		this->E[3] = F[8]*S[4] + F[1]*S[3] + F[3]*S[2];
		this->E[4] = F[0]*S[4] + F[5]*S[3] + F[4]*S[2];
		this->E[5] = F[0]*S[5] + F[5]*S[1] + F[4]*S[6];
		this->E[6] = F[7]*S[5] + F[6]*S[1] + F[2]*S[6];
		this->E[7] = F[7]*S[0] + F[6]*S[8] + F[2]*S[7];
		this->E[8] = F[8]*S[0] + F[1]*S[8] + F[3]*S[7];
	}

	inline void sub(const T* F)
	{
		this->E[0] -= F[0];
		this->E[1] -= F[1];
		this->E[2] -= F[2];
		this->E[3] -= F[3];
		this->E[4] -= F[4];
		this->E[5] -= F[5];
		this->E[6] -= F[6];
		this->E[7] -= F[7];
		this->E[8] -= F[8];
	}

	inline void transpose(const T* F)
	{
		this->E[0] = F[0];
		this->E[1] = F[1];
		this->E[2] = F[2];
		this->E[3] = F[6];
		this->E[4] = F[7];
		this->E[5] = F[8];
		this->E[6] = F[3];
		this->E[7] = F[4];
		this->E[8] = F[5];
	}

	inline void eye()
	{
		this->E[0] = this->E[1] = this->E[2] = 1;
		this->E[3] = this->E[4] = this->E[5] = this->E[6] = this->E[7] = this->E[8] = 0;
	}

	inline T trace() const
	{
		return Tensor3x3<T>::trace(this->E);
	}

	inline T det() const
	{
		return Tensor3x3<T>::det(this->E);
	}

	inline T dot(const T* A) const
	{
		return Tensor3x3<T>::dot(this->E, A);
	}

	inline T dotT(const T* A) const
	{
		return Tensor3x3<T>::dotT(this->E, A);
	}
};

template<typename T>
inline T Tensor3x3<T>::dot(const T* A, const T* B)
{
	return B[0]*A[0] + B[1]*A[1] + B[2]*A[2] + B[3]*A[3] + B[4]*A[4] + B[5]*A[5] + B[6]*A[6] + B[7]*A[7] + B[8]*A[8];
}

template<typename T>
inline T Tensor3x3<T>::dotT(const T* A, const T* B)
{
	return B[0]*A[0] + B[1]*A[1] + B[2]*A[2] + B[3]*A[6] + B[4]*A[7] + B[5]*A[8] + B[6]*A[3] + B[7]*A[4] + B[8]*A[5];
}

template<typename T>
inline T Tensor3x3<T>::det(const T* A)
{
	return   A[0]*(A[1]*A[2]-A[6]*A[3])
		-A[5]*(A[8]*A[2]-A[3]*A[7])
		+A[4]*(A[8]*A[6]-A[1]*A[7]);
}

template<typename T>
inline T Tensor3x3<T>::trace(const T* A)
{
	return (A[0] + A[1] + A[2]);
}


//! Symmetric 3x3 matrix
// the components are stored as vector with components: e11, e22, e33, e23, e13, e12
template<typename T>
class SymTensor3x3 : public Tensor<T, 6>
{
public:

	inline SymTensor3x3() : Tensor<T, 6>() { }
	inline SymTensor3x3(const T* data) : Tensor<T, 6>(data) { }
	inline SymTensor3x3(const SymTensor3x3<T>& t) : Tensor<T, 6>(t.E) { }

	//! compute determinant of A
	static inline T det(const T* A);

	//! compute inverse of A
	inline void inv(const T* A)
	{
		T invdet = 1/SymTensor3x3<T>::det(A);
		this->E[0] =  (A[1]*A[2]-A[3]*A[3])*invdet;
		this->E[1] =  (A[0]*A[2]-A[4]*A[4])*invdet;
		this->E[2] =  (A[0]*A[1]-A[5]*A[5])*invdet;
		this->E[3] = -(A[0]*A[3]-A[4]*A[5])*invdet;
		this->E[4] =  (A[5]*A[3]-A[4]*A[1])*invdet;
		this->E[5] = -(A[5]*A[2]-A[3]*A[4])*invdet;
	}

	inline void invDeriv(const T* A, const T* Ainv, const T* dA)
	{
		Tensor3x3<T> X;
		X.mult_sym_sym(Ainv, dA);
		this->mult_asym(X, Ainv);
		(*this) *= -1;
	}

	//! multiply asymmetric matrix with symmetric matrix and assume the result is symmmetric
	inline void mult_asym(const T* A, const T* S)
	{
		// 0 5 4  0 5 4
		// 8 1 3  5 1 3
		// 7 6 2  4 3 2

		this->E[0] = A[0]*S[0] + A[5]*S[5] + A[4]*S[4];
		this->E[1] = A[8]*S[5] + A[1]*S[1] + A[3]*S[3];
		this->E[2] = A[7]*S[4] + A[6]*S[3] + A[2]*S[2];
		this->E[3] = A[8]*S[4] + A[1]*S[3] + A[3]*S[2];
		this->E[4] = A[0]*S[4] + A[5]*S[3] + A[4]*S[2];
		this->E[5] = A[0]*S[5] + A[5]*S[1] + A[4]*S[3];
	}

	//! return determinant
	inline T det() const
	{
		return Tensor3x3<T>::det(this->E);
	}

	//! return contraction with A
	inline T dot(const T* A) const
	{
		return this->E[0]*A[0] + this->E[1]*A[1] + this->E[2]*A[2] + 2*(this->E[3]*A[3] + this->E[4]*A[4] + this->E[5]*A[5]);
	}

	inline void sym_prod(const T* a, const T* b)
	{
		this->E[0] = 2*(a[0]*b[0] + a[5]*b[5] + a[4]*b[4]);
		this->E[1] = 2*(a[5]*b[5] + a[1]*b[1] + a[3]*b[3]);
		this->E[2] = 2*(a[4]*b[4] + a[3]*b[3] + a[2]*b[2]);
		this->E[3] = a[5]*b[4] + a[1]*b[3] + a[3]*b[2] + b[5]*a[4] + b[1]*a[3] + b[3]*a[2];
		this->E[4] = a[0]*b[4] + a[5]*b[3] + a[4]*b[2] + b[0]*a[4] + b[5]*a[3] + b[4]*a[2];
		this->E[5] = a[0]*b[5] + a[5]*b[1] + a[4]*b[3] + b[0]*a[5] + b[5]*a[1] + b[4]*a[3];
	}

	inline void outer(const T* a)
	{
		this->E[0] = a[0]*a[0];
		this->E[1] = a[1]*a[1];
		this->E[2] = a[2]*a[2];
		this->E[3] = a[1]*a[2];
		this->E[4] = a[0]*a[2];
		this->E[5] = a[0]*a[1];
	}

	inline void dev(const T* F)
	{
		T volF = (F[0] + F[1] + F[2])/3.0;
		
		this->E[0] = F[0] - volF;
		this->E[1] = F[1] - volF;
		this->E[2] = F[2] - volF;
		this->E[3] = F[3];
		this->E[4] = F[4];
		this->E[5] = F[5];
	}

	inline void devDeriv(const T* F, const T* dF)
	{
		this->dev(dF);
	}

	inline void eye()
	{
		this->E[0] = this->E[1] = this->E[2] = 1;
		this->E[3] = this->E[4] = this->E[5] = 0;
	}

	inline void add(T c, const T* F)
	{
		this->E[0] += c*F[0];
		this->E[1] += c*F[1];
		this->E[2] += c*F[2];
		this->E[3] += c*F[3];
		this->E[4] += c*F[4];
		this->E[5] += c*F[5];
	}

	inline void sub(const T* F)
	{
		this->E[0] -= F[0];
		this->E[1] -= F[1];
		this->E[2] -= F[2];
		this->E[3] -= F[3];
		this->E[4] -= F[4];
		this->E[5] -= F[5];
	}
};

template<typename T>
inline T SymTensor3x3<T>::det(const T* A)
{
	return   A[0]*(A[1]*A[2]-A[3]*A[3])
		-A[5]*(A[5]*A[2]-A[3]*A[4])
		+A[4]*(A[5]*A[3]-A[1]*A[4]);
}

//! 9x9 matrix
template<typename T>
class Tensor9x9 : public Tensor<T, 81>
{
	inline Tensor9x9() : Tensor<T, 81>() { }
	inline Tensor9x9(const T* data) : Tensor<T, 81>(data) { }
	inline Tensor9x9(const Tensor9x9<T>& t) : Tensor<T, 81>(t.E) { }
};


//! Vector of length 3
template<typename T>
class Tensor3 : public Tensor<T, 3>
{
public:
	inline Tensor3() : Tensor<T, 3>() { }
	inline Tensor3(const T* data) : Tensor<T, 3>(data) { }
	inline Tensor3(const Tensor3<T>& t) : Tensor<T, 3>(t.E) { }

	//! cross product of two vectors
	inline void cross(const T* n1, const T* n2)
	{
		this->E[0] = n1[1]*n2[2] - n1[2]*n2[1];
		this->E[1] = n1[2]*n2[0] - n1[0]*n2[2];
		this->E[2] = n1[0]*n2[1] - n1[1]*n2[0];
	}

	//! Matrix vector product
	inline void mult(const SymTensor3x3<T>& A, const Tensor3<T>& b)
	{
		this->E[0] = A[0]*b[0] + A[5]*b[1] + A[4]*b[2];
		this->E[1] = A[5]*b[0] + A[1]*b[1] + A[3]*b[2];
		this->E[2] = A[4]*b[0] + A[3]*b[1] + A[2]*b[2];
	}

	//! Matrix vector product
	inline void mult(const Tensor3x3<T>& A, const Tensor3<T>& b)
	{
		this->E[0] = A[0]*b[0] + A[5]*b[1] + A[4]*b[2];
		this->E[1] = A[8]*b[0] + A[1]*b[1] + A[3]*b[2];
		this->E[2] = A[7]*b[0] + A[6]*b[1] + A[2]*b[2];
	}

	//! Dot product
	inline T dot(const T* b) const
	{
		return this->E[0]*b[0] + this->E[1]*b[1] + this->E[2]*b[2];
	}

	//! Normalize vector to length 1
	inline void normalize()
	{
		(*this) /= std::sqrt(dot(*this));
	}
};


//! Base class for 3d tensor fields (compatible with FFT)
class TensorFieldBase
{
public:
	TensorFieldBase(std::size_t nx, std::size_t ny, std::size_t nz, std::size_t dim = 6) :
		dim(dim), nx(nx), ny(ny), nz(nz), is_shadow(false)
	{
		// calculate number of complex points in z dimension in Fourier domain
		// (this adds eventually reqired padding to the real data, see FFTW documentation fftw_plan_dft_r2c_3d for details)
		nzc = nz/2+1;
		
		// calculate number of real points in z dimension in spatial domain (= _nz + padding)
		nzp = 2*nzc;
		
		// compute number of voxels
		nxyz = nx*ny*nz;
	
		// compute x-stride
		nyzp = ny*nzp;
		
		// compute number of voxels (incl. padding) = real data length
		n = nx*nyzp;
	}

	//! see https://stackoverflow.com/questions/827196/virtual-default-destructors-in-c/827205
	virtual ~TensorFieldBase() {}

	std::size_t dim; // number of dimensions
	std::size_t nx, ny, nz;
	std::size_t nzc, nzp, nxyz, nyzp, n;
	bool is_shadow;	// Tensor is shadow of another tensor (do not free data)
};


//! 3d tensor field
template<typename T, typename S = T>
class TensorField : public TensorFieldBase
{
public:
#ifdef USE_MANY_FFT
	T* t;	// tensor data
	std::size_t bytes;	// total size in bytes of t
	std::size_t page_size;	// size for each dimension
	std::size_t ne;		// number of elements in t
#else
	T** t;	// tensor data (component wise)
#endif

	explicit TensorField(const TensorFieldBase& base) : TensorFieldBase(base.nx, base.ny, base.nz, base.dim), t(NULL) {}

	TensorField(const TensorFieldBase& base, std::size_t dim) :
		TensorFieldBase(base.nx, base.ny, base.nz, (dim == 0) ? base.dim : dim)
	{
		init();
	}

	TensorField(std::size_t nx, std::size_t ny, std::size_t nz, std::size_t dim = 6) :
		TensorFieldBase(nx, ny, nz, dim)
	{
		init();
	}

	virtual ~TensorField()
	{
		if (is_shadow) {
			// deletion is handled elsewhere
			DEBP("Deleted shadow tensor " << this);
			t = NULL;
			return;
		}

#ifdef USE_MANY_FFT
		fftw_free(t);
#else
		for (std::size_t i = 0; i < dim; i++) {
			freeComponent(i);
		}
		delete[] t;
#endif

		t = NULL;
		DEBP("Deleted tensor " << this);
	}

	void init()
	{
#ifdef USE_MANY_FFT
		ne = n*dim;
		bytes = sizeof(T)*ne;
		page_size = n;
		t = (T*) fftw_malloc(bytes);
		if (t == NULL) {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Memory alloaction of %d bytes failed!") % bytes).str()));
		}
#else
		std::size_t bytes = sizeof(T)*n;
		
		// alloc tensor field
		t = new T*[dim];
		for (std::size_t i = 0; i < dim; i++) {
			t[i] = (T*) fftw_malloc(bytes);
			if (t[i] == NULL) {
				BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Memory alloaction of %d bytes failed!") % bytes).str()));
			}
		}
#endif

		DEBP("Allocated tensor " << this << " dim=" << dim << " n=" << n);

#ifdef NAN_FILL
		invalidatePadding();
#endif
	}

	void assert_compatible(TensorField<T,S>& t)
	{
		if (t.dim != dim || t.nx != nx || t.ny != ny || t.nz != nz) {
			BOOST_THROW_EXCEPTION(std::runtime_error("Incompatible tensors for operation"));
		}
	}

	noinline void copyTo(TensorField<T,S>& t)
	{
		Timer __timer("copy tensor", false);

		assert_compatible(t);

#ifdef USE_MANY_FFT
		memcpy(t.t, this->t, bytes);
#else
	#if 0
		#pragma omp parallel for schedule (static) collapse(2)
		for (std::size_t j = 0; j < dim; j++) {
			for (std::size_t i = 0; i < n; i++) {
				t[j][i] = this->t[j][i];
			}
		}
	#else
		#pragma omp parallel for schedule (static)
		for (std::size_t i = 0; i < dim; i++) {
			memcpy(t[i], this->t[i], sizeof(T)*n);
		}
	#endif
#endif
	}

	boost::shared_ptr< TensorField<T,S> > shadow()
	{
		boost::shared_ptr< TensorField<T,S> > f(new TensorField<T,S>(*this));
		f->is_shadow = true;
		f->t = this->t;
#ifdef USE_MANY_FFT
		f->ne = ne;
		f->page_size = page_size;
#endif

		return f;
	}

	boost::shared_ptr< TensorField< std::complex<T>, T > > complex_shadow()
	{
		boost::shared_ptr< TensorField< std::complex<T>, T > > f(new TensorField< std::complex<T>, T >(*this));
		f->is_shadow = true;
#ifdef USE_MANY_FFT
		f->t = (std::complex<T>*) this->t;
		f->ne = ne/2;
		f->page_size = page_size/2;
#else
		f->t = (std::complex<T>**) this->t;
#endif
		// adjust the size of the tensor
		f->nz = f->nzc;
		f->n = n/2;
		return f;
	}

	inline T* operator[] (const std::size_t index)
	{
		DEBP("Tensor access " << this << " index=" << index);
#ifdef USE_MANY_FFT
		return t + index*page_size;
#else
		return t[index];
#endif
	}

	inline T* operator[] (const std::size_t index) const
	{
		DEBP("Tensor const access " << this << " index=" << index);
#ifdef USE_MANY_FFT
		return t + index*page_size;
#else
		return t[index];
#endif
	}

	void freeComponent(std::size_t i)
	{
#ifdef USE_MANY_FFT
		// TODO: we can not free a component
#else
		if (t[i] == NULL) return;
		//LOG_COUT << "TensorField " << this << " free component " << i << std::endl;
		fftw_free(t[i]);
		t[i] = NULL;
#endif
	}

	void check(const std::string& name)
	{
		for (std::size_t d = 0; d < dim; d++) {
			#pragma omp parallel for schedule (static) collapse(2)
			for (std::size_t i = 0; i < nx; i++) {
				for (std::size_t j = 0; j < ny; j++) {
					std::size_t kk = i*nyzp + j*nzp;
					for (std::size_t k = 0; k < nz; k++) {
						if (std::isnan((*this)[d][kk])) {
							std::cout << i << " " << j << " " << k << " " << d << std::endl;
							BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("field '%s' contains NaN") % name).str()));
						}
						if (std::isinf((*this)[d][kk])) {
							std::cout << i << " " << j << " " << k << " " << d << std::endl;
							BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("field '%s' contains inf") % name).str()));
						}
						kk ++;
					}
				}
			}
		}
	}

	noinline void invalidatePadding()
	{
		Timer __t("invalidatePadding", false);

		T nan = 0/(T)0;

		for (std::size_t d = 0; d < dim; d++) {
			#pragma omp parallel for schedule (static) collapse(2)
			for (std::size_t i = 0; i < nx; i++) {
				for (std::size_t j = 0; j < ny; j++) {
					std::size_t kk = i*nyzp + j*nzp;
					for (std::size_t k = nz; k < nzp; k++) {
						(*this)[d][kk + k] = nan;
					}
				}
			}
		}
	}

	//! compute t = t + x
	noinline void add(const TensorField<T,S>& x)
	{
		Timer __t("add", false);

#ifdef USE_MANY_FFT
		#pragma omp parallel for schedule (static)
		for (std::size_t i = 0; i < ne; i++) {
			t[i] += x.t[i];
		}
#else
		#pragma omp parallel for schedule (static) collapse(2)
		for (std::size_t j = 0; j < dim; j++) {
			for (std::size_t i = 0; i < n; i++) {
				t[j][i] += x[j][i];
			}
		}
#endif
	}
	
	//! compute r = x + a*y
	noinline void xpay(const TensorField<T,S>& x, T a, const TensorField<T,S>& y)
	{
		Timer __t("xpay", false);

		// TODO: add specialization for a == 1
		
#ifdef USE_MANY_FFT
		#pragma omp parallel for schedule (static)
		for (std::size_t i = 0; i < ne; i++) {
			t[i] = x.t[i] + a*y.t[i];
		}
#else
		#pragma omp parallel for schedule (static) collapse(2)
		for (std::size_t j = 0; j < dim; j++) {
			for (std::size_t i = 0; i < n; i++) {
				t[j][i] = x[j][i] + a*y[j][i];
			}
		}
#endif
	}
	
	//! add constant to each component of tensor
	noinline void add(const ublas::vector<T>& c)
	{
		Timer __t("add", false);

		std::size_t m = std::min(dim, c.size());

		#pragma omp parallel for schedule (static) collapse(2)
		for (std::size_t j = 0; j < m; j++) {
			//if (c[j] == 0) continue;
			for (std::size_t i = 0; i < n; i++) {
				(*this)[j][i] += c[j];
			}
		}
	}
	
	noinline void swap(TensorField<T,S>& x)
	{
		Timer __t("add", false);

#ifdef USE_MANY_FFT
		#pragma omp parallel for schedule (static)
		for (std::size_t i = 0; i < ne; i++) {
			std::swap(t[i], x.t[i]);
		}
#else
		#pragma omp parallel for schedule (static) collapse(2)
		for (std::size_t j = 0; j < dim; j++) {
			for (std::size_t i = 0; i < n; i++) {
				std::swap(t[j][i], x[j][i]);
			}
		}
#endif
	}
	
	//! scale tensor component wise
	noinline void scale(const ublas::vector<T>& c)
	{
		Timer __t("scale", false);

		std::size_t m = std::min(dim, c.size());
		
		#pragma omp parallel for schedule (static) collapse(2)
		for (std::size_t j = 0; j < m; j++) {
			//if (c[j] == 1) continue;
			for (std::size_t i = 0; i < n; i++) {
				(*this)[j][i] *= c[j];
			}
		}
	}

	//! linear interpolation of values
	void interpolate(T i, T j, T k, T* ret)
	{
		i = std::min(-1e-5+(T)nx, std::max((T)0, i));
		j = std::min(-1e-5+(T)ny, std::max((T)0, j));
		k = std::min(-1e-5+(T)nz, std::max((T)0, k));

		std::size_t i0 = (std::size_t) i;
		std::size_t j0 = (std::size_t) j;
		std::size_t k0 = (std::size_t) k;
		std::size_t i1 = std::min(nx-1, i0+1);
		std::size_t j1 = std::min(ny-1, j0+1);
		std::size_t k1 = std::min(nz-1, k0+1);

		T x_d = i - i0;
		T y_d = j - j0;
		T z_d = k - k0;

		for (std::size_t d = 0; d < dim; d++) {
			T c00 = (*this)[d][i0*nyzp + j0*nzp + k0]*(1 - x_d) + (*this)[d][i1*nyzp + j0*nzp + k0]*x_d;
			T c01 = (*this)[d][i0*nyzp + j0*nzp + k1]*(1 - x_d) + (*this)[d][i1*nyzp + j0*nzp + k1]*x_d;
			T c10 = (*this)[d][i0*nyzp + j1*nzp + k0]*(1 - x_d) + (*this)[d][i1*nyzp + j1*nzp + k0]*x_d;
			T c11 = (*this)[d][i0*nyzp + j1*nzp + k1]*(1 - x_d) + (*this)[d][i1*nyzp + j1*nzp + k1]*x_d;
			T c0 = c00*(1 - y_d) + c10*y_d;
			T c1 = c01*(1 - y_d) + c11*y_d;
			ret[d] = c0*(1 - z_d) + c1*z_d;
		}
	}

	//! init tensors with random values
	noinline void random()
	{
		Timer __t("random", false);

		// TODO: random does not produce always different numbers in parallel

		//#pragma omp parallel for schedule (static) collapse(2)
		for (std::size_t j = 0; j < dim; j++) {
			for (std::size_t i = 0; i < n; i++) {
				(*this)[j][i] = RandomNormal01<T>::instance().rnd();
			}
		}
	}

	//! set tensors to zero
	noinline void zero()
	{
		Timer __t("zero", false);

#ifdef USE_MANY_FFT
		std::memset(t, 0, bytes);
#else
		#pragma omp parallel for schedule (static)
		for (std::size_t j = 0; j < dim; j++) {
			std::memset(t[j], 0, n*sizeof(T));
		}
#endif
	}

	//! absolute value of components
	noinline void abs()
	{
		Timer __t("abs", false);

#ifdef USE_MANY_FFT
		#pragma omp parallel for schedule (static)
		for (std::size_t i = 0; i < ne; i++) {
			t[i] = std::abs(t[i]);
		}
#else
		#pragma omp parallel for schedule (static) collapse(2)
		for (std::size_t j = 0; j < dim; j++) {
			for (std::size_t i = 0; i < n; i++) {
				t[j][i] = std::abs(t[j][i]);
			}
		}
#endif
	}

	//! scale tensors by constant
	noinline void scale(T s)
	{
		Timer __t("scale", false);

		if (s == 1) return;

#ifdef USE_MANY_FFT
		#pragma omp parallel for schedule (static)
		for (std::size_t i = 0; i < ne; i++) {
			t[i] *= s;
		}
#else
		#pragma omp parallel for schedule (static) collapse(2)
		for (std::size_t j = 0; j < dim; j++) {
			for (std::size_t i = 0; i < n; i++) {
				t[j][i] *= s;
			}
		}
#endif
	}

	// compute r = x + a*(y - z)
	noinline void xpaymz(const TensorField<T,S>& x, T a, const TensorField<T,S>& y, const TensorField<T,S>& z)
	{
		Timer __t("xpaymz", false);

#ifdef USE_MANY_FFT
		#pragma omp parallel for schedule (static)
		for (std::size_t i = 0; i < ne; i++) {
			t[i] = x.t[i] + a*(y.t[i] - z.t[i]);
		}
#else
		#pragma omp parallel for schedule (static) collapse(2)
		for (std::size_t j = 0; j < dim; j++) {
			for (std::size_t i = 0; i < n; i++) {
				t[j][i] = x[j][i] + a*(y[j][i] - z[j][i]);
			}
		}
#endif
	}

	noinline void adjustResidual(const ublas::vector<T>& E, const TensorField<T,S>& z)
	{
		Timer __t("adjustResidual", false);

		#pragma omp parallel for schedule (static) collapse(2)
		for (std::size_t j = 0; j < dim; j++) {
			for (std::size_t i = 0; i < n; i++) {
				(*this)[j][i] += E[j] - z[j][i];
			}
		}
	}

	//! Set tensor values to constant value
	// NOTE: we also write to the padding, this does not matter
	noinline void setConstant(T c)
	{
		Timer __t("setConstant", false);

#ifdef USE_MANY_FFT
		#pragma omp parallel for schedule (static)
		for (std::size_t i = 0; i < ne; i++) {
			t[i] = c;
		}
#else
		#pragma omp parallel for schedule (static) collapse(2)
		for (std::size_t j = 0; j < dim; j++) {
			for (std::size_t i = 0; i < n; i++) {
				t[j][i] = c;
			}
		}
#endif
	}

	//! Set tensor values to constant vector of length dim
	// NOTE: we also write to the padding, this does not matter
	noinline void setConstant(const ublas::vector<T>& c)
	{
		Timer __t("setConstant", false);

		#pragma omp parallel for schedule (static) collapse(2)
		for (std::size_t j = 0; j < dim; j++) {
			for (std::size_t i = 0; i < n; i++) {
				(*this)[j][i] = c[j];
			}
		}
	}

	//! Set tensor values to constant at index
	// NOTE: we also write to the padding, this does not matter
	inline void setConstant(std::size_t index, const ublas::vector<T>& c)
	{
		for (std::size_t j = 0; j < dim; j++) {
			(*this)[j][index] = c[j];
		}
	}

	//! Set tensor values to constant 1 at index
	noinline void setOne(std::size_t index)
	{
		Timer __t("setOne", false);

		T* data = (*this)[index];
		#pragma omp parallel for schedule (static)
		for (std::size_t i = 0; i < n; i++) {
			data[i] = 1;
		}
	}

	//! Assign data at index i to tensor
	inline void assign(std::size_t i, T* E) const
	{
		for (std::size_t k = 0; k < dim; k++) {
			E[k] = (*this)[k][i];
		}
	}

	noinline ublas::vector<T> component_dot(const TensorField<T,S>& b)
	{
		Timer __t("component_dot", false);

		ublas::vector<T> a = ublas::zero_vector<T>(dim);

		#pragma omp parallel
		{
			ublas::vector<T> ap = ublas::zero_vector<T>(dim);
			
			#pragma omp for schedule (static) collapse(2)
			for (std::size_t ii = 0; ii < nx; ii++)
			{
				for (std::size_t jj = 0; jj < ny; jj++)
				{
					// calculate current index in tensor
					std::size_t k = ii*nyzp + jj*nzp;
				
					for (std::size_t kk = 0; kk < nz; kk++)
					{
						for (std::size_t j = 0; j < dim; j++) {
							ap[j] += (*this)[j][k]*b[j][k];
						}
						k++;
					}
				}
			}

			// perform reduction	
			#pragma omp critical
			{
				a += ap;
			}
		}
		
		a /= nxyz;
		return a;
	}

	noinline ublas::vector<T> component_norm()
	{
		Timer __t("component_norm", false);

		ublas::vector<T> a = component_dot(*this);

		for (std::size_t j = 0; j < dim; j++) {
			a[j] = std::sqrt(a[j]);
		}

		return a;
	}

/*
	noinline T dot(const TensorField<T,S>& b)
	{
		Timer __t("dot", false);

		T s = 0;

		#pragma omp parallel for reduction(+:s) schedule (static) collapse(2)
		for (std::size_t j = 0; j < dim; j++)
		{
			for (std::size_t ii = 0; ii < nx; ii++)
			{
				for (std::size_t jj = 0; jj < ny; jj++)
				{
					// calculate current index in tensor
					std::size_t k = ii*nyzp + jj*nzp;
				
					for (std::size_t kk = 0; kk < nz; kk++) {
						s += t[j][k]*b[j][k];
						k++;
					}
				}
			}
		}
		
		s /= nxyz;
		return s;
	}
*/

	//! Returns the average value for each component of the tensor
	noinline ublas::vector<T> average() const
	{
		Timer __t("average", false);

		ublas::vector<T> a = ublas::zero_vector<T>(dim);

		#pragma omp parallel
		{
			ublas::vector<T> ap = ublas::zero_vector<T>(dim);
			
			#pragma omp for nowait schedule (static) collapse(3)
			for (std::size_t j = 0; j < dim; j++)
			{
				for (std::size_t ii = 0; ii < nx; ii++)
				{
					for (std::size_t jj = 0; jj < ny; jj++)
					{
						// calculate current index in tensor
						std::size_t k = ii*nyzp + jj*nzp;
					
						for (std::size_t kk = 0; kk < nz; kk++) {
							ap[j] += (*this)[j][k];
							k++;
						}
					}
				}
			}

			// perform reduction	
			#pragma omp critical
			{
				a += ap;
			}
		}

		a /= nxyz;
		return a;
	}

	//! Returns the max value for each component of the tensor
	noinline ublas::vector<T> max() const
	{
		Timer __t("max", false);

		ublas::vector<T> a = ublas::zero_vector<T>(dim);
		for (std::size_t j = 0; j < dim; j++) {
			a[j] = -STD_INFINITY(T);
		}

		#pragma omp parallel
		{
			ublas::vector<T> ap = ublas::zero_vector<T>(dim);
			for (std::size_t j = 0; j < dim; j++) {
				ap[j] = -STD_INFINITY(T);
			}

			#pragma omp for nowait schedule (static) collapse(3)
			for (std::size_t j = 0; j < dim; j++)
			{
				for (std::size_t ii = 0; ii < nx; ii++)
				{
					for (std::size_t jj = 0; jj < ny; jj++)
					{
						// calculate current index in tensor
						std::size_t k = ii*nyzp + jj*nzp;
					
						for (std::size_t kk = 0; kk < nz; kk++) {
							ap[j] = std::max(ap[j], (*this)[j][k]);
							k++;
						}
					}
				}
			}

			// perform reduction	
			#pragma omp critical
			{
				for (std::size_t j = 0; j < dim; j++) {
					a[j] = std::max(a[j], ap[j]);
				}
			}
		}

		return a;
	}

	//! Return checksum for tensor field
	noinline long checksum() const
	{
		Timer __t("checksum", false);

		long a = 0;

		for (std::size_t j = 0; j < dim; j++)
		{
			for (std::size_t ii = 0; ii < nx; ii++)
			{
				for (std::size_t jj = 0; jj < ny; jj++)
				{
					std::size_t k = ii*nyzp + jj*nzp;
				
					for (std::size_t kk = 0; kk < nz; kk++) {
						a ^= ((*((long*)((*this)[j] + k))) * (long) (k+1));
						k++;
					}
				}
			}
		}

		return a;
	}
};


//! Base class for material laws
template<typename T>
class MaterialLaw
{
public:
	//! see https://stackoverflow.com/questions/827196/virtual-default-destructors-in-c/827205
	virtual ~MaterialLaw() {}

	inline T log(T x) const
	{
		T y = std::log(x);

		if (std::isnan(y)) {
			set_exception((boost::format("log(x): undefined for argument (x = %g)") % x).str());
		}

		return y;
	}

	inline T pow_minus_two_third(T x) const
	{
		T y = std::pow(x, -2.0/3.0);

		if (std::isnan(y)) {
			set_exception((boost::format("pow_minus_two_third(x): undefined for argument (x = %g)") % x).str());
		}

		return y;
	}

	//! compute energy at field index i for deformation gradient F
	virtual T W(std::size_t i, const T* F) const
	{
		BOOST_THROW_EXCEPTION(std::runtime_error("MaterialLaw energy not implemented"));
	}

	//! compute P = alpha*P(F) + gamma*P, where P(F) denotes the first Piola-Kirchhoff stress and F is the deformation gradient
	//! F and P are length 9
	virtual void PK1(std::size_t i, const T* F, T alpha, bool gamma, T* P) const = 0;

	//! compute cauchy stress
	void Cauchy(std::size_t i, const T* F, T alpha, bool gamma, T* sigma) const
	{
		const T c = 1/Tensor3x3<T>::det(F);
		Tensor3x3<T> P;
		PK1(i, F, c*alpha, false, P);

		// compute sigma = P*FT/detF
		Tensor3x3<T> PFT;
		PFT.mult_t(P, F);
		
		if (gamma) {
			for (std::size_t m = 0; m < 9; m++) {
				sigma[m] += PFT[m];
			}
		}
		else {
			for (std::size_t m = 0; m < 9; m++) {
				sigma[m]  = PFT[m];
			}
		}
	}

	//! compute PK1 derivative by finite differences
	void PK1_fd(std::size_t i, const T* F, T alpha, bool gamma, T* P, std::size_t dim, T eps) const
	{
		T Feps[9];

		// evaluate W(F)
		T W0 = W(i, F);

		if (!gamma) {
			std::memset(P, 0, sizeof(T)*dim);
		}
		
		for (std::size_t j = 0; j < dim; j++) {
			// compute Feps = F + delta_jk*eps
			for (std::size_t k = 0; k < dim; k++) {
				Feps[k] = F[k] + ((j == k) ? eps : (T)0);
			}
			// evaluate W(Feps)
			T Weps = W(i, Feps);
			// compute derivative

			P[j] += alpha*(Weps - W0)/eps;
		}
	}

	//! compute the linearized PK1 dP = alpha*dP(F)/dF(F) : W + gamma*dP for all n directions in W, the list of results dP must have same length as W
	//! F ist length 9, W and dP are length 9*n
	virtual void dPK1(std::size_t i, const T* F, T alpha, bool gamma, const T* W, T* dP, std::size_t n = 1) const = 0;

	//! compute directional derivative by finite differences
	void dPK1_fd(std::size_t i, const T* F, T alpha, bool gamma, const T* W, T* dP, std::size_t n, std::size_t dim, T eps) const
	{
		T Feps[9];
		T P[9];
		T Peps[9];

		// evaluate P(F)
		PK1(i, F, alpha, false, P);

		for (std::size_t m = 0; m < n; m++)
		{
			if (!gamma) {
				std::memset(dP, 0, sizeof(T)*dim);
			}
			
			for (std::size_t j = 0; j < dim; j++) {
				// compute Feps = F + delta_jk*eps
				for (std::size_t k = 0; k < dim; k++) {
					Feps[k] = F[k] + ((j == k) ? eps : (T)0);
				}
				// evaluate P(Feps)
				PK1(i, Feps, alpha, false, Peps);
				// compute derivative
				for (std::size_t k = 0; k < dim; k++) {
					dP[k] += ((Peps[k] - P[k])/eps)*W[j];
				}
			}

			dP += dim;
			W += dim;
		}
	}

	//! compute sigma = (C0 + C)(C0 - C)^{-1} epsilon, C0 = 2*mu_0
	//! if inv is true then: sigma = -(C0 - C)^{-1} epsilon
	// Eyre, D. J., & Milton, G. W. (1999). A fast numerical scheme for computing the response of composites using grid refinement. The European Physical Journal Applied Physics, 6(1), 41–47. doi:10.1051/epjap:1999150
	virtual void calcPolarization(std::size_t i, T mu_0, const ublas::vector<T>& F, ublas::vector<T>& P,
		std::size_t dim, bool inv) const
	{
		ublas::matrix<T> C(dim, dim), C2;

		// note: for linear problems independent of F
		const T* Id;
		if (dim == 3) {
			Id = TensorIdentity<T,3>::Id;
		}
		else if (dim == 6) {
			Id = TensorIdentity<T,6>::Id;
		}
		else {
			Id = TensorIdentity<T,9>::Id;
		}

		this->dPK1(i, F.data().begin(), 1, false, Id, C.data().begin(), dim);

		// L0 = 2*mu_0
		// C1 = C - 2*mu_0*ublas::identity_matrix<T>(dim);
		C2 = C + 2.0*mu_0*ublas::identity_matrix<T>(dim);

		// Solve C2*Q = F
		P = F;
		lapack::gesv(C2, P);

		if (!inv) {
			// P = C1*C2inv*F
			P = ublas::prod(C, P) - 2.0*mu_0*P;
		}
	}

	virtual std::string str() const = 0;

	virtual void readSettings(const ptree::ptree& pt) { }
};


//! Base class for Goldberg materials
template<typename T>
class GeneralGoldbergMaterialLaw : public MaterialLaw<T>
{
public:
	// energy function
	virtual T W(T J1, T J2, T J3) const = 0;

	// derivative w.r.t. invariants
	// J1 = J3^(-2/3)*trC
	// J2 = 0.5*J3^(-4/3)*(trC*trC - tr(C*C))
	// J3 = det F
	virtual T W1(T J1, T J2, T J3) const = 0;
	virtual T W2(T J1, T J2, T J3) const = 0;
	virtual T W3(T J1, T J2, T J3) const = 0;

	// second derivatives w.r.t. invariants
	virtual T W11(T J1, T J2, T J3) const = 0;
	virtual T W22(T J1, T J2, T J3) const = 0;
	virtual T W33(T J1, T J2, T J3) const = 0;

	// utility function for invariant calculation
	inline void calcInvarinats(const T* F, const SymTensor3x3<T>& C, T& J1, T& J2, T& J3) const
	{
		T trC = C[0] + C[1] + C[2];
		T trCC = C.dot(C);
		J3 = Tensor3x3<T>::det(F);

		if (J3 < 0) {
			set_exception("detected negative det(F).");
		}

		J1 = std::pow(J3, -2.0/3.0)*trC;
		J2 = 0.5*std::pow(J3, -4.0/3.0)*(trC*trC - trCC);
	}

	// utility function for invariant derivative calculation
	inline void calcInvarinatsDeriv(const T* F, const Tensor3x3<T> Finv,
		const SymTensor3x3<T>& C, const T* dF, const SymTensor3x3<T>& dC,
		T J1, T J2, T J3, T& dJ1, T& dJ2, T& dJ3) const
	{
		T trC = C[0] + C[1] + C[2];
		T dtrC = dC[0] + dC[1] + dC[2];
		T trCC = C.dot(C);
		T half_dtrCC = C.dot(dC);
		dJ3 = J3*Finv.dotT(dF);
		dJ1 = -2.0/3.0*std::pow(J3, -5.0/3.0)*dJ3*trC + std::pow(J3, -2.0/3.0)*dtrC;
		dJ2 = -2.0/3.0*std::pow(J3, -7.0/3.0)*dJ3*(trC*trC - trCC) + std::pow(J3, -4.0/3.0)*(trC*dtrC - half_dtrCC);
	}

	// energy function
	T W(std::size_t i, const T* F) const
	{
		SymTensor3x3<T> C;
		C.rightCauchyGreen(F);

		T J1, J2, J3;
		calcInvarinats(F, C, J1, J2, J3);

		return W(J1, J2, J3);
	}

	// Y += Y + a*F.T*S
	inline void add_trans_mult(T* Y, const Tensor3x3<T>& F, const SymTensor3x3<T>& S) const
	{
		Y[0] += ( F[0]*S[0] + F[8]*S[5] + F[7]*S[4] );
		Y[1] += ( F[5]*S[5] + F[1]*S[1] + F[6]*S[3] );
		Y[2] += ( F[4]*S[4] + F[3]*S[3] + F[2]*S[2] );
		Y[3] += ( F[5]*S[4] + F[1]*S[3] + F[6]*S[2] );
		Y[4] += ( F[0]*S[4] + F[8]*S[3] + F[7]*S[2] );
		Y[5] += ( F[0]*S[5] + F[8]*S[1] + F[7]*S[3] );
		Y[6] += ( F[4]*S[5] + F[3]*S[1] + F[2]*S[3] );
		Y[7] += ( F[4]*S[0] + F[3]*S[5] + F[2]*S[4] );
		Y[8] += ( F[5]*S[0] + F[1]*S[5] + F[6]*S[4] );
	}
	
	void PK1(std::size_t _i, const T* F, T alpha, bool gamma, T* dWdF) const
	{
		//this->PK1_fd(_i, F, alpha, gamma, dWdF, 9, 1e-5);
		//return;

		if (!gamma) {
			std::memset(dWdF, 0, sizeof(T)*9);
		}

		SymTensor3x3<T> C, Cinv;
		C.rightCauchyGreen(F);
		Cinv.inv(C);

		T J1, J2, J3;
		calcInvarinats(F, C, J1, J2, J3);

		SymTensor3x3<T> S;
		S.zero();

		T w1 = W1(J1, J2, J3);
		if (w1 != 0) {
			SymTensor3x3<T> devC;
			devC.dev(C);
			S.add(2*alpha*w1*std::pow(J3, -2.0/3.0), devC);
		}

		T w2 = W2(J1, J2, J3);
		if (w2 != 0) {
			SymTensor3x3<T> devCinv;
			devCinv.dev(Cinv);
			S.add(-2*alpha*w2*std::pow(J3, 2.0/3.0), devCinv);
		}

		T w3 = W3(J1, J2, J3);
		if (w3 != 0) {
			w3 *= J3*alpha;
			S[0] += w3;
			S[1] += w3;
			S[2] += w3;
		}

		// compute inverse of F
		Tensor3x3<T> Finv;
		Finv.inv(F);

		// compute dWdF += Finv.T*S
		add_trans_mult(dWdF, Finv, S);
	}

	void dPK1(std::size_t _i, const T* F, T alpha, bool gamma, const T* dF, T* dPdF, std::size_t n = 1) const
	{
		//this->dPK1_fd(_i, F, alpha, gamma, dF, dPdF, n, 9, 1e-5);
		//return;

		SymTensor3x3<T> S, dS, C, Cinv, dC, devC, devCinv, ddevC, dCinv, ddevCinv;
		Tensor3x3<T> Finv, dFinv;

		if (!gamma) {
			std::memset(dPdF, 0, sizeof(T)*9*n);
		}

		C.rightCauchyGreen(F);
		Cinv.inv(C);
		devC.dev(C);
		devCinv.dev(Cinv);

		T J1, J2, J3;
		calcInvarinats(F, C, J1, J2, J3);

		// compute inverse of F
		Finv.inv(F);

		T w1 = alpha*W1(J1, J2, J3);
		T w2 = alpha*W2(J1, J2, J3);
		T w3 = alpha*W3(J1, J2, J3);
		T w11 = alpha*W11(J1, J2, J3);
		T w22 = alpha*W22(J1, J2, J3);
		T w33 = alpha*W33(J1, J2, J3);

		T dJ1, dJ2, dJ3, dw1, dw2, dw3;

		S.zero();
		if (w1 != 0) {
			S.add(2*w1*std::pow(J3, -2.0/3.0), devC);
		}
		if (w2 != 0) {
			S.add(-2*w2*std::pow(J3, 2.0/3.0), devCinv);
		}
		if (w3 != 0) {
			T a = w3*J3;
			S[0] += a;
			S[1] += a;
			S[2] += a;
		}

		for (std::size_t m = 0; m < n; m++)
		{
			dS.zero();

			dFinv.invDeriv(F, Finv, dF);
			dC.rightCauchyGreenDeriv(F, dF);

			calcInvarinatsDeriv(F, Finv, C, dF, dC, J1, J2, J3, dJ1, dJ2, dJ3);
			dw1 = w11*dJ1;
			dw2 = w22*dJ2;
			dw3 = w33*dJ3;

			if (w1 != 0 || dw1 != 0) {
				dS.add(-4.0/3.0*w1*std::pow(J3, -5.0/3.0)*dJ3 + 2*dw1*std::pow(J3, -2.0/3.0), devC);
				if (w1 != 0) {
					ddevC.devDeriv(devC, dC);
					dS.add(2*w1*std::pow(J3, -2.0/3.0), ddevC);
				}
			}

			if (w2 != 0 || dw2 != 0) {
				dS.add(-4.0/3.0*w2*std::pow(J3, -1.0/3.0)*dJ3 - 2*dw2*std::pow(J3, 2.0/3.0), devCinv);
				if (w2 != 0) {
					dCinv.invDeriv(C, Cinv, dC);
					ddevCinv.devDeriv(devCinv, dCinv);
					dS.add(-2*w2*std::pow(J3, 2.0/3.0), ddevCinv);
				}
			}

			T da = dw3*J3 + w3*dJ3;
			dS[0] += da;
			dS[1] += da;
			dS[2] += da;

			add_trans_mult(dPdF, dFinv, S);
			add_trans_mult(dPdF, Finv, dS);

			dF += 9;
			dPdF += 9;
		}
	}
};

//! Goldberg Matrix1 material
template<typename T>
class Matrix1GoldbergMaterialLaw : public GeneralGoldbergMaterialLaw<T>
{
public:
	// parameters
	T m1, m2;

	Matrix1GoldbergMaterialLaw() : m1(1.0), m2(10.0) { }

	// read settings from ptree
	void readSettings(const ptree::ptree& pt) {
		const ptree::ptree& attr = pt.get_child("<xmlattr>", empty_ptree);
		m1 = pt_get<T>(attr, "m1", m1);
		m2 = pt_get<T>(attr, "m2", m2);
	}

	// energy function
	T W(T J1, T J2, T J3) const { return m1*(J1 - 3.0) + m2*((J3 + 1.0/J3) - 2.0); }

	// derivatives w.r.t. invariants
	T W1(T J1, T J2, T J3) const { return m1; }
	T W2(T J1, T J2, T J3) const { return 0.0; }
	T W3(T J1, T J2, T J3) const { return m2*(1.0 - 1.0/(J3*J3)); }

	// second derivatives w.r.t. invariants
	T W11(T J1, T J2, T J3) const { return 0.0; }
	T W22(T J1, T J2, T J3) const { return 0.0; }
	T W33(T J1, T J2, T J3) const { return 2.0*m2/(J3*J3*J3); }

	std::string str() const {
		return (boost::format("Goldberg Matrix1 m1=%g m2=%g") % m1 % m2).str();
	}
};


//! Goldberg Matrix2 material
template<typename T>
class Matrix2GoldbergMaterialLaw : public GeneralGoldbergMaterialLaw<T>
{
public:
	// parameters
	T m1, m2, m3, m4;

	Matrix2GoldbergMaterialLaw() : m1(0.5), m2(0.1), m3(1.0), m4(5.0) { }

	// read settings from ptree
	void readSettings(const ptree::ptree& pt) {
		const ptree::ptree& attr = pt.get_child("<xmlattr>", empty_ptree);
		m1 = pt_get<T>(attr, "m1", m1);
		m2 = pt_get<T>(attr, "m2", m2);
		m3 = pt_get<T>(attr, "m3", m3);
		m4 = pt_get<T>(attr, "m4", m4);
	}

	// energy function
	T W(T J1, T J2, T J3) const {
		T J1m3 = J1 - 3.0;
		return (m1 + (m2 + m3*J1m3)*J1m3)*J1m3 + m4*((J3 + 1.0/J3) - 2.0);
	}

	// derivatives w.r.t. invariants
	T W1(T J1, T J2, T J3) const { T J1m3 = J1 - 3.0; return m1 + (2*m2 + 3*m3*J1m3)*J1m3; }
	T W2(T J1, T J2, T J3) const { return 0.0; }
	T W3(T J1, T J2, T J3) const { return m4*(1.0 - 1.0/(J3*J3)); }

	// second derivatives w.r.t. invariants
	T W11(T J1, T J2, T J3) const { return 2.0*m2 + 6.0*m3*(J1 - 3.0); }
	T W22(T J1, T J2, T J3) const { return 0.0; }
	T W33(T J1, T J2, T J3) const { return 2.0*m4/(J3*J3*J3); }

	std::string str() const {
		return (boost::format("Goldberg Matrix2 m1=%g m2=%g m3=%g") % m1 % m2 % m3).str();
	}
};


//! Goldberg Matrix3 material
template<typename T>
class Matrix3GoldbergMaterialLaw : public GeneralGoldbergMaterialLaw<T>
{
public:
	// parameters
	T m1, m2;

	Matrix3GoldbergMaterialLaw() : m1(1.0), m2(10.0) { }

	// read settings from ptree
	void readSettings(const ptree::ptree& pt) {
		const ptree::ptree& attr = pt.get_child("<xmlattr>", empty_ptree);
		m1 = pt_get<T>(attr, "m1", m1);
		m2 = pt_get<T>(attr, "m2", m2);
	}

	// energy function
	T W(T J1, T J2, T J3) const {
		T J3p5 = J3*J3*J3*J3*J3;
		return m1*(J1 - 3.0) + (m2/50.0)*((J3p5 + 1.0/J3p5) - 2.0);
	}

	// derivatives w.r.t. invariants
	T W1(T J1, T J2, T J3) const { return m1; }
	T W2(T J1, T J2, T J3) const { return 0.0; }
	T W3(T J1, T J2, T J3) const { T J3p4 = J3*J3*J3*J3; return (m2/10.0)*(J3p4 - 1.0/(J3p4*J3*J3)); }

	// second derivatives w.r.t. invariants
	T W11(T J1, T J2, T J3) const { return 0.0; }
	T W22(T J1, T J2, T J3) const { return 0.0; }
	T W33(T J1, T J2, T J3) const { T J3p3 = J3*J3*J3; return (m2/10.0)*(4.0*J3p3 + 6.0/(J3p3*J3p3*J3)); }

	std::string str() const {
		return (boost::format("Goldberg Matrix3 m1=%g m2=%g") % m1 % m2).str();
	}
};


//! Goldberg Matrix4 material
template<typename T>
class Matrix4GoldbergMaterialLaw : public GeneralGoldbergMaterialLaw<T>
{
public:
	// parameters
	T m1, m2, m3, m4;

	Matrix4GoldbergMaterialLaw() : m1(0.5), m2(1.0), m3(3.0), m4(50.0) { }

	// read settings from ptree
	void readSettings(const ptree::ptree& pt) {
		const ptree::ptree& attr = pt.get_child("<xmlattr>", empty_ptree);
		m1 = pt_get<T>(attr, "m1", m1);
		m2 = pt_get<T>(attr, "m2", m2);
		m3 = pt_get<T>(attr, "m3", m3);
		m4 = pt_get<T>(attr, "m4", m4);
	}

	// energy function
	T W(T J1, T J2, T J3) const {
		T J1m3 = J1 - 3.0;
		T J3p5 = J3*J3*J3*J3*J3;
		return m1*J1m3 + m2*J1m3*J1m3 + m3*J1m3*J1m3*J1m3 + (m4/50.0)*((J3p5 + 1.0/J3p5) - 2.0);
	}

	// derivatives w.r.t. invariants
	T W1(T J1, T J2, T J3) const { T J1m3 = J1 - 3.0; return m1 + 2.0*m2*J1m3 + 3.0*m3*J1m3*J1m3; }
	T W2(T J1, T J2, T J3) const { return 0.0; }
	T W3(T J1, T J2, T J3) const { T J3p4 = J3*J3*J3*J3; return (m4/10.0)*(J3p4 - 1.0/(J3p4*J3*J3)); }

	// second derivatives w.r.t. invariants
	T W11(T J1, T J2, T J3) const { return 2.0*m2 + 6.0*m3*(J1 - 3.0); }
	T W22(T J1, T J2, T J3) const { return 0.0; }
	T W33(T J1, T J2, T J3) const { T J3p3 = J3*J3*J3; return (m4/10.0)*(4.0*J3p3 + 6.0/(J3p3*J3p3*J3)); }

	std::string str() const {
		return (boost::format("Goldberg Matrix4 m1=%g m2=%g m3=%g m4=%g") % m1 % m2 % m3 % m4).str();
	}
};


//! Goldberg Fiber1 material
template<typename T>
class Fiber1GoldbergMaterialLaw : public GeneralGoldbergMaterialLaw<T>
{
public:
	// parameters
	T f1, f2;

	Fiber1GoldbergMaterialLaw() : f1(20.0), f2(200.0) { }

	// read settings from ptree
	void readSettings(const ptree::ptree& pt) {
		const ptree::ptree& attr = pt.get_child("<xmlattr>", empty_ptree);
		f1 = pt_get<T>(attr, "f1", f1);
		f2 = pt_get<T>(attr, "f2", f2);
	}

	// energy function
	T W(T J1, T J2, T J3) const { return f1*(J1 - 3.0) + f2*((J3 + 1.0/J3) - 2.0); }

	// derivatives w.r.t. invariants
	T W1(T J1, T J2, T J3) const { return f1; }
	T W2(T J1, T J2, T J3) const { return 0.0; }
	T W3(T J1, T J2, T J3) const { return f2*(1.0 - 1.0/(J3*J3)); }

	// second derivatives w.r.t. invariants
	T W11(T J1, T J2, T J3) const { return 0.0; }
	T W22(T J1, T J2, T J3) const { return 0.0; }
	T W33(T J1, T J2, T J3) const { return 2.0*f2/(J3*J3*J3); }

	std::string str() const {
		return (boost::format("Goldberg Fiber1 f1=%g f2=%g") % f1 % f2).str();
	}
};


//! Goldberg Fiber2 material
template<typename T>
class Fiber2GoldbergMaterialLaw : public GeneralGoldbergMaterialLaw<T>
{
public:
	// parameters
	T f1, f2, f3;

	Fiber2GoldbergMaterialLaw() : f1(0.8), f2(15.0), f3(500.0) { }

	// read settings from ptree
	void readSettings(const ptree::ptree& pt) {
		const ptree::ptree& attr = pt.get_child("<xmlattr>", empty_ptree);
		f1 = pt_get<T>(attr, "f1", f1);
		f2 = pt_get<T>(attr, "f2", f2);
		f3 = pt_get<T>(attr, "f3", f3);
	}

	// energy function
	T W(T J1, T J2, T J3) const { 
		T c = 1.0 - (J1 - 3.0)/f1;
		if (c <= 0) {
			set_exception("detected negative argument for log");
		}
		return -0.5*f1*f2*std::log(c) + f3*((J3 + 1.0/J3) - 2.0);
	}

	// derivatives w.r.t. invariants
	T W1(T J1, T J2, T J3) const { return 0.5*f1*f2/(f1 + (3.0 - J1)); }
	T W2(T J1, T J2, T J3) const { return 0.0; }
	T W3(T J1, T J2, T J3) const { return f3*(1.0 - 1.0/(J3*J3)); }

	// second derivatives w.r.t. invariants
	T W11(T J1, T J2, T J3) const { T c = (f1 + (3.0 - J1)); return 0.5*f1*f2/(c*c); }
	T W22(T J1, T J2, T J3) const { return 0.0; }
	T W33(T J1, T J2, T J3) const { return 2.0*f3/(J3*J3*J3); }

	std::string str() const {
		return (boost::format("Goldberg Fiber2 f1=%g f2=%g f3=%g") % f1 % f2 % f3).str();
	}
};


//! Goldberg Fiber3 material
template<typename T>
class Fiber3GoldbergMaterialLaw : public GeneralGoldbergMaterialLaw<T>
{
public:
	// parameters
	T f1, f2, f3, f4;

	Fiber3GoldbergMaterialLaw() : f1(1.0), f2(0.02), f3(100.0), f4(500.0) { }

	// read settings from ptree
	void readSettings(const ptree::ptree& pt) {
		const ptree::ptree& attr = pt.get_child("<xmlattr>", empty_ptree);
		f1 = pt_get<T>(attr, "f1", f1);
		f2 = pt_get<T>(attr, "f2", f2);
		f3 = pt_get<T>(attr, "f3", f3);
		f4 = pt_get<T>(attr, "f4", f4);
	}

	// energy function
	T W(T J1, T J2, T J3) const { return f1*J1 + f2*J1*J1*J1*J1 + f3*std::sqrt(J2) + f4*((J3 + 1.0/J3) - 2.0); }

	// derivatives w.r.t. invariants
	T W1(T J1, T J2, T J3) const { return f1 + 4.0*f2*J1*J1*J1; }
	T W2(T J1, T J2, T J3) const { return 0.5*f3/std::sqrt(J2); }
	T W3(T J1, T J2, T J3) const { return 1.0*f4*(1.0 - 1.0/(J3*J3)); }

	// second derivatives w.r.t. invariants
	T W11(T J1, T J2, T J3) const { return 12.0*f2*J1*J1; }
	T W22(T J1, T J2, T J3) const { return -0.25*std::pow(J2, -1.5); }
	T W33(T J1, T J2, T J3) const { return 2.0*f4/(J3*J3*J3); }

	std::string str() const {
		return (boost::format("Goldberg Fiber3 f1=%g f2=%g f3=%g f4=%g") % f1 % f2 % f3 % f4).str();
	}
};


//! Goldberg Fiber4 material
template<typename T>
class Fiber4GoldbergMaterialLaw : public GeneralGoldbergMaterialLaw<T>
{
public:
	// parameters
	T f1, f2;

	Fiber4GoldbergMaterialLaw() : f1(20.0), f2(200.0) { }

	// read settings from ptree
	void readSettings(const ptree::ptree& pt) {
		const ptree::ptree& attr = pt.get_child("<xmlattr>", empty_ptree);
		f1 = pt_get<T>(attr, "f1", f1);
		f2 = pt_get<T>(attr, "f2", f2);
	}

	// energy function
	T W(T J1, T J2, T J3) const {
		T J3p5 = J3*J3*J3*J3*J3;
		return f1*(J1 - 3.0) + (f2/50.0)*((J3p5 + 1.0/J3p5) - 2.0);
	}

	// derivatives w.r.t. invariants
	T W1(T J1, T J2, T J3) const { return f1; }
	T W2(T J1, T J2, T J3) const { return 0.0; }
	T W3(T J1, T J2, T J3) const { T J3p4 = J3*J3*J3*J3; return f2/10.0*(J3p4 - 1.0/(J3p4*J3*J3)); }

	// second derivatives w.r.t. invariants
	T W11(T J1, T J2, T J3) const { return 0.0; }
	T W22(T J1, T J2, T J3) const { return 0.0; }
	T W33(T J1, T J2, T J3) const { T J3p3 = J3*J3*J3; return f2/10.0*(4.0*J3p3 + 6.0/(J3p3*J3p3*J3)); }

	std::string str() const {
		return (boost::format("Goldberg Fiber4 f1=%g f2=%g") % f1 % f2).str();
	}
};


//! Goldberg Fiber5 material
template<typename T>
class Fiber5GoldbergMaterialLaw : public GeneralGoldbergMaterialLaw<T>
{
public:
	// parameters
	T f1, f2, f3;

	Fiber5GoldbergMaterialLaw() : f1(3.5), f2(2.0), f3(500.0) { }

	// read settings from ptree
	void readSettings(const ptree::ptree& pt) {
		const ptree::ptree& attr = pt.get_child("<xmlattr>", empty_ptree);
		f1 = pt_get<T>(attr, "f1", f1);
		f2 = pt_get<T>(attr, "f2", f2);
		f3 = pt_get<T>(attr, "f3", f3);
	}

	// energy function
	T W(T J1, T J2, T J3) const { return f1*(std::exp(f2*(J1 - 3.0)) - 1.0) + f3*(J3 + 1.0/J3 - 2.0); }

	// derivatives w.r.t. invariants
	T W1(T J1, T J2, T J3) const { return f1*f2*std::exp(f2*(J1 - 3.0)); }
	T W2(T J1, T J2, T J3) const { return 0.0; }
	T W3(T J1, T J2, T J3) const { return f3*(1.0 - 1.0/(J3*J3)); }

	// second derivatives w.r.t. invariants
	T W11(T J1, T J2, T J3) const { return f1*f2*f2*std::exp(f2*(J1 - 3.0)); }
	T W22(T J1, T J2, T J3) const { return 0.0; }
	T W33(T J1, T J2, T J3) const { return 2.0*f3/(J3*J3*J3); }

	std::string str() const {
		return (boost::format("Goldberg Fiber5 f1=%g f2=%g f3=%g") % f1 % f2 % f3).str();
	}
};


//! Goldberg Fiber6 material
template<typename T>
class Fiber6GoldbergMaterialLaw : public GeneralGoldbergMaterialLaw<T>
{
public:
	// parameters
	T f1, f2, f3;

	Fiber6GoldbergMaterialLaw() : f1(3.5), f2(4.0), f3(500.0) { }

	// read settings from ptree
	void readSettings(const ptree::ptree& pt) {
		const ptree::ptree& attr = pt.get_child("<xmlattr>", empty_ptree);
		f1 = pt_get<T>(attr, "f1", f1);
		f2 = pt_get<T>(attr, "f2", f2);
		f3 = pt_get<T>(attr, "f3", f3);
	}

	// energy function
	T W(T J1, T J2, T J3) const {
		T J3p5 = J3*J3*J3*J3*J3;
		return f1*(std::exp(f2*(J1 - 3.0)) - 1.0) + (f3/50.0)*(J3p5 + 1.0/J3p5 - 2.0);
	}

	// derivatives w.r.t. invariants
	T W1(T J1, T J2, T J3) const { return f1*f2*std::exp(f2*(J1 - 3.0)); }
	T W2(T J1, T J2, T J3) const { return 0.0; }
	T W3(T J1, T J2, T J3) const { T J3p4 = J3*J3*J3*J3; return (f3/10.0)*(J3p4 - 1.0/(J3p4*J3*J3)); }

	// second derivatives w.r.t. invariants
	T W11(T J1, T J2, T J3) const { return f1*f2*f2*std::exp(f2*(J1 - 3.0)); }
	T W22(T J1, T J2, T J3) const { return 0.0; }
	T W33(T J1, T J2, T J3) const { T J3p3 = J3*J3*J3; return (f3/10.0)*(4.0*J3p3 + 6.0/(J3p3*J3p3*J3)); }

	std::string str() const {
		return (boost::format("Goldberg Fiber6 f1=%g f2=%g f3=%g") % f1 % f2 % f3).str();
	}
};


//! Class for validating Goldberg material derivatives
template<typename T>
class CheckGoldbergMaterialLaw : public GeneralGoldbergMaterialLaw<T>
{
public:
	// parameters
	int coef;

	CheckGoldbergMaterialLaw(int coef) : coef(coef) { }

	// energy function
	T W(T J1, T J2, T J3) const { return (coef == 1 ? J1 : 0.0) + (coef == 2 ? J2 : 0.0) + (coef == 3 ? J3 : 0.0); }

	// derivatives w.r.t. invariants
	T W1(T J1, T J2, T J3) const { return (coef == 1 ? 1.0 : 0.0); }
	T W2(T J1, T J2, T J3) const { return (coef == 2 ? 1.0 : 0.0); }
	T W3(T J1, T J2, T J3) const { return (coef == 3 ? 1.0 : 0.0); }

	// second derivatives w.r.t. invariants
	T W11(T J1, T J2, T J3) const { return 0; }
	T W22(T J1, T J2, T J3) const { return 0; }
	T W33(T J1, T J2, T J3) const { return 0; }

	std::string str() const {
		return (boost::format("CheckGoldbergMaterialLaw coef=%d") % coef).str();
	}
};


//! Linear ainsotropic material law given by symmetric 3x3 matrix
template<typename T>
class MatrixLinearAnisotropicMaterialLaw : public MaterialLaw<T>
{
public:
	T c11, c22, c33, c23, c13, c12;

	MatrixLinearAnisotropicMaterialLaw() : c11(1.0), c22(1.0), c33(1.0), c23(0.0), c13(0.0), c12(0.0) { }

	// read settings from ptree
	void readSettings(const ptree::ptree& pt) {
		const ptree::ptree& attr = pt.get_child("<xmlattr>", empty_ptree);
		c11 = pt_get<T>(attr, "c11", c11);
		c22 = pt_get<T>(attr, "c22", c22);
		c33 = pt_get<T>(attr, "c33", c33);
		c23 = pt_get<T>(attr, "c23", c23);
		c13 = pt_get<T>(attr, "c13", c13);
		c12 = pt_get<T>(attr, "c12", c12);
	}

	T W(std::size_t i, const T* E) const
	{
		Tensor3<T> S;
		PK1(i, E, 1, false, S);
		return 0.5*S.dot(E);
	}

	void PK1(std::size_t _i, const T* E, T alpha, bool gamma, T* S) const
	{
		// compute stress S
		#define PK1_OP(OP) \
			S[0] OP alpha*(c11*E[0] + c12*E[1] + c13*E[2]); \
			S[1] OP alpha*(c12*E[0] + c22*E[1] + c23*E[2]); \
			S[2] OP alpha*(c13*E[0] + c23*E[1] + c33*E[2]);
		if (gamma) {
			PK1_OP(+=)
		}
		else {
			PK1_OP(=)
		}
		#undef PK1_OP
	}

	void dPK1(std::size_t _i, const T* E, T alpha, bool gamma, const T* W, T* dS, std::size_t n = 1) const
	{
		for (std::size_t m = 0; m < n; m++)
		{
			// compute stress S
			#define PK1_OP(OP) \
				dS[0] OP alpha*(c11*W[0] + c12*W[1] + c13*W[2]); \
				dS[1] OP alpha*(c12*W[0] + c22*W[1] + c23*W[2]); \
				dS[2] OP alpha*(c13*W[0] + c23*W[1] + c33*W[2]);
			if (gamma) {
				PK1_OP(+=)
			}
			else {
				PK1_OP(=)
			}
			#undef PK1_OP

			W += 3;
			dS += 3;
		}
	}

	std::string str() const
	{
		return (boost::format("matrix linear anisotropic c11=%g c22=%g c33=%g c23=%g c13=%g c12=%g") % c11 % c22 % c33 % c23 % c13 % c12).str();
	}
};


//! Linear isotropic material law given by single scalar value
template<typename T>
class ScalarLinearIsotropicMaterialLaw : public MaterialLaw<T>
{
public:
	T mu;
	std::size_t dim;

	ScalarLinearIsotropicMaterialLaw(std::size_t dim) : mu(1.0), dim(dim) { }

	// read settings from ptree
	void readSettings(const ptree::ptree& pt) {
		const ptree::ptree& attr = pt.get_child("<xmlattr>", empty_ptree);
		mu = pt_get<T>(attr, "mu", mu);
	}

	T W(std::size_t i, const T* E) const
	{
		Tensor3<T> S;
		PK1(i, E, 1, false, S);
		return 0.5*S.dot(E);
	}

	void PK1(std::size_t _i, const T* E, T alpha, bool gamma, T* S) const
	{
		const T alpha_mu = alpha*mu;

		// compute stress S
		#define PK1_OP(OP) \
			for (std::size_t m = 0; m < dim; m++) { \
				S[m] OP E[m]*alpha_mu; \
			}
		if (gamma) {
			PK1_OP(+=)
		}
		else {
			PK1_OP(=)
		}
		#undef PK1_OP
	}

	void dPK1(std::size_t _i, const T* E, T alpha, bool gamma, const T* W, T* dS, std::size_t n = 1) const
	{
		const T alpha_mu = alpha*mu;

		for (std::size_t m = 0; m < n; m++)
		{
			// compute stress S
			#define PK1_OP(OP) \
				for (std::size_t s = 0; s < dim; s++) { \
					dS[s] OP W[s]*alpha_mu; \
				}
			if (gamma) {
				PK1_OP(+=)
			}
			else {
				PK1_OP(=)
			}
			#undef PK1_OP

			W += dim;
			dS += dim;
		}
	}

	std::string str() const
	{
		return (boost::format("scalar linear isotropic mu=%g") % mu).str();
	}
};


//! Linear isotropic material law for elasticity defined by two Lame parameters
template<typename T>
class LinearIsotropicMaterialLaw : public MaterialLaw<T>
{
public:
	// Lame parameters
	T mu, lambda;

	// read settings from ptree
	void readSettings(const ptree::ptree& pt) {
		Material<T, 3> m;
		m.readSettings(pt);
		mu = m.mu;
		lambda = m.lambda;
	}

	T W(std::size_t i, const T* E) const
	{
		SymTensor3x3<T> S;
		PK1(i, E, 1, false, S);
		return 0.5*S.dot(E);
	}

	void PK1(std::size_t _i, const T* E, T alpha, bool gamma, T* S) const
	{
		// compute stress S
		// set S = 2*mu*E + lambda*tr(E)*I
		const T two_mu = 2*alpha*mu;
		const T lambda_tr_E = alpha*lambda*(E[0] + E[1] + E[2]);

		#define PK1_OP(OP) \
			S[0] OP E[0]*two_mu + lambda_tr_E; \
			S[1] OP E[1]*two_mu + lambda_tr_E; \
			S[2] OP E[2]*two_mu + lambda_tr_E; \
			S[3] OP E[3]*two_mu; \
			S[4] OP E[4]*two_mu; \
			S[5] OP E[5]*two_mu;
		if (gamma) {
			PK1_OP(+=)
		}
		else {
			PK1_OP(=)
		}
		#undef PK1_OP
	}

	void dPK1(std::size_t _i, const T* E, T alpha, bool gamma, const T* W, T* dS, std::size_t n = 1) const
	{
		const T two_mu = 2*alpha*mu;

		for (std::size_t m = 0; m < n; m++)
		{
			// compute stress S
			// set dS = 2*mu*W + lambda*tr(W)*I
			const T lambda_tr_W = alpha*lambda*(W[0] + W[1] + W[2]);
			#define PK1_OP(OP) \
				dS[0] OP W[0]*two_mu + lambda_tr_W; \
				dS[1] OP W[1]*two_mu + lambda_tr_W; \
				dS[2] OP W[2]*two_mu + lambda_tr_W; \
				dS[3] OP W[3]*two_mu; \
				dS[4] OP W[4]*two_mu; \
				dS[5] OP W[5]*two_mu;
			if (gamma) {
				PK1_OP(+=)
			}
			else {
				PK1_OP(=)
			}
			#undef PK1_OP

			W += 6;
			dS += 6;
		}
	}

	void calcPolarization(std::size_t i, T mu_0, const ublas::vector<T>& F, ublas::vector<T>& P,
		std::size_t dim, bool inv) const
	{
		// C = 2*mu*Id + lambda*I*I

		// L0 = 2*mu_0
		// C1 = 2*(mu-mu_0)*Id + lambda*II
		// C2 = 2*(mu+mu_0)*Id + lambda*II

		// The inverse of C2 is:
		// Simplify[Inverse[{{2*mu + L, L, L, 0, 0, 0}, {L, 2*mu + L, L, 0, 0, 0}, {L, L, 2*mu + L, 0, 0, 0}, {0, 0, 0, 2*mu, 0, 0}, {0, 0, 0, 0, 2*mu, 0}, {0, 0, 0, 0, 0, 2*mu}}]]
		// Simplify[(L + mu)/(3 L mu + 2 mu^2) - 1/(2 mu)]
		// inv(C2) = 1/(2*m)*Id - (lambda/(2*m*(3*lambda + 2*m)))*II
		// where m = mu+mu_0

		// Solve C2*P = F

		T m = 2.0*(mu+mu_0);
		T a = 1.0/m;
		T b = lambda/(m*(3.0*lambda + m));
		T tr_F = F[0] + F[1] + F[2];

		P[0] = a*F[0] - b*tr_F;
		P[1] = a*F[1] - b*tr_F;
		P[2] = a*F[2] - b*tr_F;
		P[3] = a*F[3];
		P[4] = a*F[4];
		P[5] = a*F[5];

		if (!inv) {
			// P = C1*C2inv*F
			m = 2.0*(mu-mu_0);
			T tr_P = P[0] + P[1] + P[2];
			P[0] = m*P[0] + lambda*tr_P;
			P[1] = m*P[1] + lambda*tr_P;
			P[2] = m*P[2] + lambda*tr_P;
			P[3] = m*P[3];
			P[4] = m*P[4];
			P[5] = m*P[5];
		}
	}


	std::string str() const
	{
		return (boost::format("linear isotropic lambda=%g mu=%g") % lambda % mu).str();
	}
};


//! Linear transversely isotropic material law for elasticity defined by five parameters
template<typename T, int DIM>
class LinearTransverselyIsotropicMaterialLaw : public MaterialLaw<T>
{
public:
	// material parameters
	T two_mu, lambda, alpha, beta, two_dmu;

	// direction of anisotropy field
	boost::shared_ptr< TensorField<T> > orientation;
	ublas::c_vector<T, DIM> a;
	bool have_a;

	// read settings from ptree
	void readSettings(const ptree::ptree& pt)
	{
		const ptree::ptree& attr = pt.get_child("<xmlattr>", empty_ptree);
		T E, E_a, nu, G_a, nu_ab;
		try {
			E     = pt_get<T>(attr, "E");
			nu    = pt_get<T>(attr, "nu");
			E_a   = pt_get<T>(attr, "E_a");
			G_a   = pt_get<T>(attr, "G_a");
			nu_ab = pt_get<T>(attr, "nu_a");
			read_vector(attr, a, "ax", "ay", "az", (T)0, (T)0, (T)0);
		}
		catch (boost::exception& e) {
			BOOST_THROW_EXCEPTION(std::runtime_error(
				"A transversal isotropic material requires the parameters E, nu, E_a, G_a and nu_a."));
		}

		T G = E/(2*(nu + 1));
		T nu_ba = E/E_a*nu_ab;
		T D = (1 + nu)*(1 - nu - 2*nu_ab*nu_ba);
		
		alpha = E*(nu_ab*(1 + nu - nu_ba) - nu)/D;
		beta = (E_a*(1-nu*nu) - E*(nu + nu_ab*nu_ba) - 2*E*(nu_ab*(1+nu-nu_ba)-nu))/D - 4*G_a + 2*G;
		lambda = E*(nu + nu_ab*nu_ba)/D;
		two_mu = 2*G;
		two_dmu = 2*(G_a - G);
		have_a = ublas::norm_2(a) != 0;
	}

	LinearTransverselyIsotropicMaterialLaw(boost::shared_ptr< TensorField<T> > orientation)
	{
		this->orientation = orientation;
	}

	T W(std::size_t i, const T* E) const
	{
		SymTensor3x3<T> S;
		PK1(i, E, 1, false, S);
		return 0.5*S.dot(E);
	}

	void PK1(std::size_t i, const T* E, T alpha, bool gamma, T* S) const
	{
		// compute stress S
		// set S = 2*mu*E + lambda*tr(E)*I
		const T tr_E = E[0] + E[1] + E[2];

		T a[3]; // direction of anisotropy
		if (have_a) {
			a[0] = this->a[0];
			a[1] = this->a[1];
			a[2] = this->a[2];
		}
		else {
			a[0] = (*this->orientation)[0][i];
			a[1] = (*this->orientation)[1][i];
			a[2] = (*this->orientation)[2][i];
		}

		SymTensor3x3<T> A;
		SymTensor3x3<T> AE_EA;
		A.outer(a);
		AE_EA.sym_prod(A, E);

		T aEa = A.dot(E);
		T c_I = alpha*(lambda*tr_E + this->alpha*aEa);
		T c_E = alpha*two_mu;
		T c_A = alpha*(this->alpha*tr_E + beta*aEa);
		T c_AE = alpha*two_dmu;

		#define PK1_OP(OP) \
			S[0] OP c_E*E[0] + c_I + c_A*A[0] + c_AE*AE_EA[0]; \
			S[1] OP c_E*E[1] + c_I + c_A*A[1] + c_AE*AE_EA[1]; \
			S[2] OP c_E*E[2] + c_I + c_A*A[2] + c_AE*AE_EA[2]; \
			S[3] OP c_E*E[3]       + c_A*A[3] + c_AE*AE_EA[3]; \
			S[4] OP c_E*E[4]       + c_A*A[4] + c_AE*AE_EA[4]; \
			S[5] OP c_E*E[5]       + c_A*A[5] + c_AE*AE_EA[5];
		if (gamma) {
			PK1_OP(+=)
		}
		else {
			PK1_OP(=)
		}
		#undef PK1_OP
	}

	void dPK1(std::size_t i, const T* E, T alpha, bool gamma, const T* W, T* dS, std::size_t n = 1) const
	{
		for (std::size_t m = 0; m < n; m++)
		{
			// TODO: inefficient
			PK1(m, W, alpha, gamma, dS);

			W += 6;
			dS += 6;
		}
	}

	std::string str() const
	{
		return (boost::format("linear transversely isotropic lambda=%g mu=%g mu_a=%g alpha=%g beta=%g") % lambda % (0.5*two_mu) % (0.5*(two_mu+two_dmu)) % alpha % beta).str();
	}
};


//! Saint Venant-Kirchhoff material for hyperelasticity
template<typename T>
class SaintVenantKirchhoffMaterialLaw : public MaterialLaw<T>
{
public:
	// Lame parameters
	T mu, lambda;

	// read settings from ptree
	void readSettings(const ptree::ptree& pt)
	{
		Material<T, 3> m;
		m.readSettings(pt);
		mu = m.mu;
		lambda = m.lambda;
	}

	T W(std::size_t i, const T* F) const
	{
		SymTensor3x3<T> E;
		E.greenStrain(F);
		T tr_E = E[0] + E[1] + E[2];
		return 0.5*lambda*tr_E*tr_E + mu*E.dot(E);
	}

	void PK1(std::size_t _i, const T* F, T alpha, bool gamma, T* P) const
	{
		// compute Greens strain tensor E = (F^T*F - I)/2
		SymTensor3x3<T> E;
		E.greenStrain(F);

		// compute second Piola-Kirchhoff stress S
		// set S = alpha*(2*mu*E + lambda*tr(E)*I)
		SymTensor3x3<T> S;
		const T two_mu = 2*alpha*mu;
		const T lambda_tr_E = alpha*lambda*(E[0] + E[1] + E[2]);
		S[0] = E[0]*two_mu + lambda_tr_E;
		S[1] = E[1]*two_mu + lambda_tr_E;
		S[2] = E[2]*two_mu + lambda_tr_E;
		S[3] = E[3]*two_mu;
		S[4] = E[4]*two_mu;
		S[5] = E[5]*two_mu;

		// compute first Piola-Kirchhoff stress P = alpha*F*S
		// 0 5 4  0 5 4
		// 8 1 3  5 1 3
		// 7 6 2  4 3 2
		#define PK1_OP(OP) \
			P[0] OP F[0]*S[0] + F[5]*S[5] + F[4]*S[4]; \
			P[1] OP F[8]*S[5] + F[1]*S[1] + F[3]*S[3]; \
			P[2] OP F[7]*S[4] + F[6]*S[3] + F[2]*S[2]; \
			P[3] OP F[8]*S[4] + F[1]*S[3] + F[3]*S[2]; \
			P[4] OP F[0]*S[4] + F[5]*S[3] + F[4]*S[2]; \
			P[5] OP F[0]*S[5] + F[5]*S[1] + F[4]*S[3]; \
			P[6] OP F[7]*S[5] + F[6]*S[1] + F[2]*S[3]; \
			P[7] OP F[7]*S[0] + F[6]*S[5] + F[2]*S[4]; \
			P[8] OP F[8]*S[0] + F[1]*S[5] + F[3]*S[4];
		if (gamma) {
			PK1_OP(+=)
		}
		else {
			PK1_OP(=)
		}
		#undef PK1_OP
	}

	void dPK1(std::size_t _i, const T* F, T alpha, bool gamma, const T* W, T* dP, std::size_t n = 1) const
	{
		// compute Greens strain tensor E = (F^T*F - I)/2
		SymTensor3x3<T> E;
		E.greenStrain(F);

		// compute second Piola-Kirchhoff stress S
		// set S = 2*mu*E + lambda*tr(E)*I
		SymTensor3x3<T> S;
		const T two_mu = 2*alpha*mu;
		const T lambda_tr_E = alpha*lambda*(E[0] + E[1] + E[2]);
		S[0] = E[0]*two_mu + lambda_tr_E;
		S[1] = E[1]*two_mu + lambda_tr_E;
		S[2] = E[2]*two_mu + lambda_tr_E;
		S[3] = E[3]*two_mu;
		S[4] = E[4]*two_mu;
		S[5] = E[5]*two_mu;

		SymTensor3x3<T> EdFdW;
		SymTensor3x3<T> SdFdW;

		for (std::size_t m = 0; m < n; m++)
		{
			// computes the directional derivative dE/dF : W of Greens strain tensor E = (F^T*F-I)/2
			EdFdW.greenStrainDeriv(F, W);

			// computes the directional derivative dS/dF : W
			const T lambda_tr_EdFdW = alpha*lambda*(EdFdW[0] + EdFdW[1] + EdFdW[2]);
			SdFdW[0] = EdFdW[0]*two_mu + lambda_tr_EdFdW;
			SdFdW[1] = EdFdW[1]*two_mu + lambda_tr_EdFdW;
			SdFdW[2] = EdFdW[2]*two_mu + lambda_tr_EdFdW;
			SdFdW[3] = EdFdW[3]*two_mu;
			SdFdW[4] = EdFdW[4]*two_mu;
			SdFdW[5] = EdFdW[5]*two_mu;

			#define PK1_OP(OP) \
				dP[0] OP F[0]*SdFdW[0] + F[5]*SdFdW[5] + F[4]*SdFdW[4] + W[0]*S[0] + W[5]*S[5] + W[4]*S[4]; \
				dP[1] OP F[8]*SdFdW[5] + F[1]*SdFdW[1] + F[3]*SdFdW[3] + W[8]*S[5] + W[1]*S[1] + W[3]*S[3]; \
				dP[2] OP F[7]*SdFdW[4] + F[6]*SdFdW[3] + F[2]*SdFdW[2] + W[7]*S[4] + W[6]*S[3] + W[2]*S[2]; \
				dP[3] OP F[8]*SdFdW[4] + F[1]*SdFdW[3] + F[3]*SdFdW[2] + W[8]*S[4] + W[1]*S[3] + W[3]*S[2]; \
				dP[4] OP F[0]*SdFdW[4] + F[5]*SdFdW[3] + F[4]*SdFdW[2] + W[0]*S[4] + W[5]*S[3] + W[4]*S[2]; \
				dP[5] OP F[0]*SdFdW[5] + F[5]*SdFdW[1] + F[4]*SdFdW[3] + W[0]*S[5] + W[5]*S[1] + W[4]*S[3]; \
				dP[6] OP F[7]*SdFdW[5] + F[6]*SdFdW[1] + F[2]*SdFdW[3] + W[7]*S[5] + W[6]*S[1] + W[2]*S[3]; \
				dP[7] OP F[7]*SdFdW[0] + F[6]*SdFdW[5] + F[2]*SdFdW[4] + W[7]*S[0] + W[6]*S[5] + W[2]*S[4]; \
				dP[8] OP F[8]*SdFdW[0] + F[1]*SdFdW[5] + F[3]*SdFdW[4] + W[8]*S[0] + W[1]*S[5] + W[3]*S[4];
			if (gamma) {
				PK1_OP(+=)
			}
			else {
				PK1_OP(=)
			}
			#undef PK1_OP

			W += 9;
			dP += 9;
		}
	}

	std::string str() const
	{
		return (boost::format("hyperelastic Saint Venant-Kirchhoff lambda=%g mu=%g") % lambda % mu).str();
	}
};


//! Neo Hooke material law for hyperelasticity
template<typename T>
class NeoHookeMaterialLaw : public MaterialLaw<T>
{
public:
	// Lame parameters
	T mu, lambda;

	// read settings from ptree
	void readSettings(const ptree::ptree& pt)
	{
		Material<T, 3> m;
		m.readSettings(pt);
		mu = m.mu;
		lambda = m.lambda;
	}

	T W(std::size_t i, const T* F) const
	{
		const T trC = Tensor3x3<T>::dot(F, F);
		const T J = Tensor3x3<T>::det(F);
		const T logJ = this->log(J);
		// return 0.5*mu*(trC - 3.0 - 2.0*logJ) + 0.5*lambda*logJ*logJ;
		return 0.5*(mu*((trC - 3.0) - 2.0*logJ) + lambda*logJ*logJ);
	}

	void PK1(std::size_t _i, const T* F, T alpha, bool gamma, T* P) const
	{
		// compute inverse right Cauchy Greens strain tensor Cinv = (F^T*F)^-1
		SymTensor3x3<T> C, Cinv;
		C.rightCauchyGreen(F);
		Cinv.inv(C);

		// compute second Piola-Kirchhoff stress S
		SymTensor3x3<T> S;
		const T J = Tensor3x3<T>::det(F);
		// TODO: Handle J <= 0
		const T alpha_lambda_lnJ = alpha*lambda*this->log(J);
		const T alpha_mu = alpha*mu;
		for (std::size_t m = 0; m < 6; m++) {
			S[m] = alpha_mu*(((m<3) ? 1 : 0) - Cinv[m]) + alpha_lambda_lnJ*Cinv[m];
		}

		// compute first Piola-Kirchhoff stress P = F*S
		#define PK1_OP(OP) \
			P[0] OP F[0]*S[0] + F[5]*S[5] + F[4]*S[4]; \
			P[1] OP F[8]*S[5] + F[1]*S[1] + F[3]*S[3]; \
			P[2] OP F[7]*S[4] + F[6]*S[3] + F[2]*S[2]; \
			P[3] OP F[8]*S[4] + F[1]*S[3] + F[3]*S[2]; \
			P[4] OP F[0]*S[4] + F[5]*S[3] + F[4]*S[2]; \
			P[5] OP F[0]*S[5] + F[5]*S[1] + F[4]*S[3]; \
			P[6] OP F[7]*S[5] + F[6]*S[1] + F[2]*S[3]; \
			P[7] OP F[7]*S[0] + F[6]*S[5] + F[2]*S[4]; \
			P[8] OP F[8]*S[0] + F[1]*S[5] + F[3]*S[4];
		if (gamma) {
			PK1_OP(+=)
		}
		else {
			PK1_OP(=)
		}
		#undef PK1_OP
	}

	void dPK1(std::size_t _i, const T* F, T alpha, bool gamma, const T* W, T* dP, std::size_t n = 1) const
	{
		// indices for transposed matrix
		static const std::size_t Tindices[9] = {0,1,2,6,7,8,3,4,5};

		const T J = Tensor3x3<T>::det(F);
		// TODO: Handle J <= 0
		const T alpha_mu_minus_lambda_lnJ = alpha*(mu - lambda*this->log(J));
		const T alpha_mu = alpha*mu;

		// compute inverse of F
		Tensor3x3<T> Finv;
		Finv.inv(F);
		
		for (std::size_t m = 0; m < n; m++)
		{
			// compute Finv^T * W^T
			Tensor3x3<T> FinvTWT;
			// 0 8 7  0 8 7
			// 5 1 6  5 1 6
			// 4 3 2  4 3 2

			FinvTWT[0] = Finv[0]*W[0] + Finv[8]*W[5] + Finv[7]*W[4];
			FinvTWT[1] = Finv[5]*W[8] + Finv[1]*W[1] + Finv[6]*W[3];
			FinvTWT[2] = Finv[4]*W[7] + Finv[3]*W[6] + Finv[2]*W[2];
			FinvTWT[3] = Finv[5]*W[7] + Finv[1]*W[6] + Finv[6]*W[2];
			FinvTWT[4] = Finv[0]*W[7] + Finv[8]*W[6] + Finv[7]*W[2];
			FinvTWT[5] = Finv[0]*W[8] + Finv[8]*W[1] + Finv[7]*W[3];
			FinvTWT[6] = Finv[4]*W[8] + Finv[3]*W[1] + Finv[2]*W[3];
			FinvTWT[7] = Finv[4]*W[0] + Finv[3]*W[5] + Finv[2]*W[4];
			FinvTWT[8] = Finv[5]*W[0] + Finv[1]*W[5] + Finv[6]*W[4];
			
			// compute Finv^T*W^T*Finv^T 
			Tensor3x3<T> FinvTWTFinvT;

			// 0 5 4  0 8 7
			// 8 1 3  5 1 6
			// 7 6 2  4 3 2
			FinvTWTFinvT[0] = FinvTWT[0]*Finv[0] + FinvTWT[5]*Finv[5] + FinvTWT[4]*Finv[4];
			FinvTWTFinvT[1] = FinvTWT[8]*Finv[8] + FinvTWT[1]*Finv[1] + FinvTWT[3]*Finv[3];
			FinvTWTFinvT[2] = FinvTWT[7]*Finv[7] + FinvTWT[6]*Finv[6] + FinvTWT[2]*Finv[2];
			FinvTWTFinvT[3] = FinvTWT[8]*Finv[7] + FinvTWT[1]*Finv[6] + FinvTWT[3]*Finv[2];
			FinvTWTFinvT[4] = FinvTWT[0]*Finv[7] + FinvTWT[5]*Finv[6] + FinvTWT[4]*Finv[2];
			FinvTWTFinvT[5] = FinvTWT[0]*Finv[8] + FinvTWT[5]*Finv[1] + FinvTWT[4]*Finv[3];
			FinvTWTFinvT[6] = FinvTWT[7]*Finv[8] + FinvTWT[6]*Finv[1] + FinvTWT[2]*Finv[3];
			FinvTWTFinvT[7] = FinvTWT[7]*Finv[0] + FinvTWT[6]*Finv[5] + FinvTWT[2]*Finv[4];
			FinvTWTFinvT[8] = FinvTWT[8]*Finv[0] + FinvTWT[1]*Finv[5] + FinvTWT[3]*Finv[4];
			
			T alpha_lambda_tr_FinvTWT = alpha*lambda*(FinvTWT[0] + FinvTWT[1] + FinvTWT[2]);

			#define PK1_OP(OP) \
				for (std::size_t k = 0; k < 9; k++) { \
					dP[k] OP alpha_mu*W[k] + alpha_lambda_tr_FinvTWT*Finv[Tindices[k]] + alpha_mu_minus_lambda_lnJ*FinvTWTFinvT[k]; \
				}

			if (gamma) {
				PK1_OP(+=)
			}
			else {
				PK1_OP(=)
			}
			#undef PK1_OP

			W += 9;
			dP += 9;
		}
	}

	std::string str() const
	{
		return (boost::format("hyperelastic Neo-Hooke lambda=%g mu=%g") % lambda % mu).str();
	}
};


//! Neo Hooke material law for hyperelasticity (variant 2)
template<typename T>
class NeoHooke2MaterialLaw : public MaterialLaw<T>
{
public:
	// G = mu = shear modulus, K = bulk moduls = lambda + 2/3 G
	T mu, K;

	// read settings from ptree
	void readSettings(const ptree::ptree& pt)
	{
		Material<T, 3> m;
		m.readSettings(pt);
		mu = m.mu;
		K = m.lambda + 2.0/3.0*m.mu;
	}

	T W(std::size_t i, const T* F) const
	{
		const T trC = Tensor3x3<T>::dot(F, F);
		const T J = Tensor3x3<T>::det(F);
		const T J1 = J - 1;
		return 0.5*(mu*(this->pow_minus_two_third(J)*trC - 3) + K*J1*J1);
	}

	void PK1(std::size_t _i, const T* F, T alpha, bool gamma, T* P) const
	{
		Tensor3x3<T> Finv;
		Finv.inv(F);

		const T trC = Tensor3x3<T>::dot(F, F);
		const T J = Tensor3x3<T>::det(F);
		const T muJ23 = alpha*mu*this->pow_minus_two_third(J);
		const T D = alpha*K*J*(J-1) - muJ23*(1.0/3.0)*trC;

		// compute first Piola-Kirchhoff stress P
		#define PK1_OP(OP) \
			P[0] OP muJ23*F[0] + D*Finv[0]; \
			P[1] OP muJ23*F[1] + D*Finv[1]; \
			P[2] OP muJ23*F[2] + D*Finv[2]; \
			P[3] OP muJ23*F[3] + D*Finv[6]; \
			P[4] OP muJ23*F[4] + D*Finv[7]; \
			P[5] OP muJ23*F[5] + D*Finv[8]; \
			P[6] OP muJ23*F[6] + D*Finv[3]; \
			P[7] OP muJ23*F[7] + D*Finv[4]; \
			P[8] OP muJ23*F[8] + D*Finv[5];
		if (gamma) {
			PK1_OP(+=)
		}
		else {
			PK1_OP(=)
		}
		#undef PK1_OP
	}

	void dPK1(std::size_t _i, const T* F, T alpha, bool gamma, const T* W, T* dP, std::size_t n = 1) const
	{
		// indices for transposed matrix
		static const std::size_t Tindices[9] = {0,1,2,6,7,8,3,4,5};

		// compute inverse of F and its transpose
		Tensor3x3<T> Finv, FinvT;
		Finv.inv(F);
		FinvT.transpose(Finv);

		const T J = Tensor3x3<T>::det(F);
		const T trC3 = (1.0/3.0)*Tensor3x3<T>::dot(F, F);
		const T alpha_muJ23 = alpha*mu*this->pow_minus_two_third(J);
		const T alpha_KJ = alpha*K*J;
		const T alpha_KJJ1 = alpha_KJ*(J - 1);
		const T alpha_KJJ = alpha_KJ*(2*J - 1);

		for (std::size_t m = 0; m < n; m++)
		{
			// compute Finv^T * W^T
			Tensor3x3<T> FinvTWT;
			// 0 8 7  0 8 7
			// 5 1 6  5 1 6
			// 4 3 2  4 3 2

			FinvTWT[0] = Finv[0]*W[0] + Finv[8]*W[5] + Finv[7]*W[4];
			FinvTWT[1] = Finv[5]*W[8] + Finv[1]*W[1] + Finv[6]*W[3];
			FinvTWT[2] = Finv[4]*W[7] + Finv[3]*W[6] + Finv[2]*W[2];
			FinvTWT[3] = Finv[5]*W[7] + Finv[1]*W[6] + Finv[6]*W[2];
			FinvTWT[4] = Finv[0]*W[7] + Finv[8]*W[6] + Finv[7]*W[2];
			FinvTWT[5] = Finv[0]*W[8] + Finv[8]*W[1] + Finv[7]*W[3];
			FinvTWT[6] = Finv[4]*W[8] + Finv[3]*W[1] + Finv[2]*W[3];
			FinvTWT[7] = Finv[4]*W[0] + Finv[3]*W[5] + Finv[2]*W[4];
			FinvTWT[8] = Finv[5]*W[0] + Finv[1]*W[5] + Finv[6]*W[4];
			
			// compute Finv^T*W^T*Finv^T 
			Tensor3x3<T> FinvTWTFinvT;

			// 0 5 4  0 8 7
			// 8 1 3  5 1 6
			// 7 6 2  4 3 2
			FinvTWTFinvT[0] = FinvTWT[0]*Finv[0] + FinvTWT[5]*Finv[5] + FinvTWT[4]*Finv[4];
			FinvTWTFinvT[1] = FinvTWT[8]*Finv[8] + FinvTWT[1]*Finv[1] + FinvTWT[3]*Finv[3];
			FinvTWTFinvT[2] = FinvTWT[7]*Finv[7] + FinvTWT[6]*Finv[6] + FinvTWT[2]*Finv[2];
			FinvTWTFinvT[3] = FinvTWT[8]*Finv[7] + FinvTWT[1]*Finv[6] + FinvTWT[3]*Finv[2];
			FinvTWTFinvT[4] = FinvTWT[0]*Finv[7] + FinvTWT[5]*Finv[6] + FinvTWT[4]*Finv[2];
			FinvTWTFinvT[5] = FinvTWT[0]*Finv[8] + FinvTWT[5]*Finv[1] + FinvTWT[4]*Finv[3];
			FinvTWTFinvT[6] = FinvTWT[7]*Finv[8] + FinvTWT[6]*Finv[1] + FinvTWT[2]*Finv[3];
			FinvTWTFinvT[7] = FinvTWT[7]*Finv[0] + FinvTWT[6]*Finv[5] + FinvTWT[2]*Finv[4];
			FinvTWTFinvT[8] = FinvTWT[8]*Finv[0] + FinvTWT[1]*Finv[5] + FinvTWT[3]*Finv[4];
			
			const T tr_FinvTWT = FinvTWT[0] + FinvTWT[1] + FinvTWT[2];
			const T FW23 = (2.0/3.0)*Tensor3x3<T>::dot(F, W);
			const T FinvTW = Tensor3x3<T>::dot(FinvT, W);



			#define PK1_OP(OP) \
				for (std::size_t k = 0; k < 9; k++) { \
					dP[k] OP alpha_muJ23*(-2.0/3.0*tr_FinvTWT*(F[k] - trC3*Finv[Tindices[k]]) + W[k] - FW23*FinvT[k] + trC3*FinvTWTFinvT[k]) + alpha_KJJ*FinvTW*Finv[Tindices[k]] - alpha_KJJ1*FinvTWTFinvT[k]; \
				}

			if (gamma) {
				PK1_OP(+=)
			}
			else {
				PK1_OP(=)
			}
			#undef PK1_OP

			W += 9;
			dP += 9;
		}
	}

	std::string str() const
	{
		return (boost::format("hyperelastic Neo-Hooke 2 lambda=%g mu=%g") % (K - 2.0/3.0*mu) % mu).str();
	}
};


//! Base class for phase fields (description of spatial concentration of materials)
template<typename T, typename P>
class PhaseBase
{
public:
	typedef TensorField<T> RealTensor;
	typedef boost::shared_ptr<RealTensor> pRealTensor;

	P* phi;			// phase volume fraction field
	pRealTensor _phi;	// underlying tensor

	pRealTensor _phi_cg;	// underlying tensor for coarse grid
	pRealTensor _phi_dfg;	// underlying tensor for doubly fine grid

	T vol;			// total volume fraction
	T ivf;			// interface volume fraction

	std::string name;	// name of phase

	std::string law_name;				// material law name
	boost::shared_ptr< MaterialLaw<T> > law;	// material law instance

	std::size_t index;

	PhaseBase() : vol(0), index(0)
	{
	}

	void init(std::size_t nx, std::size_t ny, std::size_t nz, bool init_dfg)
	{
		_phi_cg.reset(new RealTensor(nx, ny, nz, 1));

		if (init_dfg) {
			_phi_dfg.reset(new RealTensor(2*nx, 2*ny, 2*nz, 1));
		}

		select_dfg(false);
	}

	//! select doubly fine grid (for staggered grid method)
	void select_dfg(bool yes)
	{
		if (yes && !_phi_dfg) {
			BOOST_THROW_EXCEPTION(std::runtime_error("The doubly find grid was not initialized"));
		}

		if (yes) {
			_phi = _phi_dfg;
		}
		else {
			_phi = _phi_cg;
		}

		phi = (*_phi)[0];
	}

	void setOne(std::size_t index = 0)
	{
		_phi->setOne(index);
	}
};


//! Base class for mixed material laws (e.g. at interfaces where more than one phase is present)
template<typename T, typename P>
class MixedMaterialLawBase : public MaterialLaw<T>
{
public:
	typedef PhaseBase<T, P> Phase;
	typedef boost::shared_ptr< Phase > pPhase;

	std::vector<pPhase> phases;

	//! see https://stackoverflow.com/questions/827196/virtual-default-destructors-in-c/827205
	virtual ~MixedMaterialLawBase() {}

	virtual void getRefMaterial(TensorField<T>& F, T& mu_0, T& lambda_0, bool zero_trace, bool polarization) const = 0;
	virtual T meanW(const TensorField<T>& F) const = 0;
	virtual T minEig(const TensorField<T>& F, bool zero_trace) const = 0;
	virtual void eig(std::size_t i, const T* F, T& lambda_min_p, T& lambda_max_p, bool zero_trace) const = 0;
	virtual T maxStress(const TensorField<T>& F) const = 0;
	virtual ublas::vector<T> meanPK1(const TensorField<T>& F, T alpha) const = 0;
	virtual ublas::vector<T> meanCauchy(const TensorField<T>& F, T alpha) const = 0;
	virtual int dim() const = 0;

	void calcPolarization(std::size_t i, T mu_0, const ublas::vector<T>& F, ublas::vector<T>& _P,
		std::size_t dim, bool inv) const
	{
		for (std::size_t p = 0; p < this->phases.size(); p++) {
			P phi = this->phases[p]->phi[i];
			if (phi == 1) {
				this->phases[p]->law->calcPolarization(i, mu_0, F, _P, dim, inv);
				return;
			}
		}

		MaterialLaw<T>::calcPolarization(i, mu_0, F, _P, dim, inv);
	}

	void select_dfg(bool yes)
	{
		for (std::size_t i = 0; i < phases.size(); i++) {
			phases[i]->select_dfg(yes);
		}
	}

	void add_phase(pPhase phase)
	{
		phase->index = phases.size();
		phases.push_back(phase);
	}

	//! make 6x6 tensor symmetric, if operating in linear elasticity 
	inline void fix_dim(T* t) const
	{
		if (dim() == 6) {
			t[6] = t[3];
			t[7] = t[4];
			t[8] = t[5];
		}
		else if (dim() == 3) {
			t[3] = t[4] = t[5] = t[6] = t[7] = t[8] = 0;
		}
	}

	//! make 6x6 tensor symmetric, if operating in linear elasticity 
	inline void fix_sym(T* t) const
	{
		if (dim() == 6) {
			t[6] = t[3] = 0.5*(t[3] + t[6]);
			t[7] = t[4] = 0.5*(t[4] + t[7]);
			t[8] = t[5] = 0.5*(t[5] + t[8]);
		}
		else if (dim() == 3) {
			t[3] = t[4] = t[5] = t[6] = t[7] = t[8] = 0;
		}
	}

	virtual void init()
	{
	}
};


//! Mixed material class
template<typename T, typename P, int DIM>
class MixedMaterialLaw : public MixedMaterialLawBase<T, P>
{
public:
	//! comupute reference material
	//! i.e. (lambda_min + lambda_max)/2 for dP/dF
	noinline void getRefMaterial(TensorField<T>& F, T& mu_0, T& lambda_0, bool zero_trace, bool polarization) const
	{
		Timer __t("getRefMaterial", false);

		T lambda_min = STD_INFINITY(T);
		T lambda_max = -STD_INFINITY(T);

		#pragma omp parallel
		{
			T lambda_min_p = STD_INFINITY(T);
			T lambda_max_p = -STD_INFINITY(T);
			
			#pragma omp for schedule (dynamic) collapse(2)
			BEGIN_TRIPLE_LOOP(kk, F.nx, F.ny, F.nz, F.nzp)
			{
				Tensor<T, DIM> Fk;
				F.assign(kk, Fk);
				
				this->eig(kk, Fk, lambda_min_p, lambda_max_p, zero_trace);
			}
			END_TRIPLE_LOOP(kk)

			// perform reduction	
			#pragma omp critical
			{
				lambda_min = std::min(lambda_min, lambda_min_p);
				lambda_max = std::max(lambda_max, lambda_max_p);
			}
		}

		if (lambda_min < 0) {
			LOG_CWARN << "detected negative eigenvalue (" << lambda_min << ") in linearized 1st PK. Cutting off." << std::endl;	

#if 1
			BEGIN_TRIPLE_LOOP(kk, F.nx, F.ny, F.nz, F.nzp)
			{
				Tensor<T, DIM> Fk;
				F.assign(kk, Fk);
				T lambda_min_p = STD_INFINITY(T);
				T lambda_max_p = -STD_INFINITY(T);
				this->eig(kk, Fk, lambda_min_p, lambda_max_p, zero_trace);
				if (lambda_min_p < 0) {
					LOG_COUT << "cell index = " << kk << std::endl;
					LOG_COUT << "phi0 = " << this->phases[0]->phi[kk] << std::endl;
					ublas::c_matrix<T, DIM, DIM> dP;
					LOG_COUT << "Fk" << std::endl;
					LOG_COUT << format(Fk) << std::endl;
					
					this->dPK1(kk, Fk, 1, false, TensorIdentity<T,DIM>::Id, dP.data(), DIM);
					LOG_COUT << "dP" << std::endl;
					LOG_COUT << format(dP) << std::endl;

					this->dPK1_fd(kk, Fk, 1, false, TensorIdentity<T,DIM>::Id, dP.data(), DIM, DIM, 1e-5);
					LOG_COUT << "dP_fd (1e-5)" << std::endl;
					LOG_COUT << format(dP) << std::endl;
					/*
					VoigtMixedMaterialLaw<T, _P, DIM>* mat = dynamic_cast<VoigtMixedMaterialLaw<T, _P, DIM>*>(this);
					if (mat != NULL) {
						mat->
					}
					*/

					goto br;
				}
			}
			END_TRIPLE_LOOP(kk)
br:
#endif

			lambda_min = 0;
		}

		LOG_COUT << "lambda_min = " << lambda_min << " lambda_max = " << lambda_max << std::endl;

		if (polarization) {
			mu_0 = std::sqrt(lambda_min*lambda_max);
		}
		else {
			mu_0 = 0.5*(lambda_min + lambda_max);
			//lambda_0 = 0.5*(lambda_max - lambda_min);
		}

		lambda_0 = 0.0;
	}

	//! mean quantities for energy W 
	noinline T meanW(const TensorField<T>& F) const
	{
		Timer __t("meanW", false);

		T S = 0;

		#pragma omp parallel for schedule (static) collapse(2) reduction(+:S)
		for (std::size_t i = 0; i < F.nx; i++)
		{
			for (std::size_t j = 0; j < F.ny; j++)
			{
				std::size_t kk = i*F.nyzp + j*F.nzp;

				for (std::size_t k = 0; k < F.nz; k++)
				{
					Tensor<T, DIM> Fk;
					F.assign(kk, Fk);
					S += this->W(kk, Fk);
					kk ++;
				}
			}
		}
		
		return (S/F.nxyz);
	}

	//! Mean value for Cauchy stress 
	//! \param F strain
	//! \param alpha scaling constant
	noinline ublas::vector<T> meanCauchy(const TensorField<T>& F, T alpha) const
	{
		Timer __t("meanCauchy", false);

		ublas::c_vector<T,DIM> S = ublas::zero_vector<T>(DIM);

		alpha /= F.nxyz;

		#pragma omp parallel
		{
			ublas::c_vector<T,DIM> SP = ublas::zero_vector<T>(DIM);

			#pragma omp for schedule (static) collapse(2)
			for (std::size_t i = 0; i < F.nx; i++)
			{
				for (std::size_t j = 0; j < F.ny; j++)
				{
					std::size_t kk = i*F.nyzp + j*F.nzp;

					for (std::size_t k = 0; k < F.nz; k++)
					{
						Tensor<T, DIM> Fk;
						F.assign(kk, Fk);
						this->Cauchy(kk, Fk, alpha, true, SP.data());
						kk ++;
					}
				}
			}

			// perform reduction	
			#pragma omp critical
			{
				for (std::size_t i = 0; i < S.size(); i++) {
					S[i] += SP[i];
				}
			}
		}
		
		return S;
	}

	//! Mean value for PK1 
	//! \param F strain
	//! \param alpha scaling constant
	noinline ublas::vector<T> meanPK1(const TensorField<T>& F, T alpha) const
	{
		Timer __t("meanPK1", false);

		ublas::c_vector<T,DIM> S = ublas::zero_vector<T>(DIM);

		alpha /= F.nxyz;

		#pragma omp parallel
		{
			ublas::c_vector<T,DIM> SP = ublas::zero_vector<T>(DIM);

			#pragma omp for schedule (dynamic) collapse(2)
			for (std::size_t i = 0; i < F.nx; i++)
			{
				for (std::size_t j = 0; j < F.ny; j++)
				{
					std::size_t kk = i*F.nyzp + j*F.nzp;

					for (std::size_t k = 0; k < F.nz; k++)
					{
						Tensor<T, DIM> Fk;
						F.assign(kk, Fk);
						this->PK1(kk, Fk, alpha, true, SP.data());
						kk ++;
					}
				}
			}

			// perform reduction	
			#pragma omp critical
			{
				for (std::size_t i = 0; i < S.size(); i++) {
					S[i] += SP[i];
				}
			}
		}
		
		return S;
	}

	//! Mean value for stress 
	//! \param F strain
	noinline T maxStress(const TensorField<T>& F) const
	{
		Timer __t("maxStress", false);
		
		T m = 0;
		std::size_t imax = 0;

		#pragma omp parallel
		{
			T mp = 0;
			std::size_t imaxp = 0;
			
			#pragma omp for schedule (dynamic) collapse(2)
			for (std::size_t i = 0; i < F.nx; i++)
			{
				for (std::size_t j = 0; j < F.ny; j++)
				{
					std::size_t kk = i*F.nyzp + j*F.nzp;
					
					for (std::size_t k = 0; k < F.nz; k++)
					{
						Tensor<T, DIM> Fk;
						F.assign(kk, Fk);
						
						ublas::c_matrix<T, DIM, DIM> dP;
						this->dPK1(kk, Fk, 1, false, TensorIdentity<T,DIM>::Id, dP.data(), DIM);
						
						T norm = (T)ublas::norm_frobenius(dP);

						if (norm > mp) {
							imaxp = kk;
							mp = norm;
						}
						
						kk ++;
					}
				}
			}
			
			// perform reduction	
			#pragma omp critical
			{
				if (mp > m) {
					imax = imaxp;
					m = mp;
				}
			}
		}
		
		Tensor3x3<T> Fi;
		F.assign(imax, Fi);
		
#if 0
		LOG_COUT << "imax " << imax << std::endl;
		LOG_COUT << "F " << format(Fi) << std::endl;
		LOG_COUT << "detF " << Fi.det() << std::endl;
#endif

		return m;
	}

	//! mean quantities for PK1 
	noinline T minEig(const TensorField<T>& F, bool zero_trace) const
	{
		Timer __t("minEig", false);

		T m = STD_INFINITY(T);

		//bool pr = true;

		#pragma omp parallel
		{
			T me = STD_INFINITY(T);

			#pragma omp for schedule (dynamic) collapse(2)
			for (std::size_t i = 0; i < F.nx; i++)
			{
				for (std::size_t j = 0; j < F.ny; j++)
				{
					std::size_t kk = i*F.nyzp + j*F.nzp;

					for (std::size_t k = 0; k < F.nz; k++)
					{
						Tensor<T, DIM> Fk;
						F.assign(kk, Fk);
						
						T lambda_min_p, lambda_max_p;
						eig(kk, Fk, lambda_min_p, lambda_max_p, zero_trace);

						for (std::size_t j = 0; j < 9; j++) {
							me = std::min(me, lambda_min_p);
						}

/*
						if (me < -10 && pr) {
							LOG_COUT << "Fk=" << format(Fk) << std::endl;
							LOG_COUT << "C=\n" << format(C) << std::endl;
							pr = false;
						}
*/

						kk ++;
					}
				}
			}

			// perform reduction	
			#pragma omp critical
			{
				m = std::min(m, me);
			}
		}
		
		return m;
	}

	//! Compute eigenvalues of dP(F)/dF(F)
	inline void eig(std::size_t i, const T* F, T& lambda_min_p, T& lambda_max_p, bool zero_trace) const
	{

#if 0
		// fast estimate by upper bound

		lambda_min_p = std::min(lambda_min_p, 0.0);
		lambda_max_p = std::max(lambda_max_p, (T)ublas::norm_frobenius(C));
#else
		// exact method


		/*
		// extend to 9x9 matrix
		ublas::matrix<T> C(9, 9);

		if (DIM == 9) {
			C = dP;
		}
		else {
			for (std::size_t i = 0; i < 9; i++) {
				for (std::size_t j = i; j < 9; j++) {
					C(j,i) = C(i,j) = dP((i < 6 ? i : (i-3)), (j < 6 ? j : (j-3)));
				}
			}
		}
		*/

		// compute eigenvalues

		if (zero_trace)
		{
			ublas::c_matrix<T, DIM, DIM> C;
			this->dPK1(i, F, 1, false, TensorIdentity<T,DIM>::Id, C.data(), DIM);
			ublas::c_matrix<T,DIM-1,DIM-1> dP = subrange(C, 1, DIM, 1, DIM);

			ublas::c_vector<T, DIM-1> e;
			lapack::syev('N', 'U', dP, e, lapack::optimal_workspace());     // dP is probably overwritten

			for (std::size_t j = 0; j < (DIM-1); j++) {
				lambda_min_p = std::min(lambda_min_p, e[j]);
				lambda_max_p = std::max(lambda_max_p, e[j]);
			}
		}
		else
		{
			ublas::c_matrix<T, DIM, DIM> dP;
			this->dPK1(i, F, 1, false, TensorIdentity<T,DIM>::Id, dP.data(), DIM);

			ublas::c_vector<T, DIM> e;
			lapack::syev('N', 'U', dP, e, lapack::optimal_workspace());     // dP is probably overwritten

			for (std::size_t j = 0; j < DIM; j++) {
				lambda_min_p = std::min(lambda_min_p, e[j]);
				lambda_max_p = std::max(lambda_max_p, e[j]);
			}


#if 0
			lapack::syev((DIM == 9) ? 'V' : 'N', 'U', dP, e, lapack::optimal_workspace());     // dP is probably overwritten

			T eps = std::pow(std::numeric_limits<T>::epsilon(), 1.0/3.0);

			for (std::size_t j = 0; j < DIM; j++) {
				if (DIM == 9) {
					// check determinant
					Tensor3x3<T> X(dP.data() + j*DIM);

					if (X.det() < eps) {
#if 0
						LOG_COUT << format(dP) << std::endl;
						LOG_COUT << format(X) << std::endl;
						LOG_COUT << e[j] << std::endl;
						LOG_COUT << "det=" << X.det() << std::endl;
						exit(0);
#endif
						continue;
					}
				}
				lambda_min_p = std::min(lambda_min_p, e[j]);
				lambda_max_p = std::max(lambda_max_p, e[j]);
			}
#endif


		}
#endif
	}

	//! Dimenson of strain field the material law is acting on
	inline int dim() const { return DIM; }
};


//! Mixed material which uses the material with the maximum volume fraction 
template<typename T, typename _P, int DIM>
class MaximumMixedMaterialLaw : public MixedMaterialLaw<T, _P, DIM>
{
public:
	std::size_t get_max_phase(std::size_t i) const
	{
		std::size_t p_max = 0;
		
		for (std::size_t p = 1; p < this->phases.size(); p++) {
			if (this->phases[p]->phi[i] > this->phases[p_max]->phi[i]) {
				p_max = p;
			}
		}

		return p_max;
	}

	T W(std::size_t i, const T* F) const
	{
		return this->phases[this->get_max_phase(i)]->law->W(i, F);
	}

	void PK1(std::size_t i, const T* F, T alpha, bool gamma, T* P) const
	{
		this->phases[this->get_max_phase(i)]->law->PK1(i, F, alpha, gamma, P);
	}

	void dPK1(std::size_t i, const T* F, T alpha, bool gamma, const T* W, T* dP, std::size_t n = 1) const
	{
		this->phases[this->get_max_phase(i)]->law->dPK1(i, F, alpha, gamma, W, dP, n);
	}

	std::string str() const
	{
		return (boost::format("Maximum mixed")).str();
	}
};


//! Mixed material with some special deviatoric volumetric splitting
template<typename T, typename _P, int DIM>
class SplitMixedMaterialLaw : public MixedMaterialLaw<T, _P, DIM>
{
public:
	boost::shared_ptr< MixedMaterialLawBase<T, _P> > dev_rule;
	boost::shared_ptr< MixedMaterialLawBase<T, _P> > vol_rule;
	
	virtual void init()
	{
		dev_rule->phases = this->phases;
		vol_rule->phases = this->phases;
	}

	void PK1(std::size_t i, const T* F, T alpha, bool gamma, T* P) const
	{
		// get volumetric part of F
		T Fvol[DIM];
		Fvol[0] = Fvol[1] = Fvol[2] = (F[0] + F[1] + F[2])/3.0;
		for (std::size_t k = 3; k < DIM; k++) {
			Fvol[k] = 0;
		}
			
		// get deviatoric part of F
		T Fdev[DIM];
		for (std::size_t k = 0; k < DIM; k++) {
			Fdev[k] = F[k] - Fvol[k];
		}
			
		dev_rule->PK1(i, Fdev, alpha, gamma, P);
		vol_rule->PK1(i, Fvol, alpha, true, P);
	}

	void dPK1(std::size_t i, const T* F, T alpha, bool gamma, const T* W, T* dP, std::size_t n = 1) const
	{
		BOOST_THROW_EXCEPTION(std::runtime_error("SplitMixedMaterialLaw dPK1 not implemented"));
	}

	std::string str() const
	{
		return (boost::format("Split mixed")).str();
	}
};


//! Mixed material using the Reuss lower bound
template<typename T, typename _P, int DIM>
class ReussMixedMaterialLaw : public MixedMaterialLaw<T, _P, DIM>
{
public:
	// compute energy
	virtual T W(std::size_t i, const T* F) const
	{
		// TODO: but generialization to hyperelastic materials is difficult
		BOOST_THROW_EXCEPTION(std::runtime_error("Reuss MaterialLaw energy not implemented"));
	}

	void PK1(std::size_t i, const T* F, T alpha, bool gamma, T* P) const
	{
		ublas::c_matrix<T,DIM,DIM> C, Cinv, SumCinv = ublas::zero_matrix<T>(DIM, DIM);

		for (std::size_t p = 0; p < this->phases.size(); p++) {
			_P phi = this->phases[p]->phi[i];
			if (phi == 0) continue;
			if (phi == 1) {
				this->phases[p]->law->PK1(i, F, alpha, gamma, P);
				return;
			}
			
			this->phases[p]->law->dPK1(i, F, 1, false, TensorIdentity<T,DIM>::Id, C.data(), DIM);
			InvertMatrix<T,DIM>(C, Cinv);
			SumCinv += phi*Cinv;
		}

		InvertMatrix<T,DIM>(SumCinv, C);

		// compute P = C*F
		for (std::size_t k = 0; k < DIM; k++) {
			if (!gamma) P[k] = 0;
			for (std::size_t j = 0; j < DIM; j++) {
				P[k] += alpha*C(k,j)*F[j];
			}
		}
	}

	void dPK1(std::size_t _i, const T* F, T alpha, bool gamma, const T* W, T* dP, std::size_t n = 1) const
	{
		ublas::c_matrix<T,DIM,DIM> C, Cinv, SumCinv = ublas::zero_matrix<T>(DIM, DIM);

		for (std::size_t p = 0; p < this->phases.size(); p++) {
			_P phi = this->phases[p]->phi[_i];
			if (phi == 0) continue;
			if (phi == 1) {
				this->phases[p]->law->dPK1(_i, F, alpha, gamma, W, dP, n);
				return;
			}
			this->phases[p]->law->dPK1(_i, F, 1, false, TensorIdentity<T,DIM>::Id, C.data(), DIM);
			InvertMatrix<T,DIM>(C, Cinv);
			SumCinv += phi*Cinv;
		}

		InvertMatrix<T,DIM>(SumCinv, C);

		// compute P = C*F
		for (std::size_t i = 0; i < n; i++) {
			for (std::size_t k = 0; k < DIM; k++) {
				if (!gamma) dP[k + i*DIM] = 0;
				for (std::size_t j = 0; j < DIM; j++) {
					dP[k + i*DIM] += alpha*C(k,j)*W[j + i*DIM];
				}
			}
		}
	}

	std::string str() const
	{
		return (boost::format("Reuss mixed")).str();
	}
};


//! Mixed material using the Voigt upper bound
template<typename T, typename _P, int DIM>
class VoigtMixedMaterialLaw : public MixedMaterialLaw<T, _P, DIM>
{
public:
	T threshold;
	
	virtual void init()
	{
		threshold = 10*std::numeric_limits<T>::epsilon();
	}

	T W(std::size_t i, const T* F) const
	{
		T W = 0;

		for (std::size_t p = 0; p < this->phases.size(); p++) {
			_P phi = this->phases[p]->phi[i];
			if (phi <= threshold) continue;
			W += phi*this->phases[p]->law->W(i, F);
		}

		return W;
	}

	void PK1(std::size_t i, const T* F, T alpha, bool gamma, T* P) const
	{
		for (std::size_t p = 0; p < this->phases.size(); p++) {
			_P phi = this->phases[p]->phi[i];
			// LOG_COUT << this->phases[p]->phi << " " << p << " " << i << std::endl;
			if (phi <= threshold) continue;
			this->phases[p]->law->PK1(i, F, phi*alpha, gamma, P);
			gamma = true;
		}
	}

	void dPK1(std::size_t i, const T* F, T alpha, bool gamma, const T* W, T* dP, std::size_t n = 1) const
	{
		for (std::size_t p = 0; p < this->phases.size(); p++) {
			_P phi = this->phases[p]->phi[i];
			if (phi <= threshold) continue;
			this->phases[p]->law->dPK1(i, F, phi*alpha, gamma, W, dP, n);
			gamma = true;
		}
	}

	std::string str() const
	{
		return (boost::format("Voigt mixed")).str();
	}
};


//! Mixed material using a random material, where the probability is proportional to its concentration
template<typename T, typename _P, int DIM>
class RandomMixedMaterialLaw : public MixedMaterialLaw<T, _P, DIM>
{
public:
	T threshold, threshold2;
	
	virtual void init()
	{
		threshold = 10*std::numeric_limits<T>::epsilon();
		threshold2 = 1-threshold;
	}

	bool isInterface(std::size_t i) const
	{
		for (std::size_t p = 0; p < this->phases.size(); p++) {
			_P phi = this->phases[p]->phi[i];
			if (phi <= threshold) continue;
			if (phi < threshold2) return true;
		}

		return false;
	}

	std::size_t random(std::size_t i) const
	{
		return ((((i*1103515245 + 12345) >> 16) & RAND_MAX) % this->phases.size());
	}

	T W(std::size_t i, const T* F) const
	{
		T W = 0;

		if (this->isInterface(i)) {
			std::size_t p = this->random(i);
			W += this->phases[p]->law->W(i, F);
			return W;
		}

		for (std::size_t p = 0; p < this->phases.size(); p++) {
			_P phi = this->phases[p]->phi[i];
			if (phi <= threshold) continue;
			W += phi*this->phases[p]->law->W(i, F);
		}

		return W;
	}

	void PK1(std::size_t i, const T* F, T alpha, bool gamma, T* P) const
	{
		if (this->isInterface(i)) {
			std::size_t p = this->random(i);
			this->phases[p]->law->PK1(i, F, alpha, gamma, P);
			return;
		}

		for (std::size_t p = 0; p < this->phases.size(); p++) {
			_P phi = this->phases[p]->phi[i];
			if (phi <= threshold) continue;
			this->phases[p]->law->PK1(i, F, phi*alpha, gamma, P);
			gamma = true;
		}
	}

	void dPK1(std::size_t i, const T* F, T alpha, bool gamma, const T* W, T* dP, std::size_t n = 1) const
	{
		if (this->isInterface(i)) {
			std::size_t p = this->random(i);
			this->phases[p]->law->dPK1(i, F, alpha, gamma, W, dP, n);
			return;
		}


		for (std::size_t p = 0; p < this->phases.size(); p++) {
			_P phi = this->phases[p]->phi[i];
			if (phi <= threshold) continue;
			this->phases[p]->law->dPK1(i, F, phi*alpha, gamma, W, dP, n);
			gamma = true;
		}
	}

	std::string str() const
	{
		return (boost::format("random mixed")).str();
	}
};


//! Mixed material where materials with nonzero concentration are all weigthed equally
template<typename T, typename _P, int DIM>
class FiftyFiftyMixedMaterialLaw : public MixedMaterialLaw<T, _P, DIM>
{
public:
	T threshold, threshold2;
	
	virtual void init()
	{
		threshold = 10*std::numeric_limits<T>::epsilon();
		threshold2 = 1-threshold;
	}

	bool isInterface(std::size_t i) const
	{
		for (std::size_t p = 0; p < this->phases.size(); p++) {
			_P phi = this->phases[p]->phi[i];
			if (phi <= threshold) continue;
			if (phi < threshold2) return true;
		}

		return false;
	}

	T W(std::size_t i, const T* F) const
	{
		T W = 0;

		if (this->isInterface(i)) {
			for (std::size_t p = 0; p < this->phases.size(); p++) {
				W += this->phases[p]->law->W(i, F)/this->phases.size();
			}
			return W;
		}

		for (std::size_t p = 0; p < this->phases.size(); p++) {
			_P phi = this->phases[p]->phi[i];
			if (phi <= threshold) continue;
			W += phi*this->phases[p]->law->W(i, F);
		}

		return W;
	}

	void PK1(std::size_t i, const T* F, T alpha, bool gamma, T* P) const
	{
		if (this->isInterface(i)) {
			for (std::size_t p = 0; p < this->phases.size(); p++) {
				this->phases[p]->law->PK1(i, F, alpha/this->phases.size(), gamma, P);
				gamma = true;
			}
			return;
		}

		for (std::size_t p = 0; p < this->phases.size(); p++) {
			_P phi = this->phases[p]->phi[i];
			if (phi <= threshold) continue;
			this->phases[p]->law->PK1(i, F, phi*alpha, gamma, P);
			gamma = true;
		}
	}

	void dPK1(std::size_t i, const T* F, T alpha, bool gamma, const T* W, T* dP, std::size_t n = 1) const
	{
		if (this->isInterface(i)) {
			for (std::size_t p = 0; p < this->phases.size(); p++) {
				this->phases[p]->law->dPK1(i, F, alpha/this->phases.size(), gamma, W, dP, n);
				gamma = true;
			}
			return;
		}


		for (std::size_t p = 0; p < this->phases.size(); p++) {
			_P phi = this->phases[p]->phi[i];
			if (phi <= threshold) continue;
			this->phases[p]->law->dPK1(i, F, phi*alpha, gamma, W, dP, n);
			gamma = true;
		}
	}

	std::string str() const
	{
		return (boost::format("50/50 mixed")).str();
	}
};


//! Mixed material where materials with nonzero concentration are all weigthed equally
template<typename T, typename _P, int DIM>
class IsoMixedMaterialLaw : public MixedMaterialLaw<T, _P, DIM>
{
public:
	/*	Equations:
		
		c1*e1 + c2*e2 = e

		W = c1*W1(e1) + c2*W2(e2)
		  = c1*W1(e1) + c2*W2((e - c1*e1)/c2)

		dW/de1 = c1*C1*e1 - C2*(e - c1*e1)*c1/c2 = 0

		c1*C1*e1 - C2*(e - c1*e1)*c1/c2

		(c2*C1 + c1*C2)*e1 = C2*e
	*/

	inline void get_mix(std::size_t i, const T* F,
		typename MixedMaterialLawBase<T, _P>::pPhase& p1,
		typename MixedMaterialLawBase<T, _P>::pPhase& p2,
		T* F1, T* F2, _P& c1, _P& c2) const
	{
		for (std::size_t p = 0; p < this->phases.size(); p++) {
			_P phi = this->phases[p]->phi[i];
			if (phi == 0) continue;
			if (phi == 1) {
				c1 = phi;
				std::memcpy(F1, F, DIM*sizeof(T));
				p1 = this->phases[p];
				p2.reset();
				return;
			}
			if (!p1) { p1 = this->phases[p]; c1 = phi; continue; }
			if (!p2) { p2 = this->phases[p]; c2 = phi; continue; }
			BOOST_THROW_EXCEPTION(std::runtime_error("The laminate mixing rule supports only two phase mixtures (not more)"));
		}

		if (!p1) { 
			BOOST_THROW_EXCEPTION(std::runtime_error("The laminate mixing rule supports only two phase mixtures (not less)"));
		}

		if (!p2) { 
			std::memcpy(F1, F, DIM*sizeof(T));
			return;
		}

		// compute C = c2*C1 + c1*C2
		ublas::c_matrix<T,DIM,DIM> C;
		for (std::size_t k = 0; k < DIM; k++) {
			p1->law->PK1(i, TensorIdentity<T,DIM>::Id + k*DIM, c2, false, C.data() + k*DIM);
			p2->law->PK1(i, TensorIdentity<T,DIM>::Id + k*DIM, c1, true, C.data() + k*DIM);
		}
		
		// compute Cinv = C^{-1}
		ublas::c_matrix<T,DIM,DIM> Cinv;
		InvertMatrix<T,DIM>(C, Cinv);

		// compute C2F = C2:F
		Tensor3x3<T> C2F;
		p2->law->PK1(i, F, 1, false, C2F);
		
		// compute F1 = Cinv*C2F and
		// F2 = (F - c1*F1)/c2
		for (std::size_t k = 0; k < DIM; k++) {
			F1[k] = 0;
			for (std::size_t j = 0; j < DIM; j++) {
				F1[k] += Cinv(k,j)*C2F[j];
			}
			F2[k] = (F[k] - c1*F1[k])/c2;
		}
	}

	T W(std::size_t i, const T* F) const
	{
		typename MixedMaterialLawBase<T, _P>::pPhase p1;
		typename MixedMaterialLawBase<T, _P>::pPhase p2;
		Tensor3x3<T> F1, F2;
		T c1 = 0, c2 = 0;

		this->get_mix(i, F, p1, p2, F1, F2, c1, c2);

		T W = c1*p1->law->W(i, F1);
		if (p2) { 
			W += c2*p2->law->W(i, F2);
		}

		return W;
	}

	void PK1(std::size_t i, const T* F, T alpha, bool gamma, T* P) const
	{
		typename MixedMaterialLawBase<T, _P>::pPhase p1;
		typename MixedMaterialLawBase<T, _P>::pPhase p2;
		Tensor3x3<T> F1, F2;
		T c1 = 0, c2 = 0;

		this->get_mix(i, F, p1, p2, F1, F2, c1, c2);

		p1->law->PK1(i, F1, c1*alpha, gamma, P);
		if (p2) { 
			p2->law->PK1(i, F2, c2*alpha, true, P);
		}
	}

	void dPK1(std::size_t i, const T* F, T alpha, bool gamma, const T* W, T* dP, std::size_t n = 1) const
	{
		typename MixedMaterialLawBase<T, _P>::pPhase p1;
		typename MixedMaterialLawBase<T, _P>::pPhase p2;
		Tensor3x3<T> F1, F2;
		T c1 = 0, c2 = 0;

		this->get_mix(i, F, p1, p2, F1, F2, c1, c2);

		p1->law->dPK1(i, F1, c1*alpha, gamma, W, dP, n);
		if (p2) { 
			p2->law->dPK1(i, F2, c2*alpha, true, W, dP, n);
		}
	}

	std::string str() const
	{
		return (boost::format("iso mixed")).str();
	}
};


//! Mixed material which uses the laminate mixing rule at interfaces
template<typename T, typename _P, int DIM>
class LaminateMixedMaterialLaw : public MixedMaterialLaw<T, _P, DIM>
{
public:
	// interface normal field
	boost::shared_ptr< TensorField<T> > normals;
	T eps; 		// tolerance
	T eps_t;	// backtracking tolerance
	T eps_a;	// jump vector tolerance
	T eps_g;	// gradient norm tolerance
	T alpha;	// backtracking parameter
	T beta;		// backtracking parameter
	T delta;	// shrinking factor for projection to admissible set
	T fixed_c1;
	T fd_delta;
	std::size_t maxiter;
	bool comp_fix, backtrack, project_t, debug;
	std::string tangent;

	std::size_t _max_newtoniter, _max_backtrack, _calls;
	std::size_t _sum_newtoniter, _sum_backtrack;

	LaminateMixedMaterialLaw(boost::shared_ptr< TensorField<T> > normals)
	{
		this->normals = normals;
		eps_t = 4*std::numeric_limits<T>::epsilon();
		eps_a = std::pow(std::numeric_limits<T>::epsilon(), 2.0/3.0);
		eps_g = std::numeric_limits<T>::epsilon();
		alpha = 0.001;
		beta = 0.1;
		delta = 1 - 1024*std::numeric_limits<T>::epsilon();
		maxiter = 32;
		comp_fix = false;
		tangent = "approx";
		fd_delta = std::pow(std::numeric_limits<T>::epsilon(), 1.0/3.0);
		fixed_c1 = -1.0;
		backtrack = true;
		project_t = true;
		debug = false;

		_max_newtoniter = _max_backtrack = _calls = 0;
		_sum_newtoniter = _sum_backtrack = 0;
	}

	// read settings from ptree
	void readSettings(const ptree::ptree& _pt)
	{
		const ptree::ptree& pt = _pt.get_child("laminate_mixing", empty_ptree);

		eps_t = pt_get<T>(pt, "eps_t", eps_t);
		eps_a = pt_get<T>(pt, "eps_a", eps_a);
		eps_g = pt_get<T>(pt, "eps_g", eps_g);
		alpha = pt_get<T>(pt, "alpha", alpha);
		beta = pt_get<T>(pt, "beta", beta);
		delta = pt_get<T>(pt, "delta", delta);
		fd_delta = pt_get<T>(pt, "fd_delta", fd_delta);
		fixed_c1 = pt_get<T>(pt, "fixed_c1", fixed_c1);
		tangent = pt_get<std::string>(pt, "tangent", tangent);
		maxiter = pt_get<std::size_t>(pt, "maxiter", maxiter);
		comp_fix = pt_get<bool>(pt, "comp_fix", comp_fix);
		debug = pt_get<bool>(pt, "debug", debug);
		backtrack = pt_get<bool>(pt, "backtrack", backtrack);
		project_t = pt_get<bool>(pt, "project_t", project_t);
	}

	// newton method
	// i: cell index (should not be used)
	// law1, law2: phase material laws
	// c1, c2: concentrations
	// n: interface normal vector
	// Fbar: prescribed strain
	// F1, F2: the strains in each phase (the solution)
	virtual void solve_newton(std::size_t _i, MaterialLaw<T>& law1, MaterialLaw<T>& law2, _P c1, _P c2, const Tensor3<T>& n, const T* Fbar, T* F1, T* F2) const
	{
	//	Timer __t("laminate solve_newton", false);

		// Note: a, a_next, da live in the rotated (world) coordinate system
		
#if 0
		#define DEBUG_LAMINATE(x) LOG_COUT << x << std::endl;
#else
		#define DEBUG_LAMINATE(x)
#endif

		Tensor3x3<T> P1, P2, dP1da, dP2da, Fbarinv, RT;
		Tensor3x3<T> dF1da[3], dF2da[3];
		Tensor3<T> a, da, a_next, g, Hinvg, FbarinvHinvg, Fbarinva;
		SymTensor3x3<T> H, Hinv;
		T W_next;

		// compute rotation matrix from lamination direction (e1, e2 or e3) to n (RT*e1 = n)
#define ROT_LAMINATE 0
#if ROT_LAMINATE
		Tensor3<T> e1;
		e1[0] = 1; e1[1] = e1[2] = 0;
		RT.rot(e1, n);
#else
		RT.eye();
#endif

		// helper for voigt notation
		static const std::size_t row[9] = {0, 1, 2, 1, 0, 0, 2, 2, 1};
		static const std::size_t col[9] = {0, 1, 2, 2, 2, 1, 1, 0, 0};
		static const std::size_t row_indices[3][3] = {{0, 8, 7}, {5, 1, 6}, {4, 3, 2}};

		if (DIM == 9) {
			// compute Fbar^{-1}
			Fbarinv.inv(Fbar);
		}

		// init a
		a.zero();

		// set initial F1 and F2
		// F1 = Fbar
		// F2 = Fbar
		for (std::size_t i = 0; i < DIM; i++) {
			F1[i] = F2[i] = Fbar[i];
		}
		this->fix_dim(F1); this->fix_dim(F2);

		// compute initial energy W
		T W = c1*law1.W(_i, F1) + c2*law2.W(_i, F2);

		if (std::isnan(W) || std::isinf(W)) {
			set_exception((boost::format("laminate energy inf or NaN").str()));
		}

		DEBUG_LAMINATE("\n-------------- LAMINATE Solver");
		DEBUG_LAMINATE("n " << n[0] << " " << n[1] << " " << n[2]);
		DEBUG_LAMINATE("F1 " << format(Tensor3x3<T>(F1)));
		DEBUG_LAMINATE("F2 " << format(Tensor3x3<T>(F2)));
		DEBUG_LAMINATE("W " << W);
		DEBUG_LAMINATE("c1 " << c1);
		DEBUG_LAMINATE("c2 " << c2);

		std::size_t backtrack_iter = 0, iter = 0;

		for (; ; iter++)
		{
			DEBUG_LAMINATE("# Newton iteration " << iter);
			DEBUG_LAMINATE("W " << W);
			DEBUG_LAMINATE("a " << a[0] << " " << a[1] << " " << a[2]);
			DEBUG_LAMINATE("F1 " << format(Tensor3x3<T>(F1)));
			DEBUG_LAMINATE("F2 " << format(Tensor3x3<T>(F2)));
			DEBUG_LAMINATE("det(F1) " << Tensor3x3<T>(F1).det());
			DEBUG_LAMINATE("det(F2) " << Tensor3x3<T>(F2).det());

			// compute derivatives dF1/da and dF2/da
			for (std::size_t k = 0; k < 3; k++)
			{
				for (std::size_t i = 0; i < 9; i++) {
					dF1da[k][i] = -c2*RT[row_indices[k][row[i]]]*n[col[i]];
					dF2da[k][i] =  c1*RT[row_indices[k][row[i]]]*n[col[i]];
				}

				// symmetrize for linear elasticity
				this->fix_sym(dF1da[k]); this->fix_sym(dF2da[k]);
				
				DEBUG_LAMINATE("dF1da" << k << ": " << format(dF1da[k]));
				DEBUG_LAMINATE("dF2da" << k << ": " << format(dF2da[k]));
			}

			// compute gradient g=dW/da
			law1.PK1(_i, F1, 1, false, P1); this->fix_dim(P1);
			law2.PK1(_i, F2, 1, false, P2); this->fix_dim(P2);
			for (std::size_t k = 0; k < 3; k++) {
				g[k] = c1*P1.dot(dF1da[k]) + c2*P2.dot(dF2da[k]);
			}

			T g_norm = ublas::norm_2(g);

			DEBUG_LAMINATE("P1: " << format(P1));
			DEBUG_LAMINATE("P2: " << format(P2));
			DEBUG_LAMINATE("gradient: " << format(g));
			DEBUG_LAMINATE("gradient norm: " << g_norm);

			if (g_norm <= this->eps_g) {
				DEBUG_LAMINATE("converged g_norm <= eps_g");
				break;
			}

			// compute hessian H=d^2W/da^2
			for (std::size_t i = 0; i < 6; i++) {
				std::size_t k = row[i];
				std::size_t l = col[i];
				law1.dPK1(_i, F1, 1, false, dF1da[l], dP1da); this->fix_dim(dP1da);
				law2.dPK1(_i, F2, 1, false, dF2da[l], dP2da); this->fix_dim(dP2da);
				H[i] = c1*dP1da.dot(dF1da[k]) + c2*dP2da.dot(dF2da[k]);
			}

			DEBUG_LAMINATE("hessian: " << format(H));

#if 0
			if (DIM == 3) {
				T trH = H[0] + H[1] + H[2];
				H[0] = H[1] = H[2] = trH;
				T trg = g[0] + g[1] + g[2];
				g[0] = g[1] = g[2] = trg;
			}
#endif

			// compute H^{-1}, H^{-1}g
			Hinv.inv(H);
			Hinvg.mult(Hinv, g);

			// compute (negative) step direction da
#if ROT_LAMINATE
			da.mult(RT, Hinvg);
#else
			da = Hinvg;
#endif
			T gTda = da.dot(g);

			// compute decrement and gradient norm
#if 1
			T da_norm = ublas::norm_2(Hinvg);
			DEBUG_LAMINATE("norm da: " << da_norm);

			// check convergence
			if (da_norm <= eps_a) {
				DEBUG_LAMINATE("converged da_norm <= eps_a");
				break;
			}
#else
			T lambda2 = g.dot(Hinvg);
			DEBUG_LAMINATE("lambda2: " << lambda2);
			DEBUG_LAMINATE("lambda2/W: " << (lambda2/W));

			// check convergence
			if (lambda2 <= 2*eps_a*W) {
				DEBUG_LAMINATE("converged lambda2 <= 2*eps_a*W");
				break;
			}
#endif

			// compute maximum feasible step length (only in the nonlinear case)
			T t = 1;
			if (DIM == 9 && project_t) {
				FbarinvHinvg.mult(Fbarinv, Hinvg);
				T w = FbarinvHinvg.dot(n);

				Fbarinva.mult(Fbarinv, a);
				T x = Fbarinva.dot(n);

				DEBUG_LAMINATE("w: " << w);
				DEBUG_LAMINATE("t1: " << (delta/c1 + x)/w);
				DEBUG_LAMINATE("t2: " << (-delta/c2 + x)/w);

				if (w > 0) {
					t = std::min((T)1, (x + delta/c1)/w);
				}
				else if (w < 0) {
					t = std::min((T)1, (x - delta/c2)/w);
				}
			}

			// perform backtracking
			for (;;)
			{
				// compute next a
				for (std::size_t i = 0; i < 3; i++) {
					a_next[i] = a[i] - t*da[i];
				}
				
				// compute next F1 and F2
				// F1 = Fbar - c2*a_next*n^T
				// F2 = Fbar + c1*a_next*n^T
				for (std::size_t i = 0; i < DIM; i++) {
					F1[i] = F2[i] = Fbar[i];
				}
				this->fix_dim(F1); this->fix_dim(F2);
				for (std::size_t i = 0; i < 9; i++) {
					F1[i] -= c2*a_next[row[i]]*n[col[i]];
					F2[i] += c1*a_next[row[i]]*n[col[i]];
				}
				this->fix_sym(F1); this->fix_sym(F2);

				DEBUG_LAMINATE("det(F1) = " << Tensor3x3<T>::det(F1));
				DEBUG_LAMINATE("det(F2) = " << Tensor3x3<T>::det(F2));
				
#if 1
				if (DIM != 9) {
					// TODO: we can break here only if it is a linear problem
					return;
				}
#endif

				// compute next energy
				W_next = c1*law1.W(_i, F1) + c2*law2.W(_i, F2);
				
				if (!backtrack) {
					break;
				}

				DEBUG_LAMINATE("backtracking: t=" << t << " W=" << std::setprecision(16) << W_next);
				
				// check if we have sufficient decrease
				if (W_next < W - alpha*t*gTda) {
					break;
				}

				t *= beta;
				backtrack_iter++;

				if (t <= eps_t) {
					break;
				}
			}

			if (t <= eps_t) {
				DEBUG_LAMINATE("converged t <= eps_t");
				break;
			}

			DEBUG_LAMINATE("a " << format(a_next));

			// perform step
			a = a_next;
			W = W_next;

			if (iter >= maxiter)
			{
				DEBUG_LAMINATE("converged iter >= maxiter");
				break; // TODO: improve stopping criteria

				Tensor3x3<T> Fbar2(Fbar);
				this->fix_sym(Fbar2);
				
				LOG_COUT << "i = " << _i << std::endl;
				LOG_COUT << "c1 = " << c1 << std::endl;
				LOG_COUT << "c2 = " << c2 << std::endl;
				LOG_COUT << "c1+c2-1 = " << (c1+c2-1) << std::endl;
				LOG_COUT << "n = " << format(n) << " norm = " << ublas::norm_2(n) << std::endl;
				LOG_COUT << "Fbar = " << format(Fbar2) << std::endl;
				LOG_COUT << "det(Fbar) = " << Fbar2.det() << std::endl;

				SaintVenantKirchhoffMaterialLaw<T>* svk1 = dynamic_cast<SaintVenantKirchhoffMaterialLaw<T>*>(&law1);
				if (svk1 != NULL) {
					LOG_COUT << "mu1=" << svk1->mu << "lambda1=" << svk1->lambda << std::endl;
				}
				SaintVenantKirchhoffMaterialLaw<T>* svk2 = dynamic_cast<SaintVenantKirchhoffMaterialLaw<T>*>(&law2);
				if (svk2 != NULL) {
					LOG_COUT << "mu2=" << svk2->mu << "lambda2=" << svk2->lambda << std::endl;
				}

				set_exception((boost::format("Newton solver did not converge da_norm=%g eps_a=%g eps_t=%g") % da_norm % eps_a % eps_t).str());
			}
		}

		if (debug) {
			#pragma omp critical
			{
				*const_cast<std::size_t*>(&_max_backtrack) = std::max(backtrack_iter, _max_backtrack);
				*const_cast<std::size_t*>(&_sum_backtrack) += backtrack_iter;
				*const_cast<std::size_t*>(&_max_newtoniter) = std::max(iter, _max_newtoniter);
				*const_cast<std::size_t*>(&_sum_newtoniter) += iter;
				*const_cast<std::size_t*>(&_calls) += 1;
			}
		}

#if 0
		const T J1 = Tensor3x3<T>::det(F1);
		const T J2 = Tensor3x3<T>::det(F2);

		if (J1 <= 0 || J2 <= 0) {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("newton_solve problem").str())));
		}
#endif
	}

	inline void get_mix(std::size_t i, const T* F,
		typename MixedMaterialLawBase<T, _P>::pPhase& p1,
		typename MixedMaterialLawBase<T, _P>::pPhase& p2,
		T* F1, T* F2, _P& c1, _P& c2) const
	{
		for (std::size_t p = 0; p < this->phases.size(); p++) {
			_P phi = this->phases[p]->phi[i];
			if (phi == 0) continue;
			if (phi == 1) {
				c1 = phi;
				std::memcpy(F1, F, DIM*sizeof(T));
				p1 = this->phases[p];
				p2.reset();
				return;
			}
			if (!p1) { p1 = this->phases[p]; c1 = phi; continue; }
			if (!p2) { p2 = this->phases[p]; c2 = phi; continue; }
			BOOST_THROW_EXCEPTION(std::runtime_error("The laminate mixing rule supports only two phase mixtures"));
		}

		if (!p1) { 
			BOOST_THROW_EXCEPTION(std::runtime_error("The laminate mixing rule supports only two phase mixtures"));
		}

		if (!p2) { 
			std::memcpy(F1, F, DIM*sizeof(T));
			return;
		}

		Tensor3<T> n;
		n[0] = (*this->normals)[0][i];
		n[1] = (*this->normals)[1][i];
		n[2] = (*this->normals)[2][i];

		Tensor3x3<T> Fbar;
		for (std::size_t k = 0; k < DIM; k++) {
			Fbar[k] = F[k];
		}
		this->fix_dim(Fbar);

		if (this->fixed_c1 < -1.5) {
			#pragma omp barrier
			// automatic c1
			if (omp_get_thread_num() == 0) {
				std::size_t nc = 1;
				T c1_sum = 0;
				typename MixedMaterialLawBase<T, _P>::pPhase p1 = this->phases[0];
				TensorField<T>& phi1 = *(p1->_phi);
				BEGIN_TRIPLE_LOOP(k, phi1.nx, phi1.ny, phi1.nz, phi1.nzp) {
					_P phi = phi1[0][k];
					if (phi > 0 && phi < 1) {
						// composite voxel
						c1_sum += phi;
						nc ++;
					}
				}
				END_TRIPLE_LOOP(k)
				*const_cast<T*>(&(this->fixed_c1)) = c1_sum/nc;
				LOG_COUT << "fixed_c1=" << this->fixed_c1 << std::endl;
			}
			#pragma omp barrier
		}

		if (this->fixed_c1 > 0) {
			c1 = (_P) this->fixed_c1;
		}

		c2 = 1.0 - c1;
		solve_newton(i, *(p1->law), *(p2->law), c1, c2, n, Fbar, F1, F2);
	}

	T W(std::size_t i, const T* F) const
	{
		typename MixedMaterialLawBase<T, _P>::pPhase p1, p2;
		Tensor3x3<T> F1, F2;
		_P c1 = 0, c2 = 0;

		this->get_mix(i, F, p1, p2, F1, F2, c1, c2);

		T W = c1*p1->law->W(i, F1);
		if (p2) { 
			W += c2*p2->law->W(i, F2);
		}

		return W;
	}

	void PK1(std::size_t i, const T* F, T alpha, bool gamma, T* P) const
	{
		typename MixedMaterialLawBase<T, _P>::pPhase p1, p2;
		Tensor3x3<T> F1, F2;
		_P c1 = 0, c2 = 0;

		// NOTE: W(F) = c1*W1(F1(F)) + c2*W2(F2(F))
		// we assume dF1/dF = Id and dF2/dF = Id, i.e. the interface jump da/dF = 0
		// (ignoring higher order derivatives)

		this->get_mix(i, F, p1, p2, F1, F2, c1, c2);

		p1->law->PK1(i, F1, c1*alpha, gamma, P);
		if (p2) { 
			p2->law->PK1(i, F2, c2*alpha, true, P);
		}

#if 0
if (this->comp_fix) {
		// compressibility fix
		T P1_F[DIM];
		T P2_F[DIM];
		T tr_P12;
		
		p1->law->PK1(i, F, c1*alpha, false, P1_F);
		tr_P12 = P1_F[0] + P1_F[1] + P1_F[2];
		if (p2) { 
			p2->law->PK1(i, F, c2*alpha, false, P2_F);
			tr_P12 += P2_F[0] + P2_F[1] + P2_F[2];
		}
		
		Tensor3<T> n;
		n[0] = (*this->normals)[0][i];
		n[1] = (*this->normals)[1][i];
		n[2] = (*this->normals)[2][i];
		
		static std::size_t row[9] = {0, 1, 2, 1, 0, 0, 2, 2, 1};
		static std::size_t col[9] = {0, 1, 2, 2, 2, 1, 1, 0, 0};

		T P1[DIM];
		T P2[DIM];
		
		p1->law->PK1(i, F1, c1*alpha, false, P1);
		T tr_P = P1[0] + P1[1] + P1[2];
		if (p2) { 
			p2->law->PK1(i, F2, c2*alpha, false, P2);
			tr_P += P2[0] + P2[1] + P2[2];
		}
		
		for (int i = 0; i < DIM; i++) {
			P[i] += n[row[i]]*n[col[i]]*(tr_P12 - tr_P);
		}
}
#endif
	}

	void dPK1(std::size_t i, const T* F, T alpha, bool gamma, const T* W, T* dP, std::size_t n = 1) const
	{
		if (tangent == "fd") {
			this->dPK1_fd(i, F, alpha, gamma, W, dP, n, DIM, fd_delta);
		}

		typename MixedMaterialLawBase<T, _P>::pPhase p1, p2;
		Tensor3x3<T> F1, F2;
		_P c1 = 0, c2 = 0;

		this->get_mix(i, F, p1, p2, F1, F2, c1, c2);

		if (tangent != "exact")
		{
			// NOTE: P(F) = c1*P1(F1(F)) + c2*P2(F2(F))
			//       dP(F)/dF : W = c1*dP1(F1(F))/dF1:dF1/dF:W + c2*dP2(F2(F))/dF2:dF2/dF:W
			// but we are actually computing
			//       dP(F)/dF : W = c1*dP1(F1(F))/dF1:W + c2*dP2(F2(F))/dF2:W
			// i.e. we assume dF1/dF = Id and dF2/dF = Id, i.e. da/dF = 0

			p1->law->dPK1(i, F1, c1*alpha, gamma, W, dP, n);
			if (p2) { 
				p2->law->dPK1(i, F2, c2*alpha, true, W, dP, n);
			}

			return;
		}

		// The exact case:
		// dF1/dF = Id - c2 da/dF \otimes n
		// dF2/dF = Id + c1 da/dF \otimes n

		// get interface normal vector
		Tensor3<T> nm;
		nm[0] = (*this->normals)[0][i];
		nm[1] = (*this->normals)[1][i];
		nm[2] = (*this->normals)[2][i];

		// build equation system
		ublas::c_matrix<T,3,3> A, Ainv;

		// system matrix
		// using the solution xi = xi_1 e_1 + xi_2 e_2 + xi_3 e_3 = da/dF : W
		for (int m = 0; m < 3; m++)
		{
			Tensor3<T> ei;
			ei[0] = (m == 0);
			ei[1] = (m == 1);
			ei[2] = (m == 2);

			// direction e_i \otimes n
			Tensor3x3<T> D;
			D[0] = nm[0]*ei[0];
			D[5] = nm[1]*ei[0];
			D[4] = nm[2]*ei[0];

			D[8] = nm[0]*ei[1];
			D[1] = nm[1]*ei[1];
			D[3] = nm[2]*ei[1];
			
			D[7] = nm[0]*ei[2];
			D[6] = nm[1]*ei[2];
			D[2] = nm[2]*ei[2];

			Tensor3x3<T> L;
			p1->law->dPK1(i, F1, c2, false, D.data(), L.data(), 1);
			if (p2) { 
				p1->law->dPK1(i, F1, c2, true, D.data(), L.data(), 1);
			}

			A(0, i) = L[0]*nm[0] + L[5]*nm[1] + L[4]*nm[2];
			A(1, i) = L[8]*nm[0] + L[1]*nm[1] + L[3]*nm[2];
			A(2, i) = L[7]*nm[0] + L[6]*nm[1] + L[2]*nm[2];
		}

		InvertMatrix<T, 3>(A, Ainv);

		// TODO: we currently compute da/dF:W \otimes n instad of da/dF \otimes n : W
		// needs to befixed (

		// solve systems
		for (std::size_t m = 0; m < n; m++)
		{
			// build rhs
			Tensor3x3<T> L;
			p1->law->dPK1(i, F1, -1.0, false, W, L.data(), 1);
			if (p2) { 
				p2->law->dPK1(i, F2, 1.0, true, W, L.data(), 1);
			}
			this->fix_dim(L);
			
			ublas::c_vector<T,3> b;
			b[0] = L[0]*nm[0] + L[5]*nm[1] + L[4]*nm[2];
			b[1] = L[8]*nm[0] + L[1]*nm[1] + L[3]*nm[2];
			b[2] = L[7]*nm[0] + L[6]*nm[1] + L[2]*nm[2];

			ublas::c_vector<T,3> da_dF_W; // = ublas::prod(Ainv, b);
			ublas::axpy_prod(Ainv, b, da_dF_W, true);

			Tensor3x3<T> da_dF_W_n;

			da_dF_W_n[0] = nm[0]*da_dF_W(0);
			da_dF_W_n[5] = nm[1]*da_dF_W(0);
			da_dF_W_n[4] = nm[2]*da_dF_W(0);

			da_dF_W_n[8] = nm[0]*da_dF_W(1);
			da_dF_W_n[1] = nm[1]*da_dF_W(1);
			da_dF_W_n[3] = nm[2]*da_dF_W(1);
			
			da_dF_W_n[7] = nm[0]*da_dF_W(2);
			da_dF_W_n[6] = nm[1]*da_dF_W(2);
			da_dF_W_n[2] = nm[2]*da_dF_W(2);

			Tensor3x3<T> W1, W2;
			for (std::size_t k = 0; k < 9; k++) {
				W1[k] = c1*W[k] + c1*c2*da_dF_W_n[k];
				W2[k] = c2*W[k] - c1*c2*da_dF_W_n[k];
			}

			p1->law->dPK1(i, F1, alpha, gamma, W1.data(), dP, 1);
			if (p2) { 
				p2->law->dPK1(i, F2, alpha, true, W2.data(), dP, 1);
			}

			W += DIM;
			dP += DIM;
		}
	}

	std::string str() const
	{
		return (boost::format("laminate mixed")).str();
	}
};


//! Mixed material which uses a special laminate mixing rule at interfaces
template<typename T, typename _P, int DIM>
class InfinityLaminateMixedMaterialLaw : public LaminateMixedMaterialLaw<T, _P, DIM>
{
public:
	InfinityLaminateMixedMaterialLaw(boost::shared_ptr< TensorField<T> > normals)
		: LaminateMixedMaterialLaw<T, _P, DIM>(normals)
	{
	}

	// newton method
	// i: cell index (should not be used)
	// law1, law2: phase material laws
	// c1, c2: concentrations
	// n: interface normal vector
	// Fbar: prescribed strain
	// F1, F2: the strains in each phase (the solution)
	virtual void solve_newton(std::size_t _i, MaterialLaw<T>& law1, MaterialLaw<T>& law2, _P c1, _P c2, const Tensor3<T>& n, const T* Fbar, T* F1, T* F2) const
	{
#if 1
	//	Timer __t("laminate solve_newton", false);

		// Note: a, a_next, da live in the rotated (world) coordinate system
		
#if 0
		#define DEBUG_INFLAMINATE(x) LOG_COUT << x << std::endl;
#else
		#define DEBUG_INFLAMINATE(x)
#endif

		Tensor3x3<T> P1, P2, dP1da, dP2da, Fbarinv, RT;
		Tensor3x3<T> dF1da[3], dF2da[3];
		Tensor3<T> a, da, a_next, g, Hinvg, FbarinvHinvg, Fbarinva;
		SymTensor3x3<T> H, Hinv;
		T W_next;

		// compute rotation matrix from lamination direction (e1, e2 or e3) to n (RT*e1 = n)
#define ROT_LAMINATE 0
#if ROT_LAMINATE
		Tensor3<T> e1;
		e1[0] = 1; e1[1] = e1[2] = 0;
		RT.rot(e1, n);
#else
		RT.eye();
#endif

		// helper for voigt notation
		static const std::size_t row[9] = {0, 1, 2, 1, 0, 0, 2, 2, 1};
		static const std::size_t col[9] = {0, 1, 2, 2, 2, 1, 1, 0, 0};
		static const std::size_t row_indices[3][3] = {{0, 8, 7}, {5, 1, 6}, {4, 3, 2}};

		if (DIM == 9) {
			// compute Fbar^{-1}
			Fbarinv.inv(Fbar);
		}

		// init a
		a.zero();

		// set initial F1 and F2
		// F1 = Fbar
		// F2 = Fbar
		for (std::size_t i = 0; i < DIM; i++) {
			F1[i] = F2[i] = Fbar[i];
		}
		this->fix_dim(F1); this->fix_dim(F2);

		T w1 = c1, w2 = c2;
		T q1 = 0.5, q2 = 0.5;

		// compute initial energy W
		T W = w1*law1.W(_i, F1) + w2*law2.W(_i, F2);

		if (std::isnan(W) || std::isinf(W)) {
			set_exception((boost::format("laminate energy inf or NaN").str()));
		}

		DEBUG_INFLAMINATE("\n-------------- LAMINATE Solver");
		DEBUG_INFLAMINATE("n " << n[0] << " " << n[1] << " " << n[2]);
		DEBUG_INFLAMINATE("F1 " << format(Tensor3x3<T>(F1)));
		DEBUG_INFLAMINATE("F2 " << format(Tensor3x3<T>(F2)));
		DEBUG_INFLAMINATE("W " << W);
		DEBUG_INFLAMINATE("c1 " << c1);
		DEBUG_INFLAMINATE("c2 " << c2);

		std::size_t backtrack_iter = 0, iter = 0;

		for (; ; iter++)
		{
			DEBUG_INFLAMINATE("# Newton iteration " << iter);
			DEBUG_INFLAMINATE("W " << W);
			DEBUG_INFLAMINATE("a " << a[0] << " " << a[1] << " " << a[2]);
			DEBUG_INFLAMINATE("F1 " << format(Tensor3x3<T>(F1)));
			DEBUG_INFLAMINATE("F2 " << format(Tensor3x3<T>(F2)));
			DEBUG_INFLAMINATE("det(F1) " << Tensor3x3<T>(F1).det());
			DEBUG_INFLAMINATE("det(F2) " << Tensor3x3<T>(F2).det());

			// compute derivatives dF1/da and dF2/da
			for (std::size_t k = 0; k < 3; k++)
			{
				for (std::size_t i = 0; i < 9; i++) {
					dF1da[k][i] =  q2*RT[row_indices[k][row[i]]]*n[col[i]];
					dF2da[k][i] = -q1*RT[row_indices[k][row[i]]]*n[col[i]];
				}

				// symmetrize for linear elasticity
				this->fix_sym(dF1da[k]); this->fix_sym(dF2da[k]);

				DEBUG_INFLAMINATE("dF1da" << k << ": " << format(dF1da[k]));
				DEBUG_INFLAMINATE("dF2da" << k << ": " << format(dF2da[k]));
			}

			// compute gradient g=dW/da
			law1.PK1(_i, F1, 1, false, P1); this->fix_dim(P1);
			law2.PK1(_i, F2, 1, false, P2); this->fix_dim(P2);
			for (std::size_t k = 0; k < 3; k++) {
				g[k] = w1*P1.dot(dF1da[k]) + w2*P2.dot(dF2da[k]);
			}

			T g_norm = ublas::norm_2(g);

			DEBUG_INFLAMINATE("P1: " << format(P1));
			DEBUG_INFLAMINATE("P2: " << format(P2));
			DEBUG_INFLAMINATE("gradient: " << format(g));
			DEBUG_INFLAMINATE("gradient norm: " << g_norm);

			if (g_norm <= this->eps_g) {
				DEBUG_INFLAMINATE("converged g_norm <= eps_g");
				break;
			}

			// compute hessian H=d^2W/da^2
			for (std::size_t i = 0; i < 6; i++) {
				std::size_t k = row[i];
				std::size_t l = col[i];
				law1.dPK1(_i, F1, 1, false, dF1da[l], dP1da); this->fix_dim(dP1da);
				law2.dPK1(_i, F2, 1, false, dF2da[l], dP2da); this->fix_dim(dP2da);
				H[i] = w1*dP1da.dot(dF1da[k]) + w2*dP2da.dot(dF2da[k]);
			}

			DEBUG_INFLAMINATE("hessian: " << format(H));

#if 0
			if (DIM == 3) {
				T trH = H[0] + H[1] + H[2];
				H[0] = H[1] = H[2] = trH;
				T trg = g[0] + g[1] + g[2];
				g[0] = g[1] = g[2] = trg;
			}
#endif

			// compute H^{-1}, H^{-1}g
			Hinv.inv(H);
			Hinvg.mult(Hinv, g);

			// compute (negative) step direction da
#if ROT_LAMINATE
			da.mult(RT, Hinvg);
#else
			da = Hinvg;
#endif
			T gTda = da.dot(g);

			// compute decrement and gradient norm
#if 1
			T da_norm = ublas::norm_2(Hinvg);
			DEBUG_INFLAMINATE("norm da: " << da_norm);

			// check convergence
			if (da_norm <= this->eps_a) {
				DEBUG_INFLAMINATE("converged da_norm <= eps_a");
				break;
			}
#else
			T lambda2 = g.dot(Hinvg);
			DEBUG_INFLAMINATE("lambda2: " << lambda2);
			DEBUG_INFLAMINATE("lambda2/W: " << (lambda2/W));

			// check convergence
			if (lambda2 <= 2*eps_a*W) {
				DEBUG_INFLAMINATE("converged lambda2 <= 2*eps_a*W");
				break;
			}
#endif

			// compute maximum feasible step length (only in the nonlinear case)
			T t = 1;
			if (DIM == 9 && this->project_t) {
				/*
				TODO:
				FbarinvHinvg.mult(Fbarinv, Hinvg);
				T w = FbarinvHinvg.dot(n);

				Fbarinva.mult(Fbarinv, a);
				T x = Fbarinva.dot(n);

				DEBUG_INFLAMINATE("w: " << w);
				DEBUG_INFLAMINATE("t1: " << (delta/c1 + x)/w);
				DEBUG_INFLAMINATE("t2: " << (-delta/c2 + x)/w);

				if (w > 0) {
					t = std::min((T)1, (x + delta/c1)/w);
				}
				else if (w < 0) {
					t = std::min((T)1, (x - delta/c2)/w);
				}
				*/
			}

			// perform backtracking
			for (;;)
			{
				// compute next a
				for (std::size_t i = 0; i < 3; i++) {
					a_next[i] = a[i] - t*da[i];
				}
				
				// compute next F1 and F2
				// F1 = Fbar - c2*a_next*n^T
				// F2 = Fbar + c1*a_next*n^T
				for (std::size_t i = 0; i < DIM; i++) {
					F1[i] = F2[i] = Fbar[i];
				}
				this->fix_dim(F1); this->fix_dim(F2);
				for (std::size_t i = 0; i < 9; i++) {
					F1[i] +=  q2*a_next[row[i]]*n[col[i]];
					F2[i] += -q1*a_next[row[i]]*n[col[i]];
				}
				this->fix_sym(F1); this->fix_sym(F2);

				DEBUG_INFLAMINATE("det(F1) = " << Tensor3x3<T>::det(F1));
				DEBUG_INFLAMINATE("det(F2) = " << Tensor3x3<T>::det(F2));
				
#if 1
				if (DIM != 9) {
					DEBUG_INFLAMINATE("a " << a_next[0] << " " << a_next[1] << " " << a_next[2]);
					DEBUG_INFLAMINATE("F1 " << format(Tensor3x3<T>(F1)));
					DEBUG_INFLAMINATE("F2 " << format(Tensor3x3<T>(F2)));

					// TODO: we can break here only if it is a linear problem
					return;
				}
#endif

				// compute next energy
				W_next = w1*law1.W(_i, F1) + w2*law2.W(_i, F2);
				
				if (!this->backtrack) {
					break;
				}

				DEBUG_INFLAMINATE("backtracking: t=" << t << " W=" << std::setprecision(16) << W_next);
				
				// check if we have sufficient decrease
				if (W_next < W - this->alpha*t*gTda) {
					break;
				}

				t *= this->beta;
				backtrack_iter++;

				if (t <= this->eps_t) {
					break;
				}
			}

			if (t <= this->eps_t) {
				DEBUG_INFLAMINATE("converged t <= eps_t");
				break;
			}

			DEBUG_INFLAMINATE("a " << format(a_next));

			// perform step
			a = a_next;
			W = W_next;

			if (iter >= this->maxiter)
			{
				DEBUG_INFLAMINATE("converged iter >= maxiter");
				break; // TODO: improve stopping criteria

				Tensor3x3<T> Fbar2(Fbar);
				this->fix_sym(Fbar2);
				
				LOG_COUT << "i = " << _i << std::endl;
				LOG_COUT << "c1 = " << c1 << std::endl;
				LOG_COUT << "c2 = " << c2 << std::endl;
				LOG_COUT << "c1+c2-1 = " << (c1+c2-1) << std::endl;
				LOG_COUT << "n = " << format(n) << " norm = " << ublas::norm_2(n) << std::endl;
				LOG_COUT << "Fbar = " << format(Fbar2) << std::endl;
				LOG_COUT << "det(Fbar) = " << Fbar2.det() << std::endl;

				SaintVenantKirchhoffMaterialLaw<T>* svk1 = dynamic_cast<SaintVenantKirchhoffMaterialLaw<T>*>(&law1);
				if (svk1 != NULL) {
					LOG_COUT << "mu1=" << svk1->mu << "lambda1=" << svk1->lambda << std::endl;
				}
				SaintVenantKirchhoffMaterialLaw<T>* svk2 = dynamic_cast<SaintVenantKirchhoffMaterialLaw<T>*>(&law2);
				if (svk2 != NULL) {
					LOG_COUT << "mu2=" << svk2->mu << "lambda2=" << svk2->lambda << std::endl;
				}

				set_exception((boost::format("Newton solver did not converge da_norm=%g eps_a=%g eps_t=%g") % da_norm % this->eps_a % this->eps_t).str());
			}
		}

		if (this->debug) {
			#pragma omp critical
			{
				*const_cast<std::size_t*>(&this->_max_backtrack) = std::max(backtrack_iter, this->_max_backtrack);
				*const_cast<std::size_t*>(&this->_sum_backtrack) += backtrack_iter;
				*const_cast<std::size_t*>(&this->_max_newtoniter) = std::max(iter, this->_max_newtoniter);
				*const_cast<std::size_t*>(&this->_sum_newtoniter) += iter;
				*const_cast<std::size_t*>(&this->_calls) += 1;
			}
		}

#if 0
		const T J1 = Tensor3x3<T>::det(F1);
		const T J2 = Tensor3x3<T>::det(F2);

		if (J1 <= 0 || J2 <= 0) {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("newton_solve problem").str())));
		}
#endif

#endif
	}
};


//! Mixed material which was experimentally used for the viscosity solver
template<typename T, typename _P, int DIM>
class FluidityMixedMaterialLaw : public MixedMaterialLaw<T, _P, DIM>
{
public:
	// interface normal field
	boost::shared_ptr< TensorField<T> > normals;

	FluidityMixedMaterialLaw(boost::shared_ptr< TensorField<T> > normals)
	{
		this->normals = normals;
	}
	
	inline void get_mix(std::size_t i, const T* F, Tensor3x3<T>& gamma) const
	{
		typename MixedMaterialLawBase<T, _P>::pPhase p1, p2;
		_P c1 = 0, c2 = 0;

		for (std::size_t p = 0; p < this->phases.size(); p++) {
			_P phi = this->phases[p]->phi[i];
			if (phi == 0) continue;
			if (!p1) { p1 = this->phases[p]; c1 = phi; continue; }
			if (!p2) { p2 = this->phases[p]; c2 = phi; continue; }
			BOOST_THROW_EXCEPTION(std::runtime_error("The laminate mixing rule supports only two phase mixtures"));
		}

		if (!p1) { 
			BOOST_THROW_EXCEPTION(std::runtime_error("The laminate mixing rule supports only two phase mixtures"));
		}

		if (!p2) {
			// single phase
			p1->law->PK1(i, F, 1.0, false, gamma);
			return;
		}
		
		// mixed phase

		Tensor3x3<T> sigma;
		sigma[0] = F[0];
		sigma[1] = F[1];
		sigma[2] = F[2];
		sigma[3] = sigma[6] = F[3];
		sigma[4] = sigma[7] = F[4];
		sigma[5] = sigma[8] = F[5];

		Tensor3<T> n, e1;
		n[0] = (*this->normals)[0][i];
		n[1] = (*this->normals)[1][i];
		n[2] = (*this->normals)[2][i];
		e1[0] = 1; e1[1] = e1[2] = 0;

		ScalarLinearIsotropicMaterialLaw<T>* law1 = dynamic_cast<ScalarLinearIsotropicMaterialLaw<T>*>(p1->law.get());
		ScalarLinearIsotropicMaterialLaw<T>* law2 = dynamic_cast<ScalarLinearIsotropicMaterialLaw<T>*>(p2->law.get());
		
		if (law1 == NULL || law2 == NULL) { 
			BOOST_THROW_EXCEPTION(std::runtime_error("The laminate mixing rule supports only scalar valued material laws"));
		}
		
		T f_R = 1/(c1/law1->mu + c2/law2->mu);
		T f_V = (c1*law1->mu + c2*law2->mu);

		// compute
		// gamma = R (F : (R^T sigma R)) R^T

		Tensor3x3<T> R, RT;	// rotation from n to e1
		R.rot(n, e1);
		RT.transpose(R);

		// rotate to e_x frame
		Tensor3x3<T> Rsigma, RsigmaRT;
		Rsigma.mult(R, sigma);
		RsigmaRT.mult(Rsigma, RT);
	
#if 0
		LOG_COUT << "R=" << format(R) << std::endl;
		LOG_COUT << "f_R=" << f_R << " f_V=" << f_V << std::endl;
		LOG_COUT << "c1=" << c1 << " c2=" << c2 << std::endl;
		LOG_COUT << "f1=" << law1->mu << " f2=" << law2->mu << std::endl;
#endif

		// apply F_x
		Tensor3x3<T> Fxsigma;

		Fxsigma[0] = f_R*RsigmaRT[0];
		Fxsigma[1] = f_R*RsigmaRT[1];
		Fxsigma[2] = f_R*RsigmaRT[2];
		Fxsigma[3] = Fxsigma[6] = f_R*RsigmaRT[3];
		Fxsigma[4] = Fxsigma[7] = f_V*RsigmaRT[4];
		Fxsigma[5] = Fxsigma[8] = f_V*RsigmaRT[5];

		// rotate back
		Tensor3x3<T> FxsigmaR;
		FxsigmaR.mult(Fxsigma, R);
		gamma.mult(RT, FxsigmaR);
	}

	T W(std::size_t i, const T* E) const
	{
		Tensor3<T> S;
		PK1(i, E, 1, false, S);
		return 0.5*S.dot(E);
	}

	void PK1(std::size_t _i, const T* F, T alpha, bool _gamma, T* S) const
	{
		Tensor3x3<T> gamma;
		this->get_mix(_i, F, gamma);

		// compute mixed response S = M*F
		// 0 5 4  0 5 4
		// 8 1 3  5 1 3
		// 7 6 2  4 3 2
		#define PK1_OP(OP) \
			S[0] OP alpha*gamma[0]; \
			S[1] OP alpha*gamma[1]; \
			S[2] OP alpha*gamma[2]; \
			S[3] OP alpha*gamma[3]; \
			S[4] OP alpha*gamma[4]; \
			S[5] OP alpha*gamma[5];
		if (_gamma) {
			PK1_OP(+=)
		}
		else {
			PK1_OP(=)
		}
		#undef PK1_OP
	}

	void dPK1(std::size_t _i, const T* E, T alpha, bool gamma, const T* W, T* dS, std::size_t n = 1) const
	{
		for (std::size_t m = 0; m < n; m++)
		{
			// assuming a linear law
			PK1(_i, W, alpha, gamma, dS);

			W += DIM;
			dS += DIM;
		}
	}

	std::string str() const
	{
		return (boost::format("laminate mixed fluidity")).str();
	}
};


//! Prolongation of a tensor field c to a doubly fine field f (for staggered grid)
template<typename T>
noinline void prolongate_to_dfg(const TensorField<T>& c, const TensorField<T>& f)
{
	Timer __t("prolongate_to_dfg", false);

	const std::size_t cny = c.ny;
	const std::size_t cnzp = c.nzp;
	const std::size_t cnyzp = cny*cnzp;

	const std::size_t fnx = f.nx;
	const std::size_t fny = f.ny;
	const std::size_t fnz = f.nz;
	const std::size_t fnzp = f.nzp;
	const std::size_t fnyzp = fny*fnzp;

	const int si_vec[] = {0, 0, 0,  0,  1,  1,  0,  1,  1};
	const int sj_vec[] = {0, 0, 0,  1,  0,  1,  1,  0,  1};
	const int sk_vec[] = {0, 0, 0,  1,  1,  0,  1,  1,  0};

	for (std::size_t g = 0; g < c.dim; g++)
	{
		const int si = si_vec[g];
		const int sj = sj_vec[g];
		const int sk = sk_vec[g];

		const T* const src = c[g];
		T* const dest = f[g];

		#pragma omp parallel for schedule (static)
		for (std::size_t i = 0; i < fnx; i++)
		{
			const std::size_t ii = ((i+((int)fnx+si))%fnx)/2;
			const std::size_t i0 = ii*cnyzp;

			for (std::size_t j = 0; j < fny; j++)
			{
				const std::size_t jj = ((j+((int)fny+sj))%fny)/2;
				const std::size_t j0 = jj*cnzp;

				std::size_t dd = i*fnyzp + j*fnzp;

				for (std::size_t k = 0; k < fnz; k++)
				{
					const std::size_t kk = ((k+((int)fnz+sk))%fnz)/2;
					const std::size_t k0 = kk;

					dest[dd] = src[i0 + j0 + k0];

					dd++;
				}
			}
		}
	}
}


//! Restriction of a fine tensor field f to a coarse field c (for staggered grid)
template<typename T>
noinline void restrict_from_dfg(const TensorField<T>& f, const TensorField<T>& c)
{
	Timer __t("restrict_from_dfg", false);

	const std::size_t cnx = c.nx;
	const std::size_t cny = c.ny;
	const std::size_t cnz = c.nz;
	const std::size_t cnzp = c.nzp;
	const std::size_t cnyzp = cny*cnzp;

	const std::size_t fnx = f.nx;
	const std::size_t fny = f.ny;
	const std::size_t fnz = f.nz;
	const std::size_t fnzp = f.nzp;
	const std::size_t fnyzp = fny*fnzp;

	const int si_vec[] = {0, 0, 0,  0, -1, -1,  0, -1, -1};
	const int sj_vec[] = {0, 0, 0, -1,  0, -1, -1,  0, -1};
	const int sk_vec[] = {0, 0, 0, -1, -1,  0, -1, -1,  0};

	for (std::size_t g = 0; g < c.dim; g++)
	{
		const int si = si_vec[g];
		const int sj = sj_vec[g];
		const int sk = sk_vec[g];

		const T* const src = f[g];
		T* const dest = c[g];

		#pragma omp parallel for schedule (static)
		for (std::size_t i = 0; i < cnx; i++)
		{
			const std::size_t ii = i*2;
			const std::size_t i0 = ((ii+((int)fnx+si))%fnx)*fnyzp;
			const std::size_t i1 = ((ii+1+((int)fnx+si))%fnx)*fnyzp;

			for (std::size_t j = 0; j < cny; j++)
			{
				const std::size_t jj = j*2;
				const std::size_t j0 = ((jj+((int)fny+sj))%fny)*fnzp;
				const std::size_t j1 = ((jj+1+((int)fny+sj))%fny)*fnzp;

				std::size_t dd = i*cnyzp + j*cnzp;

				for (std::size_t k = 0; k < cnz; k++)
				{
					const std::size_t kk = k*2;
					const std::size_t k0 = ((kk+((int)fnz+sk))%fnz);
					const std::size_t k1 = ((kk+1+((int)fnz+sk))%fnz);

					dest[dd] = 0.125*(
						src[i0 + j0 + k0] +
						src[i1 + j0 + k0] +
						src[i0 + j1 + k0] +
						src[i1 + j1 + k0] +
						src[i0 + j0 + k1] +
						src[i1 + j0 + k1] +
						src[i0 + j1 + k1] +
						src[i1 + j1 + k1]
					);

					dd++;
				}
			}
		}
	}
}


//! Base class for error estimators (for abstaction of stopping criteria of iterative solvers)
template<typename T>
class ErrorEstimator
{
public:
	typedef TensorField<T> RealTensor;
	typedef boost::shared_ptr<RealTensor> pRealTensor;

	//! see https://stackoverflow.com/questions/827196/virtual-default-destructors-in-c/827205
	virtual ~ErrorEstimator() { }

	//! update the current error estimates
	virtual void update() {
		BOOST_THROW_EXCEPTION(std::runtime_error("Selected error estimator is not compatible with the selected solution method"));
	}
	
	//! update the current error estimates (used by conjugate gradients)
	virtual void update_cg(T gamma, T gamma0) {
		BOOST_THROW_EXCEPTION(std::runtime_error("Selected error estimator is not compatible with the selected solution method"));
	}

	//! returns the current relative error
	virtual T rel_error() const = 0;

	//! returns the current absolute error
	virtual T abs_error() const = 0;
};


//! Error estimator which always returns an error of 1.0
template<typename T, typename P>
class NoneErrorEstimator : public ErrorEstimator<T>
{
public:
	virtual void update() { }
	virtual void update_cg(T gamma, T gamma0) { }
	virtual T rel_error() const { return 1.0; } 
	virtual T abs_error() const { return 1.0; }
};


//! Error estimator based on residual
template<typename T, typename P>
class ResidualErrorEstimator : public ErrorEstimator<T>
{
protected:
	T _abs_err, _rel_err;

public:
	ResidualErrorEstimator()
	{
		_abs_err = STD_INFINITY(T);
		_rel_err = 1;
	}

	virtual void update_cg(T gamma, T gamma0)
	{
		_abs_err = std::sqrt(gamma);
		_rel_err = std::sqrt(gamma/gamma0);
	}

	virtual T rel_error() const { return _rel_err; } 
	virtual T abs_error() const { return _abs_err; }
};


//! Error estimator based on change in engergy
template<typename T, typename P>
class EnergyErrorEstimator : public ErrorEstimator<T>
{
protected:
	T _mean_energy, _mean_energy_prev;
	typename ErrorEstimator<T>::pRealTensor _eps, _tmp;
	boost::shared_ptr< MixedMaterialLawBase<T, P> > _mat;
	T _abs_err, _rel_err;
	std::size_t _mode, _iter;

	void update_mean_energy()
	{
		typename ErrorEstimator<T>::RealTensor* eps = _eps.get();

		if (_tmp) {
			// transfer eps to doubly fine grid
			eps = _tmp.get();
			prolongate_to_dfg(*_eps, *eps);
			_mat->select_dfg(true);
		}

		// FIXME: this does not consider boundary conditions correctly see total_energy and bc_energy
		_mean_energy = _mat->meanW(*eps);
		
		if (_tmp) {
			_mat->select_dfg(false);
		}
	}

public:
	EnergyErrorEstimator(typename ErrorEstimator<T>::pRealTensor eps, typename ErrorEstimator<T>::pRealTensor tmp, boost::shared_ptr< MixedMaterialLawBase<T, P> > mat, std::size_t mode) : _eps(eps), _tmp(tmp), _mat(mat), _mode(mode)
	{
		this->update_mean_energy();
		_mean_energy_prev = _mean_energy;
		_abs_err = STD_INFINITY(T);
		_rel_err = 1;
		_iter = 0;
	}

	virtual void update()
	{
		this->update_mean_energy();

		T small = boost::numeric::bounds<T>::smallest();

		_abs_err = std::abs(_mean_energy_prev - _mean_energy);
		_rel_err = _abs_err / (small + std::abs(_mean_energy));

		_mean_energy_prev = _mean_energy;
		_iter++;
	}

	virtual void update_cg(T gamma, T gamma0) { this->update(); }

	virtual T rel_error() const { return _rel_err; } 
	virtual T abs_error() const { return _abs_err; }
};


//! Error estimator based on stress divergence
template<typename T, typename P>
class DivSigmaErrorEstimator : public ErrorEstimator<T>
{
protected:
	typename ErrorEstimator<T>::pRealTensor _eps, _tmp;
	boost::shared_ptr< MixedMaterialLawBase<T, P> > _mat;
	T _abs_err, _rel_err;

public:
	DivSigmaErrorEstimator(typename ErrorEstimator<T>::pRealTensor eps, typename ErrorEstimator<T>::pRealTensor tmp, boost::shared_ptr< MixedMaterialLawBase<T, P> > mat) : _eps(eps), _tmp(tmp), _mat(mat)
	{
		update();
	}

	virtual void update()
	{
		typename ErrorEstimator<T>::RealTensor* eps = _eps.get();

		if (_tmp) {
			// transfer eps to doubly fine grid
			eps = _tmp.get();
			prolongate_to_dfg(*_eps, *eps);
			_mat->select_dfg(true);
		}

		//T small = boost::numeric::bounds<T>::smallest();

		// TODO: FIXME:
		_abs_err = 0; //_mat->maxDivPK1(*eps, 1);
		_rel_err = _abs_err; // (small + ublas::norm_2(_mean_sigma));

		if (_tmp) {
			_mat->select_dfg(false);
		}
	}

	virtual void update_cg(T gamma, T gamma0) { this->update(); }

	virtual T rel_error() const { return _rel_err; } 
	virtual T abs_error() const { return _abs_err; }
};


//! Error estimator based on change in stress
template<typename T, typename P>
class SigmaErrorEstimator : public ErrorEstimator<T>
{
protected:
	Tensor3x3<T> _mean_sigma, _mean_sigma_prev, _mean_sigma_prev_prev;
	typename ErrorEstimator<T>::pRealTensor _eps, _tmp;
	boost::shared_ptr< MixedMaterialLawBase<T, P> > _mat;
	T _abs_err, _rel_err;
	std::size_t _mode, _iter;

	void update_mean_sigma()
	{
		typename ErrorEstimator<T>::RealTensor* eps = _eps.get();

		if (_tmp) {
			// transfer eps to doubly fine grid
			eps = _tmp.get();
			prolongate_to_dfg(*_eps, *eps);
			_mat->select_dfg(true);
		}

		ublas::vector<T> mean = _mat->meanPK1(*eps, 1);
		
		if (_tmp) {
			_mat->select_dfg(false);
		}

		for (std::size_t i = 0; i < mean.size(); i++) {
			_mean_sigma[i] = mean[i];
		}
		_mat->fix_dim(_mean_sigma);
	}

public:
	SigmaErrorEstimator(typename ErrorEstimator<T>::pRealTensor eps, typename ErrorEstimator<T>::pRealTensor tmp, boost::shared_ptr< MixedMaterialLawBase<T, P> > mat, std::size_t mode) : _eps(eps), _tmp(tmp), _mat(mat), _mode(mode)
	{
		this->update_mean_sigma();
		_mean_sigma_prev.copyFrom(_mean_sigma);
		_mean_sigma_prev_prev.copyFrom(_mean_sigma);
		_abs_err = STD_INFINITY(T);
		_rel_err = 1;
		_iter = 0;
	}

	virtual void update()
	{
		this->update_mean_sigma();

		T small = boost::numeric::bounds<T>::smallest();

		/*
		LOG_COUT << "_mean_sigma" << format(_mean_sigma) << std::endl;
		LOG_COUT << "_mean_sigma_prev" << format(_mean_sigma_prev) << std::endl;
		LOG_COUT << "_mean_sigma_prev_prev" << format(_mean_sigma_prev_prev) << std::endl;
		*/

		if (_mode == 2 && _iter > 1) {
			_abs_err = 0.5*(ublas::norm_2(_mean_sigma_prev_prev - _mean_sigma) + ublas::norm_2(_mean_sigma_prev - _mean_sigma));
		}
		else {
			_abs_err = ublas::norm_2(_mean_sigma_prev - _mean_sigma);
		}

		_rel_err = _abs_err / (small + ublas::norm_2(_mean_sigma));

		_mean_sigma_prev_prev.copyFrom(_mean_sigma_prev);
		_mean_sigma_prev.copyFrom(_mean_sigma);
		_iter++;
	}

	virtual void update_cg(T gamma, T gamma0) { this->update(); }

	virtual T rel_error() const { return _rel_err; } 
	virtual T abs_error() const { return _abs_err; }
};


//! Error estimator based on change in strain
template<typename T, typename P>
class EpsilonErrorEstimator : public ErrorEstimator<T>
{
protected:
	Tensor3x3<T> _mean_epsilon, _mean_epsilon_prev;
	typename ErrorEstimator<T>::pRealTensor _eps;
	boost::shared_ptr< MixedMaterialLawBase<T, P> > _mat;
	T _abs_err, _rel_err;

	void update_mean_epsilon()
	{
		ublas::vector<T> mean = _eps->component_norm();
		
		for (std::size_t i = 0; i < mean.size(); i++) {
			_mean_epsilon[i] = mean[i];
		}
		
		_mat->fix_dim(_mean_epsilon);
	}

public:
	EpsilonErrorEstimator(typename ErrorEstimator<T>::pRealTensor eps, boost::shared_ptr< MixedMaterialLawBase<T, P> > mat) : _eps(eps), _mat(mat)
	{
		this->update_mean_epsilon();
		_mean_epsilon_prev.copyFrom(_mean_epsilon);
		_abs_err = STD_INFINITY(T);
		_rel_err = 1;
	}

	virtual void update()
	{
		this->update_mean_epsilon();

		T small = boost::numeric::bounds<T>::smallest();

		//_abs_err = ublas::norm_2(_mean_epsilon_prev - _mean_epsilon);
		_abs_err = std::abs(ublas::norm_2(_mean_epsilon_prev) - ublas::norm_2(_mean_epsilon));
		_rel_err = _abs_err / (small + ublas::norm_2(_mean_epsilon));

		_mean_epsilon_prev.copyFrom(_mean_epsilon);
	}

	virtual void update_cg(T gamma, T gamma0) { this->update(); }

	virtual T rel_error() const { return _rel_err; } 
	virtual T abs_error() const { return _abs_err; }
};



//! Lippmann-Schwinger solver
template<typename T, typename P, int DIM>
class LSSolver
{
public:
	typedef TensorField<T> RealTensor;
	typedef TensorField< std::complex<T>, T > ComplexTensor;
	typedef boost::shared_ptr<RealTensor> pRealTensor;
	typedef boost::shared_ptr<ComplexTensor> pComplexTensor;
	typedef boost::function<bool()> ConvergenceCallback;
	typedef boost::function<bool()> LoadstepCallback;
	typedef PhaseBase<T, P> Phase;
	typedef boost::shared_ptr< Phase > pPhase;

protected:

	typedef struct {
		T param;
	} Loadstep;

	std::size_t _nx, _ny, _nz;		// number of voxels in each dimension
	T _dx, _dy, _dz;		// RVE size in each dimension [m]
	std::size_t _nzc;			// number of complex values for z component
	std::size_t _nzp;			// number of real values of padded z componet
	std::size_t _nyzp;			// x-stride = _ny*_nzp
	std::size_t _nxyz;			// total number of values without padding nx*ny*nz
	std::size_t _n;			// total number of values including padding nx*ny*nzp
	ublas::c_vector<T, DIM> _x0;	// origin

	int _smooth_levels;	// number of subdivisions per voxel
	T _smooth_tol;		// smooth tolerance

	T _ref_scale;			// scaling of reference stiffness
	T _newton_relax;		// Newton step relaxation factor
	T _basic_relax;			// Basic scheme relaxation factor

	T _tol;				// relative error tolerance
	T _bc_tol;			// relative error for boundary conditions
	T _tol_red;			// 
	T _abs_tol;			// absolute error tolerance
	std::size_t _maxiter;		// maximum iterations
	std::string _update_ref;	// reference media update interval, "always", "loadstep"
	std::string _error_estimator;	// method for convergence check (at the moment the basic scheme uses "quick" and cg uses the residual)
	std::string _outer_error_estimator;	// method for convergence check (at the moment the basic scheme uses "quick" and cg uses the residual)
	std::string _method;		// solver method "basic" or "cg" or "nesterov" or "basic+el", "polarization"
	std::string _cg_inner_product;	// inner product for CG solver "l2" or "energy"
	std::size_t _cg_reinit;		// perform exact residual computation each _cg_reinit iteration, if ==0 residual is updated classically each iteration
	std::string _nl_cg_beta_scheme;	// 
	T _nl_cg_c;	// 
	T _nl_cg_tau;	// 
	T _nl_cg_alpha;	// 
	std::vector<Loadstep> _loadsteps;	// parameter values for each loadstep
	int _first_loadstep;		// first loadstep to compute
	std::string _loadstep_filename;	// vtk output filename for each loadstep, if empty no files are written
	bool _write_loadsteps;		// enable vtk output for each loadstep
	std::size_t _loadstep_extrapolation_order;	// 0 = none, 1 = linear, ...
	std::string _loadstep_extrapolation_method;	// polynomial, transformation
	std::string _gamma_scheme;	// scheme for gamma "collocated" or "staggered" or "full_staggered", "half_staggered" or "willot"
	std::string _mode;		// operation mode "elasticity", "hyperelasticity", "viscosity", "porous" or "heat"
	bool _parallel_fft;		// perform fft in parallel for loop (the fft itself is also parallelized)
	bool _freq_hack;		// 
	bool _debug;			// debug mode
	bool _step_mode;		// step mode
	bool _print_mean;		// print mean of solution during interation
	bool _print_detF;		// print min. of det(F)
	T _write_stress;

	std::string _material_mixing_rule;	// the material law name currently only "voigt", "reuss", "laminate"
	boost::shared_ptr< MixedMaterialLawBase<T, P> > _mat;	// the material law

	std::size_t _matrix_mat;	// index of matrix material
	T _mu_0, _lambda_0;		// Lamé parameters of reference material

	pRealTensor _epsilon;		// strain tensor components _eij in spatial domain
	pComplexTensor _tau;		// components _eij in Fourier domain
	pRealTensor _orientation;	// direction of anisotropy for transversally isotropic materials
	pRealTensor _normals;		// direction of interface normals
	pRealTensor _temp;		// temporary tensor
	pRealTensor _temp_dfg_1;	// temporary tensor for full_staggered grid
	pRealTensor _temp_dfg_2;	// temporary tensor for full_staggered grid

	ublas::vector<T> _E;		// prescribed strain tensor components
	ublas::vector<T> _S;		// prescribed stress tensor components
	ublas::vector<T> _current_E;	// current loadstep prescribed strain tensor components
	ublas::vector<T> _current_S;	// current loadstep prescribed stress tensor components
	ublas::vector<T> _Id;		// identity matrix

	ublas::matrix<T> _BC_P, _BC_Q, _BC_M, _BC_MQ, _BC_QC0;	// boundary condition projection matrix
	ublas::vector<T> _F0;		// mean stress polarization for boundary conditions
	ublas::vector<T> _F00;		// mean strain at beginning of current basic scheme loop 
	T _bc_relax;			// relaxation of boundary condition mean value adjustment

	//boost::shared_ptr< FFT3<T> > _fft;	// FFT object
	std::map<std::size_t, boost::shared_ptr< FFT3<T> > > _ffts;	// FFT objects
	std::string _fft_planner_flag;	// estimate, measure, patient, exhaustive, wisdom_only

	// offsets for computing (periodic) forward and backward finite differences along a specific component
	std::vector<int> _ffd_x, _ffd_y, _ffd_z;
	std::vector<int> _bfd_x, _bfd_y, _bfd_z;

	std::string _G0_solver;					// solver for GO operator application "fft" or "multigrid"

	// multigrid settings for solving poisson problems
	std::string _mg_scheme;					// multigrid scheme "pcg" or "direct" or "fft"
	std::size_t _mg_coarse_size;				// maximum coarse problem size
	std::string _mg_pre_smoother;				// pre smoother
	std::size_t _mg_n_pre_smooth;				// number of pre smoothings
	std::string _mg_post_smoother;				// post smoother
	std::size_t _mg_n_post_smooth;				// number of post smoothings
	std::size_t _mg_smooth_bs;				// smooth block size
	std::size_t _mg_smooth_relax;				// smooth relaxation factor
	std::string _mg_coarse_solver;				// coarse grid solver "fft" or "lu"
	std::string _mg_prolongation_op;			// prolongation operator "straight_injection" or "full_weighting"
	bool _mg_enable_timing;					// enable timing of multigrid operations on fine level
	bool _mg_residual_checking;				// enable residual check for direct solver, if disabled the solver runs _mg_maxiter iterations
	bool _mg_safe_mode;
	T _mg_tol;						// multigird relative tolerance
	std::size_t _mg_maxiter;				// multigird maximum iterations
	boost::shared_ptr< MultiGridLevel<T> > _mg_level;	// fine multigrid level

	// callbacks
	ConvergenceCallback _convergence_callback;
	LoadstepCallback _loadstep_callback;

	// statistics
	std::vector<double> _residuals;
	double _solve_time;
	double _fft_time;

	// fields for mixing rule
	/*
	std::string _mixing_rule;			// voigt, reuss or laminate
	std::vector<char> _material_index;		// indicates if voxel is a mixed material (=-1) or gets the material index otherwise
	boost::ptr_vector< ElasticityTensor<T> > _C0;	// stiffness tensors for a global constant C0
	boost::ptr_vector< ElasticityTensor<T> > _C;	// stiffness tensors C
	boost::ptr_vector< ElasticityTensor<T> > _C_C0;	// stiffness tensors C-C0
	*/
	
public:
	LSSolver(std::size_t nx, std::size_t ny, std::size_t nz, T dx, T dy, T dz, ublas::c_vector<T, DIM> x0 = ublas::zero_vector<T>(DIM)) :
		_nx(nx), _ny(ny), _nz(nz), _dx(dx), _dy(dy), _dz(dz), _x0(x0)
	{
		// calculate number of complex points in z dimension in Fourier domain
		// (this adds eventually reqired padding to the real data, see FFTW documentation fftw_plan_dft_r2c_3d for details)
		_nzc = _nz/2+1;
		
		// calculate number of real points in z dimension in spatial domain (= _nz + padding)
		_nzp = 2*_nzc;
		
		// compute number of voxels
		_nxyz = _nx*_ny*_nz;
	
		// compute x-stride
		_nyzp = _ny*_nzp;
		
		// compute number of voxels (incl. padding) = real data length
		_n = _nx*_nyzp;
		
		// set error tolerance
		_tol = 1e-4;
		_tol_red = std::sqrt(std::numeric_limits<T>::epsilon());
		_abs_tol = std::numeric_limits<T>::epsilon();
		_bc_tol = 1e-3;
		_maxiter = 10000;
		_ref_scale = 1.0;
		_newton_relax = 1.0;
		_basic_relax = 1.0;
		_error_estimator = "epsilon";
		_outer_error_estimator = "epsilon";
		_update_ref = "loadstep";
		_method = "cg";
		_cg_inner_product = "l2";
		_cg_reinit = 0;
		_nl_cg_beta_scheme = "polak_ribiere";
		_nl_cg_c = 0.5;
		_nl_cg_tau = 0.5;
		_nl_cg_alpha = 1.0;
		_bc_relax = 1.0;
		_mode = "elasticity";
		_gamma_scheme = "auto";
		_freq_hack = false;
		_parallel_fft = false;
		_debug = false;
		_step_mode = false;
		_print_mean = false;
		_print_detF = false;
		_first_loadstep = -1;
		uniform_loadsteps(1);
		_loadstep_filename = "loadstep_%02d.vtk";
		_loadstep_extrapolation_order = 0;
		_loadstep_extrapolation_method = "polynomial";
		_write_loadsteps = false;
		_write_stress = STD_INFINITY(T);
		_material_mixing_rule = "voigt";

		_mu_0 = _lambda_0 = 0;
		_matrix_mat = 0;
		
		_fft_planner_flag = "measure";

		// enable automatic interface smoothing 
		_smooth_levels = -1;
		_smooth_tol = 0.001;

		_G0_solver = "fft";

		// multigrid settings
		_mg_scheme = "direct";
		_mg_coarse_size = 4;
		_mg_tol = 1e-12;
		_mg_maxiter = 16;
		_mg_pre_smoother = "fgs";
		_mg_n_pre_smooth = 2;
		_mg_post_smoother = "bgs";
		_mg_n_post_smooth = 2;
		_mg_smooth_bs = -1;
		_mg_smooth_relax = 1.0;
		_mg_coarse_solver = "fft";
		_mg_prolongation_op = "full_weighting";
		_mg_enable_timing = false;
		_mg_residual_checking = false;
		_mg_safe_mode = false;

		_solve_time = 0;
		_fft_time = 0;

		// init forward and backward finite difference offsets
		_ffd_x.resize(_nx);
		_bfd_x.resize(_nx);
		for (std::size_t ii = 0; ii < _nx; ii++) {
			_ffd_x[ii] = (((int)((ii+1)%_nx)) - (int)ii)*(int)_nyzp;
		}
		for (std::size_t ii = 0; ii < _nx; ii++) {
			_bfd_x[ii] = -_ffd_x[_nx - 1 - ii];
		}
		_ffd_y.resize(_ny);
		_bfd_y.resize(_ny);
		for (std::size_t jj = 0; jj < _ny; jj++) {
			_ffd_y[jj] = (((int)((jj+1)%_ny)) - (int)jj)*(int)_nzp;
		}
		for (std::size_t jj = 0; jj < _ny; jj++) {
			_bfd_y[jj] = -_ffd_y[_ny - 1 - jj];
		}
		_ffd_z.resize(_nz);
		_bfd_z.resize(_nz);
		for (std::size_t kk = 0; kk < _nz; kk++) {
			_ffd_z[kk] = (((int)((kk+1)%_nz)) - (int)kk);
		}
		for (std::size_t kk = 0; kk < _nz; kk++) {
			_bfd_z[kk] = -_ffd_z[_nz - 1 - kk];
		}
	}

	inline bool use_dfg() const
	{
		return (_gamma_scheme == "half_staggered") || (_gamma_scheme == "full_staggered");
	}

	std::vector<double> get_rve_dims()
	{
		std::vector<double> dims;
		dims.push_back(_x0[0]);
		dims.push_back(_x0[1]);
		dims.push_back(_x0[2]);
		dims.push_back(_dx);
		dims.push_back(_dy);
		dims.push_back(_dz);
		return dims;
	}

	pRealTensor get_orientation()
	{
		if (!_orientation) {
			if (this->use_dfg()) {
				_orientation.reset(new RealTensor(2*_nx, 2*_ny, 2*_nz, 3));
			}
			else {
				_orientation.reset(new RealTensor(_nx, _ny, _nz, 3));
			}
		}
		
		return _orientation;
	}

	pRealTensor get_normals()
	{
		if (!_normals) {
			if (this->use_dfg()) {
				_normals.reset(new RealTensor(2*_nx, 2*_ny, 2*_nz, 3));
			}
			else {
				_normals.reset(new RealTensor(_nx, _ny, _nz, 3));
			}
		}
		
		return _normals;
	}
	
	//! Create an error estimator by name
	ErrorEstimator<T>* create_error_estimator(std::string name = "")
	{
		ErrorEstimator<T>* ee = NULL;

		if (name == "") {
			name = _error_estimator;
		}

		if (name == "sigma") {
			ee = new SigmaErrorEstimator<T, P>(_epsilon, this->use_dfg() ? _temp_dfg_1 : pRealTensor(), _mat, ((_method == "basic") ? 2 : 2));
		}
		else if (name == "div_sigma") {
			pRealTensor tmp_div(new RealTensor(*_epsilon, 0));
			ee = new DivSigmaErrorEstimator<T, P>(_epsilon, tmp_div, _mat);
		}
		else if (name == "epsilon") {
			ee = new EpsilonErrorEstimator<T, P>(_epsilon, _mat);
		}
		else if (name == "energy") {
			ee = new EnergyErrorEstimator<T, P>(_epsilon, this->use_dfg() ? _temp_dfg_1 : pRealTensor(), _mat, ((_method == "basic") ? 2 : 2));
		}
		else if (name == "residual") {
			ee = new ResidualErrorEstimator<T, P>();
		}
		else if (name == "none") {
			ee = new NoneErrorEstimator<T, P>();
		}
		else {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown error estimator '%s'") % name).str()));
		}

		return ee;
	}
	
	//! Create a mixed material by name and settings from a ptree
	MixedMaterialLawBase<T, P>* create_mixing_rule(const std::string& _name, const ptree::ptree& pt)
	{
		MixedMaterialLawBase<T, P>* ret;
		std::string name = _name;

		#define CHECK_MIX(_name, cls, args, setters) \
		else if (name == _name) { \
			if (_mode == "hyperelasticity") { \
				cls<T, P, 9>* mr = new cls<T, P, 9>(args); \
				setters; \
				ret = mr; \
			} \
			else if (_mode == "heat" || _mode == "porous") { \
				cls<T, P, 3>* mr = new cls<T, P, 3>(args); \
				setters; \
				ret = mr; \
			} \
			else { \
				cls<T, P, 6>* mr = new cls<T, P, 6>(args); \
				setters; \
				ret = mr; \
			} \
		}

		//if (_mode == "viscosity" && name == "laminate") {
		//	name = "fluidity";
		//}

		if (false) {}
		CHECK_MIX("voigt", VoigtMixedMaterialLaw, ,)
		CHECK_MIX("iso", IsoMixedMaterialLaw, ,)
		CHECK_MIX("split", SplitMixedMaterialLaw, , {
			mr->vol_rule.reset(create_mixing_rule(pt_get<std::string>(pt, "mixing_rule_vol", "reuss"), pt));
			mr->dev_rule.reset(create_mixing_rule(pt_get<std::string>(pt, "mixing_rule_dev", "voigt"), pt));
		})
		CHECK_MIX("maximum", MaximumMixedMaterialLaw, ,)
		CHECK_MIX("fiftyfifty", FiftyFiftyMixedMaterialLaw, ,)
		CHECK_MIX("random", RandomMixedMaterialLaw, ,)
		CHECK_MIX("reuss", ReussMixedMaterialLaw, ,)
		CHECK_MIX("laminate", LaminateMixedMaterialLaw, get_normals(), {
			//mr->eps = _tol;
			//T eps = std::numeric_limits<T>::epsilon();
			//mr->delta = std::max((T)0, 1 - 0.1*_tol);
		})
		CHECK_MIX("infinity-laminate", InfinityLaminateMixedMaterialLaw, get_normals(), {
			//mr->eps = _tol;
			//T eps = std::numeric_limits<T>::epsilon();
			//mr->delta = std::max((T)0, 1 - 0.1*_tol);
		})
		CHECK_MIX("fluidity", FluidityMixedMaterialLaw, get_normals(), )
		else {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown material mixing rule '%s'") % name).str()));
		}

		ret->readSettings(pt);
		
		return ret;
	}

	void uniform_loadsteps(std::size_t loadsteps)
	{
		_loadsteps.resize(loadsteps + 1);

		for (std::size_t istep = 0; istep <= loadsteps; istep++) {
			_loadsteps[istep].param = istep/(T)loadsteps;
		}
	}

	// read settings from ptree
	void readSettings(const ptree::ptree& pt)
	{
		_tol = pt_get<T>(pt, "tol", _tol);
		_tol_red = pt_get<T>(pt, "tol_red", _tol_red);
		_abs_tol = pt_get<T>(pt, "abs_tol", _abs_tol);
		_bc_tol = pt_get<T>(pt, "bc_tol", _bc_tol);
		_maxiter = pt_get(pt, "maxiter", _maxiter);
		_update_ref = pt_get(pt, "update_ref", _update_ref);
		_ref_scale = pt_get<T>(pt, "ref_scale", _ref_scale);
		_newton_relax = pt_get<T>(pt, "newton_relax", _newton_relax);
		_basic_relax = pt_get<T>(pt, "basic_relax", _basic_relax);
		_smooth_levels = pt_get(pt, "smooth_levels", _smooth_levels);
		_smooth_tol = pt_get<T>(pt, "smooth_tol", _smooth_tol);
		_error_estimator = pt_get(pt, "error_estimator", _error_estimator);
		_outer_error_estimator = pt_get(pt, "outer_error_estimator", _outer_error_estimator);
		_method = pt_get(pt, "method", _method);
		_cg_inner_product = pt_get(pt, "cg_inner_product", _cg_inner_product);
		_cg_reinit = pt_get(pt, "cg_reinit", _cg_reinit);
		_nl_cg_beta_scheme = pt_get(pt, "nl_cg_beta_scheme", _nl_cg_beta_scheme);
		_nl_cg_c = pt_get<T>(pt, "nl_cg_c", _nl_cg_c);
		_nl_cg_tau = pt_get<T>(pt, "nl_cg_tau", _nl_cg_tau);
		_nl_cg_alpha = pt_get<T>(pt, "nl_cg_alpha", _nl_cg_alpha);
		_gamma_scheme = pt_get(pt, "gamma_scheme", _gamma_scheme);
		_mode = pt_get(pt, "mode", _mode);
		if (_gamma_scheme == "full-staggered") _gamma_scheme = "full_staggered";
		else if (_gamma_scheme == "half-staggered") _gamma_scheme = "half_staggered";
		else if (_gamma_scheme == "Willot-R") _gamma_scheme = "willot";
		else if (_gamma_scheme == "auto") {
			_gamma_scheme = "staggered";
			//if (_mode == "heat" || _mode == "porous") _gamma_scheme = "collocated";
			if (_method == "polarization") _gamma_scheme = "collocated";
		}
		if (_method == "polarization" && _gamma_scheme.find("staggered") >= 0) {
			_gamma_scheme = "collocated";
			LOG_CWARN << "switching to collocated discretization for polarization method!" << std::endl;
		}
		_bc_relax = pt_get<T>(pt, "bc_relax", _bc_relax);
		_freq_hack = pt_get(pt, "freq_hack", _freq_hack);
		_debug = pt_get(pt, "debug", _debug);
		_step_mode = pt_get(pt, "step_mode", _step_mode);
		_print_mean = pt_get(pt, "print_mean", _print_mean);
		_print_detF = pt_get(pt, "print_detF", _print_detF);
		_G0_solver = pt_get(pt, "G0_solver", _G0_solver);
		_parallel_fft = pt_get(pt, "parallel_fft", _parallel_fft);
		_fft_planner_flag = pt_get(pt, "fft_planner_flag", _fft_planner_flag);
		_loadstep_filename = pt_get(pt, "loadstep_filename", _loadstep_filename);
		_loadstep_extrapolation_order = pt_get(pt, "loadstep_extrapolation_order", _loadstep_extrapolation_order);
		_loadstep_extrapolation_method = pt_get(pt, "loadstep_extrapolation_method", _loadstep_extrapolation_method);
		_write_loadsteps = pt_get(pt, "write_loadsteps", _write_loadsteps);
		_first_loadstep = pt_get(pt, "first_loadstep", _first_loadstep);
		_write_stress = pt_get(pt, "write_stress", _write_stress);
	
		// read loadsteps	
		const ptree::ptree& loadsteps = pt.get_child("loadsteps", empty_ptree);

		if (loadsteps.size() > 0)
		{
			_loadsteps.clear();

			BOOST_FOREACH(const ptree::ptree::value_type &v, loadsteps)
			{
				if (v.first != "loadstep") {
					continue;
				}

				const ptree::ptree& attr = v.second.get_child("<xmlattr>", empty_ptree);

				Loadstep ls;
				ls.param = pt_get<T>(attr, "param");
				_loadsteps.push_back(ls);
			}
		}
		else
		{
			uniform_loadsteps(loadsteps.get_value(1));
		}

		_orientation.reset();
		_normals.reset();
		_epsilon.reset();
		_temp.reset();
		_temp_dfg_1.reset();
		_temp_dfg_2.reset();
		_mat.reset();

		// init material mixing rule
		_material_mixing_rule = pt_get(pt, "mixing_rule", _material_mixing_rule);

#if 0
		if (_mode == "viscosity" && _material_mixing_rule == "laminate") {
			BOOST_THROW_EXCEPTION(std::runtime_error("Selected mixing rule does not make any sense for viscosity mode! Use Voigt mixing rule!"));
		}

		if ((_mode == "heat" || _mode == "porous") && _material_mixing_rule == "laminate") {
			BOOST_THROW_EXCEPTION(std::runtime_error("Selected mixing rule does not make any sense for heat mode! Use Reuss mixing rule!"));
		}
#endif

		_mat.reset(create_mixing_rule(_material_mixing_rule, pt));

		// init temporary variables for full staggered grid
		if (this->use_dfg()) {
			_temp_dfg_1.reset(new RealTensor(2*_nx, 2*_ny, 2*_nz, _mat->dim()));
			if (_mode == "hyperelasticity") {
				_temp_dfg_2.reset(new RealTensor(2*_nx, 2*_ny, 2*_nz, _mat->dim()));
			}
		}

		// alloc strain tensor and complex shadow _tau
		_epsilon.reset(new RealTensor(_nx, _ny, _nz, _mat->dim()));
		init_fft();
		_tau = _epsilon->complex_shadow();

		// init prescribed strain and stress
		_E = ublas::zero_vector<T>(_mat->dim());
		_S = ublas::zero_vector<T>(_mat->dim());
		_current_E = ublas::zero_vector<T>(_mat->dim());
		_current_S = ublas::zero_vector<T>(_mat->dim());

		// init boundary condition projector
		this->setBCProjector(Voigt::Id4<T>(_mat->dim()));
		_F0 = ublas::zero_vector<T>(_mat->dim());
		_F00 = ublas::zero_vector<T>(_mat->dim());

		// init identity matrix
		_Id = ublas::zero_vector<T>(_mat->dim());
		_Id(0) = _Id(1) = _Id(2) = 1;

		// read materials
		const ptree::ptree& materials = pt.get_child("materials", empty_ptree);
		bool reference_set = false;
		bool matrix_set = false;

		BOOST_FOREACH(const ptree::ptree::value_type &v, materials)
		{
			// skip comments
			if (v.first == "<xmlcomment>") {
				continue;
			}

			const ptree::ptree& attr = v.second.get_child("<xmlattr>", empty_ptree);

			if (v.first == "ref")
			{
				Material<T, DIM> m;
				m.readSettings(v.second);

				_lambda_0 = m.lambda;
				_mu_0 = m.mu;
				reference_set = true;
			}
			else
			{
				if (v.first == "matrix" || pt_get<std::size_t>(attr, "matrix", 0) != 0) {
					if (matrix_set) {
						BOOST_THROW_EXCEPTION(std::runtime_error("Matrix material already specified"));
					}
					matrix_set = true;
					_matrix_mat = _mat->phases.size();
				}
				
				pPhase p(new Phase());

				p->law_name = pt_get<std::string>(attr, "law", "iso");
				p->name = v.first;
				p->init(_nx, _ny, _nz, this->use_dfg());

				if (_mode == "elasticity" && p->law_name == "iso") {
					p->law.reset(new LinearIsotropicMaterialLaw<T>());
					p->law->readSettings(v.second);
				}
				else if (_mode == "elasticity" && p->law_name == "tiso") {
					p->law.reset(new LinearTransverselyIsotropicMaterialLaw<T, DIM>(get_orientation()));
					p->law->readSettings(v.second);
				}
				else if ((_mode == "heat" || _mode == "porous") && p->law_name == "iso") {
					ScalarLinearIsotropicMaterialLaw<T>* law = new ScalarLinearIsotropicMaterialLaw<T>(3);
					law->readSettings(v.second);
					p->law.reset(law);
				}
				else if ((_mode == "heat" || _mode == "porous") && p->law_name == "aniso") {
					MatrixLinearAnisotropicMaterialLaw<T>* law = new MatrixLinearAnisotropicMaterialLaw<T>();
					law->readSettings(v.second);
					p->law.reset(law);
				}
				else if (_mode == "viscosity" && p->law_name == "iso") {
					ScalarLinearIsotropicMaterialLaw<T>* law = new ScalarLinearIsotropicMaterialLaw<T>(6);
					law->readSettings(v.second);
					law->mu *= 0.5;		// scaling required for viscosity as working with dual quantities
					p->law.reset(law);
				}
				else if (_mode == "hyperelasticity" && p->law_name == "iso") {
					p->law.reset(new SaintVenantKirchhoffMaterialLaw<T>());
					p->law->readSettings(v.second);
				}
				else if (_mode == "hyperelasticity" && p->law_name == "nh") {
					p->law.reset(new NeoHookeMaterialLaw<T>());
					p->law->readSettings(v.second);
				}
				else if (_mode == "hyperelasticity" && p->law_name == "nh2") {
					p->law.reset(new NeoHooke2MaterialLaw<T>());
					p->law->readSettings(v.second);
				}
				else if (_mode == "hyperelasticity" && p->law_name == "gb_matrix1") {
					p->law.reset(new Matrix1GoldbergMaterialLaw<T>());
					p->law->readSettings(v.second);
				}
				else if (_mode == "hyperelasticity" && p->law_name == "gb_matrix2") {
					p->law.reset(new Matrix2GoldbergMaterialLaw<T>());
					p->law->readSettings(v.second);
				}
				else if (_mode == "hyperelasticity" && p->law_name == "gb_matrix3") {
					p->law.reset(new Matrix3GoldbergMaterialLaw<T>());
					p->law->readSettings(v.second);
				}
				else if (_mode == "hyperelasticity" && p->law_name == "gb_matrix4") {
					p->law.reset(new Matrix4GoldbergMaterialLaw<T>());
					p->law->readSettings(v.second);
				}
				else if (_mode == "hyperelasticity" && p->law_name == "gb_fiber1") {
					p->law.reset(new Fiber1GoldbergMaterialLaw<T>());
					p->law->readSettings(v.second);
				}
				else if (_mode == "hyperelasticity" && p->law_name == "gb_fiber2") {
					p->law.reset(new Fiber2GoldbergMaterialLaw<T>());
					p->law->readSettings(v.second);
				}
				else if (_mode == "hyperelasticity" && p->law_name == "gb_fiber3") {
					p->law.reset(new Fiber3GoldbergMaterialLaw<T>());
					p->law->readSettings(v.second);
				}
				else if (_mode == "hyperelasticity" && p->law_name == "gb_fiber4") {
					p->law.reset(new Fiber4GoldbergMaterialLaw<T>());
					p->law->readSettings(v.second);
				}
				else if (_mode == "hyperelasticity" && p->law_name == "gb_fiber5") {
					p->law.reset(new Fiber5GoldbergMaterialLaw<T>());
					p->law->readSettings(v.second);
				}
				else if (_mode == "hyperelasticity" && p->law_name == "gb_fiber6") {
					p->law.reset(new Fiber6GoldbergMaterialLaw<T>());
					p->law->readSettings(v.second);
				}
				else {
					BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown material law '%s'") % p->law_name).str()));
				}

				// LOG_COUT << p->name << " " << p->_phi.get() << " " << p->_phi_dfg.get() << std::endl;
				_mat->add_phase(p);
			}
		}

		if (_mat->phases.size() == 0) {
			BOOST_THROW_EXCEPTION(std::runtime_error("No materials specified"));
		}

		_mat->init();

		if (!matrix_set) {
			_matrix_mat = 0;
			matrix_set = true;
			LOG_COUT << (boost::format("Warning: selecting '%s' as matrix material") % _mat->phases[_matrix_mat]->name).str()<< std::endl;
		}

		if (!matrix_set) {
			BOOST_THROW_EXCEPTION(std::runtime_error("No matrix material specified"));
		}

		if (!reference_set)
		{
			/*
			// reference material is now computed from eigenvalues, set to NaN to avoid illegal use

			// compute reference material (half of min+max of individual parameters)

			T K_min = STD_INFINITY(T), K_max = 0;
			T mu_min = STD_INFINITY(T), mu_max = 0;

			for (std::size_t i = 0; i < _mat->phases.size(); i++) {
				T mu = _mat->phases[i]->mu;
				T K = _mat->phases[i]->lambda + (2./3.)*mu;
				K_min = std::min(K_min, K);
				K_max = std::max(K_max, K);
				mu_min = std::min(mu_min, mu);
				mu_max = std::max(mu_max, mu);
			}

			T K0 = 0.5*(K_min + K_max);
			_mu_0 = 0.5*(mu_min + mu_max);
			_lambda_0 = K0 - (2./3.)*_mu_0;
			*/
			_mu_0 = 0/(T)0;
			_lambda_0 = 0.0;
		}

		// read multigrid settings
		boost::optional< const ptree::ptree& > mg = pt.get_child_optional("multigrid");
		if (mg) {
			_mg_scheme = mg->get("scheme", _mg_scheme);
			_mg_coarse_size = mg->get("coarse_size", _mg_coarse_size);
			_mg_tol = mg->get("tol", _mg_tol);
			_mg_maxiter = mg->get("maxiter", _mg_maxiter);
			_mg_pre_smoother = mg->get("pre_smoother", _mg_pre_smoother);
			_mg_n_pre_smooth = mg->get("n_pre_smooth", _mg_n_pre_smooth);
			_mg_post_smoother = mg->get("post_smoother", _mg_post_smoother);
			_mg_n_post_smooth = mg->get("n_post_smooth", _mg_n_post_smooth);
			_mg_smooth_bs = mg->get("smooth_bs", _mg_smooth_bs);
			_mg_smooth_relax = mg->get("smooth_relax", _mg_smooth_relax);
			_mg_coarse_solver = mg->get("coarse_solver", _mg_coarse_solver);
			_mg_prolongation_op = mg->get("prolongation_op", _mg_prolongation_op);
			_mg_enable_timing = mg->get("enable_timing", _mg_enable_timing);
			_mg_residual_checking = mg->get("residual_checking", _mg_residual_checking);
			_mg_safe_mode = mg->get("safe_mode", _mg_safe_mode);
		}
	}

	//! return a list of phase names
	std::vector<std::string> getPhaseNames() const
	{
		std::vector<std::string> names;

		for (std::size_t i = 0; i < _mat->phases.size(); i++) {
			names.push_back(_mat->phases[i]->name);
		}

		return names;
	}

	//! return material index for a phase name
	std::size_t getMaterialIndex(const std::string& field) const
	{
		for (std::size_t i = 0; i < _mat->phases.size(); i++) {
			if (field == _mat->phases[i]->name) {
				return i;
			}
		}

		BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown field '%s'") % field).str()));
	}

	// FIXME: make more general, assumes 2 phases only
	double getVolumeFraction(const std::string& field) const { return _mat->phases[getMaterialIndex(field)]->vol;  }
	const std::vector<double>& getResiduals() const { return _residuals; }
	double getSolveTime() const { return _solve_time; }
	double getFFTTime() const { return _fft_time; }

	// FIXME: this is a hack
	void* get_raw_field(const std::string& field, std::vector<void*>& components, size_t& nx, size_t& ny, size_t& nz, size_t& nzp, size_t& elsize, boost::shared_ptr< FiberGenerator<T, DIM> > gen)
	{
		components.clear();
		nx = _nx;
		ny = _ny;
		nz = _nz;
		nzp = _nzp;
		elsize = sizeof(T);

		ProgressBar<T> pb(_nx);
		#define PB_UPDATE \
		if (pb.update()) { \
			pb.message() << "get field '" << field << "'" << pb.end(); \
		}

		if (field == "epsilon")
		{
			for (std::size_t i = 0; i < _epsilon->dim; i++) {
				components.push_back((*_epsilon)[i]);
			}
		}
		else if (field == "orientation")
		{
			if (_orientation) {
				for (std::size_t i = 0; i < _orientation->dim; i++) {
					components.push_back((*_orientation)[i]);
				}
			}
			else {
				RealTensor* pfield = new RealTensor(*_epsilon, 3);
				std::vector<T> data(_ny*_nz*3);
				T* pdata = &(data[0]);

				bool fast = false;
				int mat = -1;

				for (std::size_t i = 0; i < _nx; i++) {
					gen->sampleZYSlice(i, _nx, _ny, _nz, pdata, mat, FiberGenerator<T, DIM>::SampleDataTypes::ORIENTATION, 0, fast);

					int kk = i*_nyzp;
					for (std::size_t j = 0; j < _ny; j++) {
						for (std::size_t k = 0; k < _nz; k++) {
							std::size_t ii = 3*(j*_nz + k);
							(*pfield)[0][kk] = data[ii + 0];
							(*pfield)[1][kk] = data[ii + 1];
							(*pfield)[2][kk] = data[ii + 2];
							kk++;
						}
						kk += (_nzp - _nz);
					}
					PB_UPDATE;
				}

				components.push_back((*pfield)[0]);
				components.push_back((*pfield)[1]);
				components.push_back((*pfield)[2]);

				return pfield;
			}

		}
		else if (field == "normals")
		{
			if (_normals) {
				for (std::size_t i = 0; i < _normals->dim; i++) {
					components.push_back((*_normals)[i]);
				}
			}
			else {
				RealTensor* pfield = new RealTensor(*_epsilon, 3);
				std::vector<T> data(_ny*_nz*3);
				T* pdata = &(data[0]);

				bool fast = false;
				int mat = -1;

				for (std::size_t i = 0; i < _nx; i++) {
					gen->sampleZYSlice(i, _nx, _ny, _nz, pdata, mat, FiberGenerator<T, DIM>::SampleDataTypes::NORMALS, 0, fast);

					int kk = i*_nyzp;
					for (std::size_t j = 0; j < _ny; j++) {
						for (std::size_t k = 0; k < _nz; k++) {
							std::size_t ii = 3*(j*_nz + k);
							(*pfield)[0][kk] = data[ii + 0];
							(*pfield)[1][kk] = data[ii + 1];
							(*pfield)[2][kk] = data[ii + 2];
							kk++;
						}
						kk += (_nzp - _nz);
					}
					PB_UPDATE;
				}

				components.push_back((*pfield)[0]);
				components.push_back((*pfield)[1]);
				components.push_back((*pfield)[2]);

				return pfield;
			}
		}
		else if (field == "sigma")
		{
			RealTensor* psigma = new RealTensor(*_epsilon, 0);
			RealTensor& sigma = *psigma;
			
			calcStress(*_epsilon, sigma);
			
			for (std::size_t i = 0; i < sigma.dim; i++) {
				components.push_back(sigma[i]);
			}

			return psigma;
		}
		else if (field == "u")
		{
			RealTensor* psigma = new RealTensor(*_epsilon, 0);
			RealTensor& sigma = *psigma;
			pComplexTensor psigma_hat = sigma.complex_shadow();
			ComplexTensor& sigma_hat = *psigma_hat;
			std::size_t udim = 3;

			if (_mode == "elasticity")
			{
				calcStressConst(_mu_0, _lambda_0, *_epsilon, sigma);
				divOperatorStaggered(sigma, sigma);
				G0OperatorStaggered(_mu_0, _lambda_0, sigma, sigma_hat, sigma_hat, sigma, 1.0);
			}
			else if (_mode == "hyperelasticity")
			{
				calcStressDiff(*_epsilon, sigma);
				G0DivOperatorHyper(_mu_0, _lambda_0, sigma, sigma_hat, sigma_hat, sigma, 1.0);
			}
			else if (_mode == "viscosity")
			{
				// viscosity
				// solve div(2*eta_0*eps_E) = div(eta_0*(phi-phi_0)*sigma_E) - grad(p)
				calcStressDiff(*_epsilon, sigma); // calculate 0.5*(phi-phi_0)*epsilon => sigma
				divOperatorStaggered(sigma, sigma);
				G0OperatorStaggered(1/(4*_mu_0), STD_INFINITY(T), sigma, sigma_hat, sigma_hat, sigma, 1/(2*_mu_0));
			}
			else if (_mode == "heat" || _mode == "porous")
			{
				calcStressConst(_mu_0, _lambda_0, *_epsilon, sigma);
				divOperatorStaggeredHeat(sigma, sigma);
				G0OperatorStaggeredHeat(_mu_0, _lambda_0, sigma, sigma_hat, sigma_hat, sigma, 1.0);
				udim = 1;
			}

			for (std::size_t i = 0; i < udim; i++) {
				components.push_back(sigma[i]);
			}

			for (std::size_t i = udim; i < sigma.dim; i++) {
				sigma.freeComponent(i);
			}

			return psigma;
		}
		else if (field == "p")
		{
			RealTensor* psigma = new RealTensor(*_epsilon, 0);
			RealTensor& sigma = *psigma;
			
			T* f = sigma[3];
			T* p = sigma[4];

			calcStressDiff(*_epsilon, sigma);	// calculate 0.5*(phi-phi_0)*epsilon => sigma
			divOperatorStaggered(sigma, sigma);
			divVector(sigma, f, 1/(2*_mu_0));	// conatins muliplication with 2*eta0
			poisson_solve(f, p);
			components.push_back(p);

			for (std::size_t i = 0; i < sigma.dim; i++) {
				if (i != 4) sigma.freeComponent(i);
			}

			return psigma;
		}
		else if (field == "fiber_id")
		{
			RealTensor* pfield = new RealTensor(*_epsilon, 1);
			T* pdata = (*pfield)[0];

			bool fast = false;
			int mat = -1;

			for (std::size_t i = 0; i < _nx; i++) {
				int kk = i*_nyzp;
				gen->sampleZYSlice(i, _nx, _ny, _nzp, pdata + kk, mat, FiberGenerator<T, DIM>::SampleDataTypes::FIBER_ID, 0, fast);
				PB_UPDATE;
			}

			components.push_back(pdata);

			return pfield;
		}
		else if (field == "distance")
		{
			RealTensor* pfield = new RealTensor(*_epsilon, 1);
			T* pdata = (*pfield)[0];

			bool fast = false;
			int mat = -1;

			for (std::size_t i = 0; i < _nx; i++) {
				int kk = i*_nyzp;
				gen->sampleZYSlice(i, _nx, _ny, _nzp, pdata + kk, mat, FiberGenerator<T, DIM>::SampleDataTypes::DISTANCE, 0, fast);
				PB_UPDATE;
			}

			components.push_back(pdata);

			return pfield;
		}
		else if (field == "material_id")
		{
			RealTensor* pfield = new RealTensor(*_epsilon, 1);
			T* pdata = (*pfield)[0];

			bool fast = false;
			int mat = -1;

			for (std::size_t i = 0; i < _nx; i++) {
				int kk = i*_nyzp;
				gen->sampleZYSlice(i, _nx, _ny, _nzp, pdata + kk, mat, FiberGenerator<T, DIM>::SampleDataTypes::MATERIAL_ID, 0, fast);
				PB_UPDATE;
			}

			components.push_back(pdata);

			return pfield;
		}

		else if (field == "fiber_translation")
		{
			RealTensor* pfield = new RealTensor(*_epsilon, 3);
			std::vector<T> data(_ny*_nz*3);
			T* pdata = &(data[0]);

			bool fast = false;
			int mat = -1;

			for (std::size_t i = 0; i < _nx; i++) {
				gen->sampleZYSlice(i, _nx, _ny, _nz, pdata, mat, FiberGenerator<T, DIM>::SampleDataTypes::FIBER_TRANSLATION, 0, fast);

				int kk = i*_nyzp;
				for (std::size_t j = 0; j < _ny; j++) {
					for (std::size_t k = 0; k < _nz; k++) {
						std::size_t ii = 3*(j*_nz + k);
						(*pfield)[0][kk] = data[ii + 0];
						(*pfield)[1][kk] = data[ii + 1];
						(*pfield)[2][kk] = data[ii + 2];
						kk++;
					}
					kk += (_nzp - _nz);
				}
				PB_UPDATE;
			}

			components.push_back((*pfield)[0]);
			components.push_back((*pfield)[1]);
			components.push_back((*pfield)[2]);

			return pfield;
		}
		else if (field == "phi")
		{
			for (std::size_t i = 0; i < _mat->phases.size(); i++) {
				components.push_back(_mat->phases[i]->phi);
			}
			elsize = sizeof(P);
		}
		else
		{
			for (std::size_t i = 0; i < _mat->phases.size(); i++) {
				if (field == _mat->phases[i]->name) {
					components.push_back(_mat->phases[i]->phi);
					break;
				}
			}
			elsize = sizeof(P);

			if (components.size() == 0) {
				BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown field '%s'") % field).str()));
			}
		}

		return NULL;
	}

	// FIXME: this is a hack
	void free_raw_field(void* handle)
	{
		if (handle == NULL) {
			return;
		}

		delete (TensorFieldBase*)handle;
	}

	void init_fft()
	{
		get_fft(1);

#ifdef USE_MANY_FFT
		// TODO: need to initialize all cases or _epsilon data will be overwritten in get_fft when called later
		#error "TODO"
#endif
	}

	noinline boost::shared_ptr< FFT3<T> > get_fft(std::size_t howmany)
	{
		if (_ffts.count(howmany) > 0) {
			return _ffts[howmany];
		}
		
		Timer __t("init_fft", true);
		LOG_COUT << "howmany=" << howmany << " nx=" << _nx << " ny=" << _ny << " nz=" << _nz << std::endl;

		boost::shared_ptr< FFT3<T> > fft;
		fft.reset();
		fft.reset(new FFT3<T>(howmany, _nx, _ny, _nz, (*_epsilon)[0], _fft_planner_flag));
		_ffts[howmany] = fft;
		return fft;
	}

	inline void setConvergenceCallback(ConvergenceCallback cb)
	{
		_convergence_callback = cb;
	}

	inline void setLoadstepCallback(LoadstepCallback cb)
	{
		_loadstep_callback = cb;
	}

	inline T mu_matrix() const
	{
		{
			LinearIsotropicMaterialLaw<T>* law = dynamic_cast<LinearIsotropicMaterialLaw<T>*>(_mat->phases[_matrix_mat]->law.get());
			if (law != NULL) {
				return law->mu;
			}
		}

		{
			ScalarLinearIsotropicMaterialLaw<T>* law = dynamic_cast<ScalarLinearIsotropicMaterialLaw<T>*>(_mat->phases[_matrix_mat]->law.get());
			if (law != NULL) {
				return law->mu;
			}
		}

		BOOST_THROW_EXCEPTION(std::runtime_error("mu_matrix is only valid for linear isotropic materials"));
	}

	std::size_t getMaterialId(const std::string& name) const
	{
		for (std::size_t i = 0; i < _mat->phases.size(); i++) {
			if (_mat->phases[i]->name == name) {
				return i;
			}
		}

		BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown material '%s'") % name).str()));
	}


	typedef struct {
		std::vector<std::size_t> indices;
		T volume;
		ublas::c_vector<T, 3> center;
		ublas::c_matrix<T, 3, 3> moment;
		ublas::c_vector<T, 3> axis;
		T fiber_length;
		T fiber_radius;
		std::size_t checksum1;
		std::size_t checksum2;

	} segment;

	void followPath(const std::vector<bool>& graph, std::size_t start, std::vector<std::size_t>& exclude, std::size_t max_length, std::size_t nzp, std::size_t nyzp, std::size_t ny, std::size_t nx1, std::size_t ny1, std::size_t nz1)
	{
		std::size_t k = (start%nzp);
		std::size_t j = (((start - k)/nzp)%ny);
		std::size_t i = ((start - k - j*nzp)/nyzp);

		exclude.push_back(start);

		if (exclude.size() >= max_length) return;

		for (int di = ((i > 0) ? (i - 1) : 0); di <= (int)((i < nx1) ? (i + 1) : nx1); di++)
		for (int dj = ((j > 0) ? (j - 1) : 0); dj <= (int)((j < ny1) ? (j + 1) : ny1); dj++)
		for (int dk = ((k > 0) ? (k - 1) : 0); dk <= (int)((k < nz1) ? (k + 1) : nz1); dk++)
		{
			int kkk = di*nyzp + dj*nzp + dk;
			
			if (!graph[kkk]) continue;

			bool found = false;
			for (std::size_t w = 0; w < exclude.size(); w++) {
				if (((int)exclude[w]) == kkk) {
					found = true;
					break;
				}
			}

			if (found) continue;

			followPath(graph, kkk, exclude, max_length, nzp, nyzp, ny, nx1, ny1, nz1);
		}
	}

	// TODO: support periodicity
	void detectFibers(T threshold=0.5, const std::string& segment_vtk="", bool binary_vtk=true, bool overwrite_phase=false, std::size_t filter_loops=0, T convexity_threshold_low=0.0, T convexity_threshold_high=1.0, std::size_t convexity_level=0, std::size_t dir=0, std::size_t radius=1, T min_segment_volume=0.0, std::size_t max_path_length=5, T d_exponent=1, T w_exponent=2, T p_threshold=0.5, const std::vector<T>& fiber_template = std::vector<T>())
	{
		#define GET_X(p,x) \
			std::size_t _kk = p; \
			std::size_t _k = (_kk%nzp); \
			std::size_t _j = (((_kk - _k)/nzp)%ny); \
			std::size_t _i = ((_kk - _k - _j*nzp)/nyzp); \
			x[0] = _i*scale_x + _x0[0]; \
			x[1] = _j*scale_y + _x0[1]; \
			x[2] = _k*scale_z + _x0[2];


		std::size_t mat = getMaterialId("fiber");
		Phase& ph = *_mat->phases[mat];

		const int nx = (int)ph._phi->nx;
		const int ny = (int)ph._phi->ny;
		const int nz = (int)ph._phi->nz;
		const int nzp = (int)ph._phi->nzp;
		const int nyzp = ny*nzp;

		std::vector<T> pr(nx*nyzp);
		//std::vector<bool> binary_new(nx*nyzp);

		// 1. create probability image by radial template matching
		#pragma omp parallel for schedule (static)
		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < ny; j++)
			{
				int kk = i*nyzp + j*nzp;

				for (int k = 0; k < nz; k++)
				{
					T sum = 0;
					T w_sum = 0;
					int r = fiber_template.size();
					
					for (int ti = std::max(i-r, (int)0); ti < std::min(i+r, nx); ti++)
					for (int tj = std::max(j-r, (int)0); tj < std::min(j+r, ny); tj++)
					for (int tk = std::max(k-r, (int)0); tk < std::min(k+r, nz); tk++)
					{
						int kkk = ti*nyzp + tj*nzp + tk;
						int dx = ti - i;
						int dy = tj - j;
						int dz = tk - k;
						T d = std::sqrt((T)(dx*dx + dy*dy + dz*dz));
						int id = (int) d;
						if (id >= r) continue;
						T value = 0;

						if (id < r-1) {
							value = fiber_template[id]*(1-d+id) + fiber_template[id+1]*(d-id);
						}

						T diff = std::pow(std::abs(ph.phi[kkk] - value), d_exponent);
						T w = 1/(1 + std::pow(d, w_exponent));
						sum += w*(1 - diff);
						w_sum += w;
					}

					T p = sum/w_sum;

					pr[kk] = ph.phi[kk]*std::max((T)0, p - p_threshold);
					kk++;
				}
			}
		}


		// write result back to phase image (for gui visualization)
		if (overwrite_phase)
		{
			#pragma omp parallel for
			for (std::size_t i = 0; i < pr.size(); i++) {
				ph.phi[i] = pr[i];
			}
		}


		// TODO: bias field correction?
		// http://www.itk.org/ITK/applications/MRIBiasCorrection.html

	}


	// TODO: support periodicity
	void detectFibers_old(T threshold=0.5, const std::string& segment_vtk="", bool binary_vtk=true, bool overwrite_phase=false, std::size_t filter_loops=0, T convexity_threshold_low=0.0, T convexity_threshold_high=1.0, std::size_t convexity_level=0, std::size_t dir=0, std::size_t radius=1, T min_segment_volume=0.0, std::size_t max_path_length=5)
	{
		#define GET_X(p,x) \
			std::size_t _kk = p; \
			std::size_t _k = (_kk%nzp); \
			std::size_t _j = (((_kk - _k)/nzp)%ny); \
			std::size_t _i = ((_kk - _k - _j*nzp)/nyzp); \
			x[0] = _i*scale_x + _x0[0]; \
			x[1] = _j*scale_y + _x0[1]; \
			x[2] = _k*scale_z + _x0[2];


		std::size_t mat = getMaterialId("fiber");
		Phase& ph = *_mat->phases[mat];

		const std::size_t nx = ph._phi->nx;
		const std::size_t ny = ph._phi->ny;
		const std::size_t nz = ph._phi->nz;
		const std::size_t nzp = ph._phi->nzp;
		const std::size_t nyzp = ny*nzp;
		const std::size_t nx1 = nx-1;
		const std::size_t ny1 = ny-1;
		const std::size_t nz1 = nz-1;
		const T scale_x = _dx/nx;
		const T scale_y = _dy/ny;
		const T scale_z = _dz/nz;
		const T scale_xyz = scale_x*scale_y*scale_z;

		ProgressBar<T> pb(nx);

		std::vector<segment> segments;
		std::vector<std::size_t> stack;
		std::vector<bool> filled(nx*nyzp);
		std::vector<bool> binary(nx*nyzp);
		std::vector<bool> skeleton(nx*nyzp);
		//std::vector<bool> binary_new(nx*nyzp);

		// 1. binarize image
		#pragma omp parallel for schedule (static)
		for (std::size_t i = 0; i < nx; i++)
		{
			for (std::size_t j = 0; j < ny; j++)
			{
				std::size_t kk = i*nyzp + j*nzp;

				for (std::size_t k = 0; k < nz; k++)
				{
					binary[kk] = ph.phi[kk] > threshold;
					kk++;
				}
			}
		}


#ifdef ITK_ENABLED
		// 2. skeletonize binary image
		// TODO: use ITK code from demo/segmentation example
		{
			// convert to ITK image
			const   unsigned int Dimension = 3;
			typedef unsigned char PixelType;
			typedef itk::Image<PixelType, Dimension> ImageType;

			ImageType::Pointer image = ImageType::New();
			ImageType::SizeType size;
			ImageType::IndexType index;
			ImageType::RegionType region;
			size[0] = nx;
			size[1] = ny;
			size[2] = nz;
			index.Fill(0);
			region.SetSize(size);
			region.SetIndex(index);
			image->SetRegions(region);
			image->Allocate();

			// TODO: is SetPixel safe to run in parallel?
			#pragma omp parallel for schedule (static)
			for (std::size_t i = 0; i < nx; i++)
			{
				for (std::size_t j = 0; j < ny; j++)
				{
					std::size_t kk = i*nyzp + j*nzp;

					for (std::size_t k = 0; k < nz; k++)
					{
						ImageType::IndexType index;
						index[0] = i;
						index[1] = j;
						index[2] = k;
						image->SetPixel(index, binary[kk] ? 255 : 0);
						kk++;
					}
				}
			}

			LOG_COUT << "done1" << std::endl;

			typedef itk::BinaryThinningImageFilter3D<ImageType, ImageType> ThinningFilterType;
			ThinningFilterType::Pointer thinningFilter = ThinningFilterType::New();
			thinningFilter->SetInput(image);
			thinningFilter->Update();

			ImageType::Pointer output = thinningFilter->GetOutput();

			// TODO: is GetPixel safe to run in parallel?
			#pragma omp parallel for schedule (static)
			for (std::size_t i = 0; i < nx; i++)
			{
				for (std::size_t j = 0; j < ny; j++)
				{
					std::size_t kk = i*nyzp + j*nzp;

					for (std::size_t k = 0; k < nz; k++)
					{
						ImageType::IndexType index;
						index[0] = i;
						index[1] = j;
						index[2] = k;
						skeleton[kk] = output->GetPixel(index) != 0;
						kk++;
					}
				}
			}

			LOG_COUT << "done2" << std::endl;
		}
#else
		BOOST_THROW_EXCEPTION(std::runtime_error("detectFibers requires ITK"));
#endif


#if 1
		// 3. find voxels with more than 2 connections and disconnect them
		// also clear single voxels
		// TODO: the disconnection could be improved to avoid accidantially splitting fibers
		#pragma omp parallel for schedule (static)
		for (std::size_t i = 0; i < nx; i++)
		{
			for (std::size_t j = 0; j < ny; j++)
			{
				std::size_t kk = i*nyzp + j*nzp;

				for (std::size_t k = 0; k < nz; k++)
				{
					if (!skeleton[kk]) {
						kk++;
						continue;
					}

					std::vector< std::pair<std::size_t, ublas::c_vector<T, 3> > > directions;
					std::vector< std::vector<std::size_t> > paths;
					ublas::c_vector<T, 3> x1;
					GET_X(kk, x1);

					for (int di = ((i > 0) ? (i - 1) : 0); di <= (int)((i < nx1) ? (i + 1) : nx1); di++)
					for (int dj = ((j > 0) ? (j - 1) : 0); dj <= (int)((j < ny1) ? (j + 1) : ny1); dj++)
					for (int dk = ((k > 0) ? (k - 1) : 0); dk <= (int)((k < nz1) ? (k + 1) : nz1); dk++)
					{
						int kkk = di*nyzp + dj*nzp + dk;
						if (kkk == (int)kk) continue;
						if (!skeleton[kkk]) continue;
						
						std::vector<std::size_t> path;
						path.push_back(kk);
						followPath(skeleton, kkk, path, max_path_length, nzp, nyzp, ny, nx1, ny1, nz1);

						ublas::c_vector<T, 3> dirs = ublas::zero_vector<T>(3);

						for (std::size_t w = 0; w < path.size(); w++) {
							ublas::c_vector<T, 3> x2, dir;
							GET_X(path[w], x2);
							dir = x1 - x2;
							dir /= ublas::norm_2(dir);
							dirs += dir;
						}
						dirs /= ublas::norm_2(dirs);

						directions.push_back(std::pair<std::size_t, ublas::c_vector<T, 3> >(kkk, dirs));
						paths.push_back(path);
					}

					// find two directions with smallest angle to each other
					// disconnect all other directions

					if (directions.size() > 2)
					{
						T cos_angle_min = 2;
						std::size_t d1_min = 0, d2_min = 0;
						for (std::size_t d1 = 0; d1 < directions.size(); d1++) {
							for (std::size_t d2 = d1; d2 < directions.size(); d2++) {
								if (d1 == d2) continue;
								T cos_angle = ublas::inner_prod(directions[d1].second, directions[d1].second);
								if (cos_angle < cos_angle_min) {
									d1_min = d1;
									d2_min = d2;
									cos_angle_min = cos_angle;
								}
							}
						}
						
						for (std::size_t d = 0; d < directions.size(); d++) {
							if (d == d1_min) continue;
							if (d == d2_min) continue;
							for (std::size_t w = 1; w < paths[d].size(); w++) {
								skeleton[paths[d][w]] = false;
							}
						}
					}

					kk++;
				}
			}
		}

		LOG_COUT << "done3" << std::endl;
#endif



#if 0
		#pragma omp parallel for schedule (static)
		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < ny; j++)
			{
				int kk = (int)(i*nyzp + j*nzp);

				for (int k = 0; k < nz; k++)
				{
					T sum = 0;
					int n = 0;
					for (int i1 = -(int)radius; i1 <= (int)radius; i1++)
					for (int j1 = -(int)radius; j1 <= (int)radius; j1++)
					for (int k1 = -(int)radius; k1 <= (int)radius; k1++)
					{
						int r2 = i1*i1 + j1*j1 + k1*k1;
						if (r2 > (int)(radius*radius)) continue;

						int ii1 = std::min(std::max(0, i + i1), (int)nx1);
						int jj1 = std::min(std::max(0, j + j1), (int)ny1);
						int kk1 = std::min(std::max(0, k + k1), (int)nz1);
						int kkk1 = ii1*nyzp + jj1*nzp + kk1;
						
						sum += ph.phi[kkk1];
						n ++;
					}

					sum /= n;
					binary_new[kk] = sum*sum > threshold;
					kk++;
				}
			}
		}

		binary = binary_new;


		char dirs[4][3][2] = {
			{{-1, 1}, {-1, 1}, {-1, 1}},
			{{-1, 1}, {-1, 1}, {0, 0}},
			{{-1, 1}, {0, 0}, {-1, 1}},
			{{0, 0}, {-1, 1}, {-1, 1}},
		};

		std::size_t d = std::min(std::max((std::size_t)0, dir), (std::size_t)2);

		// run convexity filter
		binary_new = binary;
		for (std::size_t g = 0; g < filter_loops; g++)
		{
			// binarize image
			#pragma omp parallel for firstprivate(stack) schedule (static)
			for (int i = 0; i < nx; i++)
			{
				for (int j = 0; j < ny; j++)
				{
					std::size_t kk = i*nyzp + j*nzp;

					for (int k = 0; k < nz; k++)
					{
						stack.push_back(kk);

						while (stack.size() > 0)
						{
							int _kk = (int) stack.back();
							stack.pop_back();

							if (!binary[_kk]) continue;

							{
								int n = 0;
								T sum = 0;
								for (char i1 = dirs[d][0][0]; i1 <= dirs[d][0][1]; i1++)
								for (char j1 = dirs[d][1][0]; j1 <= dirs[d][1][1]; j1++)
								for (char k1 = dirs[d][2][0]; k1 <= dirs[d][2][1]; k1++)
								{
									int ii1 = std::min(std::max(0, i + i1), (int)nx1);
									int jj1 = std::min(std::max(0, j + j1), (int)ny1);
									int kk1 = std::min(std::max(0, k + k1), (int)nz1);
									int kkk1 = ii1*nyzp + jj1*nzp + kk1;
									sum += ph.phi[kkk1];
									n++;
								}
								
								sum /= n;
								if (sum > convexity_threshold_high) continue;
								if (sum < convexity_threshold_low) continue;
							}

							int _k = (int)(_kk%nzp);
							int _j = (int)(((_kk - _k)/nzp)%ny);
							int _i = (int)((_kk - _k - _j*nzp)/nyzp);

							std::size_t convexity_errors = 0;
							
							for (char i1 = dirs[d][0][0]; i1 <= dirs[d][0][1]; i1++)
							for (char j1 = dirs[d][1][0]; j1 <= dirs[d][1][1]; j1++)
							for (char k1 = dirs[d][2][0]; k1 <= dirs[d][2][1]; k1++)
							{
								int ii1 = std::min(std::max(0, _i + i1), (int)nx1);
								int jj1 = std::min(std::max(0, _j + j1), (int)ny1);
								int kk1 = std::min(std::max(0, _k + k1), (int)nz1);
								int kkk1 = ii1*nyzp + jj1*nzp + kk1;

								if (!binary[kkk1]) continue;

								for (char i2 = dirs[d][0][0]; i2 <= dirs[d][0][1]; i2++)
								for (char j2 = dirs[d][1][0]; j2 <= dirs[d][1][1]; j2++)
								for (char k2 = dirs[d][2][0]; k2 <= dirs[d][2][1]; k2++)
								{
									int ii2 = std::min(std::max(0, _i + i2), (int)nx1);
									int jj2 = std::min(std::max(0, _j + j2), (int)ny1);
									int kk2 = std::min(std::max(0, _k + k2), (int)nz1);
									int kkk2 = ii2*nyzp + jj2*nzp + kk2;

									if (!binary[kkk2]) continue;
									
									char di = ii2 - ii1;
									char dj = jj2 - jj1;
									char dk = kk2 - kk1;
									
									if (std::abs(di) == 1) continue;
									if (std::abs(dj) == 1) continue;
									if (std::abs(dk) == 1) continue;

									// check voxel 1/3 between the voxels
									int ii3 = std::min(std::max(0, (6*ii1 + 2*di + 3)/6), (int)nx1);
									int jj3 = std::min(std::max(0, (6*jj1 + 2*dj + 3)/6), (int)ny1);
									int kk3 = std::min(std::max(0, (6*kk1 + 2*dk + 3)/6), (int)nz1);
									int kkk3 = ii3*nyzp + jj3*nzp + kk3;

									if (!binary[kkk3]) {
										//is_convex = false;
										//goto outer_loop;
										convexity_errors++;
									}
									
									// check voxel 2/3 between the voxels
									int ii4 = std::min(std::max(0, (6*ii1 + 4*di + 3)/6), (int)nx1);
									int jj4 = std::min(std::max(0, (6*jj1 + 4*dj + 3)/6), (int)ny1);
									int kk4 = std::min(std::max(0, (6*kk1 + 4*dk + 3)/6), (int)nz1);
									int kkk4 = ii4*nyzp + jj4*nzp + kk4;

									if (!binary[kkk4]) {
										//is_convex = false;
										//goto outer_loop;
										convexity_errors++;
									}
								}
							}

							//outer_loop:

							bool is_convex = (convexity_errors <= convexity_level);

							if (!is_convex) {
#if 0
								for (char i1 = dirs[d][0][0]; i1 <= dirs[d][0][1]; i1++)
								for (char j1 = dirs[d][1][0]; j1 <= dirs[d][1][1]; j1++)
								for (char k1 = dirs[d][2][0]; k1 <= dirs[d][2][1]; k1++)
								{
									int ii1 = std::min(std::max(0, _i + i1), (int)nx1);
									int jj1 = std::min(std::max(0, _j + j1), (int)ny1);
									int kk1 = std::min(std::max(0, _k + k1), (int)nz1);
									int kkk1 = ii1*nyzp + jj1*nzp + kk1;
									if (!binary[kkk1]) continue;
									stack.push_back(kkk1);
								}
#endif
								binary_new[_kk] = false;
							}
						}

						kk++;
					}
				}
			}

			binary = binary_new;
		}

#endif


		// 4. segment the skeleton image

		#pragma omp parallel for firstprivate(stack, filled) schedule (static)
		for (std::size_t i = 0; i < nx; i++)
		{
			for (std::size_t j = 0; j < ny; j++)
			{
				std::size_t kk = i*nyzp + j*nzp;

				for (std::size_t k = 0; k < nz; k++)
				{
					segment s;
					s.volume = 0;
					s.checksum1 = 0;
					s.checksum2 = 0;
					s.center[0] = s.center[1] = s.center[2] = 0;

					stack.push_back(kk);

					while (stack.size() > 0)
					{
						std::size_t _kk = stack.back();
						stack.pop_back();
						
						if (!skeleton[_kk] || filled[_kk]) continue;
						
						// mark voxel as filled
						filled[_kk] = true;

						// compute voxel indices from _kk
						std::size_t _k = _kk%nzp;
						std::size_t _j = ((_kk - _k)/nzp)%ny;
						std::size_t _i = (_kk - _k - _j*nzp)/nyzp;
						
						// add neighbour voxels to stack (to be filled)
#if 1
						// all neighbours
						for (int di = ((_i > 0) ? (_i - 1) : _i); di <= (int)((_i < nx1) ? (_i + 1) : nx1); di++)
						for (int dj = ((_j > 0) ? (_j - 1) : _j); dj <= (int)((_j < ny1) ? (_j + 1) : ny1); dj++)
						for (int dk = ((_k > 0) ? (_k - 1) : _k); dk <= (int)((_k < nz1) ? (_k + 1) : nz1); dk++) {
							int kkk = di*nyzp + dj*nzp + dk;
							if (!filled[kkk]) stack.push_back(kkk);
						}
#else
						// only 6 neighbours
						if (_i > 0   && !filled[_kk - nyzp]) stack.push_back(_kk - nyzp);  // i-1
						if (_i < nx1 && !filled[_kk + nyzp]) stack.push_back(_kk + nyzp);  // i+1
						if (_j > 0   && !filled[_kk - nzp]) stack.push_back(_kk - nzp);  // j-1
						if (_j < ny1 && !filled[_kk + nzp]) stack.push_back(_kk + nzp);  // j+1
						if (_k > 0   && !filled[_kk - 1]) stack.push_back(_kk - 1);  // k-1
						if (_k < nz1 && !filled[_kk + 1]) stack.push_back(_kk + 1);  // k+1
#endif

						// update segment info
						T phi = ph.phi[_kk];
						s.indices.push_back(_kk);
						s.checksum1 += _kk;
						s.checksum2 ^= _kk;
						s.volume += phi;
						s.center[0] += _i*phi;
						s.center[1] += _j*phi;
						s.center[2] += _k*phi;
					}

					if (s.volume > min_segment_volume)
					{
						#pragma omp critical
						{
							bool segment_exists = false;
							for (std::size_t p = 0; p < segments.size(); p++) {
								if (segments[p].checksum1 == s.checksum1 && segments[p].checksum2 == s.checksum2) {
									segment_exists = true;
									break;
								}
							}

							if (!segment_exists) {
								segments.push_back(s);
							}
						}
					}

					kk++;
				}
			}

			if (omp_get_thread_num() == 0 && !pb.complete() && pb.update()) {
				pb.message() << ph.name << " (" << nx << "x" << ny << "x" << nz << ") fiber detection " << pb.end();
			}
		}

		LOG_COUT << "done4" << std::endl;

		// 5. create image with the segment ids and 
		std::vector<T> segment_ids(nx*nyzp);
		boost::shared_ptr< FiberCluster<float, 3> > cluster;

		for (std::size_t p = 0; p < segments.size(); p++)
		{
			segment& s = segments[p];

			for (std::size_t i = 0; i < s.indices.size(); i++)
			{
				ublas::c_vector<float, 3> x;
				GET_X(s.indices[i], x);
				boost::shared_ptr< PointFiber<float, 3> > fiber(new PointFiber<float, 3>(x));
				fiber->set_material(p+1);
				if (cluster) {
					cluster->add(fiber);
				}
				else {
					cluster.reset();
					cluster.reset(new FiberCluster<float, 3>(fiber));
				}
				segment_ids[s.indices[i]] = p+1;
			}
		}

		LOG_COUT << "done5" << std::endl;


		// 6. dilatate segment ids to whole image, only the first segment id is stored to each voxel (no overwriting)
		// and multiply the resulting image with the binary image from step 1

		if (cluster && filter_loops > 0)
		{
			for (std::size_t p = 0; p < segments.size(); p++)
			{
				segment& s = segments[p];

				// reset segment data
				s.volume = 0;
				s.center[0] = s.center[1] = s.center[2] = 0;
				s.indices.clear();
			}

			#pragma omp parallel for schedule (static)
			for (std::size_t i = 0; i < nx; i++)
			{
				for (std::size_t j = 0; j < ny; j++)
				{
					std::size_t kk = i*nyzp + j*nzp;

					for (std::size_t k = 0; k < nz; k++)
					{
						if (!binary[kk]) {
							kk++;
							continue;
						}

						ublas::c_vector<float, 3> p, x;
						boost::shared_ptr< const Fiber<float, 3> > fiber;
						p[0] = i*scale_x + _x0[0];
						p[1] = j*scale_y + _x0[1];
						p[2] = k*scale_z + _x0[2];
					
						cluster->closestFiber(p, STD_INFINITY(float), -1, x, fiber);
						if (!fiber) {
							BOOST_THROW_EXCEPTION(std::runtime_error("closestFiber failed"));
						}

						std::size_t segment_id = fiber->material();
						segment_ids[kk] = segment_id;

						// update segment info
						T phi = ph.phi[kk];
						segment& s = segments[segment_id - 1];

						#pragma omp critical
						{
							s.volume += phi;
							s.center[0] += i*phi;
							s.center[1] += j*phi;
							s.center[2] += k*phi;
							s.indices.push_back(kk);
						}

						kk++;
					}
				}
			}
		}

		LOG_COUT << "done6" << std::endl;


		// 7. finalize segments
		for (std::size_t p = 0; p < segments.size(); p++)
		{
			segment& s = segments[p];

			s.center /= s.volume;
			s.center[0] = s.center[0]*scale_x + _x0[0];
			s.center[1] = s.center[1]*scale_y + _x0[1];
			s.center[2] = s.center[2]*scale_z + _x0[2];

			s.volume *= scale_xyz;
			s.moment = ublas::zero_matrix<T>(3);

			// finalize segment
			for (std::size_t p = 0; p < s.indices.size(); p++)
			{
				T x[3];
				GET_X(s.indices[p], x);
				T phi = ph.phi[_kk];

				for (std::size_t i1 = 0; i1 < 3; i1++) {
					for (std::size_t i2 = 0; i2 < 3; i2++) {
						s.moment(i1, i2) += phi*(x[i1] - s.center[i1])*(x[i2] - s.center[i2]);
					}
				}
			}

			// compute eigenvalues/vectors of moment
			ublas::c_vector<T, 3> e;
			ublas::c_matrix<T, 3, 3> V(s.moment);
			lapack::syev('V', 'U', V, e, lapack::optimal_workspace());

			// get major axis (fiber orientation)
			std::size_t imax = 0;
			if (e[1] > e[imax]) imax = 1;
			if (e[2] > e[imax]) imax = 2;

			s.axis[0] = V(imax,0);
			s.axis[1] = V(imax,1);
			s.axis[2] = V(imax,2);

			// determine fiber length and radius
			T fiber_semi_length_min = 0;
			T fiber_semi_length_max = 0;
			T fiber_radius = 0;
			for (std::size_t p = 0; p < s.indices.size(); p++) {
				ublas::c_vector<T, 3> x;
				GET_X(s.indices[p], x.data());
				T x_a = ublas::inner_prod(x - s.center, s.axis);
				T x_r = ublas::norm_2(x - s.center - x_a*s.axis);
				fiber_semi_length_min = std::min(fiber_semi_length_min, x_a);
				fiber_semi_length_max = std::max(fiber_semi_length_max, x_a);
				fiber_radius = std::max(fiber_radius, x_r);
			}
			s.fiber_length = fiber_semi_length_max - fiber_semi_length_min;
			s.fiber_radius = fiber_radius;
		}

		LOG_COUT << "done7" << std::endl;

		// write result back to phase image (for gui visualization)
		if (overwrite_phase)
		{
			// generate permuted segment ids for better visualization
			std::vector<std::size_t> segment_id_shuffle(segments.size() + 1);
			for (std::size_t p = 0; p <= segments.size(); p++) {
				segment_id_shuffle[p] = p;
			}
			std::random_shuffle(segment_id_shuffle.begin()+1, segment_id_shuffle.end());

			#pragma omp parallel for
			for (std::size_t i = 0; i < segment_ids.size(); i++) {
				ph.phi[i] = segment_id_shuffle[segment_ids[i]];
			}
		}

		LOG_COUT << "done8" << std::endl;

		// write segmentation info
		for (std::size_t p = 0; p < segments.size(); p++)
		{
			segment& s = segments[p];

			LOG_COUT << "segment " << p << ": volume=" << s.volume
				<< " orientation x=" << s.axis(0) << " y=" << s.axis(1) << " z=" << s.axis(2) 
				<< " center x=" << s.center(0) << " y=" << s.center(1) << " z=" << s.center(2)
				<< " fiber length=" << s.fiber_length
				<< " fiber radius=" << s.fiber_radius
				<< std::endl;
		}


		// print fo tensor A2

		ublas::c_matrix<T, 3, 3> A2 = ublas::zero_matrix<T>(3);
		T volume_sum = 0;

		for (std::size_t p = 0; p < segments.size(); p++)
		{
			segment& s = segments[p];
			// update fo tensor
			A2 += s.volume*ublas::outer_prod(s.axis, s.axis);
			volume_sum += s.volume;
		}

		A2 /= volume_sum;
		LOG_COUT << "FO tensor (A2):\n" << format(A2) << std::endl;
	
		LOG_COUT << "done9" << std::endl;
	
		// save segment ids to vtk file

		if (segment_vtk.empty()) return;

		VTKCubeWriter<T> cw(segment_vtk, binary_vtk ? VTKCubeWriter<T>::WriteModes::BINARY : VTKCubeWriter<T>::WriteModes::ASCII,
			_nx, _ny, _nz, _dx, _dy, _dz, _x0[0], _x0[1], _x0[2]);
		
		cw.writeMesh();
		
		cw.template beginWriteField<T>("segment_id");
#ifdef REVERSE_ORDER
		for (std::size_t j = 0; j < _nx; j++) {
			cw.template writeZYSlice<T>(segment_ids.data() + j*_nyzp, _nzp - _nz);
		}
#else
		for (std::size_t j = 0; j < _nz; j++) {
			cw.template writeXYSlice<T>(segment_ids.data() + j, _nyzp, _nzp);
		}
#endif

		#undef GET_X
		LOG_COUT << "done10" << std::endl;
	}

	//! recursively compute volume fraction of material mat for voxel (p,dx,dy,dz)
	inline P integratePhiVoxel(const FiberGenerator<T, DIM>& fg, int levels, T tol, T r_voxel0, 
		const ublas::c_vector<T, DIM>& p, T dx, T dy, T dz, int mat, std::vector<typename FiberCluster<T, DIM>::ClosestFiberInfo>& info_list)
	{
		//LOG_COUT << "integratePhiVoxel" << " levels=" << levels << " tol=" << tol << std::endl;

		if (info_list.size() == 0) {
			return 0;
		}

		T r_voxel = 0.5*std::sqrt(dx*dx + dy*dy + dz*dz);

		// find minimum distance
		std::size_t i_min = 0;
		for (std::size_t i = 1; i < info_list.size(); i++) {
			if (info_list[i].d < info_list[i_min].d) i_min = i;
		}

		// quick check
		if (std::abs(info_list[i_min].d) >= r_voxel) {
			// voxel is completely inside/outside of fiber
			return (info_list[i_min].d < 0) ? dx*dy*dz : 0;
		}

		ublas::c_vector<T, DIM> x0;
		x0[0] = p[0] - 0.5*dx;
		x0[1] = p[1] - 0.5*dy;
		x0[2] = p[2] - 0.5*dz;

		//LOG_COUT << " p=" << format(p) << " x_min=" << format(x) << " D=" << D << std::endl;

		P V = 0;
		P V_max = dx*dy*dz;

		if (levels < 0)
		{
			// adaptive error estimator
			T K = info_list[i_min].fiber->curvature();
			T Kd = r_voxel*K;
			T err;

			if (Kd > 1) {
				err = 1;
			}
			else {
				err = Kd*Kd*std::pow(r_voxel/r_voxel0, 2.0/3.0);
			}

			if (err < tol) {
				levels = 0;
			}
		}

		if (levels == 0)
		{
			// sum up the volumes of each interface intersection
			// TODO: the volume may be overestimated actually should calculate the union of halfspaces
			
			ublas::c_vector<T, DIM> n;

			for (std::size_t i = 0; i < info_list.size(); i++)
			{
				ublas::c_vector<T, DIM>& x = info_list[i].x;

				// get interface normal
				info_list[i].fiber->distanceGrad(x, n);

				P dV = halfspace_box_cut_volume<T, DIM>(x, n, x0, dx, dy, dz);

#if 0
				if (dV < 0 || dV > V_max) {
					LOG_COUT << "x " << format(x) << std::endl;
					LOG_COUT << "n " << format(n) << std::endl;
					LOG_COUT << "x0 " << format(x0) << std::endl;
					LOG_COUT << "dx " << (dx) << std::endl;
					LOG_COUT << "V " << (V) << std::endl;
					BOOST_THROW_EXCEPTION(std::runtime_error("integratePhiVoxel unexpected error"));
				}
#endif

				V += dV;
			}

			return std::min(V, V_max);
		}

		levels--;
		dx *= 0.5;
		dy *= 0.5;
		dz *= 0.5;
		r_voxel *= 0.5;

		ublas::c_vector<T, DIM> ps;
		std::vector<typename FiberCluster<T, DIM>::ClosestFiberInfo> sub_info_list;
		sub_info_list.reserve(info_list.size());

		// recursive sub division of voxel
		for (std::size_t i = 0; i < 2; i++) {
			ps[0] = x0[0] + (i+0.5)*dx;
			for (std::size_t j = 0; j < 2; j++) {
				ps[1] = x0[1] + (j+0.5)*dy;
				for (std::size_t k = 0; k < 2; k++)
				{
					ps[2] = x0[2] + (k+0.5)*dz;

					// recalculate distances in list and build sub_list
					sub_info_list.clear();
					for (std::size_t i = 0; i < info_list.size(); i++) {
						info_list[i].d = info_list[i].fiber->distanceTo(ps, info_list[i].x);
						// quick check
						if (std::abs(info_list[i].d) >= r_voxel) {
							if (info_list[i].d < 0) {
								// voxel is completely inside of fiber
								V += dx*dy*dz;
								sub_info_list.clear();
								break;
							}
							// voxel is completely outside of fiber
							continue;
						}
						sub_info_list.push_back(info_list[i]);
					}

					if (sub_info_list.size() != 0) {
						V += integratePhiVoxel(fg, levels, tol, r_voxel0, ps, dx, dy, dz, mat, sub_info_list);
					}
				}
			}
		}

		return std::min(V, V_max);
	}

	typedef struct {
		std::size_t count;	// count of voxels
		T nx, ny, nz;		// normal sum
	}
	classinfo;


	noinline void initMultiphase(TensorField<P>& t, P discretization, std::map<std::size_t, std::size_t>& value_to_material_map, std::size_t default_mat)
	{
		Timer __t("init_multiphase", true);

		LOG_COUT << "discretization: " << discretization << std::endl;
		LOG_COUT << "mapping: " << std::endl;
		for(typename std::map<std::size_t, std::size_t>::iterator iter = value_to_material_map.begin();
			iter != value_to_material_map.end(); ++iter)
		{
			LOG_COUT << (iter->first*discretization) << " to " << _mat->phases[iter->second]->name << std::endl;
		}

		// compute downsampling factor
		if (t.nx % _nx != 0 || t.ny % _ny != 0 || t.nz % _nz != 0) {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Phase dimensions are incompatible %d %d %d %d %d %d") % t.nx % _nx % t.ny %_ny % t.nz % _nz).str()));
		}

		const std::size_t sx = std::max((std::size_t)1, t.nx / _nx);
		const std::size_t sy = std::max((std::size_t)1, t.ny / _ny);
		const std::size_t sz = std::max((std::size_t)1, t.nz / _nz);
		const std::size_t ns = sx*sy*sz;

		ProgressBar<T> pb(_nx);

		for (std::size_t kk_i = 0; kk_i < _nx; kk_i++)
		{
			#pragma omp parallel for schedule (static)
			for (std::size_t kk_j = 0; kk_j < _ny; kk_j++)
			{
				std::size_t kk = kk_i*_nyzp + kk_j*_nzp;

				for (std::size_t kk_k = 0; kk_k < _nz; kk_k++)
				{
					typename std::map<std::size_t, classinfo> cm;
					classinfo* cmax = NULL;
					P psum = 0;

					for (std::size_t i = 0; i < sx; i++) {
						for (std::size_t j = 0; j < sy; j++) {
							for (std::size_t k = 0; k < sz; k++)
							{
								std::size_t w = (sz*kk_k + k) + (sy*kk_j + j)*t.nzp + (sx*kk_i + i)*t.nyzp;

								P p = t[0][w];

								if (discretization != 0)
								{
									// compute material class value
									std::size_t value = (std::size_t) (p/discretization + 0.5);

									if (value_to_material_map.count(value) == 0) {
										BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Material value %d not mapped!") % value).str()));
									}

									std::size_t mat = value_to_material_map[value];
									if (cm.count(mat) == 0) {
										classinfo ci;
										ci.count = 0;
										ci.nx = ci.ny = ci.nz = 0;
										cm.insert(std::pair<std::size_t, classinfo>(mat, ci));
									}

									// integrate number of voxels and normals
									cmax = &(cm[mat]);
									cmax->count ++;
									cmax->nx += kk_i + i + 0.5;
									cmax->ny += kk_j + j + 0.5;
									cmax->nz += kk_k + k + 0.5;
								}

								psum += p;
							}
						}
					}

					if (cmax == NULL) {
						Phase& ph = *_mat->phases[default_mat];
						ph.phi[kk] = psum/ns;
						if (_normals) {
							BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Can't compute normals without phase discretization!")).str()));
						}
					}
					else
					{
						// assign normal
						if (_normals)
						{
							// find material with largest volume fraction in voxel
							for(typename std::map<std::size_t, classinfo>::iterator iter = cm.begin(); iter != cm.end(); ++iter) {
								if (iter->second.count > cmax->count) {
									cmax = &(iter->second);
								}
							}
						
							T nx = cmax->nx/cmax->count - (kk_i + 0.5*sx);
							T ny = cmax->ny/cmax->count - (kk_j + 0.5*sy);
							T nz = cmax->nz/cmax->count - (kk_k + 0.5*sz);

							T mag = std::sqrt(nx*nx + ny*ny + nz*nz);

							if (mag < 1e-9)
							{
								// use "random" interface normal
								
								std::size_t k3 = kk % 3;
								nx = (k3 == 0) ? 1 : 0;
								ny = (k3 == 1) ? 1 : 0;
								nz = (k3 == 2) ? 1 : 0;
								mag = 1;

								if (cm.size() > 1)
								{
#if 1
									//LOG_CWARN << (boost::format("Interface normal undefined at voxel (%d,%d,%d)") % kk_i % kk_j % kk_k).str() << std::endl;
#else
									for (std::size_t i = 0; i < sx; i++) {
										for (std::size_t j = 0; j < sy; j++) {
											for (std::size_t k = 0; k < sz; k++)
											{
												std::size_t w = (sz*kk_k + k) + (sy*kk_j + j)*t.nzp + (sx*kk_i + i)*t.nyzp;
												P p = t[0][w];
												LOG_COUT << "(" << i << "," << j << "," << k << "): " << p << std::endl;
											}
										}
									}
									BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Interface normal undefined at voxel (%d,%d,%d)") % kk_i % kk_j % kk_k).str()));
#endif
								}
							}
							
							(*_normals)[0][kk] = nx / mag;
							(*_normals)[1][kk] = ny / mag;
							(*_normals)[2][kk] = nz / mag;

							if (std::isnan(nx/mag) || std::isnan(ny/mag) || std::isnan(nz/mag)) {
								std::cout << nx << ny << nz << mag << "\n";
								BOOST_THROW_EXCEPTION(std::runtime_error(boost::format("nan").str()));
							}
						}

						for (std::size_t m = 0; m < _mat->phases.size(); m++)
						{
							Phase& ph = *_mat->phases[m];
							
							if (cm.count(m) > 0) {
								ph.phi[kk] = cm[m].count/(T)ns;
							}
							else {
								ph.phi[kk] = 0;
							}
						}
					}

					kk ++;
				}
			}
			
			if (pb.update()) {
				pb.message() << "multiphase initialization" << pb.end();
			}
		}
	}
	
	template<typename F>
	inline void readRawPhase(TensorField<P>& t, const std::string& material, std::ifstream& stream, P scale, bool col_order = true, bool compressed = false, std::size_t header_bytes = 0, P treshold = -1)
	{
		boost::iostreams::filtering_istream is;

		if (compressed) {
			is.push(boost::iostreams::gzip_decompressor());
		}
		is.push(stream);

		// skip header bytes
#if 1
		is.ignore(header_bytes);
#else
		is.seekg(header_bytes, is.beg);
#endif

		if (col_order)
		{
			ProgressBar<T> pb(t.nz);
			std::vector<F> buf(t.nx);

			for (std::size_t k = 0; k < t.nz; k++)
			{
				for (std::size_t j = 0; j < t.ny; j++)
				{
					is.read((char*)buf.data(), t.nx*sizeof(F));

					//LOG_COUT << "o " << header_bytes << std::endl;
					//LOG_COUT << (t.nx*sizeof(F)) << std::endl;

					if (!is) {
						BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Error reading raw data: %s") % strerror(errno)).str()));
					}
					
					std::size_t kk = j*t.nzp + k;

					for (std::size_t i = 0; i < t.nx; i++) {
						t[0][kk] = std::min(std::max(scale*(P)buf[i], (P)0), (P)1);
						if (treshold >= 0) t[0][kk] = (t[0][kk] > treshold) ? (P)1 : (P)0;
						kk += t.nyzp;
					}
				}

				if (pb.update()) {
					pb.message() << material << " phase initialization" << pb.end();
				}
			}
		}
		else
		{
			ProgressBar<T> pb(t.nx);
			std::vector<F> buf(t.nz);

			for (std::size_t i = 0; i < t.nx; i++)
			{
				for (std::size_t j = 0; j < t.ny; j++)
				{
					is.read((char*)buf.data(), t.nz*sizeof(F));
					if (!is) {
						BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Error reading raw data: %s") % strerror(errno)).str()));
					}
					
					std::size_t kk = i*t.nyzp + j*t.nzp;

					for (std::size_t k = 0; k < t.nz; k++) {
						t[0][kk] = std::min(std::max(scale*(P)buf[k], (P)0), (P)1);
						if (treshold >= 0) t[0][kk] = (t[0][kk] > treshold) ? (P)1 : (P)0;
						kk++;
					}
				}

				if (pb.update()) {
					pb.message() << material << " phase initialization" << pb.end();
				}
			}
		}
	}
	
	template<typename F>
	inline void writeRawPhase(const std::string& material, std::size_t material_index, std::ofstream& stream, P scale, bool col_order = true, bool compressed = false)
	{
		boost::iostreams::filtering_ostream os;

		if (compressed) {
			os.push(boost::iostreams::gzip_compressor());
		}
		os.push(stream);

		TensorField<T>& t = *(_mat->phases[material_index]->_phi);

		if (col_order)
		{
			ProgressBar<T> pb(t.nz);
			std::vector<F> buf(t.nx);

			for (std::size_t k = 0; k < t.nz; k++)
			{
				for (std::size_t j = 0; j < t.ny; j++)
				{

					std::size_t kk = j*t.nzp + k;

					for (std::size_t i = 0; i < t.nx; i++)
					{
						buf[i] = (F)(scale*t[0][kk]);
						kk += t.nyzp;
					}

					os.write((char*)buf.data(), t.nx*sizeof(F));

					if (!os) {
					//	BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Error writing raw data: %s") % strerror(errno)).str()));
					}
				}

				if (pb.update()) {
					pb.message() << "writing phase " << material << pb.end();
				}
			}
		}
		else
		{
			ProgressBar<T> pb(t.nx);
			std::vector<F> buf(t.nz);

			for (std::size_t i = 0; i < t.nx; i++)
			{
				for (std::size_t j = 0; j < t.ny; j++)
				{					
					std::size_t kk = i*t.nyzp + j*t.nzp;

					for (std::size_t k = 0; k < t.nz; k++)
					{
						buf[k] = (F)(scale*t[0][kk]);
						kk++;
					}

					os.write((char*)buf.data(), t.nz*sizeof(F));
					
					if (!os) {
					//	BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Error writing raw data: %s") % strerror(errno)).str()));
					}
				}

				if (pb.update()) {
					pb.message() << "writing phase " << material << pb.end();
				}
			}
		}
	}

	void writeData(const std::string& filename) const
	{
		std::ofstream fs;
		open_file(fs, filename);

		std::string delim = "\t";

		// write header

		fs << "i_x" << delim << "i_y" << delim << "i_z";

		if (_normals) {
			fs << delim << "n_x" << delim << "n_y" << delim << "n_z";
		}
		if (_orientation) {
			fs << delim << "a_x" << delim << "a_y" << delim << "a_z";
		}

		for (std::size_t m = 0; m < _mat->phases.size(); m++)
		{
			Phase& ph = *_mat->phases[m];
			fs << delim << ph.name;
		}

		for (std::size_t i = 0; i < _nx; i++)
		{
			for (std::size_t j = 0; j < _ny; j++)
			{
				for (std::size_t k = 0; k < _nz; k++)
				{
					int kk = i*_nyzp + j*_nzp + k;

					fs << std::endl;
					fs << i << delim << j << delim << k;

					if (_normals) {
						fs << delim << (*_normals)[0][kk] << delim << (*_normals)[1][kk] << delim << (*_normals)[2][kk];
					}
					if (_orientation) {
						fs << delim << (*_orientation)[0][kk] << delim << (*_orientation)[1][kk] << delim << (*_orientation)[2][kk];
					}
					
					for (std::size_t m = 0; m < _mat->phases.size(); m++)
					{
						Phase& ph = *_mat->phases[m];
						fs << delim << ph.phi[kk];
					}
				}
			}
		}
	}

	void setPhasesOne()
	{
		// TODO: what to do in the full_staggered scheme?
		for (std::size_t m = 0; m < _mat->phases.size(); m++) {
			_mat->phases[m]->setOne();
		}
	}

	void initRawPhi()
	{
		normalizePhi();
		initFullStageredRawPhases();
		
		if (_orientation) {
			initRawOrientation(*_orientation);
		}

		for (std::size_t i = 0; i < _mat->phases.size(); i++) {
			printTensor((boost::format("phi_%d") % i).str(), *_mat->phases[i]->_phi);
		}

	}

	//! init phase indicator and volume fractions
	void initPhi(const FiberGenerator<T, DIM>& fg, bool fast = false)
	{
		if (_gamma_scheme == "full_staggered") {
			_mat->select_dfg(true);
		}

		initPhi(fg, _mat->phases, _matrix_mat, _smooth_levels, _smooth_tol, fast);

		if (_gamma_scheme == "half_staggered") {
			// compute dfg phase from coarse grid
			this->initFullStageredRawPhases();
			_mat->select_dfg(true);
		}

		for (std::size_t i = 0; i < _mat->phases.size(); i++) {
			printTensor((boost::format("phi_%d") % i).str(), *_mat->phases[i]->_phi);
		}

		if (_orientation) {
			initOrientation(fg, *_orientation);
			printTensor("orientation", *_orientation);
		}

		if (_normals) {
			initNormals(fg, *_normals);
			printTensor("normals", *_normals);
		}

		if (_gamma_scheme == "full_staggered")
		{
			_mat->select_dfg(false);
			
			// compute coarse grid values of phase
			for (std::size_t m = 0; m < _mat->phases.size(); m++)
			{
				Phase& ph = *_mat->phases[m];
				//std::size_t nx = ph._phi_dfg->nx;
				std::size_t ny = ph._phi_dfg->ny;
				//std::size_t nz = ph._phi_dfg->nz;
				const std::size_t nzp = ph._phi_dfg->nzp;
				const std::size_t nyzp = ny*nzp;

				const P* const src = (*ph._phi_dfg)[0];
				P* const dest = (*ph._phi_cg)[0];

#if 1
				#pragma omp parallel for schedule (static)
				for (std::size_t i = 0; i < _nx; i++)
				{
					const std::size_t i0 = 2*i*nyzp;
					const std::size_t ip = i0 + nyzp;

					for (std::size_t j = 0; j < _ny; j++)
					{
						const std::size_t j0 = 2*j*nzp;
						const std::size_t jp = j0+nzp;

						std::size_t dd = i*_nyzp + j*_nzp;

						for (std::size_t k = 0; k < _nz; k++)
						{
							const std::size_t k0 = 2*k;
							const std::size_t kp = k0+1;

							dest[dd] = (1/8.0)*(
									src[i0 + j0 + k0] +
									src[ip + j0 + k0] +
									src[i0 + jp + k0] +
									src[ip + jp + k0] +
									src[i0 + j0 + kp] +
									src[ip + j0 + kp] +
									src[i0 + jp + kp] +
									src[ip + jp + kp]
								);

							dd++;
						}
					}
				}
#else
				#pragma omp parallel for schedule (static)
				for (std::size_t i = 0; i < _nx; i++)
				{
					const std::size_t ii = 2*i;
					const std::size_t im = ((ii+nx-1)%nx)*nyzp;
					const std::size_t i0 = ii*nyzp;
					const std::size_t ip = ((ii+1)%nx)*nyzp;

					for (std::size_t j = 0; j < _ny; j++)
					{
						const std::size_t jj = 2*j;
						const std::size_t jm = ((jj+ny-1)%ny)*nzp;
						const std::size_t j0 = jj*nzp;
						const std::size_t jp = ((jj+1)%ny)*nzp;

						std::size_t dd = i*_nyzp + j*_nzp;

						for (std::size_t k = 0; k < _nz; k++)
						{
							const std::size_t kk = 2*k;
							const std::size_t km = (kk+nz-1)%nz;
							const std::size_t k0 = kk;
							const std::size_t kp = (kk+1)%nz;

							dest[dd] = (1/8.0)*(
								// center
									src[i0 + j0 + k0] +
								// face centers
								(1/2.0)*(
									src[ip + j0 + k0] +
									src[i0 + jp + k0] +
									src[i0 + j0 + kp] +
									src[im + j0 + k0] +
									src[i0 + jm + k0] +
									src[i0 + j0 + km]
								) +
								// edge centers
								(1/4.0)*(
									src[ip + jp + k0] +
									src[ip + jm + k0] +
									src[ip + j0 + kp] +
									src[ip + j0 + km] +
									src[i0 + jp + kp] +
									src[i0 + jp + km] +
									src[i0 + jm + kp] +
									src[i0 + jm + km] +
									src[im + jp + k0] +
									src[im + j0 + kp] +
									src[im + jm + k0] +
									src[im + j0 + km]
								) + 
								// edges
								(1/8.0)*(
									src[ip + jp + kp] +
									src[ip + jp + km] +
									src[ip + jm + kp] +
									src[ip + jm + km] +
									src[im + jp + kp] +
									src[im + jp + km] +
									src[im + jm + kp] +
									src[im + jm + km]
								)
							);

							dd++;
						}
					}
				}
#endif
			}
		}
	}

	inline void initOrientation(const FiberGenerator<T, DIM>& fg, RealTensor& orientation)
	{
		initVectorField("orientation", FiberGenerator<T, DIM>::SampleDataTypes::ORIENTATION, fg, orientation);
	}

	inline void initNormals(const FiberGenerator<T, DIM>& fg, RealTensor& normals)
	{
		initVectorField("normals", FiberGenerator<T, DIM>::SampleDataTypes::NORMALS, fg, normals);
	}

	noinline void initRawOrientation(RealTensor& field)
	{
		std::string name = "Orientation";
		Timer __t(name + " initialization");

		const std::size_t nx = field.nx;
		const std::size_t ny = field.ny;
		const std::size_t nz = field.nz;
		const std::size_t nzp = field.nzp;
		const std::size_t nyzp = ny*nzp;
		
		ProgressBar<T> pb(nx);

		for (std::size_t i = 0; i < nx; i++)
		{
			#pragma omp parallel for schedule (static)
			for (std::size_t j = 0; j < ny; j++)
			{
				std::size_t kk = i*nyzp + j*nzp;

				for (std::size_t k = 0; k < nz; k++)
				{
					ublas::c_vector<T, DIM> a = ublas::zero_vector<T>(DIM);
					
					for (std::size_t m = 0; m < _mat->phases.size(); m++)
					{
						Phase& ph = *_mat->phases[m];
						LinearTransverselyIsotropicMaterialLaw<T, DIM>* law = dynamic_cast<LinearTransverselyIsotropicMaterialLaw<T, DIM>*>(ph.law.get());

						if (law != NULL) {
							a += ph.phi[kk]*law->a;
						}
					}

					T norm_a = ublas::norm_2(a);

					if (norm_a == 0) {
						BOOST_THROW_EXCEPTION(std::runtime_error("Orientation field undefined. Specify an orientation (ax,ay,az) for transversely isotropic material laws."));
					}

					a /= norm_a;
					field[0][kk] = a[0];
					field[1][kk] = a[1];
					field[2][kk] = a[2];
					kk++;
				}

				// this is actuall unnecessary, but it prevents valgrind from printing warnings, when the padding is accessed
				for (std::size_t k = nz; k < nzp; k++) {
					field[0][kk] = field[1][kk] = field[2][kk] = 0.0/0.0;
					kk++;
				}
			}

			if (pb.update()) {
				pb.message() << name << " (" << nx << "x" << ny << "x" << nz << ") data initialization" << pb.end();
			}
		}
	}


	noinline void initVectorField(const std::string& name, typename FiberGenerator<T, DIM>::SampleDataType type,
		const FiberGenerator<T, DIM>& fg, RealTensor& field)
	{
		Timer __t(name + " initialization");

		const std::size_t nx = field.nx;
		const std::size_t ny = field.ny;
		const std::size_t nz = field.nz;
		const std::size_t nzp = field.nzp;
		const std::size_t nyzp = ny*nzp;
		
		std::vector<T> tmp(3*ny*nz);

		ProgressBar<T> pb(nx);

		for (std::size_t i = 0; i < nx; i++)
		{
			fg.sampleZYSlice(i, nx, ny, nz, &(tmp[0]), -1, type, 0, false);

			#pragma omp parallel for schedule (static)
			for (std::size_t j = 0; j < ny; j++)
			{
				std::size_t kk = i*nyzp + j*nzp;

				for (std::size_t k = 0; k < nz; k++)
				{
					std::size_t k0 = 3*(j*nz + k);
					field[0][kk] = tmp[k0 + 0];
					field[1][kk] = tmp[k0 + 1];
					field[2][kk] = tmp[k0 + 2];
					kk++;
				}

				// this is actuall unnecessary, but it prevents valgrind from printing warnings, when the padding is accessed
				for (std::size_t k = nz; k < nzp; k++) {
					field[0][kk] = field[1][kk] = field[2][kk] = 0.0/0.0;
					kk++;
				}
			}

			if (pb.update()) {
				pb.message() << name << " (" << nx << "x" << ny << "x" << nz << ") data initialization" << pb.end();
			}
		}
	}

	std::size_t get_num_threads()
	{
		std::size_t num_threads;
		#pragma omp parallel
		#pragma omp master
		{
			num_threads = omp_get_num_threads();
		}

		return num_threads;
	}
	
	T run_tune_test(int num_threads, T t_measure)
	{
		std::size_t num_threads_org = this->get_num_threads();
		omp_set_num_threads(num_threads);

		std::size_t n = 0;
		Timer t;

		do {
			calcStressDiff(*_epsilon, *_epsilon);
			GammaOperator(_E, _mu_0, _lambda_0, *_epsilon, *_tau, *_tau, *_epsilon, -1);
			calcMeanStress();
			n++;
		}
		while (t.seconds() < t_measure || n < 2);
		
		T perf = t.seconds()/n;

		omp_set_num_threads(num_threads_org);

		return perf;
	}

	void tune_num_threads(T t_measure, T treshfac)
	{
		std::size_t max_threads = boost::thread::hardware_concurrency();
		std::size_t n_min = max_threads;

		if (max_threads > 1)
		{
			T t_min_max = run_tune_test(max_threads, t_measure);
			T t_min = t_min_max;

			for (std::size_t n = max_threads-1; n > 0; n--)
			{
				T t = run_tune_test(n, t_measure);
				T perf = t / t_min_max;

				if (t < t_min) {
					n_min = n;
					t_min = t;
				}

				LOG_COUT << (boost::format("performance at num_threads=%d: %.3f %s") % n % perf % (n == n_min ? "(*)" : "")) << std::endl;
				
				if (t > treshfac*t_min) {
					break;
				}
			}
		}

		omp_set_num_threads(n_min);
		LOG_COUT << "adjusted num_threads to " << n_min << std::endl;
	}

	noinline void initPhi(const FiberGenerator<T, DIM>& fg, std::vector<pPhase>& mat, std::size_t matrix_mat, int smooth_levels, T smooth_tol, bool fast = false)
	{
		Timer __t("phase initialization");

#ifdef TEST_DIST_EVAL
		g_dist_evals = 0;
#endif

		if (mat.size() < 1) return;
		
		// calculate length scale for smooth interface transitions
		// voxel diameter is diameter of sphere with same volume
		//P d_voxel = 2*std::pow(3/(4*M_PI)*_dx*_dy*_dz/_nxyz, 1/(T)3);

		const Phase& ph = *mat[0];
		const std::size_t nx = ph._phi->nx;
		const std::size_t ny = ph._phi->ny;
		const std::size_t nz = ph._phi->nz;
		const std::size_t nzp = ph._phi->nzp;
		const std::size_t nyzp = ny*nzp;
		
		const T dx_voxel = _dx/nx;
		const T dy_voxel = _dy/ny;
		const T dz_voxel = _dz/nz;
		const T V_voxel = dx_voxel*dy_voxel*dz_voxel;
		const P d_voxel = std::sqrt(dx_voxel*dx_voxel + dy_voxel*dy_voxel + dz_voxel*dz_voxel);
		const P r_voxel = 0.5*d_voxel;

		const std::size_t num_threads = this->get_num_threads();

		for (std::size_t m = 0; m < mat.size(); m++)
		{
			Phase& ph = *mat[m];
			std::size_t m_bits = 1 << m;

			if (m == matrix_mat) {
				// matrix material is always present (1)
				ph.setOne();
				continue;
			}

			ProgressBar<T> pb(std::max((std::size_t)1, nx/num_threads));

			// binarize slice and compute volume fraction
			#pragma omp parallel for schedule (static)
			for (std::size_t i = 0; i < nx; i++)
			{
				std::vector<typename FiberCluster<T, DIM>::ClosestFiberInfo> info_list;
				ublas::c_vector<T, DIM> x0;
				
				x0[0] = dx_voxel*(i + 0.5) + _x0[0];
				
				for (std::size_t j = 0; j < ny; j++)
				{
					x0[1] = dy_voxel*(j + 0.5) + _x0[1];
					std::size_t kk = i*nyzp + j*nzp;
					
					for (std::size_t k = 0; k < nz; k++)
					{
						x0[2] = dz_voxel*(k + 0.5) + _x0[2];
						
						info_list.clear();
						fg.closestFibers(x0, r_voxel, m_bits, info_list);

						if (info_list.size() > 0) {
							ph.phi[kk] = integratePhiVoxel(fg, smooth_levels, smooth_tol, r_voxel, x0, dx_voxel, dy_voxel, dz_voxel, m, info_list)/V_voxel;
						}
						else {
							ph.phi[kk] = 0;
						}

						kk++;
					}

					// this is actuall unnecessary, but it prevents valgrind from printing warnings, when the padding is accessed
					for (std::size_t k = nz; k < nzp; k++) {
						ph.phi[kk] = 0.0/0.0;
						kk++;
					}
				}

				if (omp_get_thread_num() == 0 && !pb.complete() && pb.update()) {
					pb.message() << ph.name << " (" << nx << "x" << ny << "x" << nz << ") phase initialization" << pb.end();
				}
			}
		}

#ifdef TEST_DIST_EVAL
		LOG_COUT << g_dist_evals << " distance evaluations (" << (g_dist_evals/((double)nx*ny*nz)) << "/pixel)" << std::endl;
#endif

		normalizePhi(mat);
	}

	inline void normalizePhi()
	{
		normalizePhi(_mat->phases);
	}

	noinline void normalizePhi(std::vector<pPhase>& mat)
	{
		Timer __t("normalize_phi");

		if (mat.size() < 1) return;

		// normalization step (volume fractions have to sum to 1)
		// the last material has highest priority
		std::vector<T> vf(mat.size());

		// interface volume fraction
		std::vector<T> ivf(mat.size());

		Phase& ph = *mat[0];
		std::size_t nx = ph._phi->nx;
		std::size_t ny = ph._phi->ny;
		std::size_t nz = ph._phi->nz;
		std::size_t nzp = ph._phi->nzp;

		#pragma omp parallel
		{
			std::vector<T> vp(mat.size());
			std::vector<T> ivfp(mat.size());

			#pragma omp for schedule (static) collapse(2)
			BEGIN_TRIPLE_LOOP(kk, nx, ny, nz, nzp)
			{
				T rem = 1;

				for (int m = ((int)mat.size())-1; m >= 0; m--)
				{
					T vol = std::min(rem, mat[m]->phi[kk]);

					mat[m]->phi[kk] = vol;
					rem -= vol;
					vp[m] += vol;
					ivfp[m] += (vol == 0 || vol == 1) ? 0 : 1;
				}
			}
			END_TRIPLE_LOOP(kk)

			// perform reduction
			#pragma omp critical
			{
				for (std::size_t m = 0; m < mat.size(); m++) {
					vf[m] += vp[m];
					ivf[m] += ivfp[m];
				}
			}
		}

		// set volume fractions
		for (std::size_t m = 0; m < mat.size(); m++) {
			Phase& ph = *mat[m];
			ph.vol = vf[m]/(nx*ny*nz);
			ph.ivf = ivf[m]/(nx*ny*nz);
			LOG_COUT << ph.name << " volume fraction: " << ph.vol << " interface volume fraction: " << ph.ivf << std::endl;
		}
	}

	noinline void initFullStageredRawPhases()
	{
		if (!this->use_dfg()) {
			return;
		}

		Timer __t("initFullStageredRawPhases");

		// init doubly fine grind from a coarse grid

		for (std::size_t m = 0; m < _mat->phases.size(); m++)
		{
			Phase& ph = *_mat->phases[m];
			std::size_t nx = ph._phi_dfg->nx;
			std::size_t ny = ph._phi_dfg->ny;
			std::size_t nz = ph._phi_dfg->nz;
			const std::size_t nzp = ph._phi_dfg->nzp;
			const std::size_t nyzp = ny*nzp;

			P* const dest = (*ph._phi_dfg)[0];
			const P* const src = (*ph._phi_cg)[0];

#if 1
			#pragma omp parallel for schedule (static)
			for (std::size_t i = 0; i < nx; i++)
			{
				const std::size_t i0 = (i/2)*_nyzp;

				for (std::size_t j = 0; j < ny; j++)
				{
					const std::size_t j0 = (j/2)*_nzp;
					std::size_t dd = i*nyzp + j*nzp;

					for (std::size_t k = 0; k < nz; k++)
					{
						const std::size_t k0 = k/2;

						dest[dd] = src[i0 + j0 + k0];
						dd++;
					}
				}
			}
#else
			// this smooths solution too much

			#pragma omp parallel for schedule (static)
			for (std::size_t i = 0; i < nx; i++)
			{
				const std::size_t ii = i/2;
				const std::size_t im = ((ii+_nx-1)%_nx)*_nyzp;
				const std::size_t i0 = ii*_nyzp;
				const std::size_t ip = ((ii+1)%_nx)*_nyzp;

				for (std::size_t j = 0; j < ny; j++)
				{
					const std::size_t jj = j/2;
					const std::size_t jm = ((jj+_ny-1)%_ny)*_nzp;
					const std::size_t j0 = jj*_nzp;
					const std::size_t jp = ((jj+1)%_ny)*_nzp;

					std::size_t dd = i*nyzp + j*nzp;

					for (std::size_t k = 0; k < nz; k++)
					{
						const std::size_t kk = k/2;
						const std::size_t km = (kk+_nz-1)%_nz;
						const std::size_t k0 = kk;
						const std::size_t kp = (kk+1)%_nz;

						dest[dd] = (1/8.0)*(
							// center
								src[i0 + j0 + k0] +
							// face centers
							(1/2.0)*(
								src[ip + j0 + k0] +
								src[i0 + jp + k0] +
								src[i0 + j0 + kp] +
								src[im + j0 + k0] +
								src[i0 + jm + k0] +
								src[i0 + j0 + km]
							) +
							// edge centers
							(1/4.0)*(
								src[ip + jp + k0] +
								src[ip + jm + k0] +
								src[ip + j0 + kp] +
								src[ip + j0 + km] +
								src[i0 + jp + kp] +
								src[i0 + jp + km] +
								src[i0 + jm + kp] +
								src[i0 + jm + km] +
								src[im + jp + k0] +
								src[im + j0 + kp] +
								src[im + jm + k0] +
								src[im + j0 + km]
							) + 
							// edges
							(1/8.0)*(
								src[ip + jp + kp] +
								src[ip + jp + km] +
								src[ip + jm + kp] +
								src[ip + jm + km] +
								src[im + jp + kp] +
								src[im + jp + km] +
								src[im + jm + kp] +
								src[im + jm + km]
							)
						);

						dd++;
					}
				}
			}
#endif
		}
	}

	T calcMeanEnergy(const RealTensor& epsilon) const
	{
		const RealTensor* eps = &epsilon;

		if (this->use_dfg()) {
			// transfer eps to doubly fine grid
			eps = _temp_dfg_1.get();
			prolongate_to_dfg(epsilon, *eps);
			_mat->select_dfg(true);
		}

		T mean = _mat->meanW(*eps);
		
		if (this->use_dfg()) {
			_mat->select_dfg(false);
		}

		return mean;
	}

	// compute <epsilon:C:epsilon>
	inline T calcMeanEnergy() const
	{
		return calcMeanEnergy(*_epsilon);
	}

	//! compute <(C-C0):epsilon>
	// mu_0, lambda_0: referenece material (C_0)
	ublas::vector<T> calcMeanStress(const RealTensor& epsilon) const
	{
		const RealTensor* eps = &epsilon;

		if (this->use_dfg()) {
			// transfer eps to doubly fine grid
			eps = _temp_dfg_1.get();
			prolongate_to_dfg(epsilon, *eps);
			_mat->select_dfg(true);
		}

		ublas::vector<T> mean = _mat->meanPK1(*eps, 1);
		
		if (this->use_dfg()) {
			_mat->select_dfg(false);
		}

		return mean;
	}

	noinline T calcMinEigH(const RealTensor& epsilon) const
	{
		Timer __t("calcMinEigH", false);

		const RealTensor* eps = &epsilon;
		T minDetF = 0;

		if (this->use_dfg()) {
			// transfer eps to doubly fine grid
			eps = _temp_dfg_1.get();
			prolongate_to_dfg(epsilon, *eps);
			_mat->select_dfg(true);
		}

		#pragma omp parallel
		{
			T minDetFP = 0;

			#pragma omp for schedule (static) collapse(2)
			for (std::size_t i = 0; i < eps->nx; i++)
			{
				for (std::size_t j = 0; j < eps->ny; j++)
				{
					std::size_t kk = i*eps->nyzp + j*eps->nzp;

					for (std::size_t k = 0; k < eps->nz; k++)
					{
						Tensor3x3<T> Fk;
						eps->assign(kk, Fk);
						
						ublas::c_matrix<T, 9, 9> dP;
						_mat->dPK1(i, Fk, 1, false, TensorIdentity<T,9>::Id, dP.data(), 9);

						
						ublas::c_matrix<T, 9, 9> dPT = ublas::trans(dP);
						ublas::c_matrix<T, 9, 9> r = dP - dPT;
						
						minDetFP = std::max(minDetFP, (T)ublas::norm_frobenius(r));
						kk ++;
					}
				}
			}

			#pragma omp critical
			{
				minDetF = std::max(minDetF, minDetFP);
			}
		}
		
		if (this->use_dfg()) {
			_mat->select_dfg(false);
		}

		return minDetF;
	}


	// compute minimum of def(F)
	noinline T calcMinDetF(const RealTensor& epsilon) const
	{
		Timer __t("calcMinDetF", false);

		const RealTensor* eps = &epsilon;
		T minDetF = STD_INFINITY(T);

		if (this->use_dfg()) {
			// transfer eps to doubly fine grid
			eps = _temp_dfg_1.get();
			prolongate_to_dfg(epsilon, *eps);
			_mat->select_dfg(true);
		}

		#pragma omp parallel
		{
			T minDetFP = STD_INFINITY(T);

			#pragma omp for schedule (static) collapse(2)
			for (std::size_t i = 0; i < eps->nx; i++)
			{
				for (std::size_t j = 0; j < eps->ny; j++)
				{
					std::size_t kk = i*eps->nyzp + j*eps->nzp;

					for (std::size_t k = 0; k < eps->nz; k++)
					{
						Tensor3x3<T> Fk;
						eps->assign(kk, Fk);
						minDetFP = std::min(minDetFP, Fk.det());
						kk ++;
					}
				}
			}

			#pragma omp critical
			{
				minDetF = std::min(minDetF, minDetFP);
			}
		}
		
		if (this->use_dfg()) {
			_mat->select_dfg(false);
		}

		return minDetF;
	}


	//! compute mean Cauchy Stress
	// mu_0, lambda_0: referenece material (C_0)
	ublas::vector<T> calcMeanCauchyStress(const RealTensor& epsilon) const
	{
		const RealTensor* eps = &epsilon;

		if (this->use_dfg()) {
			// transfer eps to doubly fine grid
			eps = _temp_dfg_1.get();
			prolongate_to_dfg(epsilon, *eps);
			_mat->select_dfg(true);
		}

		ublas::vector<T> mean = _mat->meanCauchy(*eps, 1);
		
		if (this->use_dfg()) {
			_mat->select_dfg(false);
		}

		return mean;
	}

	//! compute mean of epsilon
	inline ublas::vector<T> calcMeanStrain() const
	{
		return _epsilon->average();
	}

	//! compute mean Cauchy stress
	inline ublas::vector<T> calcMeanCauchyStress() const
	{
		return calcMeanCauchyStress(*_epsilon);
	}

	//! compute mean of C:epsilon
	inline ublas::vector<T> calcMeanStress() const
	{
		return calcMeanStress(*_epsilon);
	}

	inline T calcMinDetF() const
	{
		return calcMinDetF(*_epsilon);
	}

	inline T calcMinEigH() const
	{
		return calcMinEigH(*_epsilon);
	}

	//! compute C0:epsilon for isotropic material
	// mu_0, lambda_0: referenece material (C_0)
	// the result is stored to sigma
	noinline void calcStressConst(T mu_0, T lambda_0, const RealTensor& epsilon, RealTensor& sigma)
	{
		Timer __t("calcStressConst", false);

		T two_mu = 2*mu_0;
		T lambda = lambda_0;

		const RealTensor* peps = &epsilon;
		const RealTensor* ptau = &sigma;

#if 0
		// NOTE: for a instropic material there is no difference between stress calculation on the dfg or the original grid

		if (this->use_dfg()) {
			// transfer eps to doubly fine grid
			peps = _temp_dfg_1.get();
			ptau = _temp_dfg_1.get();
			prolongate_to_dfg(epsilon, *peps);
			_mat->select_dfg(true);
		}
#endif

		const RealTensor& eps = *peps;
		const RealTensor& tau = *ptau;

		#pragma omp parallel for schedule (static)
		for (std::size_t i = 0; i < eps.n; i++)
		{
			const T lambda_tr_e = lambda*(eps[0][i] + eps[1][i] + eps[2][i]);

			// set tau = 2*mu*epsilon + lambda*tr(epsilon)*I
			tau[0][i] = eps[0][i]*two_mu + lambda_tr_e;
			tau[1][i] = eps[1][i]*two_mu + lambda_tr_e;
			tau[2][i] = eps[2][i]*two_mu + lambda_tr_e;

			for (std::size_t k = 3; k < eps.dim; k++) {
				tau[k][i] = eps[k][i]*two_mu;
			}
		}

#if 0
		if (this->use_dfg()) {
			// get sigma from doubly fine grid
			restrict_from_dfg(*ptau, sigma);
			_mat->select_dfg(false);
		}
#endif
	}

	//! compute (C-C0):epsilon for isotropic material
	// mu_0, lambda_0: referenece material (C_0)
	// the result is stored to tau
	inline void calcStress(const RealTensor& epsilon, RealTensor& tau, T alpha = 1)
	{
		calcStress(0, 0, epsilon, tau, alpha);
	}

	inline void calcStressDiff(const RealTensor& epsilon, RealTensor& tau, T alpha = 1)
	{
		calcStress(_mu_0, _lambda_0, epsilon, tau, alpha);
	}

	inline T C0dot(T mu_0, T lambda_0, const Tensor3x3<T>& a, const Tensor3x3<T>& b)
	{
		return 2*mu_0*a.dot(b.E) + lambda_0*a.trace()*b.trace();
	}

	// Eyre, D. J., & Milton, G. W. (1999). A fast numerical scheme for computing the response of composites using grid refinement. The European Physical Journal Applied Physics, 6(1), 41–47. doi:10.1051/epjap:1999150
	//! compute sigma = (C0 + C)(C0 - C)^{-1} epsilon, C0 = 2*mu_0
	//! if inv is true then: sigma = -(C0 - C)^{-1} epsilon
	template<int N>
	noinline void calcPolarizationDim(T mu_0, T lambda_0, const RealTensor& epsilon, RealTensor& sigma, bool inv = false)
	{
		Timer __t("calc polarization", false);

		const RealTensor* eps = &epsilon;
		RealTensor* ptau = &sigma;

		if (this->use_dfg()) {
			// transfer eps to doubly fine grid
			eps = _temp_dfg_1.get();
			ptau = _temp_dfg_1.get();
			prolongate_to_dfg(epsilon, *eps);
			_mat->select_dfg(true);
		}

		ublas::c_matrix<T, N, N> C0 = 2.0*mu_0*ublas::identity_matrix<T>(N);

		#pragma omp parallel for schedule (dynamic) collapse(2)
		BEGIN_TRIPLE_LOOP(i, eps->nx, eps->ny, eps->nz, eps->nzp)
		{
			ublas::c_vector<T,N> F;
			ublas::vector<T> Q(N);
			eps->assign(i, F.data());

#if 1
			_mat->calcPolarization(i, mu_0, F, Q, eps->dim, inv);

#else
			ublas::c_matrix<T,N,N> C, C1, C2, C2inv, z;

			// note: for linear problems independent of F
			_mat->dPK1(i, F.data(), 1, false, TensorIdentity<T,N>::Id, C.data(), N);

			// L0 = 2*mu_0
			C1 = C - C0;
			C2 = C + C0;

#if 1
			// Solve C2*Q = F
			//ublas::c_matrix<T,6,1> B;
			//std::copy(F.begin(), F.end(), B.begin1());
			Q = F;
			lapack::gesv(C2, Q);
			//std::copy(B.begin1(), B.end1(), Q.begin());
			//lapack::sysv('U', C2, Q, F, lapack::optimal_workspace());

			if (inv) {
			}
			else {
				Q = ublas::prod(C1, Q);
			}
#else
			InvertMatrix<T,6>(C2, C2inv);

			if (inv) {
				Q = ublas::prod(C2inv, F);
			}
			else {
				z = ublas::prod(C1, C2inv);
				Q = ublas::prod(z, F);
			}
#endif
#endif
			for (std::size_t k = 0; k < ptau->dim; k++) {
				(*ptau)[k][i] = Q[k];
			}
		}
		END_TRIPLE_LOOP(i)

		if (this->use_dfg()) {
			// get sigma from doubly fine grid
			restrict_from_dfg(*ptau, sigma);
			_mat->select_dfg(false);
		}
	}

	inline void calcPolarization(T mu_0, T lambda_0, const RealTensor& epsilon, RealTensor& sigma, bool inv = false)
	{
		if (epsilon.dim == 3) {
			calcPolarizationDim<3>(mu_0, lambda_0, epsilon, sigma, inv);
		}
		else if (epsilon.dim == 6) {
			calcPolarizationDim<6>(mu_0, lambda_0, epsilon, sigma, inv);
		}
		else if (epsilon.dim == 9) {
			calcPolarizationDim<9>(mu_0, lambda_0, epsilon, sigma, inv);
		}
	}

	
	noinline void calcStress(T mu_0, T lambda_0, const RealTensor& epsilon, RealTensor& sigma, T alpha = 1)
	{
		Timer __t("calc stress", false);

		const T beta = -alpha*2*mu_0;
		const T gamma = -alpha*lambda_0;
		const RealTensor* eps = &epsilon;
		RealTensor* ptau = &sigma;

		if (this->use_dfg()) {
			// transfer eps to doubly fine grid
			eps = _temp_dfg_1.get();
			ptau = _temp_dfg_1.get();
			prolongate_to_dfg(epsilon, *eps);
			_mat->select_dfg(true);
		}

#ifndef TEST_MERGE_ISSUE

		#pragma omp parallel for schedule (dynamic) collapse(2)
		BEGIN_TRIPLE_LOOP(i, eps->nx, eps->ny, eps->nz, eps->nzp)
		{
			Tensor3x3<T> F, _P;

			eps->assign(i, F);
			
			_mat->PK1(i, F, alpha, false, _P);

			if (beta != 0) {
				for (std::size_t k = 0; k < ptau->dim; k++) {
					_P[k] += beta*F[k];
				}
			}
			if (gamma != 0) {
				T trF = F[0] + F[1] + F[2];
				for (std::size_t k = 0; k < 3; k++) {
					_P[k] += gamma*trF;
				}
			}

			for (std::size_t k = 0; k < ptau->dim; k++) {
				(*ptau)[k][i] = _P[k];
			}

			// TODO: compute quantities for error estimator
			// Matti: NHaR.pdf 4.3.13
			//_eps_C0_norm_square += C0dot(F, F);
			//_mean_sigma_eps += F.dot(_P.E);
			//_mean_sigma_E += _P.dot(_E);
		}
		END_TRIPLE_LOOP(i)

#else
		sigma.setConstant(_E);
		T* phi0 = _mat->phases[0]->phi;
		T* phi1 = _mat->phases[1]->phi;

		T* sigma0 = sigma[0];
		T* sigma1 = sigma[1];

		_mat->phases[0]->_phi->setConstant(0.5);
		_mat->phases[1]->_phi->setConstant(0.5);

		std::size_t n = _nx*_ny*_nzp;

		#pragma omp parallel for schedule(static)
		for (std::size_t i = 0; i < n; i++)
	//	for (std::size_t _i = 0; _i < _nx; _i++)
		{ 
	//		for (std::size_t _j = 0; _j < _ny; _j++)
		//	{ 
	//			std::size_t i = _i*_ny*_nzp + _j*_nzp; 
				
		//		for (std::size_t _k = 0; _k < _nz; _k++)
	//			{
					T Y[6];

					T phi = 2*(phi0[i] + phi1[i]);
					Y[0] = phi;
					Y[1] = phi;

					#pragma omp critical
					{
						int w = std::abs(rand() % 10);
					}

	//				if (std::abs(Y[1] - 2) > 1e-6) {
	//					LOG_COUT << "problem 1 " << i << " " << Y[1] << std::endl;
	//				}

		#pragma omp flush
					sigma0[i] = Y[0];
		#pragma omp flush
					sigma1[i] = Y[1];
		#pragma omp flush

	//				if (std::abs(sigma1[i] - 2) > 1e-6) {
	//					LOG_COUT << "problem 2 " << i << " " << sigma1[i] << std::endl;
	//				}

					//chk[0][i] = (T)(((int)chk[0][i]) | (1 << omp_get_thread_num()));
					//chk[1][i] = omp_get_thread_num();

		//			i++;
	//			}
	//		}
		}


		bool has_fault = false;

		// find the faulty value
		for (std::size_t _i = 0; _i < (_nx); _i++)
		{ 
			for (std::size_t _j = 0; _j < (_ny); _j++)
			{ 
				std::size_t i = _i*(_ny)*(_nzp) + _j*(_nzp); 

				for (std::size_t _k = 0; _k < (_nz); _k++)
				{
					T* addr0 = sigma0 + i;
					T* addr1 = sigma1 + i;

					if (std::abs(*addr0 - 2) > 1e-6 || std::abs(*addr1 - 2) > 1e-6) {
						LOG_COUT << "fault address " << addr0 << " " << _i << " " << _j << " " << _k << " value=" << (*addr0) << std::endl;
						LOG_COUT << "fault address " << addr1 << " " << _i << " " << _j << " " << _k << " value=" << (*addr1) << std::endl;
						has_fault = true; break;
					}

					i++;
				}
				if (has_fault) break;
			}
			if (has_fault) break;
		}

		if (has_fault)
		{
			check_var("phi0 calcStressDiff", *_mat->phases[0]->_phi);
			check_var("phi1 calcStressDiff", *_mat->phases[1]->_phi);

			{
				VTKCubeWriter<T> cw("chk.vtk", VTKCubeWriter<T>::WriteModes::BINARY,
						_nx, _ny, _nz, _dx, _dy, _dz, _x0[0], _x0[1], _x0[2]);

				cw.writeMesh();

				for (std::size_t i = 0; i < 2; i++) {
					cw.template beginWriteField<T>((boost::format("sigma_%d") % i).str());
					for (std::size_t k = 0; k < _nz; k++) {
						cw.template writeXYSlice<T>(sigma[i] + k, _ny*_nzp, _nzp);
					}
				}

				/*
				   for (std::size_t i = 0; i < 6; i++) {
				   cw.template beginWriteField<T>((boost::format("epsilon_%d") % i).str());
				   for (std::size_t k = 0; k < _nz; k++) {
				   cw.template writeXYSlice<T>(epsilon[i] + k, _ny*_nzp, _nzp);
				   }
				   }
				 */


				cw.template beginWriteField<T>("phi0");

				for (std::size_t k = 0; k < _nz; k++) {
					cw.template writeXYSlice<T>(_mat->phases[0]->phi + k, _ny*_nzp, _nzp);
				}

				cw.template beginWriteField<T>("phi1");

				for (std::size_t k = 0; k < _nz; k++) {
					cw.template writeXYSlice<T>(_mat->phases[1]->phi + k, _ny*_nzp, _nzp);
				}

				_mat->phases[0]->_phi->add(*(_mat->phases[1]->_phi));

				cw.template beginWriteField<T>("phisum");

				for (std::size_t k = 0; k < _nz; k++) {
					cw.template writeXYSlice<T>(_mat->phases[0]->phi + k, _ny*_nzp, _nzp);
				}

/*
				cw.template beginWriteField<T>("x");

				for (std::size_t k = 0; k < _nz; k++) {
					cw.template writeXYSlice<T>(chk[0] + k, _ny*_nzp, _nzp);
				}

				cw.template beginWriteField<T>("y");

				for (std::size_t k = 0; k < _nz; k++) {
					cw.template writeXYSlice<T>(chk[1] + k, _ny*_nzp, _nzp);
				}
*/
			}
			exit(1);
		}


	//	check_var("phi0 calcStressDiff", *_mat->phases[0]->_phi);
		//check_var("phi1 calcStressDiff", *_mat->phases[1]->_phi);
	//	check_var("after calcStressDiff", *ptau);

#endif


		if (this->use_dfg()) {
			// get sigma from doubly fine grid
			restrict_from_dfg(*ptau, sigma);
			_mat->select_dfg(false);
		}
	}

	// 
	noinline void calcDetF(RealTensor& F, T* detF)
	{
		Timer __t("calc detF", false);

		#pragma omp parallel for schedule (dynamic) collapse(2)
		BEGIN_TRIPLE_LOOP(i, F.nx, F.ny, F.nz, F.nzp)
		{
			// load tensor F and W
			Tensor3x3<T> Fi;
			F.assign(i, Fi);
			
			detF[i] = Fi.det();
		}
		END_TRIPLE_LOOP(i)
	}

	// 
	noinline void calcDetC(RealTensor& F, T* detC)
	{
		Timer __t("calc detC", false);

		#pragma omp parallel for schedule (dynamic) collapse(2)
		BEGIN_TRIPLE_LOOP(i, F.nx, F.ny, F.nz, F.nzp)
		{
			// load tensor F and W
			Tensor3x3<T> Fi;
			F.assign(i, Fi);
	
			ublas::c_matrix<T, 9, 9> dP;
			_mat->dPK1(i, Fi, 1, false, TensorIdentity<T,9>::Id, dP.data(), 9);

			ublas::c_vector<std::complex<T>, 9> e;
			ublas::matrix<T, ublas::column_major> vl(9,9);
			ublas::matrix<T, ublas::column_major> vr(9,9);

			lapack::geev(dP, e, &vl, &vr, lapack::optimal_workspace());

#if 1
			detC[i] = STD_INFINITY(T);

			for (std::size_t j = 0; j < 9; j++) {
				T imag = std::abs(std::imag(e[j]));
				if (imag > 1e-12) {
					detC[i] = std::min(detC[i], -imag);
				}
				else if (std::abs(std::real(e[j])) > 1e-12) {
					detC[i] = std::min(detC[i], std::real(e[j]));
				}
			}
/*
			if (detC[i] < 0) {
				#pragma omp critical
				{
					_mat->dPK1(i, Fi, 1, false, TensorIdentity<T,9>::Id, dP.data(), 9);
					LOG_COUT << format(dP) << std::endl;
					LOG_COUT << std::endl;
					for (std::size_t j = 0; j < 9; j++) {
						LOG_COUT << e[j] << std::endl;
					}
				}
			}
*/
#else
			detC[i] = 1.0;

			for (std::size_t j = 0; j < 9; j++) {
				detC[i] *= e[j];
			}
#endif
		}
		END_TRIPLE_LOOP(i)
	}

	//! compute sigma = alpha*(dP/dF(F) - C0) : W
	noinline void calcStressDeriv(T mu_0, T lambda_0, RealTensor& F, RealTensor& W, RealTensor& sigma, T alpha = 1)
	{
		Timer __t("calc stress deriv", false);

		const T beta = -alpha*2*mu_0;
		const T gamma = -alpha*lambda_0;
		const RealTensor* pF = &F;
		const RealTensor* pW = &W;
		const RealTensor* ptau = &sigma;

		if (this->use_dfg()) {
			// transfer eps to doubly fine grid
			pF = _temp_dfg_1.get();
			pW = _temp_dfg_2.get();
			ptau = pF;
			prolongate_to_dfg(F, *pF);
			prolongate_to_dfg(W, *pW);
			_mat->select_dfg(true);
		}

		#pragma omp parallel for schedule (dynamic) collapse(2)
		BEGIN_TRIPLE_LOOP(i, pF->nx, pF->ny, pF->nz, pF->nzp)
		{
			// load tensor F and W
			Tensor3x3<T> F, W, dP;
			pF->assign(i, F);
			pW->assign(i, W);
			
			_mat->dPK1(i, F, alpha, false,  W, dP);

			if (beta != 0) {
				for (std::size_t k = 0; k < ptau->dim; k++) {
					dP[k] += beta*W[k];
				}
			}
			if (gamma != 0) {
				T trW = W[0] + W[1] + W[2];
				for (std::size_t k = 0; k < 3; k++) {
					dP[k] += gamma*trW;
				}
			}

			for (std::size_t k = 0; k < ptau->dim; k++) {
				(*ptau)[k][i] = dP[k];
			}
		}
		END_TRIPLE_LOOP(i)
	
		if (this->use_dfg()) {
			// get sigma from doubly fine grid
			restrict_from_dfg(*ptau, sigma);
			_mat->select_dfg(false);
		}
	}

	//! perform inplace forward FFT of vector (first 3 compontents of x)
	noinline void fftVector(const RealTensor& x, ComplexTensor& y, std::size_t dims=3)
	{
		Timer __t("fftVector", false);
		Timer dt_fft;

		const T scale = 1/(T)_nxyz;

#ifdef USE_MANY_FFT
		get_fft(dims)->forward(x.t, y.t);

		#pragma omp parallel for schedule (static)
		for (std::size_t i = 0; i < y.ne; i++) {
			y.t[i] *= scale;
		}
#else
		#pragma omp parallel for schedule (static) if(_parallel_fft)
		for (std::size_t i = 0; i < dims; i++) {
			get_fft(1)->forward(x[i], y[i]);
		}

		#pragma omp parallel for schedule (static) collapse(2)
		for (std::size_t i = 0; i < dims; i++) {
			for (std::size_t j = 0; j < y.n; j++) {
				y[i][j] *= scale;
			}
		}
#endif

		_fft_time += dt_fft;
	}

	//! perform inplace backward FFT of vector (first 3 compontents of x)
	noinline void fftInvVector(const ComplexTensor& x, RealTensor& y, std::size_t dims=3)
	{
		Timer __t("fftInvVector", false);
		Timer dt_fft;

#ifdef USE_MANY_FFT
		get_fft(dims)->backward(x.t, y.t);
#else
		#pragma omp parallel for schedule (static) if(_parallel_fft)
		for (std::size_t i = 0; i < dims; i++) {
			get_fft(1)->backward(x[i], y[i]);
		}
#endif

		_fft_time += dt_fft;
	}

	//! perform inplace forward FFT of tensor
	noinline void fftTensor(const RealTensor& x, ComplexTensor& y, bool zero_trace = false)
	{
		Timer __t("fftTensor", false);
		Timer dt_fft;

		const T scale = 1/(T)_nxyz;
		std::size_t i0 = (zero_trace ? 1 : 0);

#ifdef USE_MANY_FFT
		get_fft(x.dim - i0)->forward(x[i0], y[i0]);
#else
		#pragma omp parallel for schedule (static) if(_parallel_fft)
		for (std::size_t i = i0; i < x.dim; i++) {
			get_fft(1)->forward(x[i], y[i]);
		}
#endif

		#pragma omp parallel for schedule (static) collapse(2)
		for (std::size_t i = i0; i < x.dim; i++) {
			for (std::size_t j = 0; j < y.n; j++) {
				y[i][j] *= scale;
			}
		}

		_fft_time += dt_fft;

		if (zero_trace) {
			mxpyTensor(reinterpret_cast<T*>(y[0]), reinterpret_cast<T*>(y[1]), reinterpret_cast<T*>(y[2]));
		}
	}

	//! perform inplace backward FFT of tensor
	noinline void fftInvTensor(const ComplexTensor& x, RealTensor& y, bool zero_trace = false)
	{
		Timer __t("fftInvTensor", false);
		Timer dt_fft;

		std::size_t i0 = (zero_trace ? 1 : 0);

#ifdef USE_MANY_FFT
		get_fft(x.dim - i0)->backward(x[i0], y[i0]);
#else
		#pragma omp parallel for schedule (static) if(_parallel_fft)
		for (std::size_t i = i0; i < x.dim; i++) {
			get_fft(1)->backward(x[i], y[i]);
		}
#endif

		_fft_time += dt_fft;

		if (zero_trace) {
			mxpyTensor(y[0], y[1], y[2]);
		}
	}

	// set solid parts of the results to zero (for viscosity mode)
/*
	void maskSolidRegions()
	{
		#pragma omp parallel for schedule (static) collapse(2)
		for (std::size_t i = 0; i < 6; i++) {
			for (std::size_t j = 0; j < _n; j++) {
				_epsilon[i][j] *= 1-_phi[j];
			}
		}
	}

	void fixTrace(RealTensor& x)
	{
		#pragma omp parallel for schedule (static)
		for (std::size_t i = 0; i < _n; i++) {
			T tr_by_3 = (x[0][i] + x[1][i] + x[2][i])/3;
			// TODO: check if really going to 6 is correct
			for (std::size_t j = 0; j < 6; j++) {
				x[j][i] -= tr_by_3;
			}
		}
	}
*/

	//! compute the symmetric gradient of the vector x (first 3 components) add E
	//! and store result in tensor y
	// x and y may be the same for inplace operation
	noinline void epsOperatorStaggered(const ublas::vector<T>& E, const RealTensor& x, const RealTensor& y)
	{
		Timer __t("epsOperatorStaggered", false);

		const T hx = _nx/_dx;
		const T hy = _ny/_dy;
		const T hz = _nz/_dz;

		#pragma omp parallel
		{
			// compute D_y^-(x[z]) + D_z^-(x[y]) => y[yz]
			// compute D_x^-(x[z]) + D_z^-(x[x]) => y[xz]
			// compute D_x^-(x[y]) + D_y^-(x[x]) => y[xy]
			#pragma omp for nowait schedule (static) collapse(2)
			for (std::size_t ii = 0; ii < _nx; ii++) {
				for (std::size_t jj = 0; jj < _ny; jj++) {
					int k = ii*_nyzp + jj*_nzp;
					for (std::size_t kk = 0; kk < _nz; kk++) {
						y[3][k] = E[3] + 0.5*((x[2][k] - x[2][k + _bfd_y[jj]])*hy + (x[1][k] - x[1][k + _bfd_z[kk]])*hz);
						y[4][k] = E[4] + 0.5*((x[2][k] - x[2][k + _bfd_x[ii]])*hx + (x[0][k] - x[0][k + _bfd_z[kk]])*hz);
						y[5][k] = E[5] + 0.5*((x[1][k] - x[1][k + _bfd_x[ii]])*hx + (x[0][k] - x[0][k + _bfd_y[jj]])*hy);
						k ++;
					}
				}
			}

			// we need to make sure x[0...2] is not overwritten by the following loops
			// before the loop above finishes (could have removed the nowait attribute also)
			#pragma omp barrier

			// compute D_x^+(x[x]) => y[xx]
			#pragma omp for nowait schedule (static) collapse(2)
			for (std::size_t kk = 0; kk < _nz; kk++) {
				for (std::size_t jj = 0; jj < _ny; jj++) {
					std::size_t k = jj*_nzp + kk;
					T x0 = x[0][k];
					k += _nx*_nyzp;
					for (std::size_t ii = 0; ii < _nx; ii++) {
						k -= _nyzp;
						T x1 = x[0][k];
						y[0][k] = E[0] + (x0 - x1)*hx;
						x0 = x1;
					}
				}
			}

			// compute D_y^+(x[yy]) => y[yy]
			#pragma omp for nowait schedule (static) collapse(2)
			for (std::size_t kk = 0; kk < _nz; kk++) {
				for (std::size_t ii = 0; ii < _nx; ii++) {
					std::size_t k = ii*_nyzp + kk;
					T x0 = x[1][k];
					k += _ny*_nzp;
					for (std::size_t jj = 0; jj < _ny; jj++) {
						k -= _nzp;
						T x1 = x[1][k];
						y[1][k] = E[1] + (x0 - x1)*hy;
						x0 = x1;
					}
				}
			}

			// compute D_z^+(x[zz]) => y[zz]
			#pragma omp for nowait schedule (static) collapse(2)
			for (std::size_t jj = 0; jj < _ny; jj++) {
				for (std::size_t ii = 0; ii < _nx; ii++) {
					std::size_t k = ii*_nyzp + jj*_nzp;
					T x0 = x[2][k];
					k += _nz;
					for (std::size_t kk = 0; kk < _nz; kk++) {
						k --;
						T x1 = x[2][k];
						y[2][k] = E[2] + (x0 - x1)*hz;
						x0 = x1;
					}
				}
			}
		}
	}

	//! compute the gradient of the vector x (first 3 components) add E
	//! and store result in tensor y
	// x and y may be the same for inplace operation
	noinline void epsOperatorStaggeredHeat(const ublas::vector<T>& E, const RealTensor& x, const RealTensor& y)
	{
		Timer __t("epsOperatorStaggeredHeat", false);

		const T hx = _nx/_dx;
		const T hy = _ny/_dy;
		const T hz = _nz/_dz;

		#pragma omp parallel
		{
			// compute D_z^+(x[zz]) => y[zz]
			#pragma omp for nowait schedule (static) collapse(2)
			for (std::size_t jj = 0; jj < _ny; jj++) {
				for (std::size_t ii = 0; ii < _nx; ii++) {
					std::size_t k = ii*_nyzp + jj*_nzp;
					T x0 = x[0][k];
					k += _nz;
					for (std::size_t kk = 0; kk < _nz; kk++) {
						k --;
						T x1 = x[0][k];
						y[2][k] = E[2] + (x0 - x1)*hz;
						x0 = x1;
					}
				}
			}

			// compute D_y^+(x[yy]) => y[yy]
			#pragma omp for nowait schedule (static) collapse(2)
			for (std::size_t kk = 0; kk < _nz; kk++) {
				for (std::size_t ii = 0; ii < _nx; ii++) {
					std::size_t k = ii*_nyzp + kk;
					T x0 = x[0][k];
					k += _ny*_nzp;
					for (std::size_t jj = 0; jj < _ny; jj++) {
						k -= _nzp;
						T x1 = x[0][k];
						y[1][k] = E[1] + (x0 - x1)*hy;
						x0 = x1;
					}
				}
			}

			// make sure x[0] is not used anymore as input for y[1] and y[2]
			#pragma omp barrier

			// compute D_x^+(x[x]) => y[xx]
			#pragma omp for nowait schedule (static) collapse(2)
			for (std::size_t kk = 0; kk < _nz; kk++) {
				for (std::size_t jj = 0; jj < _ny; jj++) {
					std::size_t k = jj*_nzp + kk;
					T x0 = x[0][k];
					k += _nx*_nyzp;
					for (std::size_t ii = 0; ii < _nx; ii++) {
						k -= _nyzp;
						T x1 = x[0][k];
						y[0][k] = E[0] + (x0 - x1)*hx;
						x0 = x1;
					}
				}
			}
		}
	}

	//! compute the gradient of the vector x (first 3 components) add E
	//! and store result in tensor y
	// x and y may be the same for inplace operation
	noinline void epsOperatorStaggeredHyper(const ublas::vector<T>& E, const RealTensor& x, const RealTensor& y)
	{
		Timer __t("epsOperatorStaggeredHyper", false);

		const T hx = _nx/_dx;
		const T hy = _ny/_dy;
		const T hz = _nz/_dz;

		#pragma omp parallel
		{
			// compute D_y^-(x[z]) + D_z^-(x[y]) => y[yz]
			// compute D_x^-(x[z]) + D_z^-(x[x]) => y[xz]
			// compute D_x^-(x[y]) + D_y^-(x[x]) => y[xy]
			// compute D_y^-(x[z]) + D_z^-(x[y]) => y[yz]
			// compute D_x^-(x[z]) + D_z^-(x[x]) => y[xz]
			// compute D_x^-(x[y]) + D_y^-(x[x]) => y[xy]
			#pragma omp for nowait schedule (static) collapse(2)
			for (std::size_t ii = 0; ii < _nx; ii++) {
				for (std::size_t jj = 0; jj < _ny; jj++) {
					int k = ii*_nyzp + jj*_nzp;
					for (std::size_t kk = 0; kk < _nz; kk++) {
						y[3][k] = E[3] + (x[1][k] - x[1][k + _bfd_z[kk]])*hz;
						y[4][k] = E[4] + (x[0][k] - x[0][k + _bfd_z[kk]])*hz;
						y[5][k] = E[5] + (x[0][k] - x[0][k + _bfd_y[jj]])*hy;
						y[6][k] = E[6] + (x[2][k] - x[2][k + _bfd_y[jj]])*hy;
						y[7][k] = E[7] + (x[2][k] - x[2][k + _bfd_x[ii]])*hx;
						y[8][k] = E[8] + (x[1][k] - x[1][k + _bfd_x[ii]])*hx;
						k ++;
					}
				}
			}

			// we need to make sure x[0...2] is not overwritten by the following loops
			// before the loop above finishes (could have removed the nowait attribute also)
			#pragma omp barrier

			// compute D_x^+(x[x]) => y[xx]
			#pragma omp for nowait schedule (static) collapse(2)
			for (std::size_t kk = 0; kk < _nz; kk++) {
				for (std::size_t jj = 0; jj < _ny; jj++) {
					std::size_t k = jj*_nzp + kk;
					T x0 = x[0][k];
					k += _nx*_nyzp;
					for (std::size_t ii = 0; ii < _nx; ii++) {
						k -= _nyzp;
						T x1 = x[0][k];
						y[0][k] = E[0] + (x0 - x1)*hx;
						x0 = x1;
					}
				}
			}

			// compute D_y^+(x[yy]) => y[yy]
			#pragma omp for nowait schedule (static) collapse(2)
			for (std::size_t kk = 0; kk < _nz; kk++) {
				for (std::size_t ii = 0; ii < _nx; ii++) {
					std::size_t k = ii*_nyzp + kk;
					T x0 = x[1][k];
					k += _ny*_nzp;
					for (std::size_t jj = 0; jj < _ny; jj++) {
						k -= _nzp;
						T x1 = x[1][k];
						y[1][k] = E[1] + (x0 - x1)*hy;
						x0 = x1;
					}
				}
			}

			// compute D_z^+(x[zz]) => y[zz]
			#pragma omp for nowait schedule (static) collapse(2)
			for (std::size_t jj = 0; jj < _ny; jj++) {
				for (std::size_t ii = 0; ii < _nx; ii++) {
					std::size_t k = ii*_nyzp + jj*_nzp;
					T x0 = x[2][k];
					k += _nz;
					for (std::size_t kk = 0; kk < _nz; kk++) {
						k --;
						T x1 = x[2][k];
						y[2][k] = E[2] + (x0 - x1)*hz;
						x0 = x1;
					}
				}
			}
		}
	}

	//! compute the divergence of x using backward differences for the diagonal elements
	//! and forward differences for the off-diagonal elements
	//! the result is stored in the first 3 components of y, however the other 3 components are used internally
	// x and y may be the same for inplace operation
	noinline void divOperatorStaggered(const RealTensor& x, const RealTensor& y)
	{
		Timer __t("divOperatorStaggered", false);

		const T hx = _nx/_dx;
		const T hy = _ny/_dy;
		const T hz = _nz/_dz;
		
		#pragma omp parallel
		{
			// compute D_x^-(x[xx]) + D_y^+(x[xy]) + D_z^+(x[xz]) => y[0]
			#pragma omp for nowait schedule (static) collapse(2)
			for (std::size_t kk = 0; kk < _nz; kk++) {
				for (std::size_t jj = 0; jj < _ny; jj++) {
					int k = jj*_nzp + kk;
					T x0 = x[0][k + (_nx-1)*_nyzp];
					for (std::size_t ii = 0; ii < _nx; ii++) {
						T x1 = x[0][k];
						y[0][k] = (x1 - x0)*hx + (x[5][k + _ffd_y[jj]] - x[5][k])*hy + (x[4][k + _ffd_z[kk]] - x[4][k])*hz;
						x0 = x1;
						k += _nyzp;
					}
				}
			}

			// compute D_x^+(x[xy]) + D_y^-(x[yy]) + D_z^+(x[yz]) => y[1]
			#pragma omp for nowait schedule (static) collapse(2)
			for (std::size_t kk = 0; kk < _nz; kk++) {
				for (std::size_t ii = 0; ii < _nx; ii++) {
					int k = ii*_nyzp + kk;
					T x0 = x[1][k + (_ny-1)*_nzp];
					for (std::size_t jj = 0; jj < _ny; jj++) {
						T x1 = x[1][k];
						y[1][k] = (x[5][k + _ffd_x[ii]] - x[5][k])*hx + (x1 - x0)*hy + (x[3][k + _ffd_z[kk]] - x[3][k])*hz;
						x0 = x1;
						k += _nzp;
					}
				}
			}

			// compute D_x^+(x[xz]) + D_y^+(x[yz]) + D_z^-(x[zz]) => y[2]
			#pragma omp for nowait schedule (static) collapse(2)
			for (std::size_t jj = 0; jj < _ny; jj++) {
				for (std::size_t ii = 0; ii < _nx; ii++) {
					int k = ii*_nyzp + jj*_nzp;
					T x0 = x[2][k + (_nz-1)];
					for (std::size_t kk = 0; kk < _nz; kk++) {
						T x1 = x[2][k];
						y[2][k] = (x[4][k + _ffd_x[ii]] - x[4][k])*hx + (x[3][k + _ffd_y[jj]] - x[3][k])*hy + (x1 - x0)*hz;
						x0 = x1;
						k ++;
					}
				}
			}
		}
	}

	//! compute the divergence of x using backward differences for the diagonal elements
	//! and forward differences for the off-diagonal elements
	//! the result is stored in the first 3 components of y, however the other 3 components are used internally
	// x and y may be the same for inplace operation
	noinline void divOperatorStaggeredHeat(const RealTensor& x, const RealTensor& y)
	{
		Timer __t("divOperatorStaggeredHeat", false);

		const T hx = _nx/_dx;
		const T hy = _ny/_dy;
		const T hz = _nz/_dz;
		
		#pragma omp parallel
		{
			// compute D_x^-(x[xx]) + D_y^+(x[xy]) + D_z^+(x[xz]) => y[0]
			#pragma omp for schedule (static) collapse(2)
			for (std::size_t kk = 0; kk < _nz; kk++) {
				for (std::size_t jj = 0; jj < _ny; jj++) {
					int k = jj*_nzp + kk;
					T x0 = x[0][k + (_nx-1)*_nyzp];
					for (std::size_t ii = 0; ii < _nx; ii++) {
						T x1 = x[0][k];
						y[0][k] = (x1 - x0)*hx;
						x0 = x1;
						k += _nyzp;
					}
				}
			}

			// compute D_x^+(x[xy]) + D_y^-(x[yy]) + D_z^+(x[yz]) => y[1]
			#pragma omp for schedule (static) collapse(2)
			for (std::size_t kk = 0; kk < _nz; kk++) {
				for (std::size_t ii = 0; ii < _nx; ii++) {
					int k = ii*_nyzp + kk;
					T x0 = x[1][k + (_ny-1)*_nzp];
					for (std::size_t jj = 0; jj < _ny; jj++) {
						T x1 = x[1][k];
						y[0][k] += (x1 - x0)*hy;
						x0 = x1;
						k += _nzp;
					}
				}
			}

			// compute D_x^+(x[xz]) + D_y^+(x[yz]) + D_z^-(x[zz]) => y[2]
			#pragma omp for schedule (static) collapse(2)
			for (std::size_t jj = 0; jj < _ny; jj++) {
				for (std::size_t ii = 0; ii < _nx; ii++) {
					int k = ii*_nyzp + jj*_nzp;
					T x0 = x[2][k + (_nz-1)];
					for (std::size_t kk = 0; kk < _nz; kk++) {
						T x1 = x[2][k];
						y[0][k] += (x1 - x0)*hz;
						x0 = x1;
						k ++;
					}
				}
			}
		}

/*

		const T hx = _nx/_dx;
		const T hy = _ny/_dy;
		const T hz = _nz/_dz;
		
#if 1
		// compute D_x^+(x[xx]) + D_y^+(x[xy]) + D_z^+(x[xz]) => y[0]
		#pragma omp parallel for schedule (static) collapse(2)
		for (std::size_t kk = 0; kk < _nz; kk++) {
			for (std::size_t jj = 0; jj < _ny; jj++) {
				int k = jj*_nzp + kk;
				T x1 = x[0][k];
				k += _nyzp*_nx;
				for (std::size_t ii = 0; ii < _nx; ii++) {
					k -= _nyzp;
					T x0 = x[0][k];
					y[0][k] = (x1 - x0)*hx + (x[1][k + _ffd_y[jj]] - x[1][k])*hy + (x[2][k + _ffd_z[kk]] - x[2][k])*hz;
					x1 = x0;
				}
			}
		}
#else
		// compute D_x^-(x[xx]) + D_y^+(x[xy]) + D_z^+(x[xz]) => y[0]
		#pragma omp parallel for schedule (static) collapse(2)
		for (std::size_t kk = 0; kk < _nz; kk++) {
			for (std::size_t jj = 0; jj < _ny; jj++) {
				int k = jj*_nzp + kk;
				T x0 = x[0][k + (_nx-1)*_nyzp];
				for (std::size_t ii = 0; ii < _nx; ii++) {
					T x1 = x[0][k];
					y[0][k] = (x1 - x0)*hx + (x[1][k + _ffd_y[jj]] - x[1][k])*hy + (x[2][k + _ffd_z[kk]] - x[2][k])*hz;
					x0 = x1;
					k += _nyzp;
				}
			}
		}
#endif
*/
	}


	//! compute the divergence of x using backward differences for the diagonal elements
	//! and forward differences for the off-diagonal elements
	//! the result is stored in the first 3 components of y, however the other 3 components are used internally
	// x and y may be the same for inplace operation
	noinline void divOperatorStaggeredHyper(const RealTensor& x, const RealTensor& y)
	{
		Timer __t("divOperatorStaggeredHyper", false);

		const T hx = _nx/_dx;
		const T hy = _ny/_dy;
		const T hz = _nz/_dz;
		
		#pragma omp parallel
		{
			// compute D_x^-(x[xx]) + D_y^+(x[xy]) + D_z^+(x[xz]) => y[0]
			#pragma omp for nowait schedule (static) collapse(2)
			for (std::size_t kk = 0; kk < _nz; kk++) {
				for (std::size_t jj = 0; jj < _ny; jj++) {
					int k = jj*_nzp + kk;
					T x0 = x[0][k + (_nx-1)*_nyzp];
					for (std::size_t ii = 0; ii < _nx; ii++) {
						T x1 = x[0][k];
						y[0][k] = (x1 - x0)*hx + (x[5][k + _ffd_y[jj]] - x[5][k])*hy + (x[4][k + _ffd_z[kk]] - x[4][k])*hz;
						x0 = x1;
						k += _nyzp;
					}
				}
			}

			// compute D_x^+(x[yx]) + D_y^-(x[yy]) + D_z^+(x[yz]) => y[1]
			#pragma omp for nowait schedule (static) collapse(2)
			for (std::size_t kk = 0; kk < _nz; kk++) {
				for (std::size_t ii = 0; ii < _nx; ii++) {
					int k = ii*_nyzp + kk;
					T x0 = x[1][k + (_ny-1)*_nzp];
					for (std::size_t jj = 0; jj < _ny; jj++) {
						T x1 = x[1][k];
						y[1][k] = (x[8][k + _ffd_x[ii]] - x[8][k])*hx + (x1 - x0)*hy + (x[3][k + _ffd_z[kk]] - x[3][k])*hz;
						x0 = x1;
						k += _nzp;
					}
				}
			}

			// compute D_x^+(x[zx]) + D_y^+(x[zy]) + D_z^-(x[zz]) => y[2]
			#pragma omp for nowait schedule (static) collapse(2)
			for (std::size_t jj = 0; jj < _ny; jj++) {
				for (std::size_t ii = 0; ii < _nx; ii++) {
					int k = ii*_nyzp + jj*_nzp;
					T x0 = x[2][k + (_nz-1)];
					for (std::size_t kk = 0; kk < _nz; kk++) {
						T x1 = x[2][k];
						y[2][k] = (x[7][k + _ffd_x[ii]] - x[7][k])*hx + (x[6][k + _ffd_y[jj]] - x[6][k])*hy + (x1 - x0)*hz;
						x0 = x1;
						k ++;
					}
				}
			}
		}
	}

	//! compute eta_hat = Delta_hat : tau_hat, eta_hat(0) = E
	//! i.e. compute eta_hat = 2*alpha*mu_0*(tau_hat - 2*mu_0 * Gamma_hat : tau_hat), eta_hat(0) = E
	void applyDeltaFourier(const ublas::vector<T>& E, T mu_0, const ComplexTensor& tau_hat, ComplexTensor& eta_hat, T alpha = -1)
	{
		initBCProjector(tau_hat);
		GammaOperatorFourierCollocated(E, -1.0/(4*mu_0), STD_INFINITY(T), tau_hat, eta_hat, alpha, 2*alpha*mu_0);
		applyBCProjector(eta_hat, alpha);
	}

	//! compute eta_hat = alpha * Gamma_hat : tau_hat + beta*tau_hat, eta_hat(0) = E
	noinline void GammaOperatorFourierWillotR(const ublas::vector<T>& E, T mu_0, T lambda_0, const ComplexTensor& tau_hat, ComplexTensor& eta_hat,
		T alpha = -1, T beta = 0)
	{
		Timer __t("GammaOperatorFourierWillotR", false);

		//const T xi0_0 = 2*M_PI/_dx, xi1_0 = 2*M_PI/_dy, xi2_0 = 2*M_PI/_dz;
		const T xi0_0 = 2*M_PI/_dx, xi1_0 = 2*M_PI/_dy, xi2_0 = 2*M_PI/_dz;
		const T small = boost::numeric::bounds<T>::smallest();
		const T mu_lambda_0 = mu_0/lambda_0;

		const bool nx_even = (_nx & 1) == 0;
		const bool ny_even = (_ny & 1) == 0;
		const bool nz_even = (_nz & 1) == 0;
		const std::size_t ii_half = nx_even ? (_nx/2 - 1) : _nx/2;
		const std::size_t jj_half = ny_even ? (_ny/2 - 1) : _ny/2;
		const std::size_t kk_half = nz_even ? (_nz/2 - 1) : _nz/2;
		//const std::size_t ii_filt = (_freq_hack && nx_even) ? _nx/2 : _nx;
		//const std::size_t jj_filt = (_freq_hack && ny_even) ? _ny/2 : _ny;
		//const std::size_t kk_filt = (_freq_hack && nz_even) ? _nz/2 : _nz;

		std::complex<T> ey[6];

#if 0
#else
		// "safe" version, but slow

		std::complex<T> c;
		ublas::c_matrix<std::complex<T>,6,6> gamma;
		const ublas::c_matrix<T,3,3> delta(ublas::identity_matrix<T>(3));
		ublas::c_vector<T,3> xi;
		ublas::c_vector<T,3> qi;
		ublas::c_vector<T,3> wi;
		wi(0) = _dx/_nx;
		wi(1) = _dy/_ny;
		wi(2) = _dz/_nz;

		// indices for Voigt notation
		const int vi[6] = {0, 1, 2, 1, 0, 0};
		const int vj[6] = {0, 1, 2, 2, 2, 1};

		ublas::c_vector<std::complex<T>, 3> kvec, r, rc;

#define WILLOT_ALLOW_NONZERO_LAMBDA 1

		#pragma omp parallel for private(c, ey, gamma, xi, qi, kvec, r, rc)
		for (size_t ii = 0; ii < _nx; ii++)
		{
			xi(0) = xi0_0*((ii <= ii_half) ? ii : ((T)ii - (T)_nx));
			qi(0) = xi(0)*wi(0);
			std::complex<T> exp0 = (T)1.0 + std::polar<T>(1, qi(0));

			for (size_t jj = 0; jj < _ny; jj++)
			{
				xi(1) = xi1_0*((jj <= jj_half) ? jj : ((T)jj - (T)_ny));
				qi(1) = xi(1)*wi(1);

				// calculate current index in complex tensor tau[*]
				std::size_t k0 = ii*_ny*_nzc + jj*_nzc;
				std::complex<T> exp1 = (T)1.0 + std::polar<T>(1, qi(1));

				for (size_t kk = 0; kk < _nzc; kk++)
				{
					xi(2) = xi2_0*((kk <= kk_half) ? kk : ((T)kk - (T)_nz));
					qi(2) = xi(2)*wi(2);
					
					std::complex<T> exp2 = (T)1.0 + std::polar<T>(1, qi(2));
					std::complex<T> exp012 = exp0*exp1*exp2;

					for (size_t i = 0; i < 3; i++) {
						kvec[i] = std::complex<T>(0, 0.25*std::tan(0.5*qi(i)))*exp012/wi(i);
					}

					T mag_k = std::sqrt(std::norm(kvec(0)) + std::norm(kvec(1)) + std::norm(kvec(2))) + small;
					r = kvec/mag_k;
					rc(0) = std::conj(r(0));
					rc(1) = std::conj(r(1));
					rc(2) = std::conj(r(2));
					
#if WILLOT_ALLOW_NONZERO_LAMBDA
					T r2 = std::norm(r(0)*r(0) + r(1)*r(1) + r(2)*r(2));
#endif

					for (size_t iv = 0; iv < 6; iv++)
					{
						for (size_t jv = iv; jv < 6; jv++)
						{
							std::size_t i = vi[iv];
							std::size_t j = vj[iv];
							std::size_t k = vi[jv];
							std::size_t l = vj[jv];
							
#if !WILLOT_ALLOW_NONZERO_LAMBDA
							gamma(iv, jv) = (
								0.5*(r[i]*rc[l]*delta(j,k) + r[j]*rc[l]*delta(i,k) + r[i]*rc[k]*delta(j,l) + r[j]*rc[k]*delta(i,l)) - r[i]*r[j]*rc[k]*rc[l]
							) / (2*mu_0);
#else
							// Fall: lambda_0 != 0

							T sjk, sik, sjl, sil;

							if (k == j) {
								T rik = std::imag(r[i]*std::conj(r[k]));
								sjk = 4.0*rik*rik;
							}
							else {
								sjk = -4.0*std::imag(r[k]*std::conj(r[j]))*std::imag(r[k]*std::conj(r[i]));
							}

							if (l == j) {
								T ril = std::imag(r[i]*std::conj(r[l]));
								sjl = 4.0*ril*ril;
							}
							else {
								sjl = -4.0*std::imag(r[l]*std::conj(r[j]))*std::imag(r[l]*std::conj(r[i]));
							}


							if (k == i) {
								T rjk = std::imag(r[j]*std::conj(r[k]));
								sik = 4.0*rjk*rjk;
							}
							else {
								sik = -4.0*std::imag(r[k]*std::conj(r[i]))*std::imag(r[k]*std::conj(r[j]));
							}

							if (l == i) {
								T rjl = std::imag(r[j]*std::conj(r[l]));
								sil = 4.0*rjl*rjl;
							}
							else {
								sil = -4.0*std::imag(r[l]*std::conj(r[i]))*std::imag(r[l]*std::conj(r[j]));
							}

			/*
							ublas::c_matrix<T,3,3> s(ublas::zero_matrix<T>(3));
							for (std::size_t sk = 0; sk < 3; sk++) {
								for (std::size_t sj = 0; sj < 3; sj++) {
									if (sk == sj) {
										T rik = std::imag(r[i]*std::conj(r[sk]));
										s(sj,sk) = 4.0*rik*rik;
									}
									else {
										s(sj,sk) = -4.0*std::imag(r[sk]*std::conj(r[sj]))*std::imag(r[sk]*std::conj(r[i]));
									}
								}
							}
*/

							#if 1
							// defined for lambda_0 -> infinity
							gamma(iv, jv) = (
								(1 + 2*mu_lambda_0)*((T)0.25)*(r[i]*rc[l]*delta(j,k) + r[j]*rc[l]*delta(i,k) + r[i]*rc[k]*delta(j,l) + r[j]*rc[k]*delta(i,l))
								+ (((T)0.25)*(r[i]*rc[l]*sjk + r[j]*rc[l]*sik + r[i]*rc[k]*sjl + r[j]*rc[k]*sil) - std::real(r[i]*rc[j])*std::real(r[k]*rc[l]))
								- mu_lambda_0*r[i]*r[j]*rc[k]*rc[l]
							) /
							(
								mu_0*(2*(1 + mu_lambda_0) - r2)
							);
							#else
							// undefined for lambda_0 -> infinity
							gamma(iv, jv) = (
								(lambda_0 + 2*mu_0)*((T)0.25)*(r[i]*rc[l]*delta(j,k) + r[j]*rc[l]*delta(i,k) + r[i]*rc[k]*delta(j,l) + r[j]*rc[k]*delta(i,l))
								+ lambda_0*(((T)0.25)*(r[i]*rc[l]*sjk + r[j]*rc[l]*sik + r[i]*rc[k]*sjl + r[j]*rc[k]*sil) - std::real(r[i]*rc[j])*std::real(r[k]*rc[l]))
								- mu_0*r[i]*r[j]*rc[k]*rc[l]
							) /
							(
								mu_0*(2*(lambda_0 + mu_0) - lambda_0*r2)
							);
							#endif

#endif
							
							gamma(jv, iv) = std::conj(gamma(iv, jv));
						}

						// sum up the components
						// we need to scale the last 3 components of tau_hat by 2 (Voigt notation)
						c = 0;
						for (size_t j = 3; j < 6; j++) {
							c += gamma(iv, j)*tau_hat[j][k0];
						}
						c *= 2;
						for (size_t j = 0; j < 3; j++) {
							c += gamma(iv, j)*tau_hat[j][k0];
						}
						ey[iv] = c;
					}

					/*
					LOG_COUT << "gamma matrix: " << std::endl;

					for (size_t i = 0; i < 6; i++)
					{
						for (size_t j = 0; j < 6; j++) {
							LOG_COUT << gamma(i,j) << " ";
						}
						LOG_COUT << std::endl;
					}
					*/

					// assign result to eta_hat
					// we do this seperately since eta_hat and tau_hat may point to the same memory
					for (size_t j = 0; j < 6; j++) {
						eta_hat[j][k0] = alpha*ey[j] + beta*tau_hat[j][k0];
					}
					
					k0++;
				}
			}
		}
#endif

		// set zero component
		for (std::size_t j = 0; j < 6; j++) {
			eta_hat[j][0] = E[j];
		}
	}

	//! compute eta_hat = alpha * Gamma_hat : tau_hat + beta*tau_hat, eta_hat(0) = E
	noinline void GammaOperatorFourierCollocatedHeat(const ublas::vector<T>& E, T mu_0, T lambda_0, const ComplexTensor& tau_hat, ComplexTensor& eta_hat,
		T alpha = -1, T beta = 0)
	{
		Timer __t("GammaOperatorFourierCollocatedHeat", false);

		const T xi0_0 = 1/_dx, xi1_0 = 1/_dy, xi2_0 = 1/_dz;	// constant factor 2*M_PI actually does not matter
		// non-symmetrized version
		const T c10 = alpha/(2*mu_0);

		const bool nx_even = (_nx & 1) == 0;
		const bool ny_even = (_ny & 1) == 0;
		const bool nz_even = (_nz & 1) == 0;
		const std::size_t ii_half = nx_even ? (_nx/2 - 1) : _nx/2;
		const std::size_t jj_half = ny_even ? (_ny/2 - 1) : _ny/2;
		const std::size_t kk_half = nz_even ? (_nz/2 - 1) : _nz/2;
		//const std::size_t ii_filt = (_freq_hack && nx_even) ? _nx/2 : _nx;
		//const std::size_t jj_filt = (_freq_hack && ny_even) ? _ny/2 : _ny;
		//const std::size_t kk_filt = (_freq_hack && nz_even) ? _nz/2 : _nz;

		std::complex<T> ey[3];

		// "safe" version, but slow

		std::complex<T> c;
		ublas::c_vector<T,3> xi;

		#pragma omp parallel for private(c, ey, xi)
		for (size_t ii = 0; ii < _nx; ii++)
		{
			xi(0) = xi0_0*((ii <= ii_half) ? ii : ((T)ii - (T)_nx));

			for (size_t jj = 0; jj < _ny; jj++)
			{
				xi(1) = xi1_0*((jj <= jj_half) ? jj : ((T)jj - (T)_ny));

				// calculate current index in complex tensor tau[*]
				size_t k = ii*_ny*_nzc + jj*_nzc;

				for (size_t kk = 0; kk < _nzc; kk++)
				{
					xi(2) = xi2_0*((kk <= kk_half) ? kk : ((T)kk - (T)_nz));

					T norm_xi2 = xi(0)*xi(0) + xi(1)*xi(1) + xi(2)*xi(2);
					T c1 = c10/(norm_xi2);

					// perform multiplication in Voigt notation
					// ey = -Gamma_0 : tau_hat = -Gamma_0^v*tau_hat^v

					for (size_t i = 0; i < 3; i++)
					{
						// sum up the components
						c = 0;
						for (size_t j = 0; j < 3; j++)
						{
							T gamma_ij = c1*xi(i)*xi(j);

							c += gamma_ij*tau_hat[j][k];
						}

						ey[i] = c;
					}

					// assign result to eta_hat
					// we do this seperately since eta_hat and tau_hat may point to the same memory
					for (size_t j = 0; j < 3; j++) {
						eta_hat[j][k] = ey[j] + beta*tau_hat[j][k];
					}
					
					k++;
				}
			}
		}

		// set zero component
		eta_hat.setConstant(0, E);
	}


	//! compute eta_hat = alpha * Gamma_hat : tau_hat + beta*tau_hat, eta_hat(0) = E
	noinline void GammaOperatorFourierCollocated(const ublas::vector<T>& E, T mu_0, T lambda_0, const ComplexTensor& tau_hat, ComplexTensor& eta_hat,
		T alpha = -1, T beta = 0)
	{
		Timer __t("GammaOperatorFourierCollocated", false);

		const T xi0_0 = 1/_dx, xi1_0 = 1/_dy, xi2_0 = 1/_dz;		// constant factor 2*M_PI actually does not matter
		const T c10 = alpha/(4*mu_0);
		const T c20 = -alpha/(mu_0*(1 + mu_0/(lambda_0 + mu_0)));	// == -alpha*(lambda_0 + mu_0)/(mu_0*(lambda_0 + 2*mu_0))

		const bool nx_even = (_nx & 1) == 0;
		const bool ny_even = (_ny & 1) == 0;
		const bool nz_even = (_nz & 1) == 0;
		const std::size_t ii_half = nx_even ? (_nx/2 - 1) : _nx/2;
		const std::size_t jj_half = ny_even ? (_ny/2 - 1) : _ny/2;
		const std::size_t kk_half = nz_even ? (_nz/2 - 1) : _nz/2;
		const std::size_t ii_filt = (_freq_hack && nx_even) ? _nx/2 : _nx;
		const std::size_t jj_filt = (_freq_hack && ny_even) ? _ny/2 : _ny;
		const std::size_t kk_filt = (_freq_hack && nz_even) ? _nz/2 : _nz;

		std::complex<T> ey[6];

#if 1
		#pragma omp parallel for private(ey) schedule (static)
		for (std::size_t ii = 0; ii < _nx; ii++)
		{
			const T xi0 = xi0_0*((ii <= ii_half) ? ii : ((T)ii - (T)_nx));
			const T xi00 = xi0*xi0;

			for (std::size_t jj = 0; jj < _ny; jj++)
			{
				const T xi1 = xi1_0*((jj <= jj_half) ? jj : ((T)jj - (T)_ny));
				const T xi01 = xi0*xi1;
				const T xi11 = xi1*xi1;

				// calculate current index in complex tensor tau[*]
				std::size_t k = ii*_ny*_nzc + jj*_nzc;

				for (std::size_t kk = 0; kk < _nzc; kk++)
				{
					const T xi2 = xi2_0*((kk <= kk_half) ? kk : ((T)kk - (T)_nz));

					const T xi02 = xi0*xi2;
					const T xi12 = xi1*xi2;
					const T xi22 = xi2*xi2;

					const T norm_xi2 = xi00 + xi11 + xi22;
					const T c1 = c10/(norm_xi2);
					const T c12 = c1*2;
					const T c2 = c20/(norm_xi2*norm_xi2);
					const T c3 = (c12 + c2*xi00);
					const T c4 = (c12 + c2*xi11);
					const T c5 = (c12 + c2*xi22);

					T g[6][6];
#define APPLY_GAMMA_CALC_G(OP,S0,S1,S2) \
					g[0][0] OP (c12 + c3)*xi00; \
					g[1][0] OP c2*xi00*xi11; \
					g[2][0] OP c2*xi00*xi22; \
					g[3][0] OP c2*xi00*xi12*S1*S2; \
					g[4][0] OP c3*xi02*S0*S2; \
					g[5][0] OP c3*xi01*S0*S1; \
					g[1][1] OP (c12 + c4)*xi11; \
					g[2][1] OP c2*xi11*xi22; \
					g[3][1] OP c4*xi12*S1*S2; \
					g[4][1] OP c2*xi11*xi02*S0*S2; \
					g[5][1] OP c4*xi01*S0*S1; \
					g[2][2] OP (c12 + c5)*xi22; \
					g[3][2] OP c5*xi12*S1*S2; \
					g[4][2] OP c5*xi02*S0*S2; \
					g[5][2] OP c2*xi22*xi01*S0*S1; \
					g[3][3] OP (c1*(xi11 + xi22) + c2*xi11*xi22); \
					g[4][3] OP (c1 + c2*xi22)*xi01*S0*S1; \
					g[5][3] OP (c1 + c2*xi11)*xi02*S0*S2; \
					g[4][4] OP (c1*(xi00 + xi22) + c2*xi00*xi22); \
					g[5][4] OP (c1 + c2*xi00)*xi12*S1*S2; \
					g[5][5] OP (c1*(xi00 + xi11) + c2*xi00*xi11);

					if (ii == ii_filt || jj == jj_filt || kk == kk_filt) {
						std::memset(&(g[0][0]), 0, sizeof(T)*36);
						T s = 1;
						if (ii == ii_filt) s *= 0.5;
						if (jj == jj_filt) s *= 0.5;
						if (kk == kk_filt) s *= 0.5;
						for (int i = 1; i >= (ii == ii_filt ? -1 : 1); i -= 2) {
							for (int j = 1; j >= (jj == jj_filt ? -1 : 1); j -= 2) {
								for (int k2 = 1; k2 >= (kk == kk_filt ? -1 : 1); k2 -= 2) {
									APPLY_GAMMA_CALC_G(+=s*,i,j,k2);
								}
							}
						}
					}
					else {
						APPLY_GAMMA_CALC_G(=,1,1,1);
					}

					g[0][1] = g[1][0];
					g[0][2] = g[2][0];
					g[0][3] = g[3][0];
					g[0][4] = g[4][0];
					g[0][5] = g[5][0];

					g[1][2] = g[2][1];
					g[1][3] = g[3][1];
					g[1][4] = g[4][1];
					g[1][5] = g[5][1];

					g[2][3] = g[3][2];
					g[2][4] = g[4][2];
					g[2][5] = g[5][2];

					g[3][4] = g[4][3];
					g[3][5] = g[5][3];

					g[4][5] = g[5][4];

					// perform multiplication
					// ey = -Gamma_0 : tau_hat = -Gamma_0^v*tau_hat^v
					// last 3 components must be scaled by 2 (since using Voigt notation for Gamma)
					for (std::size_t i = 0; i < 6; i++) {
						ey[i] = tau_hat[0][k]*g[i][0] + tau_hat[1][k]*g[i][1] + tau_hat[2][k]*g[i][2] +
							(tau_hat[3][k]*g[i][3] + tau_hat[4][k]*g[i][4] + tau_hat[5][k]*g[i][5])*(T)2;
					}

					// assign result to eta_hat
					// we do this seperately since eta_hat and tau_hat may point to the same memory
					for (std::size_t j = 0; j < 6; j++) {
						eta_hat[j][k] = ey[j] + beta*tau_hat[j][k];
					}
					
					k++;
				}
			}
		}
#else
		// "safe" version, but slow

		std::complex<T> c;
		ublas::c_matrix<T,6,6> gamma;
		const ublas::c_matrix<T,3,3> I(ublas::identity_matrix<T>(3));
		ublas::c_vector<T,3> xi;

		// indices for Voigt notation
		const int vi[6] = {0, 1, 2, 1, 0, 0};
		const int vj[6] = {0, 1, 2, 2, 2, 1};

		#pragma omp parallel for private(c, ey, gamma, xi)
		for (size_t ii = 0; ii < _nx; ii++)
		{
			xi(0) = xi0_0*((ii <= ii_half) ? ii : ((T)ii - (T)_nx));

			for (size_t jj = 0; jj < _ny; jj++)
			{
				xi(1) = xi1_0*((jj <= jj_half) ? jj : ((T)jj - (T)_ny));

				// calculate current index in complex tensor tau[*]
				size_t k = ii*_ny*_nzc + jj*_nzc;

				for (size_t kk = 0; kk < _nzc; kk++)
				{
					xi(2) = xi2_0*((kk <= kk_half) ? kk : ((T)kk - (T)_nz));

					T norm_xi2 = xi(0)*xi(0) + xi(1)*xi(1) + xi(2)*xi(2);
					T c1 = c10/(norm_xi2);
					T c2 = c20/(norm_xi2*norm_xi2);

					// perform multiplication in Voigt notation
					// ey = -Gamma_0 : tau_hat = -Gamma_0^v*tau_hat^v

#if 0
					for (size_t i = 0; i < 6; i++)
					{
						for (size_t j = i; j < 6; j++) {
							LOG_COUT << "gamma[" << i << "][" << j << "] =";
							if (i != j) LOG_COUT << " gamma[" << j << "][" << i << "] =";
							LOG_COUT << " c1*(";
							if (I(vi[j],vi[i]) != 0) LOG_COUT << "+ xi" << vj[j] << "*xi" << vj[i] << "";
							if (I(vj[j],vi[i]) != 0) LOG_COUT << "+ xi" << vi[j] << "*xi" << vj[i] << "";
							if (I(vi[j],vj[i]) != 0) LOG_COUT << "+ xi" << vj[j] << "*xi" << vi[i] << "";
							if (I(vj[j],vj[i]) != 0) LOG_COUT << "+ xi" << vi[j] << "*xi" << vi[i] << "";
							LOG_COUT << ") + c2*xi" << vi[i] << "*xi" << vj[i] << "*xi" << vi[j] << "*xi" << vj[j] << ";" << std::endl;
						}
					}
#endif

					for (size_t i = 0; i < 6; i++)
					{
						for (size_t j = i; j < 6; j++)
						{
							gamma(i, j) = gamma(j, i) = c1*(
									I(vi[j],vi[i])*xi(vj[j])*xi(vj[i])
									+ I(vj[j],vi[i])*xi(vi[j])*xi(vj[i])
									+ I(vi[j],vj[i])*xi(vj[j])*xi(vi[i])
									+ I(vj[j],vj[i])*xi(vi[j])*xi(vi[i])
								) + c2*(
									xi(vi[i])*xi(vj[i])*xi(vi[j])*xi(vj[j])
								);
						}

						// sum up the components
						// we need to scale the last 3 components of tau_hat by 2 (Voigt notation)
						c = 0;
						for (size_t j = 3; j < 6; j++) {
							c += gamma(i, j)*tau_hat[j][k];
						}
						c *= 2;
						for (size_t j = 0; j < 3; j++) {
							c += gamma(i, j)*tau_hat[j][k];
						}
						ey[i] = c;
					}

					// assign result to eta_hat
					// we do this seperately since eta_hat and tau_hat may point to the same memory
					for (size_t j = 0; j < 6; j++) {
						eta_hat[j][k] = ey[j] + beta*tau_hat[j][k];
					}
					
					k++;
				}
			}
		}
#endif

		// set zero component
		for (std::size_t j = 0; j < 6; j++) {
			eta_hat[j][0] = E[j];
		}
	}

	noinline void GammaOperatorFourierStaggeredHyper(const ublas::vector<T>& E, T mu_0, T lambda_0, const ComplexTensor& tau_hat, ComplexTensor& eta_hat,
		T alpha = -1, T beta = 0)
	{
		Timer __t("GammaOperatorFourierStaggeredHyper", false);

		BOOST_THROW_EXCEPTION(std::runtime_error("GammaOperatorFourierStaggeredHyper not implemented"));
	}

	//! compute eta_hat = alpha * Gamma_hat : tau_hat + beta*tau_hat, eta_hat(0) = E
	noinline void GammaOperatorFourierCollocatedHyper(const ublas::vector<T>& E, T mu_0, T lambda_0, const ComplexTensor& tau_hat, ComplexTensor& eta_hat,
		T alpha = -1, T beta = 0)
	{
		Timer __t("GammaOperatorFourierCollocatedHyper", false);

		const T xi0_0 = 1/_dx, xi1_0 = 1/_dy, xi2_0 = 1/_dz;	// constant factor 2*M_PI actually does not matter
		// non-symmetrized version
		const T c10 = alpha/(2*mu_0);
		const T c20 = -alpha/(2*mu_0*(1 + 2*mu_0/lambda_0));	// == -alpha*lambda_0/(2*mu_0*(lambda_0 + 2*mu_0))

		const bool nx_even = (_nx & 1) == 0;
		const bool ny_even = (_ny & 1) == 0;
		const bool nz_even = (_nz & 1) == 0;
		const std::size_t ii_half = nx_even ? (_nx/2 - 1) : _nx/2;
		const std::size_t jj_half = ny_even ? (_ny/2 - 1) : _ny/2;
		const std::size_t kk_half = nz_even ? (_nz/2 - 1) : _nz/2;
		//const std::size_t ii_filt = (_freq_hack && nx_even) ? _nx/2 : _nx;
		//const std::size_t jj_filt = (_freq_hack && ny_even) ? _ny/2 : _ny;
		//const std::size_t kk_filt = (_freq_hack && nz_even) ? _nz/2 : _nz;

		std::complex<T> ey[9];

#if 0
#else
		// "safe" version, but slow

		std::complex<T> c;
		const ublas::c_matrix<T,3,3> I(ublas::identity_matrix<T>(3));
		ublas::c_vector<T,3> xi;

		// indices for Voigt notation
		const int vi[9] = {0, 1, 2, 1, 0, 0, 2, 2, 1};
		const int vj[9] = {0, 1, 2, 2, 2, 1, 1, 0, 0};

		#pragma omp parallel for private(c, ey, xi)
		for (size_t ii = 0; ii < _nx; ii++)
		{
			xi(0) = xi0_0*((ii <= ii_half) ? ii : ((T)ii - (T)_nx));

			for (size_t jj = 0; jj < _ny; jj++)
			{
				xi(1) = xi1_0*((jj <= jj_half) ? jj : ((T)jj - (T)_ny));

				// calculate current index in complex tensor tau[*]
				size_t k = ii*_ny*_nzc + jj*_nzc;

				for (size_t kk = 0; kk < _nzc; kk++)
				{
					xi(2) = xi2_0*((kk <= kk_half) ? kk : ((T)kk - (T)_nz));

					T norm_xi2 = xi(0)*xi(0) + xi(1)*xi(1) + xi(2)*xi(2);
					T c1 = c10/(norm_xi2);
					T c2 = c20/(norm_xi2*norm_xi2);

					// perform multiplication in Voigt notation
					// ey = -Gamma_0 : tau_hat = -Gamma_0^v*tau_hat^v

#if 0
					for (size_t i = 0; i < 9; i++)
					{
						for (size_t j = i; j < 9; j++) {
							LOG_COUT << "gamma[" << i << "][" << j << "] =";
							LOG_COUT << " c1*(";
							if (I(vi[j],vi[i]) != 0) LOG_COUT << "+ xi" << vj[j] << "*xi" << vj[i] << "";
							LOG_COUT << ") + c2*xi" << vi[i] << "*xi" << vj[i] << "*xi" << vi[j] << "*xi" << vj[j] << ";" << std::endl;
						}
					}
#endif

					for (size_t i = 0; i < 9; i++)
					{
						// sum up the components
						c = 0;
						for (size_t j = 0; j < 9; j++)
						{
							T gamma_ij = c1*(I(vi[i],vi[j])*xi(vj[i])*xi(vj[j]))
								+ c2*(xi(vi[i])*xi(vj[i])*xi(vi[j])*xi(vj[j]));

							c += gamma_ij*tau_hat[j][k];
						}

#if 0
						// sum up the components
						std::complex<T> s1 = 0;
						for (size_t j = 4; j < 9; j++)
						{
							T gamma_ij = c1*(I(vi[i],vi[j])*xi(vj[i])*xi(vj[j]))
								+ c2*(xi(vi[i])*xi(vj[i])*xi(vi[j])*xi(vj[j]));

							s1 += gamma_ij*tau_hat[j][k];
						}
						std::complex<T> s2 = 0;
						for (size_t j = 0; j < 4; j++)
						{
							T gamma_ij = c1*(I(vi[i],vi[j])*xi(vj[i])*xi(vj[j]))
								+ c2*(xi(vi[i])*xi(vj[i])*xi(vi[j])*xi(vj[j]));

							s2 += gamma_ij*tau_hat[j][k];
						}

						std::complex<T> s= s1 + s2;

						T err = std::max(std::abs(c/(s+1e-20)), std::abs(s/(c+1e-20))) - 1;

						if (err > 0.1) {
							LOG_CWARN << "cancelation " << c << " " << s << std::endl;
						}
#endif

						ey[i] = c;
					}

					// assign result to eta_hat
					// we do this seperately since eta_hat and tau_hat may point to the same memory
					for (size_t j = 0; j < 9; j++) {
						eta_hat[j][k] = ey[j] + beta*tau_hat[j][k];
					}
					
					k++;
				}
			}
		}
#endif

		// set zero component
		eta_hat.setConstant(0, E);
	}

	//! compute eta_hat = alpha*G0_hat(tau_hat)
	// eta_hat and tau_hat are vectors (only first 3 components are used)
	void G0OperatorFourierStaggered(T mu_0, T lambda_0, const ComplexTensor& tau_hat, ComplexTensor& eta_hat, T alpha = -1)
	{
		const T c10 = -alpha/(mu_0);
		const T c20 = -alpha/(mu_0*(1 + mu_0/(lambda_0 + mu_0)));	// == (lambda_0 + mu_0)/(mu_0*(lambda_0 + 2*mu_0))

		G0OperatorFourierStaggeredGeneral(mu_0, lambda_0, tau_hat, eta_hat, c10, c20);
	}

	//! compute eta_hat = alpha*G0_hat(tau_hat)
	// eta_hat and tau_hat are vectors (only first 3 components are used)
	void G0OperatorFourierStaggeredHeat(T mu_0, T lambda_0, const ComplexTensor& tau_hat, ComplexTensor& eta_hat, T alpha = -1)
	{
		const T c10 = -alpha/(2*mu_0);

		G0OperatorFourierStaggeredGeneralHeat(mu_0, lambda_0, tau_hat, eta_hat, c10);
	}

	//! compute eta_hat = alpha*G0_hat(tau_hat)
	// eta_hat and tau_hat are vectors (only first 3 components are used)
	void G0OperatorFourierStaggeredHyper(T mu_0, T lambda_0, const ComplexTensor& tau_hat, ComplexTensor& eta_hat, T alpha = -1)
	{
		const T c10 = -alpha/(2*mu_0);
		const T c20 = -alpha/(2*mu_0*(1 + 2*mu_0/lambda_0));	// == lambda_0/(2*mu_0*(lambda_0 + 2*mu_0))

		G0OperatorFourierStaggeredGeneral(mu_0, lambda_0, tau_hat, eta_hat, c10, c20);
	}

	//! compute eta_hat = alpha*G0_hat(tau_hat)
	// eta_hat and tau_hat are vectors (only first 3 components are used)
	noinline void G0OperatorFourierStaggeredGeneralHeat(T mu_0, T lambda_0, const ComplexTensor& tau_hat, ComplexTensor& eta_hat, T c10)
	{
		Timer __t("G0OperatorFourierStaggeredHeat", false);
		
		const T h0 = _dx/(2*_nx), h1 = _dy/(2*_ny), h2 = _dz/(2*_nz);
		const T xi0_0 = 2*M_PI*h0/(_dx), xi1_0 = 2*M_PI*h1/(_dy), xi2_0 = 2*M_PI*h2/(_dz);

		const bool nx_even = (_nx & 1) == 0;
		const bool ny_even = (_ny & 1) == 0;
		const bool nz_even = (_nz & 1) == 0;
		const std::size_t ii_half = nx_even ? (_nx/2 - 1) : _nx/2;
		const std::size_t jj_half = ny_even ? (_ny/2 - 1) : _ny/2;
		const std::size_t kk_half = nz_even ? (_nz/2 - 1) : _nz/2;
	
		std::complex<T> km[3], kp[3], tmp[3];

		#pragma omp parallel for private(km,kp,tmp) schedule (static)
		for (std::size_t ii = 0; ii < _nx; ii++)
		{
			const T xi0 = xi0_0*((ii <= ii_half) ? ii : ((T)ii - (T)_nx));
			const T kpm0 = std::sin(xi0)/h0;
			kp[0] = kpm0*std::exp(std::complex<T>(0, xi0));
			km[0] = std::complex<T>(-kp[0].real(), kp[0].imag());
			
			for (std::size_t jj = 0; jj < _ny; jj++)
			{
				const T xi1 = xi1_0*((jj <= jj_half) ? jj : ((T)jj - (T)_ny));
				const T kpm1 = std::sin(xi1)/h1;
				kp[1] = kpm1*std::exp(std::complex<T>(0, xi1));
				km[1] = std::complex<T>(-kp[1].real(), kp[1].imag());
				
				// calculate current index in complex tensor tau[*]
				std::size_t k = ii*_ny*_nzc + jj*_nzc;
				
				for (std::size_t kk = 0; kk < _nzc; kk++)
				{
					const T xi2 = xi2_0*((kk <= kk_half) ? kk : ((T)kk - (T)_nz));
					const T kpm2 = std::sin(xi2)/h2;
					kp[2] = kpm2*std::exp(std::complex<T>(0, xi2));
					km[2] = std::complex<T>(-kp[2].real(), kp[2].imag());
					
					const T norm_kp2 = kpm0*kpm0 + kpm1*kpm1 + kpm2*kpm2;
					const T c1 = c10/(norm_kp2);

					eta_hat[0][k] = c1*tau_hat[0][k];
					k++;
				}
			}
		}

		// set zero component
		eta_hat[0][0] = 0;
	}

	//! compute eta_hat = alpha*G0_hat(tau_hat)
	// eta_hat and tau_hat are vectors (only first 3 components are used)
	noinline void G0OperatorFourierStaggeredGeneral(T mu_0, T lambda_0, const ComplexTensor& tau_hat, ComplexTensor& eta_hat, T c10, T c20)
	{
		Timer __t("G0OperatorFourierStaggered", false);
		
		const T h0 = _dx/(2*_nx), h1 = _dy/(2*_ny), h2 = _dz/(2*_nz);
		const T xi0_0 = 2*M_PI*h0/(_dx), xi1_0 = 2*M_PI*h1/(_dy), xi2_0 = 2*M_PI*h2/(_dz);

		const bool nx_even = (_nx & 1) == 0;
		const bool ny_even = (_ny & 1) == 0;
		const bool nz_even = (_nz & 1) == 0;
		const std::size_t ii_half = nx_even ? (_nx/2 - 1) : _nx/2;
		const std::size_t jj_half = ny_even ? (_ny/2 - 1) : _ny/2;
		const std::size_t kk_half = nz_even ? (_nz/2 - 1) : _nz/2;
//		const std::size_t ii_filt = (_freq_hack && nx_even) ? _nx/2 : _nx;
//		const std::size_t jj_filt = (_freq_hack && ny_even) ? _ny/2 : _ny;
//		const std::size_t kk_filt = (_freq_hack && nz_even) ? _nz/2 : _nz;
	
		std::complex<T> km[3], kp[3], tmp[3];

		#pragma omp parallel for private(km,kp,tmp) schedule (static)
		for (std::size_t ii = 0; ii < _nx; ii++)
		{
			const T xi0 = xi0_0*((ii <= ii_half) ? ii : ((T)ii - (T)_nx));
			const T kpm0 = std::sin(xi0)/h0;
			kp[0] = kpm0*std::exp(std::complex<T>(0, xi0));
			km[0] = std::complex<T>(-kp[0].real(), kp[0].imag());
			
			for (std::size_t jj = 0; jj < _ny; jj++)
			{
				const T xi1 = xi1_0*((jj <= jj_half) ? jj : ((T)jj - (T)_ny));
				const T kpm1 = std::sin(xi1)/h1;
				kp[1] = kpm1*std::exp(std::complex<T>(0, xi1));
				km[1] = std::complex<T>(-kp[1].real(), kp[1].imag());
				
				// calculate current index in complex tensor tau[*]
				std::size_t k = ii*_ny*_nzc + jj*_nzc;
				
				for (std::size_t kk = 0; kk < _nzc; kk++)
				{
					const T xi2 = xi2_0*((kk <= kk_half) ? kk : ((T)kk - (T)_nz));
					const T kpm2 = std::sin(xi2)/h2;
					kp[2] = kpm2*std::exp(std::complex<T>(0, xi2));
					km[2] = std::complex<T>(-kp[2].real(), kp[2].imag());
					
					const T norm_kp2 = kpm0*kpm0 + kpm1*kpm1 + kpm2*kpm2;
					const T c1 = c10/(norm_kp2);
					const T c2 = c20/(norm_kp2*norm_kp2);

/*
					if (false && (ii == ii_filt || jj == jj_filt || kk == kk_filt)) {
						T s = 1;
						if (ii == ii_filt) s *= 0.5;
						if (jj == jj_filt) s *= 0.5;
						if (kk == kk_filt) s *= 0.5;
						std::memset(&(tmp[0]), 0, sizeof(T)*6);
						for (int i = 1; i >= (ii == ii_filt ? -1 : 1); i -= 2) {
							if (i < 0) std::swap(kp[0], km[0]);
							for (int j = 1; j >= (jj == jj_filt ? -1 : 1); j -= 2) {
								if (j < 0) std::swap(kp[1], km[1]);
								for (int k2 = 1; k2 >= (kk == kk_filt ? -1 : 1); k2 -= 2) {
									if (k2 < 0) std::swap(kp[2], km[2]);
									const std::complex<T> c2_fkp = c2*(tau_hat[0][k]*kp[0] + tau_hat[1][k]*kp[1] + tau_hat[2][k]*kp[2]);
									for (std::size_t h = 0; h < 3; h++) {
										tmp[h] += c2_fkp*km[h];
									}
									if (k2 < 0) std::swap(kp[2], km[2]);
								}
								if (j < 0) std::swap(kp[1], km[1]);
							}
							if (i < 0) std::swap(kp[0], km[0]);
						}
						for (std::size_t j = 0; j < 3; j++) {
							eta_hat[j][k] = c1*tau_hat[j][k] + s*tmp[j];
						}
					}
					else {
*/
						const std::complex<T> c2_fkp = c2*(tau_hat[0][k]*kp[0] + tau_hat[1][k]*kp[1] + tau_hat[2][k]*kp[2]);
						for (std::size_t j = 0; j < 3; j++) {
							eta_hat[j][k] = c1*tau_hat[j][k] + c2_fkp*km[j];
						}

//					}
					
					k++;
				}
			}
		}

		// set zero component
		for (std::size_t j = 0; j < 3; j++) {
			eta_hat[j][0] = 0;
		}
	}

	//! solve Laplace x = b
	//! \param r temporary vector for residual
	//! \param x the solution
	//! \param b the rhs
	noinline void mg_solve(T* r, T* x, T* b)
	{
		Timer __t(_mg_scheme + " poisson solver");

		if (!_mg_level)
		{
			// init multigrid solver
			bool alloc_rxb = (_mg_scheme == "pcg");
			_mg_level.reset(new MultiGridLevel<T>(_nx, _ny, _nz, _nzp, _dx, _dy, _dz, alloc_rxb));
			_mg_level->safe_mode = _mg_safe_mode;
			_mg_level->post_smoother = _mg_post_smoother;
			_mg_level->pre_smoother = _mg_pre_smoother;
			_mg_level->n_post_smooth = _mg_n_post_smooth;
			_mg_level->n_pre_smooth = _mg_n_pre_smooth;
			_mg_level->smooth_bs = _mg_smooth_bs;
			_mg_level->smooth_relax = _mg_smooth_relax;
			_mg_level->coarse_solver = _mg_coarse_solver;
			_mg_level->prolongation_op = _mg_prolongation_op;
			_mg_level->enable_timing = _mg_enable_timing;
			_mg_level->residual_checking = _mg_residual_checking;
			_mg_level->init_levels(_mg_coarse_size);
		}

		if (_mg_scheme == "fft") {
			_mg_level->solve_direct_fft(b, x);

		}
		else if (_mg_scheme == "direct") {
			// TODO: remove following comment and memset in G0OperatorMultigridStaggered _temp is then no longer needed
//			_mg_level->zero(x);
			_mg_level->run_direct(r, b, x, _mg_tol, _mg_maxiter);
			_mg_level->project_zero(x);
		}
		else if (_mg_scheme == "pcg") {
			// we use 3 vectors from the mg level for the temporaries, since they are not used
			// TODO: make this more clean, it is just coincidence that there are 3 unused vectors...
			T* z = _mg_level->_r;
			T* d = _mg_level->_x;
			T* h = _mg_level->_b;
			_mg_level->zero(x);
			_mg_level->run_pcg(z, d, r, h, b, x, _mg_tol, _mg_maxiter);
			_mg_level->project_zero(x);
		}
		else {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown multigrid scheme '%s'") % _mg_scheme).str()));
		}
	}

	//! compute divergence of vector tau and store to b
	// eta_hat and tau_hat are vectors (only first 3 components are used)
	noinline void divVector(const RealTensor& tau, T* b, T alpha = 1)
	{
		Timer __t("divVector", false);

		const T chx = alpha*_nx/_dx;
		const T chy = alpha*_ny/_dy;
		const T chz = alpha*_nz/_dz;

		#pragma omp parallel for schedule (static) collapse(2)
		for (std::size_t ii = 0; ii < _nx; ii++) {
			for (std::size_t jj = 0; jj < _ny; jj++) {
				int k = ii*_nyzp + jj*_nzp;
				for (std::size_t kk = 0; kk < _nz; kk++) {
					b[k] =  (tau[0][k] - tau[0][k + _ffd_x[ii]])*chx +
						(tau[1][k] - tau[1][k + _ffd_y[jj]])*chy +
						(tau[2][k] - tau[2][k + _ffd_z[kk]])*chz;
					k ++;
				}
			}
		}
	}

	//! compute eta_hat = alpha*G0_hat(tau_hat)
	// eta_hat and tau_hat are vectors (only first 3 components are used)
	void G0OperatorMultigridStaggered(T mu_0, T lambda_0, const RealTensor& tau, RealTensor& eta, T alpha = -1)
	{
		T* u = eta[0];
		T* v = eta[1];
		T* w = eta[2];
		T* r = eta[3];
		T* p = eta[4];
		T* b = eta[5];

		const T hx = _nx/_dx;
		const T hy = _ny/_dy;
		const T hz = _nz/_dz;

		// compute divergence b = 1/(2*mu_0 + lambda_0) div^- f
		// and solve Laplace p = b
		{
			divVector(tau, b, alpha);

			//memcpy(p, _temp[0], sizeof(T)*_n);
			memset(p, 0, sizeof(T)*_n);
			mg_solve(r, p, b);
			//memcpy(_temp[0], p, sizeof(T)*_n);
		}

		// compute b = 1/(mu_0) (f - (lambda_0 + mu_0) D_x^+ p)
		// and solve Laplace u = b
		{
			const T c1 = alpha/mu_0;
			const T c2 = -(1/mu_0)*(1 - mu_0/(2*mu_0 + lambda_0))*hx;

			#pragma omp parallel for schedule (static) collapse(2)
			for (std::size_t ii = 0; ii < _nx; ii++) {
				for (std::size_t jj = 0; jj < _ny; jj++) {
					int k = ii*_nyzp + jj*_nzp;
					for (std::size_t kk = 0; kk < _nz; kk++) {
						b[k] = c1*tau[0][k] + c2*(p[k + _bfd_x[ii]] - p[k]);
						k ++;
					}
				}
			}

			//memcpy(u, _temp[1], sizeof(T)*_n);
			memset(u, 0, sizeof(T)*_n);
			mg_solve(r, u, b);
			//memcpy(_temp[1], u, sizeof(T)*_n);
		}

		// compute b = 1/(mu_0) (g - (lambda_0 + mu_0) D_y^+ p)
		// and solve Laplace v = b
		{
			const T c1 = alpha/mu_0;
			const T c2 = -(1/mu_0)*(1 - mu_0/(2*mu_0 + lambda_0))*hy;

			#pragma omp parallel for schedule (static) collapse(2)
			for (std::size_t ii = 0; ii < _nx; ii++) {
				for (std::size_t jj = 0; jj < _ny; jj++) {
					int k = ii*_nyzp + jj*_nzp;
					for (std::size_t kk = 0; kk < _nz; kk++) {
						b[k] = c1*tau[1][k] + c2*(p[k + _bfd_y[jj]] - p[k]);
						k ++;
					}
				}
			}

			//memcpy(v, _temp[2], sizeof(T)*_n);
			memset(v, 0, sizeof(T)*_n);
			mg_solve(r, v, b);
			//memcpy(_temp[2], v, sizeof(T)*_n);
		}

		// compute b = 1/(mu_0) (g - (lambda_0 + mu_0) D_z^+ p)
		// and solve Laplace w = b
		{
			const T c1 = alpha/mu_0;
			const T c2 = -(1/mu_0)*(1 - mu_0/(2*mu_0 + lambda_0))*hz;

			#pragma omp parallel for schedule (static) collapse(2)
			for (std::size_t ii = 0; ii < _nx; ii++) {
				for (std::size_t jj = 0; jj < _ny; jj++) {
					int k = ii*_nyzp + jj*_nzp;
					for (std::size_t kk = 0; kk < _nz; kk++) {
						b[k] = c1*tau[2][k] + c2*(p[k + _bfd_z[kk]] - p[k]);
						k ++;
					}
				}
			}

			//memcpy(w, _temp[3], sizeof(T)*_n);
			memset(w, 0, sizeof(T)*_n);
			mg_solve(r, w, b);
			//memcpy(_temp[3], w, sizeof(T)*_n);
		}
	}

	void G0OperatorStaggered(T mu_0, T lambda_0, RealTensor& tau, ComplexTensor& tau_hat, ComplexTensor& eta_hat, RealTensor& eta, T alpha = 1)
	{
		if (_G0_solver == "fft")
		{
			fftVector(tau, tau_hat);
			G0OperatorFourierStaggered(mu_0, lambda_0, tau_hat, eta_hat, alpha);
			fftInvVector(eta_hat, eta);
		}
		else if (_G0_solver == "multigrid")
		{
			G0OperatorMultigridStaggered(mu_0, lambda_0, tau, eta, alpha);
		}
		else {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown G0-solver '%s'") % _G0_solver).str()));
		}
	}

	void G0OperatorStaggeredHeat(T mu_0, T lambda_0, RealTensor& tau, ComplexTensor& tau_hat, ComplexTensor& eta_hat, RealTensor& eta, T alpha = 1)
	{
		if (_G0_solver == "fft")
		{
			fftVector(tau, tau_hat, 1);
			G0OperatorFourierStaggeredHeat(mu_0, lambda_0, tau_hat, eta_hat, alpha);
			fftInvVector(eta_hat, eta, 1);
		}
		/*
		else if (_G0_solver == "multigrid")
		{
			G0OperatorMultigridStaggered(mu_0, lambda_0, tau, eta, alpha);
		}
		*/
		else {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown G0-solver '%s'") % _G0_solver).str()));
		}
	}

	void G0OperatorStaggeredHyper(T mu_0, T lambda_0, RealTensor& tau, ComplexTensor& tau_hat, ComplexTensor& eta_hat, RealTensor& eta, T alpha = 1)
	{
		if (_G0_solver == "fft")
		{
			fftVector(tau, tau_hat);
			G0OperatorFourierStaggeredHyper(mu_0, lambda_0, tau_hat, eta_hat, alpha);
			fftInvVector(eta_hat, eta);
		}
		else if (_G0_solver == "multigrid")
		{
			BOOST_THROW_EXCEPTION(std::runtime_error("G0OperatorStaggeredHyper multigrid not implemented"));
			// G0OperatorMultigridStaggeredHyper(mu_0, lambda_0, tau, eta, alpha);
		}
		else {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown G0-solver '%s'") % _G0_solver).str()));
		}
	}

	noinline void G0DivOperatorFourierHyper(T mu_0, T lambda_0, ComplexTensor& tau_hat, ComplexTensor& eta_hat, T alpha = 1)
	{
		Timer __t("G0DivOperatorFourierHyper", false);

		const T xi0_0 = 2*M_PI/_dx, xi1_0 = 2*M_PI/_dy, xi2_0 = 2*M_PI/_dz;
		
		// non-symmetrized version
		const T c10 = -alpha/(2*mu_0);
		const T c20 = alpha/(2*mu_0*(1 + 2*mu_0/lambda_0));	// == alpha*lambda_0/(2*mu_0*(lambda_0 + 2*mu_0))

		const bool nx_even = (_nx & 1) == 0;
		const bool ny_even = (_ny & 1) == 0;
		const bool nz_even = (_nz & 1) == 0;
		const std::size_t ii_half = nx_even ? (_nx/2 - 1) : _nx/2;
		const std::size_t jj_half = ny_even ? (_ny/2 - 1) : _ny/2;
		const std::size_t kk_half = nz_even ? (_nz/2 - 1) : _nz/2;

		const std::complex<T> imag(0, 1);
		ublas::c_vector<T,3> xi;
		std::complex<T> f1, f2, f3;

		#pragma omp parallel for private(f1, f2, f3, xi)
		for (size_t ii = 0; ii < _nx; ii++)
		{
			xi(0) = xi0_0*((ii <= ii_half) ? ii : ((T)ii - (T)_nx));

			for (size_t jj = 0; jj < _ny; jj++)
			{
				xi(1) = xi1_0*((jj <= jj_half) ? jj : ((T)jj - (T)_ny));

				// calculate current index in complex tensor tau[*]
				size_t k = ii*_ny*_nzc + jj*_nzc;

				for (size_t kk = 0; kk < _nzc; kk++)
				{
					xi(2) = xi2_0*((kk <= kk_half) ? kk : ((T)kk - (T)_nz));

					const T norm_xi2 = xi(0)*xi(0) + xi(1)*xi(1) + xi(2)*xi(2);
					const T c1 = c10/(norm_xi2);
					const T c2 = c20/(norm_xi2*norm_xi2);
					
					// compute Div
#if 0
					f1 = imag*(xi(0)*tau_hat[0][k] + xi(1)*tau_hat[8][k] + xi(2)*tau_hat[7][k]);
					f2 = imag*(xi(0)*tau_hat[5][k] + xi(1)*tau_hat[1][k] + xi(2)*tau_hat[6][k]);
					f3 = imag*(xi(0)*tau_hat[4][k] + xi(1)*tau_hat[3][k] + xi(2)*tau_hat[2][k]);
#else
					f1 = imag*(xi(0)*tau_hat[0][k] + xi(1)*tau_hat[5][k] + xi(2)*tau_hat[4][k]);
					f2 = imag*(xi(0)*tau_hat[8][k] + xi(1)*tau_hat[1][k] + xi(2)*tau_hat[3][k]);
					f3 = imag*(xi(0)*tau_hat[7][k] + xi(1)*tau_hat[6][k] + xi(2)*tau_hat[2][k]);
#endif
					
					// apply G0
					eta_hat[0][k] = c1*f1 + c2*(xi(0)*xi(0)*f1 + xi(0)*xi(1)*f2 + xi(0)*xi(2)*f3);
					eta_hat[1][k] = c1*f2 + c2*(xi(1)*xi(0)*f1 + xi(1)*xi(1)*f2 + xi(1)*xi(2)*f3);
					eta_hat[2][k] = c1*f3 + c2*(xi(2)*xi(0)*f1 + xi(2)*xi(1)*f2 + xi(2)*xi(2)*f3);

					k++;
				}
			}
		}

		eta_hat[0][0] = eta_hat[1][0] = eta_hat[2][0] = 0;
	}

	void initBCProjector(const ComplexTensor& tau_hat)
	{
		// save mean value
		for (std::size_t i = 0; i < tau_hat.dim; i++) {
			_F0[i] = tau_hat[i][0].real();
		}
	}

	noinline void initBCProjector(const RealTensor& tau)
	{
		Timer __t("initBCProjector", false);

		// do not compute the average if the projector _BC_MQ is zero anyway 
		if (ublas::norm_frobenius(_BC_MQ) < std::numeric_limits<T>::epsilon()) {
			_F0 = ublas::zero_vector<T>(_mat->dim());
			return;
		}

		_F0 = tau.average();
	}

	// compute bc projection mean constant
	ublas::vector<T> calcBCMean(const ublas::vector<T>& E, const ublas::vector<T>& S)
	{
		return E + _bc_relax*Voigt::dyad4(_BC_M, ublas::vector<T>(S - Voigt::dyad4(_BC_QC0, E)));

		/*
		LOG_COUT << "Id:BC_Q = " << format(Voigt::dyad4(Id4, _BC_Q)) << std::endl;
		LOG_COUT << "BC_Q:Id = " << format(Voigt::dyad4(_BC_Q, Id4)) << std::endl;
		LOG_COUT << "C0:BC_Q = " << format(Voigt::dyad4(C0, _BC_Q)) << std::endl;
		LOG_COUT << "BC_Q = " << format(_BC_Q) << std::endl;
		LOG_COUT << "BC_P = " << format(_BC_P) << std::endl;
		LOG_COUT << "VT = " << format(VT) << std::endl;
		LOG_COUT << "s = " << format(s) << std::endl;
		LOG_COUT << "BC_M = " << format(BC_M) << std::endl;
		*/
	}
	
	ublas::vector<T> calcBCProjector()
	{
		return _bc_relax*Voigt::dyad4(_BC_MQ, _F0) - (1-_bc_relax)*Voigt::dyad4(_BC_M, Voigt::dyad4(_BC_QC0, _F00));
	}

	noinline void applyBCProjector(RealTensor& eta, T alpha)
	{
		Timer __t("applyBCProjector", false);

		ublas::vector<T> R = alpha*calcBCProjector();
		// adjust mean component
		eta.add(R);
	}

	void applyBCProjector(ComplexTensor& eta_hat, T alpha)
	{		
		ublas::vector<T> R = alpha*calcBCProjector();
		// adjust mean component
		for (std::size_t i = 0; i < eta_hat.dim; i++) {
			eta_hat[i][0] += R[i];
		}
	}

	void G0DivOperatorHyper(T mu_0, T lambda_0, RealTensor& tau, ComplexTensor& tau_hat, ComplexTensor& eta_hat, RealTensor& eta, T alpha = -1)
	{
		fftTensor(tau, tau_hat);
		G0DivOperatorFourierHyper(mu_0, lambda_0, tau_hat, eta_hat, alpha);
		fftInvVector(eta_hat, eta);
	}

	void GammaOperatorStaggered(const ublas::vector<T>& E, T mu_0, T lambda_0,
		RealTensor& tau, ComplexTensor& tau_hat, ComplexTensor& eta_hat, RealTensor& eta, T alpha = -1, T beta = 0)
	{
		initBCProjector(tau);
		printTensor("tau", tau);
		divOperatorStaggered(tau, tau);
		printTensor("f", tau);
		G0OperatorStaggered(mu_0, lambda_0, tau, tau_hat, eta_hat, eta, alpha);
		printTensor("u", eta);
		epsOperatorStaggered(E, eta, eta);
		printTensor("eps", eta);
		applyBCProjector(eta, alpha);
	}

	void GammaOperatorCollocated(const ublas::vector<T>& E, T mu_0, T lambda_0,
		RealTensor& tau, ComplexTensor& tau_hat, ComplexTensor& eta_hat, RealTensor& eta, T alpha = -1, T beta = 0)
	{
		fftTensor(tau, tau_hat);
		initBCProjector(tau_hat);
		GammaOperatorFourierCollocated(E, mu_0, lambda_0, tau_hat, eta_hat, alpha, beta);
		applyBCProjector(eta_hat, alpha);
		fftInvTensor(eta_hat, eta);
	}

	void GammaOperatorCollocatedHeat(const ublas::vector<T>& E, T mu_0, T lambda_0,
		RealTensor& tau, ComplexTensor& tau_hat, ComplexTensor& eta_hat, RealTensor& eta, T alpha = -1, T beta = 0)
	{
		fftTensor(tau, tau_hat);
		initBCProjector(tau_hat);
		GammaOperatorFourierCollocatedHeat(E, mu_0, lambda_0, tau_hat, eta_hat, alpha, beta);
		applyBCProjector(eta_hat, alpha);
		fftInvTensor(eta_hat, eta);
	}

	void GammaOperatorWillotR(const ublas::vector<T>& E, T mu_0, T lambda_0,
		RealTensor& tau, ComplexTensor& tau_hat, ComplexTensor& eta_hat, RealTensor& eta, T alpha = -1, T beta = 0)
	{
		fftTensor(tau, tau_hat);
		initBCProjector(tau_hat);
		GammaOperatorFourierWillotR(E, mu_0, lambda_0, tau_hat, eta_hat, alpha, beta);
		applyBCProjector(eta_hat, alpha);
		fftInvTensor(eta_hat, eta);
	}

	void GammaOperatorCollocatedHyper(const ublas::vector<T>& E, T mu_0, T lambda_0,
		RealTensor& tau, ComplexTensor& tau_hat, ComplexTensor& eta_hat, RealTensor& eta, T alpha = -1, T beta = 0)
	{
		fftTensor(tau, tau_hat);
		initBCProjector(tau_hat);
		GammaOperatorFourierCollocatedHyper(E, mu_0, lambda_0, tau_hat, eta_hat, alpha, beta);
		applyBCProjector(eta_hat, alpha);
		fftInvTensor(eta_hat, eta);
	}

	void GammaOperatorStaggeredHeat(const ublas::vector<T>& E, T mu_0, T lambda_0,
		RealTensor& tau, ComplexTensor& tau_hat, ComplexTensor& eta_hat, RealTensor& eta, T alpha = -1, T beta = 0)
	{
#if 1
		initBCProjector(tau);
		divOperatorStaggeredHeat(tau, eta);
		G0OperatorStaggeredHeat(_mu_0, _lambda_0, eta, eta_hat, eta_hat, eta, alpha);
		epsOperatorStaggeredHeat(E, eta, eta);
		applyBCProjector(eta, alpha);
#else
		fftTensor(tau, tau_hat);
		initBCProjector(tau_hat);
		GammaOperatorFourierStaggeredHeat(E, mu_0, lambda_0, tau_hat, eta_hat, alpha);
		applyBCProjector(eta_hat, alpha);
		fftInvTensor(eta_hat, eta);
#endif
	}

	void GammaOperatorStaggeredHyper(const ublas::vector<T>& E, T mu_0, T lambda_0,
		RealTensor& tau, ComplexTensor& tau_hat, ComplexTensor& eta_hat, RealTensor& eta, T alpha = -1, T beta = 0)
	{
#if 1
		initBCProjector(tau);
		divOperatorStaggeredHyper(tau, eta);
		G0OperatorStaggeredHyper(_mu_0, _lambda_0, eta, eta_hat, eta_hat, eta, alpha);
		epsOperatorStaggeredHyper(E, eta, eta);
		applyBCProjector(eta, alpha);
#else
		fftTensor(tau, tau_hat);
		initBCProjector(tau_hat);
		GammaOperatorFourierStaggeredHyper(E, mu_0, lambda_0, tau_hat, eta_hat, alpha);
		applyBCProjector(eta_hat, alpha);
		fftInvTensor(eta_hat, eta);
#endif
	}

	//! compute eta = 2*alpha*mu_0*(tau - mu_0 * Gamma^0 : tau), mean eta = E
	//! where lambda_0 -> infinity, mu_0 -> mu_0/2 is used for Gamma^0
	void DeltaOperatorWillotR(const ublas::vector<T>& E, T mu_0, T lambda_0,
		RealTensor& tau, ComplexTensor& tau_hat, ComplexTensor& eta_hat, RealTensor& eta, T alpha = -1)
	{
		pRealTensor tau_copy;

		/*
		ublas::vector<T> tau0 = tau.average();
		LOG_COUT << "E = " << format(E) << std::endl;
		LOG_COUT << "tau0 = " << format(tau0) << std::endl;
		*/
		
		// fluidity -> viscosity
		mu_0 = 1/(4*mu_0);

		if (tau[0] == eta[0]) {
			// create copy of tau since tau gets lost after inplace GammaOperatorStaggered
			tau_copy = _temp->shadow();
			tau.copyTo(*tau_copy);
		}
		else {
			// operating out of place, no copy necessary
			tau_copy = tau.shadow();
		}

		// calculate mean constant such that <eta> = E
		ublas::c_vector<T, 6> adj = E - 2*alpha*mu_0*tau_copy->average();

		// compute eta = -4*alpha * mu_0^2 * Gamma^0 : tau
		GammaOperatorWillotR(adj, -1.0/(4*mu_0), STD_INFINITY(T), tau, tau_hat, eta_hat, eta, alpha);

		// compute eta += 2*alpha*mu_0*tau
		eta.xpay(eta, 2*alpha*mu_0, *tau_copy);

		/*
		ublas::vector<T> tau1 = eta.average();
		LOG_COUT << "tau1 = " << format(tau1) << std::endl;
		LOG_COUT << "eta0*(gamma_bar - gamma_0) = " << format((_S - tau0)*(2*mu_0)) << std::endl;
		*/
	}

	//! compute eta = 2*alpha*mu_0*(tau - mu_0 * Gamma^0 : tau), mean eta = E
	//! where lambda_0 -> infinity, mu_0 -> mu_0/2 is used for Gamma^0
	void DeltaOperatorStaggered(const ublas::vector<T>& E, T mu_0, T lambda_0,
		RealTensor& tau, ComplexTensor& tau_hat, ComplexTensor& eta_hat, RealTensor& eta, T alpha = -1)
	{
		pRealTensor tau_copy;

		/*
		ublas::vector<T> tau0 = tau.average();
		LOG_COUT << "E = " << format(E) << std::endl;
		LOG_COUT << "tau0 = " << format(tau0) << std::endl;
		*/
		
		// fluidity -> viscosity
		mu_0 = 1/(4*mu_0);

		if (tau[0] == eta[0]) {
			// create copy of tau since tau gets lost after inplace GammaOperatorStaggered
			tau_copy = _temp->shadow();
			tau.copyTo(*tau_copy);
		}
		else {
			// operating out of place, no copy necessary
			tau_copy = tau.shadow();
		}

		// calculate mean constant such that <eta> = E
		ublas::c_vector<T, 6> adj = E - 2*alpha*mu_0*tau_copy->average();

		// compute eta = -4*alpha * mu_0^2 * Gamma^0 : tau
		GammaOperatorStaggered(adj, -1.0/(4*mu_0), STD_INFINITY(T), tau, tau_hat, eta_hat, eta, alpha);

		// compute eta += 2*alpha*mu_0*tau
		eta.xpay(eta, 2*alpha*mu_0, *tau_copy);

		/*
		ublas::vector<T> tau1 = eta.average();
		LOG_COUT << "tau1 = " << format(tau1) << std::endl;
		LOG_COUT << "eta0*(gamma_bar - gamma_0) = " << format((_S - tau0)*(2*mu_0)) << std::endl;
		*/
	}

	//! compute eta = 2*alpha*mu_0*(tau - mu_0 * Gamma^0 : tau), mean eta = E
	//! where lambda_0 -> infinity, mu_0 -> mu_0/2 is used for Gamma^0
	void DeltaOperatorCollocated(const ublas::vector<T>& E, T mu_0, T lambda_0,
		RealTensor& tau, ComplexTensor& tau_hat, ComplexTensor& eta_hat, RealTensor& eta, T alpha = -1)
	{
		bool zero_trace = true;
		fftTensor(tau, tau_hat, zero_trace);
		applyDeltaFourier(E, 1/(4*mu_0), tau_hat, eta_hat, alpha);
		fftInvTensor(eta_hat, eta, zero_trace);
		//if (!zero_trace) fixTrace(eta);
	}

	void DeltaOperator(const ublas::vector<T>& E, T mu_0, T lambda_0,
		RealTensor& tau, ComplexTensor& tau_hat, ComplexTensor& eta_hat, RealTensor& eta, T alpha = -1)
	{
		if (_gamma_scheme == "collocated") {
			DeltaOperatorCollocated(E, mu_0, lambda_0, tau, tau_hat, eta_hat, eta, alpha);
		}
		else if (_gamma_scheme == "staggered" || _gamma_scheme == "half_staggered" || _gamma_scheme == "full_staggered") {
			DeltaOperatorStaggered(E, mu_0, lambda_0, tau, tau_hat, eta_hat, eta, alpha);
		}
		else if (_gamma_scheme == "willot") {
			DeltaOperatorWillotR(E, mu_0, lambda_0, tau, tau_hat, eta_hat, eta, alpha);
		}
	}

	void GammaOperator(const ublas::vector<T>& E, T mu_0, T lambda_0,
		RealTensor& tau, ComplexTensor& tau_hat, ComplexTensor& eta_hat, RealTensor& eta, T alpha = -1, T beta = 0)
	{
		if (_mode == "elasticity") {
			if (_gamma_scheme == "collocated") {
				GammaOperatorCollocated(E, mu_0, lambda_0, tau, tau_hat, eta_hat, eta, alpha, beta);
				return;
			}
			else if (_gamma_scheme == "staggered" || _gamma_scheme == "half_staggered" || _gamma_scheme == "full_staggered") {
				GammaOperatorStaggered(E, mu_0, lambda_0, tau, tau_hat, eta_hat, eta, alpha, beta);
				return;
			}
			else if (_gamma_scheme == "willot") {
				GammaOperatorWillotR(E, mu_0, lambda_0, tau, tau_hat, eta_hat, eta, alpha, beta);
				return;
			}
		}
		else if (_mode == "hyperelasticity") {
			if (_gamma_scheme == "collocated") {
				GammaOperatorCollocatedHyper(E, mu_0, lambda_0, tau, tau_hat, eta_hat, eta, alpha, beta);
				return;
			}
			else if (_gamma_scheme == "staggered" || _gamma_scheme == "half_staggered" || _gamma_scheme == "full_staggered") {
				GammaOperatorStaggeredHyper(E, mu_0, lambda_0, tau, tau_hat, eta_hat, eta, alpha, beta);
				return;
			}
		}
		else if (_mode == "viscosity") {
			DeltaOperator(E, mu_0, lambda_0, tau, tau_hat, eta_hat, eta, alpha);
			return;
		}
		else if (_mode == "heat" || _mode == "porous") {
			if (_gamma_scheme == "collocated") {
				GammaOperatorCollocatedHeat(E, mu_0, lambda_0, tau, tau_hat, eta_hat, eta, alpha, beta);
				return;
			}
			else if (_gamma_scheme == "staggered" || _gamma_scheme == "half_staggered" || _gamma_scheme == "full_staggered") {
				GammaOperatorStaggeredHeat(E, mu_0, lambda_0, tau, tau_hat, eta_hat, eta, alpha, beta);
				return;
			}
		}

		BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown gamma scheme '%s'") % _gamma_scheme).str()));
	}

	//! wrapper for polarization scheme
	// operation depends on the current mode of operation
	// Eyre, D. J., & Milton, G. W. (1999). A fast numerical scheme for computing the response of composites using grid refinement. The European Physical Journal Applied Physics, 6(1), 41–47. doi:10.1051/epjap:1999150
	void polarizationScheme(const ublas::vector<T>& P0, const RealTensor& epsilon,
		RealTensor& tau, ComplexTensor& tau_hat, ComplexTensor& eta_hat, RealTensor& eta)
	{
		printTensor("eps", epsilon);

		if (_bc_relax != 1.0) {
			_F00 = epsilon.average();
		}

		calcPolarization(_mu_0, 0, epsilon, tau);	// tau = Q

		ublas::vector<T> P00 = tau.average();

		// L0 = 2*mu_0
		// compute eta_hat = alpha * Gamma_hat : tau_hat + beta*tau_hat, eta_hat(0) = E

		printTensor("sigma", tau);
		GammaOperator(P00 + P0, _mu_0, _lambda_0, tau, tau_hat, eta_hat, eta, -4*_mu_0, 1.0);
	}

	//! wrapper for basic scheme
	// operation depends on the current mode of operation
	void basicScheme(const ublas::vector<T>& E, const RealTensor& epsilon,
		RealTensor& tau, ComplexTensor& tau_hat, ComplexTensor& eta_hat, RealTensor& eta)
	{
		printTensor("eps", epsilon);

		if (_bc_relax != 1.0) {
			_F00 = epsilon.average();
		}

		calcStressDiff(epsilon, tau);
		//LOG_COUT << "eps " << format(epsilon.average()) << std::endl;
		//LOG_COUT << "lambda,mu " << _lambda_0 << " " << _mu_0 << std::endl;
		//LOG_COUT << "calcStressDiff " << format(tau.average()) << std::endl;
		// Tensor<T, 6> Fk;
		// tau.assign(0, Fk);
		//LOG_COUT << "W0 " << Fk << std::endl;
		printTensor("sigma", tau);
		GammaOperator(E, _mu_0, _lambda_0, tau, tau_hat, eta_hat, eta, -1);
		// eta.assign(0, Fk);
		// LOG_COUT << "eta0 " << Fk << std::endl;
	}

	//! compute res = -Gamma_0 * (C-C0):epsilon  if (_mode == "elasticity" or "hyperelasticity")
	//! compute res = -Delta_0 * (phi-phi0):epsilon if (_mode == "viscosity")
	// note: tau, tau_hat, eta_hat, eta must not be equal to epsilon (i.e. no "in place" operation is possible)
	void krylovOperator(const RealTensor& epsilon, RealTensor& tau, ComplexTensor& tau_hat, ComplexTensor& eta_hat,
		RealTensor& eta, RealTensor& res)
	{
		basicScheme(ublas::zero_vector<T>(epsilon.dim), epsilon, tau, tau_hat, eta_hat, eta);
	}

	//! compute r = -(x + y)
	void mxpyTensor(T* r, const T* x, const T* y)
	{
		#pragma omp parallel for schedule (static)
		for (std::size_t i = 0; i < _n; i++) {
			r[i] = -(x[i] + y[i]);
		}
	}

	//! set boundary projector
	void setBCProjector(const ublas::matrix<T>& _P)
	{
		std::size_t dim = _P.size1();
		T eps = std::sqrt(std::numeric_limits<T>::epsilon());
		
		if (_P.size2() != dim || ublas::norm_frobenius(_P - ublas::trans(_P)) > eps) {
			LOG_COUT << "P = " << std::endl;
			LOG_COUT << format(_P) << std::endl;
			BOOST_THROW_EXCEPTION(std::runtime_error("Projector is not symmetric"));
		}
		
		if (ublas::norm_frobenius(_P - Voigt::dyad4(_P, _P)) > eps) {
			LOG_COUT << "P = " << std::endl;
			LOG_COUT << format(_P) << std::endl;
			LOG_COUT << "P*P = " << std::endl;
			LOG_COUT << format(Voigt::dyad4(_P, _P)) << std::endl;
			BOOST_THROW_EXCEPTION(std::runtime_error("Specified Projector is not a projector"));
		}

		// NOTE: Assuming C0 is a multiple of the identity
		ublas::matrix<T> C0 = 2*_mu_0*Voigt::Id4<T>(dim) + _lambda_0*Voigt::II4<T>(dim);
		
		_BC_P = _P;
		_BC_Q = Voigt::Id4<T>(dim) - _BC_P;
		_BC_QC0 = Voigt::dyad4(_BC_Q, C0);

		// compute SVD of Q:C0:Q = VT*S*U
		std::size_t edim = (dim == 6) ? 9 : dim;
		ublas::matrix<T> VT = ublas::zero_matrix<T>(edim), U = ublas::zero_matrix<T>(edim), Sinv = ublas::zero_matrix<T>(edim);
		ublas::vector<T> s = ublas::zero_vector<T>(edim);
		ublas::matrix<T> QC0Q = Voigt::dyad4(_BC_QC0, _BC_Q);

		if (dim == 6) {
			QC0Q.resize(9, 9);
			// extend to 9x9 matrix
			for (std::size_t i = 0; i < 9; i++) {
				for (std::size_t j = i; j < 9; j++) {
					QC0Q(j,i) = QC0Q(i,j) = QC0Q((i < 6 ? i : (i-3)), (j < 6 ? j : (j-3)));
				}
			}
		}

		//LOG_COUT << "A = " << format(QC0Q) << std::endl;
		lapack::gesvd(QC0Q, s, U, VT);
		
		// compute Moore-Penrose pseudo inverse BC_M = UT*Sinv*V
		T alpha = std::sqrt(std::numeric_limits<T>::epsilon())*ublas::norm_2(s);
		for (std::size_t i = 0; i < edim; i++) {
			if (std::abs(s(i)) > alpha) {
				Sinv(i,i) = 1.0/s(i);
			}
		}

		_BC_M = ublas::prod(ublas::matrix<T>(ublas::prod(VT, Sinv)), U);
		if (dim == 6) {
			// reduce to symmetric 6x6 matrix
			for (std::size_t i = 0; i < 3; i++) {
				for (std::size_t j = 0; j < 6; j++) {
					_BC_M(j, 3+i) = 0.5*(_BC_M(j, 3+i) + _BC_M(j, 6+i));
					_BC_M(3+i, j) = 0.5*(_BC_M(3+i, j) + _BC_M(6+i, j));
				}
			}
			_BC_M.resize(6, 6);
		}

		_BC_MQ = Voigt::dyad4(_BC_M, _BC_Q);
	}

	//! set prescribed stress
	void setStress(const ublas::vector<T>& e)
	{
		if (e.size() == 3) {
			// vector
			_S = e;
		}
		else if (e.size() == 6) {
			// symmetric "matrix"
			ublas::subrange(_S, 0, 6) = e;
			if (_S.size() == 9) {
				ublas::subrange(_S, 6, 9) = ublas::subrange(e, 3, 6);
			}
		}
		else if (e.size() == 9) {
			// full "matrix"
			_S = e;
		}
		else {
			BOOST_THROW_EXCEPTION(std::runtime_error("Invalid size of stress vector"));
		}

	}

	//! set prescribed strain
	void setStrain(const ublas::vector<T>& e)
	{
		if (e.size() == 3) {
			// vector
			_E = e;
		}
		else if (e.size() == 6) {
			// symmetric "matrix"
			ublas::subrange(_E, 0, 6) = e;
			if (_E.size() == 9) {
				ublas::subrange(_E, 6, 9) = ublas::subrange(e, 3, 6);
			}
		}
		else if (e.size() == 9) {
			// full "matrix"
			_E = e;
		}
		else {
			BOOST_THROW_EXCEPTION(std::runtime_error("Invalid size of strain vector"));
		}
	}
	
	//! returns the current average value for epsilon
	inline ublas::vector<T> averageValue()
	{
		return _epsilon->average();
	}

	//! return current operation mode
	inline std::string mode()
	{
		return _mode;
	}

	//! calculate |norm1 - norm2|
	noinline T calcNormDiff(const RealTensor& a, const RealTensor& b)
	{
		Timer __t("calcNormDiff", false);

		T d = 0;
	
		#pragma omp parallel for reduction(+:d) schedule (static) collapse(3)
		for (std::size_t j = 0; j < a.dim; j++)
		{
			for (std::size_t ii = 0; ii < _nx; ii++)
			{
				for (std::size_t jj = 0; jj < _ny; jj++)
				{
					// calculate current index in tensor epsilon
					std::size_t k = ii*_nyzp + jj*_nzp;
				
					for (std::size_t kk = 0; kk < _nz; kk++)
					{
						T diff = a[j][k] - b[j][k];
						d += diff*diff;
						k++;
					}
				}
			}
		}

		d = std::sqrt(d/_nxyz);

		return d;
	}

	//! compute sum over a:(b - c)
	noinline T innerProduct(RealTensor& a, RealTensor& b, RealTensor& c)
	{
		Timer __t("innerProductDiff", false);

		if (_cg_inner_product == "energy") {
			return innerProductEnergyC0(a, b, c);
		}
		else if (_cg_inner_product == "l2") {
			return innerProductL2(a, b, c);
		}
		else {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown inner product '%s'") % _cg_inner_product).str()));
		}
	}

	//! compute sum over a:b
	noinline T innerProduct(RealTensor& a, RealTensor& b)
	{
		Timer __t("innerProduct", false);

		if (_cg_inner_product == "energy") {
			return innerProductEnergyC0(a, b);
		}
		else if (_cg_inner_product == "l2") {
			return innerProductL2(a, b);
		}
		else {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown inner product '%s'") % _cg_inner_product).str()));
		}
	}

	noinline T innerProductEnergyC0(RealTensor& a, RealTensor& b)
	{
		BOOST_THROW_EXCEPTION(std::runtime_error("not implemented"));

		Timer __t("innerProductEnergyC0", false);

		T s = 0;
		T mu = _mu_0;
		T two_mu = 2*mu;
		T half_lambda = 0.5*_lambda_0;

		#pragma omp parallel for reduction(+:s) schedule (static) collapse(2)
		for (std::size_t ii = 0; ii < _nx; ii++)
		{
			for (std::size_t jj = 0; jj < _ny; jj++)
			{
				// calculate current index in tensor epsilon
				std::size_t k = ii*_nyzp + jj*_nzp;

				for (std::size_t kk = 0; kk < _nz; kk++)
				{
					const T half_lambda_tr_a = half_lambda*(a[0][k] + a[1][k] + a[2][k]);

					s +=      (a[0][k]*mu + half_lambda_tr_a)*(b[0][k])
						+ (a[1][k]*mu + half_lambda_tr_a)*(b[1][k])
						+ (a[2][k]*mu + half_lambda_tr_a)*(b[2][k])
						+ (a[3][k]*two_mu)*(b[3][k])
						+ (a[4][k]*two_mu)*(b[4][k])
						+ (a[5][k]*two_mu)*(b[5][k]);

					k++;
				}
			}
		}

		s *= 2;
		s /= _nxyz;
		return s;
	}

	noinline T innerProductEnergyC0(RealTensor& a, RealTensor& b, RealTensor& c)
	{
		BOOST_THROW_EXCEPTION(std::runtime_error("not implemented"));

		Timer __t("innerProductEnergyC0", false);

		T s = 0;
		T mu = _mu_0;
		T two_mu = 2*mu;
		T half_lambda = 0.5*_lambda_0;

		#pragma omp parallel for reduction(+:s) schedule (static) collapse(2)
		for (std::size_t ii = 0; ii < _nx; ii++)
		{
			for (std::size_t jj = 0; jj < _ny; jj++)
			{
				// calculate current index in tensor epsilon
				std::size_t k = ii*_nyzp + jj*_nzp;
				
				for (std::size_t kk = 0; kk < _nz; kk++)
				{
					const T half_lambda_tr_a = half_lambda*(a[0][k] + a[1][k] + a[2][k]);

					s +=      (a[0][k]*mu + half_lambda_tr_a)*(b[0][k] - c[0][k])
						+ (a[1][k]*mu + half_lambda_tr_a)*(b[1][k] - c[1][k])
						+ (a[2][k]*mu + half_lambda_tr_a)*(b[2][k] - c[2][k])
						+ (a[3][k]*two_mu)*(b[3][k] - c[3][k])
						+ (a[4][k]*two_mu)*(b[4][k] - c[4][k])
						+ (a[5][k]*two_mu)*(b[5][k] - c[5][k]);

					k++;
				}
			}
		}

		s *= 2;
		s /= _nxyz;
		return s;
	}

	//! compute sum over a:(b - c)
	noinline T innerProductL2(RealTensor& a, RealTensor& b, RealTensor& c)
	{
		Timer __t("innerProductL2", false);

		T s = 0;

		if (a.dim == 3)
		{
			#pragma omp parallel for reduction(+:s) schedule (static) collapse(2)
			for (std::size_t ii = 0; ii < _nx; ii++)
			{
				for (std::size_t jj = 0; jj < _ny; jj++)
				{
					// calculate current index in tensor epsilon
					std::size_t k = ii*_nyzp + jj*_nzp;
					
					for (std::size_t kk = 0; kk < _nz; kk++)
					{
						s +=	  a[0][k]*(b[0][k] - c[0][k])
							+ a[1][k]*(b[1][k] - c[1][k])
							+ a[2][k]*(b[2][k] - c[2][k]);
						k++;
					}
				}
			}
		}
		else if (a.dim == 6)
		{
			#pragma omp parallel for reduction(+:s) schedule (static) collapse(2)
			for (std::size_t ii = 0; ii < _nx; ii++)
			{
				for (std::size_t jj = 0; jj < _ny; jj++)
				{
					// calculate current index in tensor epsilon
					std::size_t k = ii*_nyzp + jj*_nzp;
					
					for (std::size_t kk = 0; kk < _nz; kk++)
					{
						s +=	  a[0][k]*(b[0][k] - c[0][k])
							+ a[1][k]*(b[1][k] - c[1][k])
							+ a[2][k]*(b[2][k] - c[2][k])
							+ 2*(a[3][k]*(b[3][k] - c[3][k])
							+    a[4][k]*(b[4][k] - c[4][k])
							+    a[5][k]*(b[5][k] - c[5][k]));
						k++;
					}
				}
			}
		}
		else if (a.dim == 9)
		{
			#pragma omp parallel for reduction(+:s) schedule (static) collapse(2)
			for (std::size_t ii = 0; ii < _nx; ii++)
			{
				for (std::size_t jj = 0; jj < _ny; jj++)
				{
					// calculate current index in tensor epsilon
					std::size_t k = ii*_nyzp + jj*_nzp;
					
					for (std::size_t kk = 0; kk < _nz; kk++)
					{
						s +=	  a[0][k]*(b[0][k] - c[0][k])
							+ a[1][k]*(b[1][k] - c[1][k])
							+ a[2][k]*(b[2][k] - c[2][k])
							+ a[3][k]*(b[3][k] - c[3][k])
							+ a[4][k]*(b[4][k] - c[4][k])
							+ a[5][k]*(b[5][k] - c[5][k])
							+ a[6][k]*(b[6][k] - c[6][k])
							+ a[7][k]*(b[7][k] - c[7][k])
							+ a[8][k]*(b[8][k] - c[8][k]);
						k++;
					}
				}
			}
		}
		else {
			BOOST_THROW_EXCEPTION(std::runtime_error("not implemented"));
		}

		s /= _nxyz;
		return s;
	}

	//! compute sum over a:b
	noinline T innerProductL2(RealTensor& a, RealTensor& b)
	{
		Timer __t("innerProductL2", false);

		T s = 0;

		if (a.dim == 3)
		{
			#pragma omp parallel for reduction(+:s) schedule (static) collapse(2)
			for (std::size_t ii = 0; ii < _nx; ii++)
			{
				for (std::size_t jj = 0; jj < _ny; jj++)
				{
					// calculate current index in tensor epsilon
					std::size_t k = ii*_nyzp + jj*_nzp;
					
					for (std::size_t kk = 0; kk < _nz; kk++)
					{
						s +=	  a[0][k]*b[0][k]
							+ a[1][k]*b[1][k]
							+ a[2][k]*b[2][k];
						k++;
					}
				}
			}
		}
		else if (a.dim == 6)
		{
			#pragma omp parallel for reduction(+:s) schedule (static) collapse(2)
			for (std::size_t ii = 0; ii < _nx; ii++)
			{
				for (std::size_t jj = 0; jj < _ny; jj++)
				{
					// calculate current index in tensor epsilon
					std::size_t k = ii*_nyzp + jj*_nzp;
					
					for (std::size_t kk = 0; kk < _nz; kk++)
					{
						s +=	  a[0][k]*b[0][k]
							+ a[1][k]*b[1][k]
							+ a[2][k]*b[2][k]
							+ 2*(a[3][k]*b[3][k]
							+    a[4][k]*b[4][k]
							+    a[5][k]*b[5][k]);
						k++;
					}
				}
			}
		}
		else if (a.dim == 9)
		{
			#pragma omp parallel for reduction(+:s) schedule (static) collapse(2)
			for (std::size_t ii = 0; ii < _nx; ii++)
			{
				for (std::size_t jj = 0; jj < _ny; jj++)
				{
					// calculate current index in tensor epsilon
					std::size_t k = ii*_nyzp + jj*_nzp;
					
					for (std::size_t kk = 0; kk < _nz; kk++)
					{
						s +=	  a[0][k]*b[0][k]
							+ a[1][k]*b[1][k]
							+ a[2][k]*b[2][k]
							+ a[3][k]*b[3][k]
							+ a[4][k]*b[4][k]
							+ a[5][k]*b[5][k]
							+ a[6][k]*b[6][k]
							+ a[7][k]*b[7][k]
							+ a[8][k]*b[8][k];
						k++;
					}
				}
			}
		}
		else {
			BOOST_THROW_EXCEPTION(std::runtime_error("not implemented"));
		}

		s /= _nxyz;
		return s;
	}


	//! calculate t:t pointwise
	noinline void calcNorm(RealTensor& t, T* norm)
	{
		BOOST_THROW_EXCEPTION(std::runtime_error("not implemented"));

		Timer __timer("calcNorm", false);

		#pragma omp parallel for schedule (static) collapse(2)
		for (std::size_t ii = 0; ii < _nx; ii++)
		{
			for (std::size_t jj = 0; jj < _ny; jj++)
			{
				// calculate current index in tensor epsilon
				std::size_t k = ii*_nyzp + jj*_nzp;
				
				for (std::size_t kk = 0; kk < _nz; kk++)
				{
					norm[k] = t[0][k]*t[0][k]
						+ t[1][k]*t[1][k]
						+ t[2][k]*t[2][k]
						+ 2*(t[3][k]*t[3][k]
						+    t[4][k]*t[4][k]
						+    t[5][k]*t[5][k]);
					k++;
				}
			}
		}
	}

	void check_var(const std::string& name, const RealTensor& t) const
	{
		LOG_COUT << (boost::format("%s: %d") % name % t.checksum()) << std::endl;
	}

	void printTensor(const std::string& name, const RealTensor& t, std::size_t dims = 0, std::size_t start = 0)
	{
		if (!_debug) return;

		if (dims < 1) dims = t.dim;

		for (std::size_t i = start; i < std::min(start + dims, t.dim); i++) {
			LOG_COUT << (boost::format("%s[%d]:") % name % i).str() << "(" << t.nx << "x" << t.ny << "x" << t.nz << "+" << (t.nzp-t.nz) << ")" << std::endl
				<< format(t[i], t.nx, t.ny, t.nzp, t.nzp) << std::endl;
		}
	}

	// general method for convergence check
	noinline void printMeanValues() const
	{
		Timer __t("printMeanValues", false);

		std::string strain, stress;

		if (_mode == "elasticity")
		{
			strain  = "elastic strain";
			stress = "average elastic stress";
		}
		else if (_mode == "hyperelasticity")
		{
			strain  = "deformation gradient";
			stress = "1st Piola-Kirchhoff stress";
		}
		else if (_mode == "viscosity")
		{
			strain  = "fluid stress";
			stress = "fluid shear";
		}
		else if (_mode == "heat")
		{
			strain  = "temperture gradient";
			stress = "heat flux";
		}
		else if (_mode == "porous")
		{
			strain  = "pressure gradient";
			stress = "volumetric flux";
		}

		ublas::vector<T> Emean = _epsilon->average();
		ublas::vector<T> Smean = calcMeanStress();
		ublas::vector<T> Eerror = Voigt::dyad4(_BC_P, Emean) - _current_E;
		ublas::vector<T> Serror = Voigt::dyad4(_BC_Q, Smean) - _current_S;
		
		LOG_COUT << "mean " << strain << " = " << format(Emean) << std::endl;
		// LOG_COUT << strain << " error = " << format(Eerror) << std::endl;
		LOG_COUT << "mean " << stress << " = " << format(Smean) << std::endl;
		LOG_COUT << stress << " error = " << format(Serror) << std::endl;
	}

	T bc_error() const
	{
		// calculate relative bc error
		ublas::vector<T> Emean = _epsilon->average();
		ublas::vector<T> Smean = calcMeanStress();
		ublas::vector<T> P_Emean(Voigt::dyad4(_BC_P, Emean));
		ublas::vector<T> Q_Smean(Voigt::dyad4(_BC_Q, Smean));
		ublas::vector<T> _P_current_E_minus_Id(Voigt::dyad4(_BC_P, _current_E));

		if (_current_E.size() == 9) {
			_P_current_E_minus_Id -= ublas::vector<T>(Voigt::dyad4(_BC_P, _Id));
		}
		
		//T eps = std::sqrt(std::numeric_limits<T>::epsilon());
		T norm_E = Voigt::norm_2(_P_current_E_minus_Id);

		T err_F = Voigt::norm_2(ublas::vector<T>(P_Emean - _current_E)) / ((norm_E < _bc_tol) ? 1 : norm_E);
		//T err_S = Voigt::norm_2(ublas::vector<T>(Q_Smean - _current_S)) / (std::max(Voigt::norm_2(_current_S), Voigt::norm_2(Q_Smean)*2/_bc_tol));
		T norm_S = Voigt::norm_2(_current_S);
		T err_S = Voigt::norm_2(ublas::vector<T>(Q_Smean - _current_S)) / ((norm_S < _bc_tol) ? 1 : norm_S);

/*
		LOG_COUT << "bc err: " << err_F << " " << err_S << std::endl;
		LOG_COUT << "Q_Smean - _current_S: " << format(Q_Smean - _current_S) << std::endl;
		LOG_COUT << "P_Emean - _current_E: " << format(P_Emean - _current_E) << std::endl;
		LOG_COUT << "_P_Emean_minus_Id: " << format(_P_current_E_minus_Id) << std::endl;
		LOG_COUT << "_P_Emean_minus_Id: " << Voigt::norm_2(_P_current_E_minus_Id) << std::endl;
		LOG_COUT << "_Q_Smean_minus_Id: " << format(_Q_current_S) << std::endl;
		LOG_COUT << "_Q_Smean_minus_Id: " << Voigt::norm_2(_Q_current_S) << std::endl;
*/

		return std::max(err_F, err_S);
	}

	//! general method for convergence check
	bool converged(std::size_t& iter, T abs_err, T rel_err, bool check_bc = true)
	{
		bool ret = _converged(iter, abs_err, rel_err, check_bc);

		if (_step_mode) {
			LOG_COUT << "Press the ENTER key\r";
			std::cin.ignore();
		}

		return ret;
	}

	//! general method for convergence check
	noinline bool _converged(std::size_t& iter, T abs_err, T rel_err, bool check_bc = true)
	{
		Timer __t("converged", false);
 

		LOG_COUT << "# Iteration " << iter << ": " << _error_estimator << " error abs. = " << abs_err << " rel. = " << rel_err << std::endl;

		Logger::instance().incIndent();
		if (_print_mean) {
			printMeanValues();
		}
		if (_print_detF && _mode == "hyperelasticity") {
			LOG_COUT << "min det(F) = " << calcMinDetF() << std::endl;
		}
		Logger::instance().decIndent();

		LaminateMixedMaterialLaw<T, P, 9>* lm = dynamic_cast<LaminateMixedMaterialLaw<T, P, 9>* >(_mat.get());
		if (lm != NULL && lm->debug) {
			LOG_COUT << "LaminateMixedMaterialLaw: calls=" << lm->_calls << " call_iter=" << (lm->_sum_newtoniter /(T) lm->_calls) << " call_backtrack=" << (lm->_sum_backtrack /(T) lm->_calls) << " max_iter=" << (lm->_max_newtoniter) << " max_backtrack=" << (lm->_max_backtrack) << std::endl;
			lm->_max_newtoniter = lm->_max_backtrack = lm->_calls = 0;
			lm->_sum_newtoniter = lm->_sum_backtrack = 0;
		}


		// convergence check	
		if (std::isnan(rel_err)) {
			set_exception("NaN detected in solution. Aborting.");
		}

		if (_except) {
			return true;
		}

		// store residual
		_residuals.push_back(rel_err);

		// call custom convergence check
		
		if (_convergence_callback && _convergence_callback()) {
			LOG_COUT << "Custom convergence test returned true." << std::endl;
			return true;
		}
		
		// maximum iteration check	
		if (iter >= _maxiter) {
			LOG_COUT << "Maximum number of iterations reached." << std::endl;
			return true;
		}
	
		// convergence check	
		if (rel_err <= _tol || abs_err <= _abs_tol) {
			// check convergence of boundary conditions
			T bc_err = 0;
			if (check_bc) {
				bc_err = bc_error();
				LOG_COUT << "Boundary condition error = " << bc_err << std::endl;
			}
			if (bc_err <= _bc_tol) {
				LOG_COUT << "Converged." << std::endl;
				return true;
			}
		}

		// next iteration
		iter ++;
		
		return false;
	}

	//! run the solver
	noinline bool run()
	{
		Timer __t("running solver");

		_solve_time = 0;
		_fft_time = 0;
		_residuals.clear();

		// check strain
		//if (ublas::norm_2(_E) == 0) {
		//	BOOST_THROW_EXCEPTION(std::runtime_error("no average strain applied"));
		//}

		if (_mode == "elasticity")
		{
			LOG_COUT << "prescribed elastic strain: " << _E << std::endl;
			LOG_COUT << "prescribed elastic stress: " << _S << std::endl;
		}
		else if (_mode == "hyperelasticity")
		{
			LOG_COUT << "prescribed deformation gradient: " << _E << std::endl;
			LOG_COUT << "prescribed 1st PK: " << _S << std::endl;
		}
		else if (_mode == "viscosity")
		{
			LOG_COUT << "prescribed fluid stress: " << _E << std::endl;
			LOG_COUT << "prescribed fluid strain: " << _S << std::endl;
		}
		else if (_mode == "heat")
		{
			LOG_COUT << "prescribed temperature gradient: " << _E << std::endl;
			LOG_COUT << "prescribed heat flux: " << _S << std::endl;
		}
		else if (_mode == "porous")
		{
			LOG_COUT << "prescribed pressure gradient: " << _E << std::endl;
			LOG_COUT << "prescribed volumetric flux: " << _S << std::endl;
		}

		// print some general information
		LOG_COUT << "RVE: dims=" << _dx << "x" << _dy << "x" << _dz << " voxels=" << _nx << "x" << _ny << "x" << _nz << " (" << _nxyz << ")" << std::endl;
		LOG_COUT << "mode: " << _method << " " << _gamma_scheme << " " << _mode << " " << _cg_inner_product
			<< (_freq_hack ? " freq_hack" : "")
			<< (_debug ? " debug" : "")
			<< (" G0-" + _G0_solver)
			<< std::endl;
		LOG_COUT << "tolerances: relative=" << _tol << " absolute=" << _abs_tol << std::endl;

		// list materials
		LOG_COUT << "materials:" << std::endl;
		for (std::size_t m = 0; m < _mat->phases.size(); m++) {
			LOG_COUT << " - " << _mat->phases[m]->name << ": " << _mat->phases[m]->law->str() << std::endl;
		}
		LOG_COUT << " - reference: K=" << (_lambda_0 + 2.0/3.0*_mu_0) << " mu=" << _mu_0 << " lambda=" << _lambda_0 << std::endl;
		LOG_COUT << "interfaces: " << _mat->str() << std::endl;

		/*
		// TODO: replace by something more general
		if (_mat->phases.size() == 2) {
			// print HS bounds
			T kl, ku, mul, muu;
			HashinBounds<T>::get(_mat->phases[0]->mu, _mat->phases[0]->lambda, _mat->phases[0]->vol, _mat->phases[1]->mu, _mat->phases[1]->lambda, _mat->phases[1]->vol, kl, mul, ku, muu);

			LOG_COUT << "HS lower bounds: K=" << kl << " mu=" << mul << " lambda=" << (kl - 2.0/3.0*mul) << std::endl;
			LOG_COUT << "HS upper bounds: K=" << ku << " mu=" << muu << " lambda=" << (ku - 2.0/3.0*muu) << std::endl;
		}
		*/

		// check if we need additional temporary vector
		bool needTemp = ((_mode == "viscosity") && (_gamma_scheme == "staggered" || _gamma_scheme == "half_staggered" || _gamma_scheme == "full_staggered" || _gamma_scheme == "willot")); // || (_G0_solver == "multigrid");
		if (needTemp) {
			_temp.reset(new RealTensor(*_epsilon, 0));
			_temp->setConstant(ublas::zero_vector<T>(_temp->dim));
		}

		// print phase field for debugging
		/*
		for (std::size_t m = 0; m < _mat->phases.size(); m++) {
			if (this->use_dfg()) {
				for (std::size_t i = 0; i < 4; i++) {
					printField((boost::format("phi_fs%d_%s") % i % _mat->phases[m]->name).str(), _mat->phases[m]->phi_fs[i]);
				}
			}
			else {
				printField("phi_" + _mat->phases[m]->name, _mat->phases[m]->phi);
			}
		}
		*/

#ifdef TEST_MERGE_ISSUE
		if (_normals) {
			check_var("normals", *_normals);
		}
		for (std::size_t m = 0; m < _mat->phases.size(); m++) {
			check_var(_mat->phases[m]->name, *_mat->phases[m]->_phi);
		}
#endif

		LOG_COUT << "projection matrix P:" << format(_BC_P) << std::endl;
		//LOG_COUT << "projection matrix Q:" << format(_BC_Q) << std::endl;
		//LOG_COUT << "projection matrix M:" << format(_BC_M) << std::endl;
		//LOG_COUT << "projection matrix QC0:" << format(_BC_QC0) << std::endl;


		// check boundary conditions
		T eps = std::sqrt(std::numeric_limits<T>::epsilon());

		this->setBCProjector(_BC_P);

		if (ublas::norm_2(Voigt::dyad4(_BC_P, _S)) > eps*ublas::norm_2(_S)) {
			LOG_COUT << "P*S = " << format(Voigt::dyad4(_BC_P, _S)) << " != 0" << std::endl;
			BOOST_THROW_EXCEPTION(std::runtime_error("Incompatible stress boundary condition specified"));
		}

		if (ublas::norm_2(Voigt::dyad4(_BC_Q, _E)) > eps*ublas::norm_2(_E)) {
			LOG_COUT << "Q*E = " << format(Voigt::dyad4(_BC_Q, _E)) << " != 0" << std::endl;
			BOOST_THROW_EXCEPTION(std::runtime_error("Incompatible strain boundary condition specified"));
		}

		Timer dt_solve;

		if (_mode == "hyperelasticity")
		{
			// init epsilon to identity
			//T t = 1/(T)_loadsteps;
			//ublas::vector<T> E = t*_E + (1-t)*Voigt::dyad4(_BC_P, _Id);
			//_epsilon->setConstant(E + Voigt::dyad4(_BC_Q, _Id));
			_epsilon->setConstant(_Id);

		}
		else {
			// init epsilon to prescribed strain
			_epsilon->setConstant(ublas::zero_vector<T>(6));
		}

		// check for NaN, inf, etc.
		if (_normals) _normals->check("normals");
		_epsilon->check("epsilon");

		// run solver
		bool ret = runLoadsteppingSolver(_E, _S);

		_solve_time += dt_solve;

		// free temporary 
		_temp.reset();

		// delete multigrid level
		_mg_level.reset();

		return ret;
	}

	//! run the solver with prescribed strain and stress
	void runSolver(const ublas::vector<T>& E, const ublas::vector<T>& S)
	{
		_current_E = E;
		_current_S = S;

		// adjust solution mean value (saves one interation)
		// _epsilon->add(EC - _epsilon->average());

		if (_method == "basic") {
			runBasic(E, S);
		}
		else if (_method == "polarization") {
			runPolarization(E, S);
		}
		else if (_method == "basic+el") {
			runBasicEL(E, S);
		}
		else if (_method == "nesterov") {
			runNesterov(E, S);
		}
		else if (_method == "cg") {
			runCG(E, S);
		}
		else if (_method == "nl_cg") {
			runNLCG(E, S);
		}
		else {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown solver method '%s'") % _method).str()));
		}

		// print results
		printMeanValues();
	}

	bool performLoadstepActions(std::size_t istep)
	{
		if (_write_loadsteps && !_loadstep_filename.empty()) {
			writeVTK<float>((boost::format(_loadstep_filename) % istep).str(), true);
		}

		if (_loadstep_callback && _loadstep_callback()) {
			LOG_COUT << "Loadstep callback break request." << std::endl;
			return true;
		}

		return false;
	}

	typedef struct {
		T t;
		boost::shared_ptr<RealTensor> eps;
	} loadstep_data;

	void extrapolateLoadstep(const std::list<loadstep_data>& last_loadsteps, const RealTensor& F, T t)
	{
		if (_loadstep_extrapolation_method == "polynomial") {
			extrapolateLoadstepPolynomial(last_loadsteps, F, t);
		}
		else if (_loadstep_extrapolation_method == "transformation") {
			extrapolateLoadstepTransformation(last_loadsteps, F, t);
		}
		else {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown loadstep extrapolation method '%s'") %
				_loadstep_extrapolation_method).str()));
		}
	}

	void extrapolateLoadstepPolynomial(const std::list<loadstep_data>& last_loadsteps, const RealTensor& F, T t)
	{
		std::size_t n = last_loadsteps.size();
		ublas::matrix<T> V(n, n); // vandermonde matrix
		ublas::vector<T> tpowers(n);

		typename std::list<loadstep_data>::const_iterator lsd = last_loadsteps.begin();

		for (std::size_t i = 0; i < n; i++) {
			for (std::size_t j = 0; j < n; j++) {
				V(i,j) = std::pow(lsd->t, j);
			}
			tpowers(i) = std::pow(t, i);
			++lsd;
		}


		// compute inverse of E
		ublas::matrix<T> Vinv = ublas::identity_matrix<T>(n);
		int err = lapack::gesv(V, Vinv);
		if (err != 0) {
			BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Error inverting Vandermonde matrix")).str()));
		}

		LOG_COUT << format(Vinv) << std::endl;

		#pragma omp parallel for schedule (static) collapse(2)
		BEGIN_TRIPLE_LOOP(kk, F.nx, F.ny, F.nz, F.nzp)
		{
			for (std::size_t c = 0; c < F.dim; c++)
			{
				ublas::vector<T> f(n); // rhs

				typename std::list<loadstep_data>::const_iterator lsd = last_loadsteps.begin();
				for (std::size_t i = 0; i < n; i++) {
					f(i) = (*lsd->eps)[c][kk];
					++lsd;
				}

				// solve for polynomial coefficients
				ublas::vector<T> p(n);
				p = ublas::prod(Vinv, f);
				F[c][kk] = ublas::inner_prod(tpowers, p);
			}
		}
		END_TRIPLE_LOOP(kk)
	}

	void extrapolateLoadstepTransformation(const std::list<loadstep_data>& last_loadsteps, const RealTensor& F, T t)
	{
		if (last_loadsteps.size() < 2) return;

		#pragma omp parallel for schedule (static) collapse(2)
		BEGIN_TRIPLE_LOOP(kk, F.nx, F.ny, F.nz, F.nzp)
		{
			typename std::list<loadstep_data>::const_reverse_iterator lsd1 = last_loadsteps.rbegin(); ++lsd1;
			typename std::list<loadstep_data>::const_reverse_iterator lsd2 = last_loadsteps.rbegin();

			Tensor3x3<T> F1, F2, F1inv, TR;
			T t1 = lsd1->t;
			T t2 = lsd2->t;
			for (std::size_t i = 0; i < 9; i++) {
				if (i >= F.dim && i >= 3) {
					F1[i] = F1[i-3];
					F2[i] = F2[i-3];
				}
				else {
					F1[i] = (*lsd1->eps)[i][kk];
					F2[i] = (*lsd2->eps)[i][kk];
				}
			}

			F1inv.inv(F1);
			TR.mult(F2, F1inv);

			//T tt = (t - t2)/(t2 - t1);
			//T tt = std::log(t/t2)/std::log(t2/t1);
			T tt = std::log(3.0)/std::log(2.0) - 1.0;

			// compute SVD of TR
			ublas::matrix<T> TRm = ublas::zero_matrix<T>(3), V = ublas::zero_matrix<T>(3), U = ublas::zero_matrix<T>(3);
			ublas::vector<T> s = ublas::zero_vector<T>(3);
			TR.copyTo(TRm);
			// TRm = U S VT
			lapack::gesvd(TRm, s, U, V);
			
			// power TR**tt
			ublas::matrix<T> TRtt = ublas::zero_matrix<T>(3);
			TRtt(0,0) = std::pow(s(0), tt);
			TRtt(1,1) = std::pow(s(1), tt);
			TRtt(2,2) = std::pow(s(2), tt);
			TRtt = ublas::prod(U, TRtt);
			TRtt = ublas::prod(TRtt, ublas::trans(V));

			// compute extrapolation
			Tensor3x3<T> Fi;
			Fi.mult(Tensor3x3<T>(TRtt), F2);

			// assign result
			for (std::size_t i = 0; i < F.dim; i++) {
				F[i][kk] = Fi[i];
			}

			if (kk == 0) {
				LOG_COUT << "F1" << format(F1) << std::endl;
				LOG_COUT << "F2" << format(F2) << std::endl;
				LOG_COUT << "TRm" << format(TRm) << std::endl;
				LOG_COUT << "TRtt" << format(TRtt) << std::endl;
				LOG_COUT << "tt " << tt << std::endl;
				LOG_COUT << "Fi" << format(Fi) << std::endl;
			}
		}
		END_TRIPLE_LOOP(kk)
	}

	// solver with loadstepping
	noinline bool runLoadsteppingSolver(const ublas::vector<T>& Emax, const ublas::vector<T>& Smax)
	{
		std::list<loadstep_data> last_loadsteps;

		// clear errors
		_except.reset();

		std::size_t first_loadstep = (_first_loadstep >= 0) ? ((std::size_t) _first_loadstep) : ((_loadsteps.size() > 2) ? 0 : 1);

		for (std::size_t istep = first_loadstep; istep < _loadsteps.size(); istep++)
		{
			Timer __timer(((boost::format(_YELLOW_TEXT "Loadstep %d of %d" _DEFAULT_TEXT) % istep) % (_loadsteps.size()-1)).str(), true, false);

			T t = _loadsteps[istep].param;
			ublas::vector<T> E = t*Emax;
			ublas::vector<T> S = t*Smax;

			if (_mode == "hyperelasticity") {
				E += (1-t)*Voigt::dyad4(_BC_P, _Id);
			}

			LOG_COUT << "***********************************************" << std::endl;
			LOG_COUT << "loadstep parameter: " << t << std::endl;

			if (_mode == "elasticity")
			{
				LOG_COUT << "prescribed elastic strain: " << E << std::endl;
				LOG_COUT << "prescribed elastic stress: " << S << std::endl;
			}
			else if (_mode == "hyperelasticity")
			{
				LOG_COUT << "prescribed deformation gradient: " << E << std::endl;
				LOG_COUT << "prescribed 1st PK: " << S << std::endl;
			}
			else if (_mode == "viscosity")
			{
				LOG_COUT << "prescribed fluid stress: " << E << std::endl;
				LOG_COUT << "prescribed fluid strain: " << S << std::endl;
			}
			else if (_mode == "heat")
			{
				LOG_COUT << "prescribed temperature gradient : " << E << std::endl;
				LOG_COUT << "prescribed heat flux: " << S << std::endl;
			}
			else if (_mode == "porous")
			{
				LOG_COUT << "prescribed pressure gradient: " << E << std::endl;
				LOG_COUT << "prescribed volumetric flux: " << S << std::endl;
			}

			if (_loadstep_extrapolation_order > 0 && istep > first_loadstep)
			{
				Timer __t("extrapolateLoadstep");

				while (last_loadsteps.size() > _loadstep_extrapolation_order) {
					last_loadsteps.pop_front();
				}
				loadstep_data lsd;
				lsd.t = _loadsteps[istep-1].param;
				lsd.eps.reset(new RealTensor(*_epsilon, 0));
				_epsilon->copyTo(*lsd.eps);
				last_loadsteps.push_back(lsd);

				if (last_loadsteps.size() >= 2) {
					extrapolateLoadstep(last_loadsteps, *_epsilon, t);
				}
			}
			
			try {
				runSolver(E, S);
			}
			catch (boost::exception& e) {
				LOG_CERR << "boost::exception: " << boost::diagnostic_information(e) << std::endl;
				set_exception("runSolver failed");
			}
			catch(...) {
				set_exception("runSolver failed");
#if 0
				if (istep == 0) throw;
				// split the loadstep
				LOG_COUT << "loadstep split: " << S << std::endl;
				Loadstep ls;
				ls.param = 0.5*(_loadsteps[istep].param + _loadsteps[istep - 1].param);

				_loadsteps.insert(_loadsteps.begin() + istep, ls);
				istep --;
				continue;
#endif
			}

			if (_except) {
				LOG_CERR << "Loadsteps canceled because of previous error!" << std::endl;
				return true;
			}

			if (performLoadstepActions(istep)) {
				return true;
			}
		}

		return false;
	}

	//! compute sum over a:b
	noinline T l2_norm(const RealTensor& a) const
	{
		Timer __t("l2_norm", false);

		T s = 0;
		
		if (a.dim == 6) {
			#pragma omp parallel for schedule (static) collapse(2) reduction(+:s)
			BEGIN_TRIPLE_LOOP(k, a.nx, a.ny, a.nz, a.nzp) {
				s += a[0][k]*a[0][k] + a[1][k]*a[1][k] + a[2][k]*a[2][k] + 2*(a[3][k]*a[3][k] + a[4][k]*a[4][k] + a[5][k]*a[5][k]);
			}
			END_TRIPLE_LOOP(k)
		}
		else if (a.dim == 9) {
			#pragma omp parallel for schedule (static) collapse(2) reduction(+:s)
			BEGIN_TRIPLE_LOOP(k, a.nx, a.ny, a.nz, a.nzp) {
				s += a[0][k]*a[0][k] + a[1][k]*a[1][k] + a[2][k]*a[2][k] + a[3][k]*a[3][k] + a[4][k]*a[4][k] + a[5][k]*a[5][k] + a[6][k]*a[6][k] + a[7][k]*a[7][k] + a[8][k]*a[8][k];
			}
			END_TRIPLE_LOOP(k)
		}
		else {
			throw "problem l2_norm";
		}

		return std::sqrt(s*_dx*_dy*_dz/_nxyz);
	}

	//! run the basic algorithm with stain E0 and stress S0
	void runBasic(const ublas::vector<T>& E0, const ublas::vector<T>& S0)
	{
		std::size_t iter = 1;

		RealTensor& epsilon = *_epsilon;
		ComplexTensor& tau = *_tau;
		boost::shared_ptr< ErrorEstimator<T> > ee(create_error_estimator());

#ifdef TEST_MERGE_ISSUE
		for (int i = 0; i < _maxiter; i++) {
			basicScheme(E, epsilon, epsilon, tau, tau, epsilon);
		}
		return;
#endif

		bool update_ref = true;
		ublas::vector<T> E;

//		pRealTensor epsilon_old;
		//if (_basic_relax != 1) {
//			epsilon_old.reset(new RealTensor(epsilon, 0));
		//}
	
		for(;;)
		{
			if (update_ref) {
				calcRefMaterial(_mu_0, _lambda_0, *_epsilon);
				E = calcBCMean(E0, S0);
				//LOG_COUT << "F(0) = " << format(E) << std::endl;
				update_ref = false;
			}

/*
			if (epsilon_old && iter > 1)
			{
				epsilon.copyTo(*epsilon_old);

				T W_old = total_energy(*epsilon_old, S0);
				//bool increase = false;

				for (;;) {
					basicScheme(E, epsilon, epsilon, tau, tau, epsilon);

					T W = total_energy(epsilon, S0);
					T W_rel = (W - W_old)/W_old;
					LOG_COUT << "W_old = " << W_old << ", W = " << W << " rel diff=" << W_rel << std::endl;
					if (W_rel > 1e-3*_tol || std::isnan(std::abs(W_rel))) {
						_mu_0 *= 2;
						LOG_COUT << "increase _mu_0 = " << _mu_0 << std::endl;
						this->setBCProjector(_BC_P);
						E = calcBCMean(E0, S0);
						epsilon_old->copyTo(epsilon);
						increase = true;
						_except.reset();
					}
					else {
#if 0
						if (!increase) {
							_mu_0 *= 0.5;
							LOG_COUT << "decrease _mu_0 = " << _mu_0 << std::endl;
							this->setBCProjector(_BC_P);
							E = calcBCMean(E0, S0);
						}
#endif
						break;
					}
				}
			}
			else {
*/
				basicScheme(E, epsilon, epsilon, tau, tau, epsilon);
//			}


			ee->update();

			// check convergence 
			if (converged(iter, ee->abs_error(), ee->rel_error())) {
				break;
			}


			/*
			if (epsilon_old) {
				epsilon_old->scale(1 - _basic_relax);
				epsilon.xpay(*epsilon_old, _basic_relax, epsilon);
			}
			*/
		}
	}

	//! run the polarization algorithm with strain E0 and stres S0
	void runPolarization(const ublas::vector<T>& E0, const ublas::vector<T>& S0)
	{
		std::size_t iter = 1;

		RealTensor& epsilon = *_epsilon;
		ComplexTensor& tau = *_tau;
		boost::shared_ptr< ErrorEstimator<T> > ee(create_error_estimator());

		// TODO: bc checking disabled, because bc_error does need epsilon and not polarization
		// in general mixed bc need to be done correctly

		bool check_bc = false;
		bool update_ref = false;
		ublas::vector<T> E, P0;

		calcRefMaterial(_mu_0, _lambda_0, *_epsilon);
		E = calcBCMean(E0, S0);

		epsilon.setConstant(4*_mu_0*E);

		// convert epsilon to polarization: eps <- (C + C0) : eps
		//calcStress(_mu_0, _lambda_0, epsilon, epsilon, -1);

		for(;;)
		{
			if (update_ref) {
				calcRefMaterial(_mu_0, _lambda_0, *_epsilon);
				E = calcBCMean(E0, S0);
				//LOG_COUT << "F(0) = " << format(E) << std::endl;
				update_ref = false;
			}

			// compute mean polarization 2*C0:E
			P0 = 4*_mu_0*E;

			polarizationScheme(P0, epsilon, epsilon, tau, tau, epsilon);

			ee->update();

			// check convergence 
			if (converged(iter, ee->abs_error(), ee->rel_error(), check_bc)) {
				break;
			}
		}

		// convert polarization to epsilon

		calcPolarization(_mu_0, 0, epsilon, epsilon, true);
	}

	noinline T calcStep(const RealTensor& epsilon, RealTensor& depsilon) const
	{
		Timer __t("calc step", false);

		const RealTensor* eps = &epsilon;
		const RealTensor* deps = &depsilon;

		/*
		if (this->use_dfg()) {
			// transfer eps to doubly fine grid
			eps = _temp_dfg_1.get();
			ptau = _temp_dfg_1.get();
			prolongate_to_dfg(epsilon, *eps);
			_mat->select_dfg(true);
		}
		*/

		T s1 = 0, s2 = 0;

		#pragma omp parallel for schedule (dynamic) collapse(2) reduction(+:s1,s2)
		for (std::size_t ii = 0; ii < _nx; ii++)
		{
			for (std::size_t jj = 0; jj < _ny; jj++)
			{
				Tensor3x3<T> F, dF, S;

				// calculate current index in tensor epsilon
				std::size_t k = ii*_nyzp + jj*_nzp;
				
				for (std::size_t kk = 0; kk < _nz; kk++)
				{
					eps->assign(k, F);
					deps->assign(k, dF);
					
					_mat->fix_dim(F);
					_mat->fix_dim(dF);

					//LOG_COUT << "F = " << format(F) << std::endl;
					//LOG_COUT << "dF = " << format(dF) << std::endl;

					_mat->PK1(k, dF, 1.0, false, S);
					_mat->fix_dim(S);

					//LOG_COUT << "S = " << format(S) << std::endl;
					
					s1 += F.dot(S);
					s2 += dF.dot(S);

					k++;
				}
			}
		}

		if (s2 == 0) {
			return 0;
		}

		//LOG_COUT << "s1 = " << s1 << std::endl;
		//LOG_COUT << "s2 = " << s2 << std::endl;

		return (-s1/s2);
	}


	//! run the basic scheme + exact line search
	void runBasicEL(const ublas::vector<T>& E0, const ublas::vector<T>& S0)
	{
		std::size_t iter = 1;
		std::size_t n = 0;
		std::size_t n_min = 5;
		T q = 0, q_old = 0;

		pRealTensor pdepsilon(new RealTensor(*_epsilon, 0));
		RealTensor& depsilon = *pdepsilon;
		RealTensor& epsilon = *_epsilon;
		pComplexTensor pdepsilon_hat = depsilon.complex_shadow();
		ComplexTensor& depsilon_hat = *pdepsilon_hat;

		/*
		RealTensor& tau = *_epsilon;
		pRealTensor peps(new RealTensor(*_epsilon, 0));
		RealTensor& epsilon = *peps;

		bool update_ref = true;

		tau.copyTo(epsilon);
		*/

		boost::shared_ptr< ErrorEstimator<T> > ee(create_error_estimator());
		ublas::vector<T> ZERO = ublas::zero_vector<T>(E0.size());

		epsilon.setConstant(E0);
		calcRefMaterial(_mu_0, _lambda_0, epsilon);

		basicScheme(ZERO, epsilon, depsilon, depsilon_hat, depsilon_hat, depsilon);

		for(;;)
		{

			/*
			if (update_ref) {
				calcRefMaterial(_mu_0, _lambda_0, tau);
				E = calcBCMean(E0, S0);
				//LOG_COUT << "F(0) = " << format(E) << std::endl;
				update_ref = false;
			}
			*/

			T alpha = calcStep(epsilon, depsilon);

			LOG_COUT << "alpha = " << alpha << std::endl;

			epsilon.xpay(epsilon, alpha, depsilon);
			basicScheme(ZERO, depsilon, depsilon, depsilon_hat, depsilon_hat, depsilon);


			/*
			n += 1;
			basicScheme(E, tau, tau, *_tau, *_tau, tau);
			epsilon.xpay(tau, -1.0, epsilon);
			q_old = q;
			q = this->l2_norm(tau); q *= q;

			if ((q_old > q) && (n > n_min)) {
				n = 0;
				tau.copyTo(epsilon);
			}
			else {
				epsilon.xpay(tau, (n - 1.0)/(n + 2.0), epsilon);
				// TODO: replace by soft swap
				epsilon.swap(tau);
			}
			*/
			
			ee->update();

			// check convergence 
			if (converged(iter, ee->abs_error(), ee->rel_error())) {
				break;
			}
		}
	}

	//! run the Nesterov's method
	void runNesterov(const ublas::vector<T>& E0, const ublas::vector<T>& S0)
	{
		std::size_t iter = 1;
		std::size_t n = 0;
		std::size_t n_min = 5;
		T q = 0, q_old = 0;

		RealTensor& tau = *_epsilon;
		pRealTensor peps(new RealTensor(*_epsilon, 0));
		RealTensor& epsilon = *peps;
		boost::shared_ptr< ErrorEstimator<T> > ee(create_error_estimator());

		bool update_ref = true;
		ublas::vector<T> E;

		tau.copyTo(epsilon);

		for(;;)
		{
			if (update_ref) {
				calcRefMaterial(_mu_0, _lambda_0, tau);
				E = calcBCMean(E0, S0);
				//LOG_COUT << "F(0) = " << format(E) << std::endl;
				update_ref = false;
			}

			n += 1;
			basicScheme(E, tau, tau, *_tau, *_tau, tau);
			epsilon.xpay(tau, -1.0, epsilon);
			q_old = q;
			q = this->l2_norm(tau); q *= q;

			if ((q_old > q) && (n > n_min)) {
				n = 0;
				tau.copyTo(epsilon);
			}
			else {
				epsilon.xpay(tau, (n - 1.0)/(n + 2.0), epsilon);
				// TODO: replace by soft swap
				epsilon.swap(tau);
			}
			
			ee->update();

			// check convergence 
			if (converged(iter, ee->abs_error(), ee->rel_error())) {
				break;
			}
		}
	}


	//! run the NLCG algorithm
	void runNLCG(const ublas::vector<T>& E, const ublas::vector<T>& S)
	{
		if (_mode == "hyperelasticity")
		{
			runNLCGHyper(E, S);
		}
	}

	//! run the CG algorithm
	void runCG(const ublas::vector<T>& E, const ublas::vector<T>& S)
	{
		if (_mode == "hyperelasticity")
		{
			runCGHyper(E, S);
		}
		else
		{
			runCGElasticity(E, S);
		}
	}

	//! compute W_hat = GRAD_hat q_hat
	noinline void GradOperatorFourierHyper(ComplexTensor& q_hat, ComplexTensor& W_hat)
	{
		Timer __t("GradOperatorFourierHyper", false);

		const T xi0_0 = 2*M_PI/_dx, xi1_0 = 2*M_PI/_dy, xi2_0 = 2*M_PI/_dz;

		const bool nx_even = (_nx & 1) == 0;
		const bool ny_even = (_ny & 1) == 0;
		const bool nz_even = (_nz & 1) == 0;
		const std::size_t ii_half = nx_even ? (_nx/2 - 1) : _nx/2;
		const std::size_t jj_half = ny_even ? (_ny/2 - 1) : _ny/2;
		const std::size_t kk_half = nz_even ? (_nz/2 - 1) : _nz/2;

		const std::complex<T> imag(0, 1);

		#pragma omp parallel for
		for (size_t ii = 0; ii < _nx; ii++)
		{
			const T xi0 = xi0_0*((ii <= ii_half) ? ii : ((T)ii - (T)_nx));

			for (size_t jj = 0; jj < _ny; jj++)
			{
				const T xi1 = xi1_0*((jj <= jj_half) ? jj : ((T)jj - (T)_ny));

				// calculate current index in complex tensor tau[*]
				size_t k = ii*_ny*_nzc + jj*_nzc;

				for (size_t kk = 0; kk < _nzc; kk++)
				{
					const T xi2 = xi2_0*((kk <= kk_half) ? kk : ((T)kk - (T)_nz));
					const std::complex<T> q0 = q_hat[0][k];
					const std::complex<T> q1 = q_hat[1][k];
					const std::complex<T> q2 = q_hat[2][k];
					
					W_hat[0][k] = imag*xi0*q0;
					W_hat[1][k] = imag*xi1*q1;
					W_hat[2][k] = imag*xi2*q2;
					W_hat[3][k] = imag*xi2*q1;
					W_hat[4][k] = imag*xi2*q0;
					W_hat[5][k] = imag*xi1*q0;
					W_hat[6][k] = imag*xi1*q2;
					W_hat[7][k] = imag*xi0*q2;
					W_hat[8][k] = imag*xi0*q1;

					k++;
				}
			}
		}
	}

#if 0
	// compute W_hat = GRAD_hat q_hat
	void GradOperatorFourierHyperII(ComplexTensor& q_hat, ComplexTensor& W_hat)
	{
		const T xi0_0 = 2*M_PI/_nx, xi1_0 = 2*M_PI/_ny, xi2_0 = 2*M_PI/_nz;

		const bool nx_even = (_nx & 1) == 0;
		const bool ny_even = (_ny & 1) == 0;
		const bool nz_even = (_nz & 1) == 0;
		const std::size_t ii_half = nx_even ? (_nx/2 - 1) : _nx/2;
		const std::size_t jj_half = ny_even ? (_ny/2 - 1) : _ny/2;
		const std::size_t kk_half = nz_even ? (_nz/2 - 1) : _nz/2;

		// non-symmetrized version
		const T c10 = alpha/(2*mu_0);
		const T c20 = -alpha/(2*mu_0*(1 + 2*mu_0/lambda_0));	// == -alpha*lambda_0/(2*mu_0*(lambda_0 + 2*mu_0))

		ublas::c_vector<T,3> xi;
		std::complex<T> f1, f2, f3;
		const std::complex<T> imag(0, 1);

		#pragma omp parallel for
		for (size_t ii = 0; ii < _nx; ii++)
		{
			const T xi0 = xi0_0*((ii <= ii_half) ? ii : ((T)ii - (T)_nx));

			for (size_t jj = 0; jj < _ny; jj++)
			{
				const T xi1 = xi1_0*((jj <= jj_half) ? jj : ((T)jj - (T)_ny));

				// calculate current index in complex tensor tau[*]
				size_t k = ii*_ny*_nzc + jj*_nzc;

				for (size_t kk = 0; kk < _nzc; kk++)
				{
					const T xi2 = xi2_0*((kk <= kk_half) ? kk : ((T)kk - (T)_nz));
					
					const T norm_xi2 = xi(0)*xi(0) + xi(1)*xi(1) + xi(2)*xi(2);
					const T c1 = c10/(norm_xi2);
					const T c2 = c20/(norm_xi2*norm_xi2);
					
					// compute Div
					f1 = imag*(xi(0)*tau_hat[0][k] + xi(1)*tau_hat[5][k] + xi(2)*tau_hat[4][k]);
					f2 = imag*(xi(0)*tau_hat[8][k] + xi(1)*tau_hat[1][k] + xi(2)*tau_hat[3][k]);
					f3 = imag*(xi(0)*tau_hat[7][k] + xi(1)*tau_hat[6][k] + xi(2)*tau_hat[2][k]);
					
					// apply G0
					const std::complex<T> q0 = c1*f1 + c2*(xi(0)*xi(0)*f1 + xi(0)*xi(1)*f2 + xi(0)*xi(2)*f3);
					const std::complex<T> q1 = c1*f2 + c2*(xi(1)*xi(0)*f1 + xi(1)*xi(1)*f2 + xi(1)*xi(2)*f3);
					const std::complex<T> q2 = c1*f3 + c2*(xi(2)*xi(0)*f1 + xi(2)*xi(1)*f2 + xi(2)*xi(2)*f3);

					W_hat[0][k] = imag*xi0*q0;
					W_hat[1][k] = imag*xi1*q1;
					W_hat[2][k] = imag*xi2*q2;
					W_hat[3][k] = imag*xi2*q1;
					W_hat[4][k] = imag*xi2*q0;
					W_hat[5][k] = imag*xi1*q0;
					W_hat[6][k] = imag*xi1*q2;
					W_hat[7][k] = imag*xi0*q2;
					W_hat[8][k] = imag*xi0*q1;

					k++;
				}
			}
		}

		W_hat.setConstant(0, ublas::zero_vector<T>(9));


		// DIFFERENT: 
		const T xi0_0 = 1/(_dx), xi1_0 = 1/(_dy), xi2_0 = 1/(_dz);	// constant factor 2*M_PI actually does not matter
		// non-symmetrized version
		const T c10 = alpha/(2*mu_0);
		const T c20 = -alpha/(2*mu_0*(1 + 2*mu_0/lambda_0));	// == -alpha*lambda_0/(2*mu_0*(lambda_0 + 2*mu_0))

		const bool nx_even = (_nx & 1) == 0;
		const bool ny_even = (_ny & 1) == 0;
		const bool nz_even = (_nz & 1) == 0;
		const std::size_t ii_half = nx_even ? (_nx/2 - 1) : _nx/2;
		const std::size_t jj_half = ny_even ? (_ny/2 - 1) : _ny/2;
		const std::size_t kk_half = nz_even ? (_nz/2 - 1) : _nz/2;
		//const std::size_t ii_filt = (_freq_hack && nx_even) ? _nx/2 : _nx;
		//const std::size_t jj_filt = (_freq_hack && ny_even) ? _ny/2 : _ny;
		//const std::size_t kk_filt = (_freq_hack && nz_even) ? _nz/2 : _nz;

		std::complex<T> ey[9];

		// "safe" version, but slow

		std::complex<T> c;
		const ublas::c_matrix<T,3,3> I(ublas::identity_matrix<T>(3));
		ublas::c_vector<T,3> xi;

		// indices for Voigt notation
		const int vi[9] = {0, 1, 2, 1, 0, 0, 2, 2, 1};
		const int vj[9] = {0, 1, 2, 2, 2, 1, 1, 0, 0};

		#pragma omp parallel for private(c, ey, xi)
		for (size_t ii = 0; ii < _nx; ii++)
		{
			xi(0) = xi0_0*((ii <= ii_half) ? ii : ((T)ii - (T)_nx));

			for (size_t jj = 0; jj < _ny; jj++)
			{
				xi(1) = xi1_0*((jj <= jj_half) ? jj : ((T)jj - (T)_ny));

				// calculate current index in complex tensor tau[*]
				size_t k = ii*_ny*_nzc + jj*_nzc;

				for (size_t kk = 0; kk < _nzc; kk++)
				{
					xi(2) = xi2_0*((kk <= kk_half) ? kk : ((T)kk - (T)_nz));

					T norm_xi2 = xi(0)*xi(0) + xi(1)*xi(1) + xi(2)*xi(2);
					T c1 = c10/(norm_xi2);
					T c2 = c20/(norm_xi2*norm_xi2);

					// perform multiplication in Voigt notation
					// ey = -Gamma_0 : tau_hat = -Gamma_0^v*tau_hat^v

					for (size_t i = 0; i < 9; i++)
					{
						for (size_t j = i; j < 9; j++) {
							LOG_COUT << "gamma[" << i << "][" << j << "] =";
							LOG_COUT << " c1*(";
							if (I(vi[j],vi[i]) != 0) LOG_COUT << "+ xi" << vj[j] << "*xi" << vj[i] << "";
							LOG_COUT << ") + c2*xi" << vi[i] << "*xi" << vj[i] << "*xi" << vi[j] << "*xi" << vj[j] << ";" << std::endl;
						}
					}

					for (size_t i = 0; i < 9; i++)
					{
						// sum up the components
						c = 0;
						for (size_t j = 0; j < 9; j++)
						{
							T gamma_ij = c1*(I(vi[i],vi[j])*xi(vj[i])*xi(vj[j]))
								+ c2*(xi(vi[i])*xi(vj[i])*xi(vi[j])*xi(vj[j]));

							c += gamma_ij*tau_hat[j][k];
						}

						ey[i] = c;
					}

					// assign result to eta_hat
					// we do this seperately since eta_hat and tau_hat may point to the same memory
					for (size_t j = 0; j < 9; j++) {
						eta_hat[j][k] = ey[j] + beta*tau_hat[j][k];
					}
					
					k++;
				}
			}
		}

		// set zero component
		eta_hat.setConstant(0, E);

	}
#endif


	//! compute reference material parameters
	noinline void calcRefMaterial(T& mu_0, T& lambda_0, RealTensor& F)
	{
		Timer __t("calc ref material");

		/*
		if (this->use_dfg()) {
			_mat->select_dfg(true);
		}
		*/

#if 0
		_mat->getRefMaterial(F, mu_0, lambda_0);
		mu_0 *= 0.5*_ref_scale;
		lambda_0 *= 0.5*_ref_scale;
#else
		T x;
		_mat->getRefMaterial(F, mu_0, x, _mode == "viscosity", _method == "polarization");
		mu_0 *= 0.5*_ref_scale;
#endif

		LOG_COUT << "adjusting mu_ref=" << mu_0 << ", lambda_ref=" << lambda_0 << std::endl;

		/*
		if (this->use_dfg()) {
			_mat->select_dfg(false);
		}
		*/
	
		// update boundary condition projector, as it depends on the reference material
		this->setBCProjector(_BC_P);
	}

	void ReducedOperator(RealTensor& F, ComplexTensor& q_hat, RealTensor& W, ComplexTensor& W_hat, ComplexTensor& w_hat)
	{
		GradOperatorFourierHyper(q_hat, W_hat);	// q_hat = GRAD_hat q_hat
		fftInvTensor(W_hat, W);		// W = invFFT W_hat
		calcStressDeriv(_mu_0, _lambda_0, F, W, W, 1);	// W = -(dP/dF(F) - C0) : W
		fftTensor(W, W_hat);		// W_hat = FFT(W)
		G0DivOperatorFourierHyper(_mu_0, _lambda_0, W_hat, w_hat, 1);	// w_hat = G0_hat Div_hat W_hat
	}

	//! compute sum_xi |xi|**2 q:conj(r)
	noinline std::complex<T> innerProductHyper(ComplexTensor& q, ComplexTensor& r)
	{
		Timer __t("innerProductHyper", false);

		const T xi0_0 = 2*M_PI/_nx, xi1_0 = 2*M_PI/_ny, xi2_0 = 2*M_PI/_nz;

		const bool nx_even = (_nx & 1) == 0;
		const bool ny_even = (_ny & 1) == 0;
		const bool nz_even = (_nz & 1) == 0;
		const std::size_t ii_half = nx_even ? (_nx/2 - 1) : _nx/2;
		const std::size_t jj_half = ny_even ? (_ny/2 - 1) : _ny/2;
		const std::size_t kk_half = nz_even ? (_nz/2 - 1) : _nz/2;

		T sr = 0, si = 0;

		#pragma omp parallel for reduction(+:sr,si)
		for (size_t ii = 0; ii < _nx; ii++)
		{
			const T xi0 = xi0_0*((ii <= ii_half) ? ii : ((T)ii - (T)_nx));
			const T xi00 = xi0*xi0;

			for (size_t jj = 0; jj < _ny; jj++)
			{
				const T xi1 = xi1_0*((jj <= jj_half) ? jj : ((T)jj - (T)_ny));
				const T xi0011 = xi00 + xi1*xi1;

				// calculate current index in complex tensor tau[*]
				size_t k = ii*_ny*_nzc + jj*_nzc;

				for (size_t kk = 0; kk < _nzc; kk++)
				{
					const T xi2 = xi2_0*((kk <= kk_half) ? kk : ((T)kk - (T)_nz));
					const T norm_xi2 = xi0011 + xi2*xi2;

					std::complex<T> c = norm_xi2*(q[0][k]*std::conj(r[0][k]) + q[1][k]*std::conj(r[1][k]) + q[2][k]*std::conj(r[2][k]));
					sr += c.real();
					si += c.imag();

					k++;
				}
			}
		}

		std::complex<T> s(sr, si);
		s /= _nxyz;
		return s;
	}

	//! compute innerProductHyper(q, q-r)
	noinline std::complex<T> innerProductDiffHyper(ComplexTensor& q, ComplexTensor& r)
	{
		Timer __t("innerProductDiffHyper", false);

		const T xi0_0 = 2*M_PI/_nx, xi1_0 = 2*M_PI/_ny, xi2_0 = 2*M_PI/_nz;

		const bool nx_even = (_nx & 1) == 0;
		const bool ny_even = (_ny & 1) == 0;
		const bool nz_even = (_nz & 1) == 0;
		const std::size_t ii_half = nx_even ? (_nx/2 - 1) : _nx/2;
		const std::size_t jj_half = ny_even ? (_ny/2 - 1) : _ny/2;
		const std::size_t kk_half = nz_even ? (_nz/2 - 1) : _nz/2;

		T sr = 0, si = 0;

		#pragma omp parallel for reduction(+:sr,si)
		for (size_t ii = 0; ii < _nx; ii++)
		{
			const T xi0 = xi0_0*((ii <= ii_half) ? ii : ((T)ii - (T)_nx));
			const T xi00 = xi0*xi0;

			for (size_t jj = 0; jj < _ny; jj++)
			{
				const T xi1 = xi1_0*((jj <= jj_half) ? jj : ((T)jj - (T)_ny));
				const T xi0011 = xi00 + xi1*xi1;

				// calculate current index in complex tensor tau[*]
				size_t k = ii*_ny*_nzc + jj*_nzc;

				for (size_t kk = 0; kk < _nzc; kk++)
				{
					const T xi2 = xi2_0*((kk <= kk_half) ? kk : ((T)kk - (T)_nz));
					const T norm_xi2 = xi0011 + xi2*xi2;

					std::complex<T> c = norm_xi2*(q[0][k]*std::conj(q[0][k]-r[0][k]) + q[1][k]*std::conj(q[1][k]-r[1][k]) + q[2][k]*std::conj(q[2][k]-r[2][k]));
					sr += c.real();
					si += c.imag();

					k++;
				}
			}
		}

		std::complex<T> s(sr, si);
		s /= _nxyz;
		return s;
	}

	T bc_energy(const RealTensor& eps, const ublas::vector<T>& S0) const
	{
		return Voigt::dyad(S0, eps.average());
	}

	T total_energy(const RealTensor& eps, const ublas::vector<T>& S0) const
	{
		return _mat->meanW(eps) - bc_energy(eps, S0);
	}


	//! compute gradient
	void calcGrad(const ublas::vector<T>& E0, const ublas::vector<T>& S0, RealTensor& epsilon, ComplexTensor& grad_hat, RealTensor& grad, T alpha)
	{
		calcStress(0, 0, epsilon, grad, 1);

#if 1
		// assumes P<F> = E0
		ublas::vector<T> E = Voigt::dyad4(_BC_M, S0);
#else
		ublas::vector<T> F0 = epsilon.average();
		ublas::vector<T> E = E0 - F0 + Voigt::dyad4(_BC_M, ublas::vector<T>(S0 - Voigt::dyad4(_BC_QC0, ublas::vector<T>(E0 - F0))));
#endif
		E *= -alpha;
		GammaOperator(E, _mu_0, _lambda_0, grad, grad_hat, grad_hat, grad, alpha);
	}


	T alphaEstimate(const RealTensor& X, const RealTensor& dX, const ublas::vector<T>& S0)
	{
		T alpha = STD_INFINITY(T);
		T Winf = STD_INFINITY(T);

		BEGIN_TRIPLE_LOOP(kk, X.nx, X.ny, X.nz, X.nzp)
		{
			Tensor3x3<T> Xk;
			X.assign(kk, Xk);

			Winf = std::min(Winf, Voigt::dyad(ublas::vector<T>(Xk), S0));
		}
		END_TRIPLE_LOOP(kk)

		BEGIN_TRIPLE_LOOP(kk, dX.nx, dX.ny, dX.nz, dX.nzp)
		{
			Tensor3x3<T> dXk, Xk;
			X.assign(kk, Xk);
			dX.assign(kk, dXk);
			T W = _mat->W(kk, Xk);

			alpha = std::min(alpha, 1/dXk.dot(dXk)*(W - Winf));
		}
		END_TRIPLE_LOOP(kk)

		return alpha;
	}

	//! run the nonlinear CG algorithm
	// https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method
	void runNLCGHyper(const ublas::vector<T>& E0, const ublas::vector<T>& S0)
	{
		T beta = 0;
		std::size_t iter = 0;
		T dX_norm2 = 0;
		T dX_norm2_initial = -1;

		// alloc variables
		RealTensor& X = *_epsilon;
		RealTensor X_old(X, 0);
		RealTensor dX(X, 0), dX_old(X, 0);
		RealTensor s(X, 0), s_old(X, 0);
		pComplexTensor pdX_hat = dX.complex_shadow();
		ComplexTensor& dX_hat = *pdX_hat;

		std::string beta_scheme = _nl_cg_beta_scheme;
		T alpha = _nl_cg_alpha;		// current step size
		T c = _nl_cg_c;		// backtracking parameter 
		T tau = _nl_cg_tau;	// backtracking parameter for alpha reduction

		// satisfy <X> = E0
#if 1
		calcRefMaterial(_mu_0, _lambda_0, *_epsilon);
		basicScheme(calcBCMean(E0, S0), *_epsilon, *_epsilon, *_tau, *_tau, *_epsilon);
#else
		ublas::vector<T> dE = E0 - Voigt::dyad4(_BC_P, X.average());
		X.add(dE);
#endif

		// compute initial total energy
		T W = total_energy(X, S0);

		// compute reference material
		calcRefMaterial(_mu_0, _lambda_0, X);
		
		for (;;)
		{
			// store previous solutions
			T W_old = W;
			T dX_old_norm2 = dX_norm2;
			X.copyTo(X_old);
			s.copyTo(s_old);

			// calculate the steepest direction
			calcGrad(E0, S0, X, dX_hat, dX, -1);

			// compute gradient norm
			dX_norm2 = innerProductL2(dX, dX);
			//T dX_norm = std::sqrt(dX_norm2);
			if (dX_norm2_initial < 0) {
				dX_norm2_initial = dX_norm2 + boost::numeric::bounds<T>::smallest();
			}

	//		LOG_COUT << "iter = " << iter << "\talpha = " << alpha << "\tW_old = " << W_old << "\tdX_norm = " << std::sqrt(dX_norm2) << std::endl;

#if 1
			// check convergence
			T abs_err = std::sqrt(dX_norm2);
			T rel_err = std::sqrt(dX_norm2/dX_norm2_initial);
			if (converged(iter, abs_err, rel_err, false)) {
				break;
			}
#else
			T abs_err = std::sqrt(dX_norm2);
			if (abs_err < _tol) return;
			iter++;
#endif

			// compute \beta_n according to one of the formulas below
			if (iter > 1) {
				if (beta_scheme == "steepest_descent") {
					beta = 0;
				}
				else if (beta_scheme == "polak_ribiere") {
					T dXdX_old = innerProductL2(dX, dX_old);
					if (dXdX_old > 0.2*dX_norm2) {
						beta = 0;
					}
					else {
						beta = (dX_norm2 - dXdX_old) / dX_old_norm2;
					}
				}
				else if (beta_scheme == "fletcher_reeves") {
					beta = dX_norm2 / dX_old_norm2;
				}
				else if (beta_scheme == "hestenes_stiefel") {
					beta = (dX_norm2 - innerProductL2(dX, dX_old)) / innerProductL2(s_old, dX, dX_old);
				}
				else if (beta_scheme == "day_yuan") {
					beta = dX_norm2 / innerProductL2(s_old, dX, dX_old);
				}
				else {
					BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown beta scheme '%s'") % beta_scheme).str()));
				}
			}

			// reset search direction if beta is negative
			beta = std::max((T)0, beta);


/*
			if (iter % 10 == 0) {
				alpha = _nl_cg_alpha;
				beta = 0;
			}
*/


			// update the conjugate direction
			if (beta != 0) {
				s.xpay(dX, beta, s_old);
			}
			else {
				dX.copyTo(s);
			}

			X.xpay(X_old, alpha, s);
			continue;

			// perform line search
			T m = -c*innerProductL2(s, dX);

/*
			if (std::abs(alpha*m) < dX_norm) {
				// reset step
				m = -c*dX_norm;
				dX.copyTo(s);
				beta = 0;
			}
*/

			T alpha_new = 10*alpha;
			T Wmin = STD_INFINITY(T);
			T alpha_min = alpha_new;
			for (int k = 0; k < 50; k++) {
				X.xpay(X_old, alpha_new, s);
				T Wk = total_energy(X, S0);
				if (Wk < Wmin) {
					alpha_min = alpha_new;
					Wmin = Wk;
				}
				else if (!std::isnan(Wk)) {
					break;
				}
				//LOG_COUT << "alpha=" << alpha_new << " W=" << Wk << std::endl;
				alpha_new *= 0.5;
			}

			if (Wmin >= W_old) {
				LOG_COUT << "no descent direction" << std::endl;
				break;
			}

			W = Wmin;
			alpha = alpha_min;
			// perform the step
			X.xpay(X_old, alpha, s);
			

			LOG_COUT << "iter = " << iter << "\talpha = " << alpha << "\tW_old = " << W_old << "\tW = " << W << "\tdW = " << (W-W_old) << "\talpha*m = " << (alpha*m) << "\tdX_norm = " << std::sqrt(dX_norm2) << std::endl;
			continue;


			bool alpha_reduced = false;
			for (;;)
			{
				// perform the step
				X.xpay(X_old, alpha, s);

				// calculate energy (decrease)
				W = total_energy(X, S0);

				T alpha_est = alphaEstimate(X, dX, S0);
				LOG_COUT << "iter = " << iter << "\talpha = " << alpha << "\tW_old = " << W_old << "\tW = " << W << "\tdW = " << (W-W_old) << "\talpha*m = " << (alpha*m) << "\tdX_norm = " << std::sqrt(dX_norm2) << "\talpha_est = " << alpha_est << std::endl;
				if ((W-W_old) < alpha*m) {
					if (!alpha_reduced) {
						// try to increase alpha
						//alpha = std::min(alpha/tau, (T)1);
						//alpha = alpha/tau;
					}
					break;
				}
				
				// clear errors
				_except.reset();

				// decrease step length
				alpha *= tau;
				alpha_reduced = true;

				/*
				if (beta == 0) {
					// decrease step length
					alpha *= tau;
					alpha_reduced = true;
				}
				else {
					// try the descent direction
					dX.copyTo(s);
					m = -c*dX_norm;
					beta = 0;
				}
				*/

				if (alpha <= 1e-20) {
					// no descent direction
					LOG_COUT << "no descent direction" << std::endl;
					break;
					beta = 0;
					alpha = 1;
					s.zero();
					break;
				}
			}
		}
	}

	//! run the CG algorithm
	// NRCGsolver2
	noinline void runCGHyper(const ublas::vector<T>& E0, const ublas::vector<T>& S0)
	{
		Timer __t("runCGHyper", false);

		T small = boost::numeric::bounds<T>::smallest();

		// alloc F
		RealTensor F(*_epsilon, 0);

		// alloc X
		RealTensor X(F, 0);
		pComplexTensor pX_hat = X.complex_shadow();
		ComplexTensor& X_hat = *pX_hat;

		// alloc R
		RealTensor R(F, 0);
		pComplexTensor pR_hat = R.complex_shadow();
		ComplexTensor& R_hat = *pR_hat;

		// alloc Q
		RealTensor Q(F, 0);

		// alloc W
		RealTensor W(F, 0);
		pComplexTensor pW_hat = W.complex_shadow();
		ComplexTensor& W_hat = *pW_hat;


	//	pRealTensor epsilon_old;
	//	epsilon_old.reset(new RealTensor(*_epsilon, 0));

//#define CG_HYPER_PRINT_W_RESIDUAL
#ifdef CG_HYPER_PRINT_W_RESIDUAL
		RealTensor Z(F, 0);
		pComplexTensor pZ_hat = Z.complex_shadow();
		ComplexTensor& Z_hat = *pZ_hat;
#endif



		// satisfy P : <_epsilon> = E
#if 0
		calcRefMaterial(_mu_0, _lambda_0, *_epsilon);
		basicScheme(calcBCMean(E0, S0), *_epsilon, *_epsilon, *_tau, *_tau, *_epsilon);
#else
		ublas::vector<T> dE = E0 - Voigt::dyad4(_BC_P, _epsilon->average());
		_epsilon->add(dE);
#endif

		//ublas::vector<T> E = calcBCMean(E0, S0);	// boundary condition corrected mean value



		boost::shared_ptr< ErrorEstimator<T> > ee_outer(create_error_estimator(_outer_error_estimator));
		std::size_t iter_outer = 0;

		// we need the first residual for convergence check
		T gamma0 = -1;


//		T mu_scale = 1;

		for(;;)
		{
//cg_restart:

			if (gamma0 < 0 || _update_ref == "always") {
				calcRefMaterial(_mu_0, _lambda_0, *_epsilon);
			}

//			_mu_0 *= mu_scale;
//			this->setBCProjector(_BC_P);


			// solve linearized Problem (calculate Newton step using CG)
			{
				Timer __t("CG loop");

				_epsilon->copyTo(F);	// F = current F

				// NOTE: the Lippman-Schwinger equation is:
				// (I + Gamma^0 : (dP/dF(F)-C0)) X = dE - Gamma^0 P(F), i.e. the residual is
				// it follows the initial guess for X
				// X = dE - Gamma^0 P(F)
				// and the residual
				// r := -Gamma^0 : (dP/dF(F)-C0) X

				//LOG_COUT << "E0 " << format(E0) << std::endl;
				//LOG_COUT << "S0 " << format(S0) << std::endl;
				//LOG_COUT << "F0 " << format(F0) << std::endl;

				calcStress(F, X, 1);	// X = P(F)
				//LOG_COUT << "P(F) " << X.average() << std::endl;
				
				// calculate constant part
				
				//ublas::vector<T> P0 = X.average();
				//ublas::vector<T> XX = Voigt::dyad4(_BC_MQ, ublas::vector<T>(P0 - Voigt::dyad4(_BC_QC0, F0)));
				//LOG_COUT << "MQ<P-C0F> " << format(XX) << std::endl;

				//LOG_COUT << "P0 " << format(P0) << std::endl;
				//ublas::vector<T> X0 = Voigt::dyad4(_BC_M, ublas::vector<T>(S0 - P0)) + Voigt::dyad4(_BC_MQ, P0);
				//ublas::vector<T> X0 = Voigt::dyad4(_BC_M, S0);

#if 1
				ublas::vector<T> X0 = Voigt::dyad4(_BC_M, S0);
#else
				ublas::vector<T> F0 = F.average();
				ublas::vector<T> X0 = E0 - F0 + Voigt::dyad4(_BC_M, ublas::vector<T>(S0 - Voigt::dyad4(_BC_QC0, ublas::vector<T>(E0 - F0))));
#endif

//				LOG_COUT << "### X0: " << format(X0) << std::endl;
//				LOG_COUT << "### <X>: " << format(X.average()) << std::endl;
//				LOG_COUT << "### S0: " << format(S0) << std::endl;
//		LOG_COUT << "projection matrix M:" << std::endl;
//		LOG_COUT << format(_BC_M) << std::endl;

				GammaOperator(X0, _mu_0, _lambda_0, X, X_hat, X_hat, X, -1);	// X = -Gamma0 X, <X> = X0
				printTensor("X", X);
				ApplyOperator(F, X, R, R_hat);
				printTensor("R", R);
				R.copyTo(Q);	// Q = R
				T gamma = innerProductL2(R, R) + small; // gamma = <R, R>
				if (gamma0 < 0) gamma0 = gamma;

				//LOG_COUT << "lambda,mu " << _lambda_0 << " " << _mu_0 << std::endl;
//				LOG_COUT << "gamma0 " << gamma << std::endl;
				//LOG_COUT << "R " << R.average() << std::endl;

				//Tensor<T, 9> Fk;
				//R.assign(0, Fk);
				//LOG_COUT << "R0 " << Fk << std::endl;

				boost::shared_ptr< ErrorEstimator<T> > ee(create_error_estimator());
				std::size_t iter = 0;
		
//				T energy_last = STD_INFINITY(T);
//				T energy_initial = total_energy(*_epsilon, S0);


//				T me = _mat->minEig(F, false);
//				LOG_COUT << "minEig: " << me << std::endl;
			//	LOG_COUT << " sym error=" << calcMinEigH() << std::endl;


				for(;;)
				{
					// printTensor("Q", Q);

					ApplyOperator(F, Q, W, W_hat);
					

/*
					W = -(Gamma0 + MQ<>) (dP/dF(F) - C0) : Q
					T alpha = innerProductL2(Q, Q, W) + small;

					alpha = <(I + (Gamma0 + MQ<>) (dP/dF(F) - C0)) : Q, Q>
					LOG_COUT << "Q:Q=" << innerProductL2(Q, Q) << std::endl;
					LOG_COUT << "Q:0=" << innerProductL2(Q, Q, Q) << std::endl;
					LOG_COUT << "Q:Q-W=" << innerProductL2(Q, Q, W) << std::endl;

*/

// check if operator is self adjoint
#if 0
		{
/*
			_lambda_0 = 0;
			_mu_0 = 1.234;
			this->setBCProjector(_BC_P);
*/

			Tensor3x3<T> eye;
			eye.eye();

			/*
			F.random();
			F.scale(0.1);
			F.add(eye);
			*/

			T minDetF = calcMinDetF(F);
			LOG_COUT << "minDetF=" << minDetF << std::endl;

			RealTensor dF(F, 0);
			pComplexTensor pdF_hat = dF.complex_shadow();
			ComplexTensor& dF_hat = *pdF_hat;
			RealTensor dG(F, 0);
			pComplexTensor pdG_hat = dG.complex_shadow();
			ComplexTensor& dG_hat = *pdG_hat;

			Q.copyTo(dF);
			Q.copyTo(dG);
			
	//		dF.random();
	//		dG.random();

	//		GammaOperator(ublas::zero_vector<T>(9), _mu_0, _lambda_0, dF, dF_hat, dF_hat, dF);
	//		GammaOperator(ublas::zero_vector<T>(9), _mu_0, _lambda_0, dG, dG_hat, dG_hat, dG);

			calcStressDeriv(_mu_0, _lambda_0, F, dF, Z, 1);	// Z = (dP/dF(F) - C0) : dF
			T rhs = innerProductL2(Z, dG);

			GammaOperator(ublas::zero_vector<T>(9), _mu_0, _lambda_0, Z, Z_hat, Z_hat, Z, 1);  // Z = (Gamma0 + MQ<>) : Z

			T lhs = 2*_mu_0*innerProductL2(Z, dG);

			LOG_COUT << "lhs=" << lhs << " rhs=" << rhs << std::endl;
			LOG_COUT << "lhs-rhs=" << (lhs-rhs) << std::endl;

			//exit(0);
		}
#endif


					T alpha = innerProductL2(Q, Q, W) + small;



					if (alpha <= 0) {
						set_exception((boost::format("indefinite operator (alpha=%g) canceling CG!") % alpha).str());
						return;
						/*
						throw "indefinite operator";
						
						// fix stuff
						mu_scale *= 2.0;
						goto cg_restart;
						break;
						*/
					}


//					printTensor("F", F);
//					printTensor("X", X);
//					LOG_COUT << "alpha: " << alpha << std::endl;
//					LOG_COUT << "gamma: " << gamma << std::endl;

#ifdef CG_HYPER_PRINT_W_RESIDUAL
					// compute Operator weigthed norm of residual
					ApplyOperator(F, R, Z, Z_hat);
					T r = innerProductL2(R, R, Z);
					LOG_COUT << "w-residual: " << r << std::endl;
					LOG_COUT << "alpha: " << alpha << std::endl;
					LOG_COUT << "gamma: " << gamma << std::endl;
					LOG_COUT << "gamma/alpha: " << (gamma/alpha) << std::endl;
					T me = _mat->minEig(F, false);
					LOG_COUT << "minEig: " << me << std::endl;
					if (alpha < 0) {
						LOG_COUT << "###############################################################################" << std::endl;
					}
#endif

					alpha = gamma/alpha;
					X.xpay(X, alpha, Q);		// X = X + alpha*Q

					//_epsilon->copyTo(*epsilon_old);
					//T W_old = total_energy(*epsilon_old, S0);

//					for (;;)
//					{

						//LOG_COUT << "dF " << format(X.average()) << std::endl;

						//_epsilon->xpay(F, _newton_relax, X);	// next F = current F + dF


					//	T energy_initial = total_energy(*_epsilon, S0);
					//	LOG_COUT << "W=" << energy_initial << " sym error=" << calcMinEigH() << std::endl;

		
#if 0
						T ms = _mat->maxStress(*_epsilon);
						LOG_COUT << "maxStress: " << ms << std::endl;

						if (ms > _write_stress) {
							this->writeVTK<float>((boost::format("y_%d.vtk") % iter).str(), true);
						}
#endif

						/*
						{
							// TODO: this is only required for the sigma error estimator
							_epsilon->xpay(F, relax*_newton_relax, X);	// next F = current F + dF
							//calcRefMaterial(_mu_0, _lambda_0, *_epsilon);


							T minDetF = calcMinDetF(*_epsilon);
							LOG_COUT << "minDetF = " << minDetF << " relax = " << relax << std::endl;

							if (minDetF < 1e-3) {
								relax *= 0.5;
							}
							else {
								break;
							}
						}
						*/


#if 0
						T _W = total_energy(*_epsilon, S0);
						T W_rel = (_W - W_old)/W_old;
						LOG_COUT << "W_old = " << W_old << ", W = " << _W << " rel diff=" << W_rel << std::endl;
						if (W_rel > 1e-3*_tol || std::isnan(std::abs(W_rel))) {
							_mu_0 *= 2;
							LOG_COUT << "increase _mu_0 = " << _mu_0 << std::endl;
							this->setBCProjector(_BC_P);
							//E = calcBCMean(E0, S0);
							epsilon_old->copyTo(*_epsilon);
							//increase = true;
							_except.reset();
							goto cg_restart;
						}
						else {
#if 0
							if (!increase) {
								_mu_0 *= 0.5;
								LOG_COUT << "decrease _mu_0 = " << _mu_0 << std::endl;
								this->setBCProjector(_BC_P);
								E = calcBCMean(E0, S0);
							}
#endif
//							break;
						}
#endif


/*

						// compute current total energy
						T energy = total_energy(*_epsilon, S0);
						LOG_COUT << "W = " << energy << std::endl;

						break;

						if (energy > energy_initial) {
						//	_newton_relax *= 0.5;
						//	LOG_CWARN << "energy increase detected! Relaxing to " << _newton_relax << std::endl;
						//	LOG_CWARN << "energy increase detected, canceling inner CG loop!" << std::endl;
						//	break;
						}
						else {
							energy_last = energy;
							break;
						}

					}
*/
					
					_epsilon->xpay(F, _newton_relax, X);	// next F = current F + dF


					ee->update_cg(gamma, gamma0);

					printTensor("F", F);

					// check convergence
					if (converged(iter, ee->abs_error(), ee->rel_error(), false)) {
						break;
					}


/*
					// r = r - alpha*(p - w)
					if (_cg_reinit > 0 && (iter % _cg_reinit) == 0) {
						LOG_COUT << "cg_reinit" << std::endl;
						// compute residual exactly
						ApplyOperator(F, X, R, R_hat);	// R = -Gamma0 (dP/dF(F) - C0) X
						R.xpay(R, -1.0, X);		// R = -(I + Gamma^0 : (dP/dF(F)-C0)) X
						X.copyTo(Z);
						calcStress(F, Z, 1);	// Z = P(F)
						ublas::vector<T> Z0 = Voigt::dyad4(_BC_M, ublas::vector<T>(S0 - Z.average()));
						GammaOperator(Z0, _mu_0, _lambda_0, Z, Z_hat, Z_hat, Z, -1);	// Z = -Gamma0 Z, <Z> = Z0
						R.xpay(R, 1.0, Z);		// R = dE - Gamma0 P(F) - (I + Gamma^0 : (dP/dF(F)-C0)) X
						R.copyTo(Q);	// Q = R
						gamma = innerProductL2(R, R) + small; // gamma = <R, R>
						continue;
					}
*/



					R.xpaymz(R, -alpha, Q, W);	// R = R - alpha*(Q - W)

					T delta = innerProductL2(R, R) + small;	// delta = <R, R>
					T beta = delta/gamma;		// beta = delta/gamma
					gamma = delta;			// gamma = delta
					Q.xpay(R, beta, Q);		// Q = R + beta*Q
				}
			}

#if 0
			// perform backtracking line search
			T W0 = calcMinDetF(F);
			T alpha = 1.0;
			
			for (;;)
			{
				T W = calcMinDetF(*_epsilon);
				if (W > small && W/W0 > 0.9) {
					break;
				}

				alpha *= 0.5;
				if (alpha < _tol) {
					break;
				}
				
				_epsilon->xpay(F, alpha*_newton_relax, X);	// next F = current F + alpha*dF

				LOG_COUT << "backtracking: alpha=" << alpha << " detF0 = " << W0 << " detF = " << W << std::endl;
			}

			if (alpha < _tol) {
				LOG_CWARN << "no further backtracking possible canceling CG!" << std::endl;
				break;
			}
#endif

			ee_outer->update();

			// check convergence
			if (converged(iter_outer, ee_outer->abs_error(), ee_outer->rel_error())) {
				break;
			}

			//dE *= (1 - _newton_relax);
		}

		//printMeanValues();
	}

	noinline void ApplyOperator(RealTensor& F, RealTensor& Q, RealTensor& W, ComplexTensor& W_hat)
	{
		Timer __t("ApplyOperator", false);

		calcStressDeriv(_mu_0, _lambda_0, F, Q, W, 1);	// W = (dP/dF(F) - C0) : Q

		printTensor("W", W);
		//Tensor<T, 9> Fk;
		//W.assign(0, Fk);
		//LOG_COUT << "W0 " << Fk << std::endl;
		//LOG_COUT << "eps " << format(F.average()) << std::endl;
		//LOG_COUT << "deps " << format(Q.average()) << std::endl;
		//LOG_COUT << "calcStressDeriv " << format(W.average()) << std::endl;
		//LOG_COUT << "lambda,mu " << _lambda_0 << " " << _mu_0 << std::endl;
		GammaOperator(ublas::zero_vector<T>(9), _mu_0, _lambda_0, W, W_hat, W_hat, W, -1);	// W = -Gamma0 W
		printTensor("G0 W", W);
		//W.assign(0, Fk);
		//LOG_COUT << "W0 " << Fk << std::endl;
	}

	//! run the CG algorithm
	noinline void runCGElasticity(const ublas::vector<T>& E0, const ublas::vector<T>& S0)
	{
		Timer __t("runCGElasticity", false);

		T small = boost::numeric::bounds<T>::smallest();
	
		calcRefMaterial(_mu_0, _lambda_0, *_epsilon);
		ublas::vector<T> E = calcBCMean(E0, S0);

		// allocate required tensors

		RealTensor& epsilon = *_epsilon;
		boost::shared_ptr< ErrorEstimator<T> > ee(create_error_estimator());

		RealTensor r(epsilon, 0);
		RealTensor p(epsilon, 0);
		RealTensor w(epsilon, 0);
		pComplexTensor pr_hat = r.complex_shadow();
		pComplexTensor pw_hat = w.complex_shadow();
		ComplexTensor& r_hat = *pr_hat;
		ComplexTensor& w_hat = *pw_hat;

		// NOTE: the Lippman-Schwinger equation is:
		// (I + Gamma^0 : (C-C0)) epsilon = E, i.e. the residual is
		// r := E - epsilon - Gamma^0 : (C-C0) : epsilon
		
		// compute initial residual
		// r = E - epsilon + MinusB(epsilon)
		epsilon.setConstant(E);
		krylovOperator(epsilon, r, r_hat, r_hat, r, r);
		r.adjustResidual(E, epsilon);

		printTensor("r", r);

		// compute gamma = r:r
		T gamma = innerProduct(r, r) + small;
		T gamma0 = gamma;

		//LOG_COUT << "eps " << format(epsilon.average()) << std::endl;
		//LOG_COUT << "lambda,mu " << _lambda_0 << " " << _mu_0 << std::endl;
		//LOG_COUT << "gamma0 " << gamma << std::endl;

		//Tensor<T, 6> Fk;
		//r.assign(0, Fk);
		//LOG_COUT << "R0 " << Fk << std::endl;

		// set p = r
		r.copyTo(p);

		std::size_t iter = 0;

		for(;;)
		{
			// w = MinusB(p)
			krylovOperator(p, w, w_hat, w_hat, w, w);

			T alpha = innerProduct(p, p, w) + small;

			//LOG_COUT << "alpha = " << alpha << std::endl;
			//LOG_COUT << "gamma = " << gamma << std::endl;
			//LOG_COUT << "gamma/alpha = " << (gamma/alpha) << std::endl;

			// alpha = gamma / alpha
			alpha = gamma/alpha;

			// epsilon = epsilon + alpha*p
			epsilon.xpay(epsilon, alpha, p);

			ee->update_cg(gamma, gamma0);

			// check convergence
			if (converged(iter, ee->abs_error(), ee->rel_error())) {
				break;
			}

			// r = r - alpha*(p - w)
			if (_cg_reinit > 0 && (iter % _cg_reinit) == 0) {
				// compute residual exactly
				krylovOperator(epsilon, r, r_hat, r_hat, r, r);
				r.adjustResidual(E, epsilon);
			}
			else {
				r.xpaymz(r, -alpha, p, w);
			}

			T delta = innerProduct(r, r) + small;
			T beta = delta / gamma;
			gamma = delta;

			// p = r + beta*p
			p.xpay(r, beta, p);
		}
	}

	template< typename R >
	void writeScalarVTK(VTKCubeWriter<T>& cw, const T* t, const char* name)
	{		
		cw.template beginWriteField<R>(name);
#ifdef REVERSE_ORDER
		for (std::size_t j = 0; j < _nx; j++) {
			cw.template writeZYSlice<R>(t + j*_nyzp, _nzp - _nz);
		}
#else
		for (std::size_t j = 0; j < _nz; j++) {
			cw.template writeXYSlice<R>(t + j, _nyzp, _nzp);
		}
#endif
	}

	template< typename R >
	void writeTensorVTK(VTKCubeWriter<T>& cw, const RealTensor& t, size_t n, const char** names)
	{		
		for (std::size_t i = 0; i < n; i++) {
			writeScalarVTK<R>(cw, t[i], names[i]);
		}
	}

	template< typename R >
	void writeVectorVTK(VTKCubeWriter<T>& cw, const RealTensor& t, const char* name)
	{
		cw.template beginWriteField<R>(name, VTKCubeWriter<T>::FieldTypes::VECTORS);

		T* adata[3];

#ifdef REVERSE_ORDER
		for (std::size_t j = 0; j < _nx; j++) {
			adata[0] = t[0] + j*_nyzp;
			adata[1] = t[1] + j*_nyzp;
			adata[2] = t[2] + j*_nyzp;
			cw.template writeZYSlice<R>(adata, _nzp - _nz);
		}
#else
		for (std::size_t j = 0; j < _nz; j++) {
			adata[0] = t[0] + j;
			adata[1] = t[1] + j;
			adata[2] = t[2] + j;
			cw.template writeXYSlice<R>(adata, _nyzp, _nzp);
		}
#endif	
	}

	//! write strain fields and phase field to VTK
	template< typename R >
	void writeVTKPhase(const std::string& filename, std::size_t m, bool binary = true)
	{
		VTKCubeWriter<T> cw(filename, binary ? VTKCubeWriter<T>::WriteModes::BINARY : VTKCubeWriter<T>::WriteModes::ASCII,
			_nx, _ny, _nz, _dx, _dy, _dz, _x0[0], _x0[1], _x0[2]);
		
		cw.writeMesh();
		
		cw.template beginWriteField<R>("phi_" + _mat->phases[m]->name);
#ifdef REVERSE_ORDER
		for (std::size_t j = 0; j < _nx; j++) {
			cw.template writeZYSlice<R>(_mat->phases[m]->phi + j*_nyzp, _nzp - _nz);
		}
#else
		for (std::size_t j = 0; j < _nz; j++) {
			cw.template writeXYSlice<R>(_mat->phases[m]->phi + j, _nyzp, _nzp);
		}
#endif
	}
		
	//! write strain fields and phase field to VTK
	template< typename R >
	void writeVTK(const std::string& filename, bool binary = true)
	{
		VTKCubeWriter<T> cw(filename, binary ? VTKCubeWriter<T>::WriteModes::BINARY : VTKCubeWriter<T>::WriteModes::ASCII,
			_nx, _ny, _nz, _dx, _dy, _dz, _x0[0], _x0[1], _x0[2]);
		const char* F_names[] = {"F_11", "F_22", "F_33", "F_23", "F_13", "F_12", "F_32", "F_31", "F_21"};
		const char* epsilon_names[] = {"epsilon_11", "epsilon_22", "epsilon_33", "epsilon_23", "epsilon_13", "epsilon_12", "epsilon_32", "epsilon_31", "epsilon_21"};
		const char* sigma_names[] = {"sigma_11", "sigma_22", "sigma_33", "sigma_23", "sigma_13", "sigma_12", "sigma_32", "sigma_31", "sigma_21"};
		const char* P_names[] = {"P_11", "P_22", "P_33", "P_23", "P_13", "P_12", "P_32", "P_31", "P_21"};
		const char* g_names[] = {"g_11", "g_22", "g_33", "g_23", "g_13", "g_12", "g_32", "g_31", "g_21"};
		//const char* u_names[] = {"u_1", "u_2", "u_3"};
		const char* u_name = "u";
		
		cw.writeMesh();
		
		// write phase field
		for (std::size_t m = 0; m < _mat->phases.size(); m++)
		{
			cw.template beginWriteField<R>("phi_" + _mat->phases[m]->name);
#ifdef REVERSE_ORDER
			for (std::size_t j = 0; j < _nx; j++) {
				cw.template writeZYSlice<R>(_mat->phases[m]->phi + j*_nyzp, _nzp - _nz);
			}
#else
			for (std::size_t j = 0; j < _nz; j++) {
				cw.template writeXYSlice<R>(_mat->phases[m]->phi + j, _nyzp, _nzp);
			}
#endif
		}
		
		RealTensor& epsilon = *_epsilon;
		RealTensor sigma(epsilon, 0);
		pComplexTensor psigma_hat = sigma.complex_shadow();
		ComplexTensor& sigma_hat = *psigma_hat;

		if (_mode == "elasticity")
		{
			// write strain fields
			writeTensorVTK<R>(cw, epsilon, 6, epsilon_names);

			// write stress
			calcStress(epsilon, sigma);
			writeTensorVTK<R>(cw, sigma, 6, sigma_names);

			// write displacements
			calcStressConst(_mu_0, _lambda_0, epsilon, sigma);
			divOperatorStaggered(sigma, sigma);
			G0OperatorStaggered(_mu_0, _lambda_0, sigma, sigma_hat, sigma_hat, sigma, 1.0);
			writeVectorVTK<R>(cw, sigma, u_name);
		}
		else if (_mode == "hyperelasticity")
		{
			// write strain fields
			writeTensorVTK<R>(cw, epsilon, 9, F_names);

			// write stress
			calcStress(epsilon, sigma);
			writeTensorVTK<R>(cw, sigma, 9, P_names);

			// write displacements
			calcStressDiff(epsilon, sigma);
			G0DivOperatorHyper(_mu_0, _lambda_0, sigma, sigma_hat, sigma_hat, sigma, 1.0);
			writeVectorVTK<R>(cw, sigma, u_name);
			//writeTensorVTK<R>(cw, sigma, 3, u_names);

			// write div sigma forces
			calcStressDiff(epsilon, sigma);
			divOperatorStaggeredHyper(sigma, sigma);
			writeVectorVTK<R>(cw, sigma, "f");

			// write result of Gamma Operator
			calcStressDiff(epsilon, sigma);
			GammaOperator(ublas::zero_vector<T>(9.0), _mu_0, _lambda_0, sigma, sigma_hat, sigma_hat, sigma, 1.0);
			writeTensorVTK<R>(cw, sigma, 9, g_names);

			T* detF = sigma[0];
			calcDetF(epsilon, detF);
			writeScalarVTK<R>(cw, detF, "detF");

			#if 0
			T* detC = sigma[0];
			calcDetC(epsilon, detC);
			writeScalarVTK<R>(cw, detC, "detC");
			#endif
		}
		else if (_mode == "viscosity")
		{
			// we work in the dual scheme, so stresses are stored in epsilon!

			// write strain fields fluidity*epsilon
			calcStress(epsilon, sigma);
			writeTensorVTK<R>(cw, sigma, 6, epsilon_names);

			// write stress
			writeTensorVTK<R>(cw, epsilon, 6, sigma_names);

			// write velocities
			// solve div(2*eta_0*eps_E) = div(eta_0*(phi-phi_0)*sigma_E) - grad(p)
			calcStressDiff(epsilon, sigma); // calculate 0.5*(phi-phi_0)*epsilon => sigma
			divOperatorStaggered(sigma, sigma);
			G0OperatorStaggered(1/(4*_mu_0), STD_INFINITY(T), sigma, sigma_hat, sigma_hat, sigma, 1/(2*_mu_0));
			writeVectorVTK<R>(cw, sigma, u_name);

			T* f = sigma[3];
			T* p = sigma[4];

			// write divergence of u
			//divVector(sigma, f, 1.0);
			//writeScalarVTK<R>(cw, f, "div_u");

			// write pressure
			// need to compute laplace(p) = div(div(eta_0*(phi-phi_0)*sigma_E))
			calcStressDiff(epsilon, sigma);	// calculate 0.5*(phi-phi_0)*epsilon => sigma
			divOperatorStaggered(sigma, sigma);
			divVector(sigma, f, 1/(2*_mu_0));	// conatins muliplication with 2*eta0
			poisson_solve(f, p);
			writeScalarVTK<R>(cw, p, "p");
		}
		else if (_mode == "heat" || _mode == "porous")
		{
			// write temperature gradient
			writeTensorVTK<R>(cw, epsilon, 3, epsilon_names);

			// write heat flux field
			calcStress(epsilon, sigma);
			writeTensorVTK<R>(cw, sigma, 3, sigma_names);

			// write temperature field
			calcStressConst(_mu_0, _lambda_0, epsilon, sigma);
			divOperatorStaggeredHeat(sigma, sigma);
			G0OperatorStaggeredHeat(_mu_0, _lambda_0, sigma, sigma_hat, sigma_hat, sigma, 1.0);
			writeScalarVTK<R>(cw, sigma[0], _mode == "heat" ? "T" : "p");
		}
	}

	//! solve the Laplace u = f, mean u = 0
	void poisson_solve(const T* f, T* u)
	{
		std::complex<T>* uc = (std::complex<T>*) u;

		// compute FFT of rhs
		get_fft(1)->forward(f, uc);

		// calculate solution if Fourier domain
		const T xi0_0 = 2.0*M_PI/_nx, xi1_0 = 2.0*M_PI/_ny, xi2_0 = 2.0*M_PI/_nz;
		const T c = 2*(T)_nxyz;
		const T hax = _nx*_nx/(_dx*_dx);
		const T hay = _ny*_ny/(_dy*_dy);
		const T haz = _nz*_nz/(_dz*_dz);

		#pragma omp parallel for schedule (static)
		for (std::size_t ii = 0; ii < _nx; ii++)
		{
			const T xi0 = xi0_0*ii;

			for (std::size_t jj = 0; jj < _ny; jj++)
			{
				const T xi1 = xi1_0*jj;

				// calculate current index in complex tensor tau[*]
				std::size_t k = ii*_ny*_nzc + jj*_nzc;

				for (std::size_t kk = 0; kk < _nzc; kk++)
				{
					const T xi2 = xi2_0*kk;

					uc[k] /= c*(
						hax*(std::cos(xi0) - (T)1) +
						hay*(std::cos(xi1) - (T)1) +
						haz*(std::cos(xi2) - (T)1)
					);

					k++;
				}
			}
		}

		// set zero component
		uc[0] = (T)0;

		// transform solution to spatial domain
		get_fft(1)->backward(uc, u);
	}

	int check_tol(const std::string& test, T r, T tol = 0)
	{
		if (tol == 0) {
			tol = std::sqrt(std::numeric_limits<T>::epsilon());
		}

		if (r > tol || std::isnan(r)) {
#if 0
			BOOST_THROW_EXCEPTION(std::runtime_error(((((boost::format("TEST FAILED: '%s' test exceeds tolerance (%g) by %g%% (residual=%g)") % test) % tol) % (100*(r-tol)/tol)) % r).str()));
#else
			LOG_COUT << RED_TEXT << ((((boost::format("TEST FAILED: '%s' test exceeds tolerance (%g) by %g%% (residual=%g)") % test) % tol) % (100*(r-tol)/tol)) % r).str() << DEFAULT_TEXT << std::endl;
#endif
			// BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Test failed")).str()));
			return 1;
		}

		//LOG_COUT << GREEN_TEXT << "TEST PASSED: " << test << " (residual=" << r << ")" << DEFAULT_TEXT << std::endl;
		return 0;
	}

	T error(T a, T b, T c)
	{
		return std::abs(a - b)/c;
	}

	int test_law(MaterialLaw<T>& law, bool second_deriv = true)
	{
		return 0;

		int nfail = 0;

		T delta = 1e-5;
		T tol = 1e-2;
		T alpha = 3.2345;

		Tensor3x3<T> eps;
		Tensor3x3<T> deps;
		Tensor3x3<T> sigma1;
		Tensor3x3<T> sigma2;
		Tensor3x3<T> eye;

		eye.eye();

#if 0
		// rank one convexity check
		for (int k = 0; k < 10000; k++) {
			ublas::c_vector<T,3> e;
			T r = RandomNormal01<T>::instance().rnd();
			e[0] = r*RandomNormal01<T>::instance().rnd();
			e[1] = r*RandomNormal01<T>::instance().rnd();
			e[2] = r*RandomNormal01<T>::instance().rnd();
			Tensor3x3<T> ee(ublas::matrix<T>(ublas::outer_prod(e, e)));
			Tensor3x3<T> F1; F1.random();
			Tensor3x3<T> F2(ublas::vector<T>(F1 + ee));
			Tensor3x3<T> F12(ublas::vector<T>(F1 + 0.5*ee));

			if (F1.det() <= 0.01 || F2.det() <= 0.01 || F12.det() <= 0.01) {
				continue;
			}

			T W1 = 0.5*(law.W(0, F1) + law.W(0, F2));
			T W2 = law.W(0, F12);

			if (std::isnan(W1) || std::isnan(W2)) {
				LOG_COUT << law.str() << std::endl;
				LOG_COUT << "W1=" << W1 << std::endl;
				LOG_COUT << "W2=" << W2 << std::endl;
				LOG_COUT << format(F1) << " " << F1.det() << std::endl;
				LOG_COUT << format(F2) << " " << F2.det() << std::endl;
				LOG_COUT << format(F12) << " " << F12.det() << std::endl;
				BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("NaN detected")).str()));
			}

			if (W2 - W1 > 1e-9) {
				LOG_COUT << "W1=" << W1 << std::endl;
				LOG_COUT << "W2=" << W2 << std::endl;
				LOG_COUT << "W2-W1=" << (W2-W1) << std::endl;
				LOG_COUT << law.str() << std::endl;
				LOG_COUT << format(F1) << " " << F1.det() << std::endl;
				LOG_COUT << format(F2) << " " << F2.det() << std::endl;
				LOG_COUT << format(F12) << " " << F12.det() << std::endl;
				BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("law not strictly rank-one konvex")).str()));
			}
		}
#endif
		//return;

		for (int i = 0; i < 100; i++)
		{
			if (i == 0) {
				deps.zero();
				deps[0] = 1.0;
				eps.zero();
				eps[0] = eps[1] = eps[2] = 1.0;
			}
	/*		else if (i == 1) {
				deps.zero();
				deps[0] = 1.0;
				eps.zero();
				eps[0] = eps[1] = eps[2] = eps[5] = 1.0;
			} */
			else {
				deps.random();
				do {
					eps.random();
					eps[3] = eps[4] = eps[5] = 0.0;
					eps[6] = eps[7] = eps[8] = 0.0;
				}
				while (eps.det() <= 0.05);
			}

			bool gamma = i > 0;

			T W = law.W(0, eps);
			law.PK1(0, eps, alpha, gamma, sigma1);
			law.PK1_fd(0, eps, alpha, gamma, sigma2, 9, delta);
			T norm_sigma = ublas::norm_2(sigma2);

			//LOG_COUT << "F = " << eps << " dF = " << deps << std::endl;
			//LOG_COUT << "det(F) = " << eps.det() << " W(F) = " << W << std::endl;

			//sigma1.print("P(F)");
			//sigma2.print("P(F) (FD)");

			T max_err = 0;
			for (int i = 0; i < 9; i++) {
				max_err = std::max(max_err, error(sigma1[i], sigma2[i], norm_sigma));
			}
			nfail += check_tol(law.str() + " material derivative", max_err, tol);

			if (second_deriv)
			{
				law.dPK1(0, eps, alpha, gamma, deps, sigma1);
				law.dPK1_fd(0, eps, alpha, gamma, deps, sigma2, 1, 9, delta);
				norm_sigma = ublas::norm_2(sigma2);

				//sigma1.print("dP(F)");
				//sigma2.print("dP(F) (FD)");

				T max_err = 0;
				for (int i = 0; i < 9; i++) {
					max_err = std::max(max_err, error(sigma1[i], sigma2[i], norm_sigma));
				}

				nfail += check_tol(law.str() + " material 2nd derivative", max_err, tol);
				//nfail += check_tol(law.str() + " material 2nd derivative", ublas::norm_2(sigma1 - sigma2)/ublas::norm_2(sigma2), tol);
			}
		}

		return nfail;
	}
	
	//! run test routines
	int run_tests()
	{
		int nfail = 0;
		nfail += this->run_tests_math();
		nfail += this->run_tests_heat();
		nfail += this->run_tests_elasticity();
		nfail += this->run_tests_hyperelasticity();
		return nfail;
	}

	//! run test routines for math operations
	int run_tests_math()
	{
		int nfail = 0;

		{
			SymTensor3x3<T> ep, tau;
			Tensor3x3<T> id;
			tau.random();
			ep.inv(tau);

			id.mult_sym_sym(ep, tau);
			id[0] -= 1;
			id[1] -= 1;
			id[2] -= 1;
			nfail += check_tol("symmetric left inverse", id.dot(id));

			id.mult_sym_sym(tau, ep);
			id[0] -= 1;
			id[1] -= 1;
			id[2] -= 1;
			nfail += check_tol("symmetric right inverse", id.dot(id));
		}

		{
			Tensor3x3<T> ep, tau, id;
			tau.random();
			ep.inv(tau);

			id.mult(ep, tau);
			id[0] -= 1;
			id[1] -= 1;
			id[2] -= 1;
			nfail += check_tol("left inverse", id.dot(id));

			id.mult(tau, ep);
			id[0] -= 1;
			id[1] -= 1;
			id[2] -= 1;
			nfail += check_tol("right inverse", id.dot(id));
		}

		// check matrix calculus
	
		for (std::size_t k = 0; k < 10; k ++)	
		{
			Tensor3<T> n1, n2;
			n1.random(); n1.normalize();
			n2.random(); n2.normalize();

			Tensor3x3<T> R, RRT;
			Tensor3<T> Rn1;
			R.rot(n1, n2);

			RRT.mult_t(R, R);
			RRT[0] -= 1;
			RRT[1] -= 1;
			RRT[2] -= 1;
			nfail += check_tol("vector rotation I", RRT.dot(RRT));

			Rn1.mult(R, n1);
			Rn1 -= n2;
			nfail += check_tol("vector rotation II", Rn1.dot(Rn1));
		}

		{
			SymTensor3x3<T> eps;
			eps.zero();
			eps[0] = 1; eps[1] = 2; eps[2] = 3;

			nfail += check_tol("sym determinant", eps.det() - 6);
		}

		{
			Tensor3x3<T> a, b, c;
			a.zero();
			a[0] = a[1] = a[2] = 1;
			b.random();

			c.mult(a, b);
			c.sub(b);

			nfail += check_tol("mul by identity", c.dot(c));
		}

		{
			Tensor3x3<T> ep;
			ep.zero();
			ep[0] = 1; ep[1] = 2; ep[2] = 3;

			nfail += check_tol("determinant", ep.det() - 6);
		}

		// check box halfspace cutting algorithm
		#if 1
		{
			ublas::c_vector<T, DIM> p = ublas::zero_vector<T>(DIM);
			ublas::c_vector<T, DIM> x = ublas::zero_vector<T>(DIM);
			ublas::c_vector<T, DIM> n = ublas::zero_vector<T>(DIM);
			ublas::c_vector<T, DIM> x0 = ublas::zero_vector<T>(DIM);
			ublas::c_vector<T, DIM> dim = ublas::zero_vector<T>(DIM);
			RandomUniform01<T> rnd;

			for (int k = 0; k < 100; k++)
			{
				for (int i = 0; i < 3; i++) {
					dim[i] = 0.01 + rnd.rnd();
					n[i] = rnd.rnd() - 0.5;
					x0[i] = dim[i]*rnd.rnd();
					x[i] = x0[i] + 0.5*dim[i] + 3*dim[i]*(rnd.rnd() - 0.5);
				}

				n /= ublas::norm_2(n);

				T V1 = halfspace_box_cut_volume<T, DIM>(x, n, x0, dim[0], dim[1], dim[2]);
				T V2 = halfspace_box_cut_volume_old<T, DIM>(x, n, x0, dim[0], dim[1], dim[2]);
				//LOG_COUT << format(x) << " " << format(n) << " " << format(x0) << " " << format(dim) << " " << V1 << " " << V2 << std::endl;

				/*
				int res = 1000
				int nx = (int)(dim[0]*res + 1);
				int ny = (int)(dim[1]*res + 1);
				int nz = (int)(dim[2]*res + 1);
				T V2 = 0;
				T dV = dim[0]*dim[1]*dim[2]/(nx*ny*nz);

				for (int x = 0; x < nx; x++) {
					p[0] = dim[0]*(x + 0.5)/nx;
					for (int y = 0; y < ny; y++) {
						p[1] = dim[1]*(y + 0.5)/ny;
						for (int z = 0; z < nz; z++) {
							p[2] = dim[2]*(z + 0.5)/nz;
							if (ublas::inner_prod(p - x, n) < 0) {
								V2 += dV;
							}
						}
					}
				}
				*/

				nfail += check_tol("halfspace cutting I", V1 - V2);
			}
		}
		#endif

		{
			T dim[3]; dim[0] = 1; dim[1] = 2; dim[2] = 3;
			ublas::c_vector<T, DIM> x = ublas::zero_vector<T>(DIM);
			ublas::c_vector<T, DIM> n = ublas::zero_vector<T>(DIM);
			ublas::c_vector<T, DIM> x0 = ublas::zero_vector<T>(DIM);

			#if 0
			T dx = 0.015625;
			x0[0] = 9.375e-02; x0[1] = 4.062e-01; x0[2] = 4.844e-01;
			n[0] = -9.773e-01; n[1] = -2.108e-01; n[2] = -1.916e-02;
			x[0] = 1.091e-01; x[1] = 4.157e-01; x[2] = 4.923e-01;

			T V = halfspace_box_cut_volume<T, DIM>(x, n, x0, dx, dx, dx);
			LOG_COUT << V << std::endl;

			x -= x0;
			x0 = ublas::zero_vector<T>(DIM);
			V = halfspace_box_cut_volume<T, DIM>(x, n, x0, dx, dx, dx);
			LOG_COUT << V << std::endl;

			x /= dx;
			dx = 1;
			V = halfspace_box_cut_volume<T, DIM>(x, n, x0, dx, dx, dx);
			LOG_COUT << V << std::endl;
			#endif

			for (int k = -10; k < 30; k ++) {
				for (int j = 0; j < 3; j++) {
					T t = dim[j]*k/30.0;
					x = ublas::zero_vector<T>(DIM);
					n = ublas::zero_vector<T>(DIM);
					x0 = ublas::zero_vector<T>(DIM);
					n[j] = 1;
					x0[j] = -t;
					T V = halfspace_box_cut_volume<T, DIM>(x, n, x0, dim[0], dim[1], dim[2]);
					//LOG_COUT << format(x0) << " " << format(n) << " " << V << " " << std::min(std::max((T)0, t), dim[j])*dim[(j+1)%3]*dim[(j+2)%3] << std::endl;
					nfail += check_tol("halfspace cutting II", V - std::min(std::max((T)0, t), dim[j])*dim[(j+1)%3]*dim[(j+2)%3]);
				}
			}
			
			for (int k = -10; k < 30; k ++) {
				for (int j = 0; j < 3; j++) {
					x = ublas::zero_vector<T>(DIM);
					n[0] = n[1] = n[2] = 1/std::sqrt(3);
					x0[0] = x0[1] = x0[2] = -2.0*k/30.0;
					T V1 = halfspace_box_cut_volume<T, DIM>(x, n, x0, dim[0], dim[1], dim[2]);
					x0[j] *= -1;
					n[j] *= -1;
					x[j] += dim[j];
					T V2 = halfspace_box_cut_volume<T, DIM>(x, n, x0, dim[0], dim[1], dim[2]);
					//LOG_COUT << k << " " << j << " " << V1 << " " << V2 << std::endl;
					nfail += check_tol("halfspace cutting III", V1 - V2);
				}
			}
		}

		return nfail;
	}

	//! run test routines for heat equation
	int run_tests_heat()
	{
		int nfail = 0;

		// TODO: perform separate tests for elastic and hyperelastic case
		omp_set_num_threads(1);

		_method = "cg";
		_mode = "heat";
		_gamma_scheme = "collocated";
		//_debug = true;
		_G0_solver = "fft";

		_mat.reset(new VoigtMixedMaterialLaw<T, P, 3>());
		_temp_dfg_1.reset(new RealTensor(2*_nx, 2*_ny, 2*_nz, _mat->dim()));
		_temp_dfg_2.reset(new RealTensor(2*_nx, 2*_ny, 2*_nz, _mat->dim()));
		_epsilon.reset(new RealTensor(_nx, _ny, _nz, _mat->dim()));
		init_fft();
		_tau = _epsilon->complex_shadow();
		this->setBCProjector(Voigt::Id4<T>(_mat->dim()));
		_F0 = ublas::zero_vector<T>(_mat->dim());
		_F00 = ublas::zero_vector<T>(_mat->dim());
		_E = ublas::zero_vector<T>(_mat->dim());
		_Id = ublas::zero_vector<T>(_mat->dim());
		_Id(0) = _Id(1) = _Id(2) = 1;
		_matrix_mat = 0;
		_lambda_0 = 0;
		_mu_0 = 1.0;
		
		pPhase p1(new Phase());

		p1->name = "matrix";
		p1->init(_nx, _ny, _nz, true);

		ScalarLinearIsotropicMaterialLaw<T>* law1 = new ScalarLinearIsotropicMaterialLaw<T>(3);
		law1->mu = 1.56*_mu_0;
		p1->law.reset(law1);

		_mat->add_phase(p1);


		pPhase p2(new Phase());

		p2->name = "fiber";
		p2->init(_nx, _ny, _nz, true);

		ScalarLinearIsotropicMaterialLaw<T>* law2 = new ScalarLinearIsotropicMaterialLaw<T>(3);
		law2->mu = 1.56*_mu_0;
		p2->law.reset(law2);

		_mat->add_phase(p2);

# if 1
		boost::shared_ptr< FiberGenerator<T, DIM> > gen;
		gen.reset(new FiberGenerator<T, DIM>());

		boost::shared_ptr< const Fiber<T, DIM> > fiber;
		ublas::c_vector<T, DIM> c;
		ublas::c_vector<T, DIM> a;
		c[0] = c[1] = c[2] = 0.5;
		a[0] = a[1] = a[2] = 0.5;
		fiber.reset(new CapsuleFiber<T, DIM>(c, a, 0.6, 0.3));
		fiber->set_material(1);
		gen->addFiber(fiber);

		//get_normals();
		initPhi(*gen);
#else
		p->_phi->random();
		initRawPhi();
		//printField("phi", p->phi);
#endif

		RealTensor& F = *_epsilon;

		// checking staggered grid operators for heat
		{
			RealTensor tau(F, 3);
			RealTensor tau_org(F, 3);
			pComplexTensor ptau_hat = tau.complex_shadow();
			ComplexTensor& tau_hat = *ptau_hat;

			tau.random();
			//printTensor("tau", tau, 1);
			GammaOperatorStaggeredHeat(ublas::zero_vector<T>(tau.dim), _mu_0, _lambda_0, tau, tau_hat, tau_hat, tau, 1);
			tau.copyTo(tau_org);
			//printTensor("eps", tau, 3);

			calcStressConst(_mu_0, _lambda_0, tau, tau);
			//printTensor("sigma", tau, 3);
			divOperatorStaggeredHeat(tau, tau);
			//printTensor("divsigma", tau, 1);
			G0OperatorStaggeredHeat(_mu_0, _lambda_0, tau, tau_hat, tau_hat, tau, 1);
			//printTensor("G0divsigma", tau, 1);
			epsOperatorStaggeredHeat(ublas::zero_vector<T>(tau.dim), tau, tau);
			//printTensor("epsG0divsigma", tau, 3);

			//printTensor("tau", tau, 3);
			//printTensor("tau_org", tau_org, 3);

			tau.xpay(tau, -1, tau_org);
			tau.abs();
			//printTensor("tau-tau_org", tau, 3);
			nfail += check_tol("staggered epsG0div identity", ublas::norm_2(tau.max()));
		}

		return nfail;
	}


	//! run test routines for elasticity
	int run_tests_elasticity()
	{
		int nfail = 0;

		// TODO: perform separate tests for elastic and hyperelastic case
		omp_set_num_threads(1);

		_method = "cg";
		_mode = "elasticity";
		_gamma_scheme = "collocated";
		//_debug = true;
		_G0_solver = "fft";

		_mat.reset(new VoigtMixedMaterialLaw<T, P, 6>());
		_temp_dfg_1.reset(new RealTensor(2*_nx, 2*_ny, 2*_nz, _mat->dim()));
		_temp_dfg_2.reset(new RealTensor(2*_nx, 2*_ny, 2*_nz, _mat->dim()));
		_epsilon.reset(new RealTensor(_nx, _ny, _nz, _mat->dim()));
		init_fft();
		_tau = _epsilon->complex_shadow();
		_E = ublas::zero_vector<T>(_mat->dim());
		this->setBCProjector(Voigt::Id4<T>(_mat->dim()));
		_F0 = ublas::zero_vector<T>(_mat->dim());
		_F00 = ublas::zero_vector<T>(_mat->dim());
		_Id = ublas::zero_vector<T>(_mat->dim());
		_Id(0) = _Id(1) = _Id(2) = 1;
		_matrix_mat = 0;
		_lambda_0 = 324.2;
		_mu_0 = 1324.3;
		
		pPhase p1(new Phase());

		p1->name = "matrix";
		p1->init(_nx, _ny, _nz, true);

		LinearIsotropicMaterialLaw<T>* law1 = new LinearIsotropicMaterialLaw<T>();
		law1->lambda = 1.23*_lambda_0;
		law1->mu = 1.56*_mu_0;
		p1->law.reset(law1);

		_mat->add_phase(p1);


		pPhase p2(new Phase());

		p2->name = "fiber";
		p2->init(_nx, _ny, _nz, true);

		LinearIsotropicMaterialLaw<T>* law2 = new LinearIsotropicMaterialLaw<T>();
		law2->lambda = 1.23*_lambda_0;
		law2->mu = 1.56*_mu_0;
		p2->law.reset(law2);

		_mat->add_phase(p2);

# if 1
		boost::shared_ptr< FiberGenerator<T, DIM> > gen;
		gen.reset(new FiberGenerator<T, DIM>());

		boost::shared_ptr< const Fiber<T, DIM> > fiber;
		ublas::c_vector<T, DIM> c;
		ublas::c_vector<T, DIM> a;
		c[0] = c[1] = c[2] = 0.5;
		a[0] = a[1] = a[2] = 0.5;
		fiber.reset(new CapsuleFiber<T, DIM>(c, a, 0.6, 0.3));
		fiber->set_material(1);
		gen->addFiber(fiber);

		get_normals();
		initPhi(*gen);
#else
		p->_phi->random();
		initRawPhi();
//		printField("phi", p->phi);
#endif

		RealTensor& F = *_epsilon;

		// check linear material laws
		{
			T delta = std::sqrt(std::numeric_limits<T>::epsilon());
			
			LinearIsotropicMaterialLaw<T> law;
			law.lambda = 1234.3;
			law.mu = 134.4;
			
			SymTensor3x3<T> eps;
			SymTensor3x3<T> deps;
			SymTensor3x3<T> sigma1;
			SymTensor3x3<T> sigma2;
			SymTensor3x3<T> sigma3;

			eps.random();
			deps.random();
			law.dPK1(0, eps, 1, false, deps, sigma1);
			law.PK1(0, deps, 1, false, sigma2);
			law.dPK1_fd(0, eps, 1, false, deps, sigma3, 1, 6, delta);

		//	LOG_COUT << sigma1 << std::endl;
		//	LOG_COUT << sigma2 << std::endl;
		//	LOG_COUT << sigma3 << std::endl;
		
			nfail += check_tol("isotropic material derivative", ublas::norm_2(sigma1 - sigma2) + ublas::norm_2(sigma1 - sigma3), 12*std::sqrt(delta));
		}

		// checking collocated operators for elasticity
		{
			RealTensor tau(F, 6);
			RealTensor tau_org(F, 6);
			pComplexTensor ptau_hat = tau.complex_shadow();
			ComplexTensor& tau_hat = *ptau_hat;

			tau.random();
			GammaOperatorCollocated(ublas::zero_vector<T>(tau.dim), _mu_0, _lambda_0, tau, tau_hat, tau_hat, tau, 1.0);
			tau.copyTo(tau_org);

			calcStressConst(_mu_0, _lambda_0, tau, tau);
			GammaOperatorCollocated(ublas::zero_vector<T>(tau.dim), _mu_0, _lambda_0, tau, tau_hat, tau_hat, tau, 1.0);

			//printTensor("tau", tau, 1);
			//printTensor("tau_org", tau_org, 1);

			tau.xpay(tau, -1, tau_org);
			tau.abs();
			nfail += check_tol("collocated epsG0div identity", ublas::norm_2(tau.max()));
		}

		// checking WillotR operators for elasticity
		{
			RealTensor tau(F, 6);
			RealTensor tau_org(F, 6);
			pComplexTensor ptau_hat = tau.complex_shadow();
			ComplexTensor& tau_hat = *ptau_hat;

			tau.random();
			GammaOperatorWillotR(ublas::zero_vector<T>(tau.dim), _mu_0, _lambda_0, tau, tau_hat, tau_hat, tau, 1.0);
			tau.copyTo(tau_org);

			calcStressConst(_mu_0, _lambda_0, tau, tau);
			GammaOperatorWillotR(ublas::zero_vector<T>(tau.dim), _mu_0, _lambda_0, tau, tau_hat, tau_hat, tau, 1.0);

			//printTensor("tau", tau, 1);
			//printTensor("tau_org", tau_org, 1);

			tau.xpay(tau, -1, tau_org);
			tau.abs();
			nfail += check_tol("WillotR epsG0div identity", ublas::norm_2(tau.max()));
		}

		// checking staggered grid operators for elasticity
		{
			RealTensor tau(F, 6);
			RealTensor tau_org(F, 6);
			pComplexTensor ptau_hat = tau.complex_shadow();
			ComplexTensor& tau_hat = *ptau_hat;

			tau.random();
			epsOperatorStaggered(ublas::zero_vector<T>(tau.dim), tau, tau);
			tau.copyTo(tau_org);

			calcStressConst(_mu_0, _lambda_0, tau, tau);
			divOperatorStaggered(tau, tau);
			G0OperatorStaggered(_mu_0, _lambda_0, tau, tau_hat, tau_hat, tau, 1);
			epsOperatorStaggered(ublas::zero_vector<T>(tau.dim), tau, tau);

			//printTensor("tau", tau, 1);
			//printTensor("tau_org", tau_org, 1);

			tau.xpay(tau, -1, tau_org);
			tau.abs();
			nfail += check_tol("staggered epsG0div identity", ublas::norm_2(tau.max()));
		}

		// check full staggered grid scheme
		{
			_gamma_scheme = "full_staggered";

			LinearIsotropicMaterialLaw<T>* law = new LinearIsotropicMaterialLaw<T>();
			law->lambda = 1.23*_lambda_0;
			law->mu = 1.56*_mu_0;
			p2->law.reset(law);

			RealTensor tau(F, 6);
			RealTensor tau_org(F, 6);
			pComplexTensor ptau_hat = tau.complex_shadow();
			ComplexTensor& tau_hat = *ptau_hat;

			tau.random();
			epsOperatorStaggered(ublas::zero_vector<T>(tau.dim), tau, tau);
			tau.copyTo(tau_org);

			calcStressConst(_mu_0, _lambda_0, tau, tau);
			divOperatorStaggered(tau, tau);
			G0OperatorStaggered(_mu_0, _lambda_0, tau, tau_hat, tau_hat, tau, 1);
			epsOperatorStaggered(ublas::zero_vector<T>(tau.dim), tau, tau);

			//printTensor("tau", tau, 1);
			//printTensor("tau_org", tau_org, 1);

			tau.xpay(tau, -1, tau_org);
			tau.abs();
			nfail += check_tol("full_staggered epsG0div identity", ublas::norm_2(tau.max()));
		}

		return nfail;
	}

	//! run test routines for hyperelasticity
	int run_tests_hyperelasticity()
	{
		int nfail = 0;

		// TODO: perform separate tests for elastic and hyperelastic case
		omp_set_num_threads(1);

		_method = "cg";
		_mode = "hyperelasticity";
		_gamma_scheme = "collocated";
		//_debug = true;
		_G0_solver = "fft";

		_mat.reset(new VoigtMixedMaterialLaw<T, P, 9>());
		_temp_dfg_1.reset(new RealTensor(2*_nx, 2*_ny, 2*_nz, _mat->dim()));
		_temp_dfg_2.reset(new RealTensor(2*_nx, 2*_ny, 2*_nz, _mat->dim()));
		_epsilon.reset(new RealTensor(_nx, _ny, _nz, _mat->dim()));
		init_fft();
		_tau = _epsilon->complex_shadow();
		this->setBCProjector(Voigt::Id4<T>(_mat->dim()));
		_F0 = ublas::zero_vector<T>(_mat->dim());
		_F00 = ublas::zero_vector<T>(_mat->dim());
		_E = ublas::zero_vector<T>(_mat->dim());
		_Id = ublas::zero_vector<T>(_mat->dim());
		_Id(0) = _Id(1) = _Id(2) = 1;
		_matrix_mat = 0;
		_lambda_0 = 324.2;
		_mu_0 = 1324.3;

		
		pPhase p1(new Phase());

		p1->name = "matrix";
		p1->init(_nx, _ny, _nz, true);

		SaintVenantKirchhoffMaterialLaw<T>* law1 = new SaintVenantKirchhoffMaterialLaw<T>();
		law1->lambda = 1.23*_lambda_0;
		law1->mu = 1.56*_mu_0;
		p1->law.reset(law1);

		_mat->add_phase(p1);


		pPhase p2(new Phase());

		p2->name = "fiber";
		p2->init(_nx, _ny, _nz, true);

		SaintVenantKirchhoffMaterialLaw<T>* law2 = new SaintVenantKirchhoffMaterialLaw<T>();
		law2->lambda = 1.23*_lambda_0;
		law2->mu = 1.56*_mu_0;
		p2->law.reset(law2);

		_mat->add_phase(p2);


# if 1
		boost::shared_ptr< FiberGenerator<T, DIM> > gen;
		gen.reset(new FiberGenerator<T, DIM>());

		boost::shared_ptr< const Fiber<T, DIM> > fiber;
		ublas::c_vector<T, DIM> c;
		ublas::c_vector<T, DIM> a;
		c[0] = c[1] = c[2] = 0.5;
		a[0] = a[1] = a[2] = 0.5;
		fiber.reset(new CapsuleFiber<T, DIM>(c, a, 0.6, 0.3));
		fiber->set_material(1);
		gen->addFiber(fiber);

		get_normals();
		initPhi(*gen);
#else
		p->_phi->random();
		initRawPhi();
//		printField("phi", p->phi);
#endif

		RealTensor& F = *_epsilon;

		// check material laws

		{
			Fiber5GoldbergMaterialLaw<T> law;
			//law.f1 = 234.3;
			//law.f2 = 134.4;
			//law.f3 = 34.5;
			//law.f4 = 47.7;
			
			nfail += test_law(law, true);
		}

		{
			Matrix4GoldbergMaterialLaw<T> law;
			//law.m1 = 234.3;
			//law.m2 = 134.4;
			//law.m3 = 34.5;
			//law.m4 = 47.7;
			
			nfail += test_law(law, true);
		}

		{
			NeoHookeMaterialLaw<T> law;
			law.lambda = 1.0;
			law.mu = 10.0;
			
			nfail += test_law(law);
		}

		{
			Fiber6GoldbergMaterialLaw<T> law;
			//law.f1 = 234.3;
			//law.f2 = 134.4;
			//law.f3 = 34.5;
			//law.f4 = 47.7;
			
			nfail += test_law(law, true);
		}

		{
			Matrix1GoldbergMaterialLaw<T> law;
			//law.m1 = 1234.3;
			//law.m2 = 134.4;
			
			nfail += test_law(law, true);
		}

		{
			Matrix2GoldbergMaterialLaw<T> law;
			//law.m1 = 234.3;
			//law.m2 = 134.4;
			//law.m3 = 34.5;
			//law.m4 = 47.7;
			
			nfail += test_law(law, true);
		}

		{
			Matrix3GoldbergMaterialLaw<T> law;
			//law.m1 = 234.3;
			//law.m2 = 134.4;
			//law.m3 = 34.5;
			//law.m4 = 47.7;
			
			nfail += test_law(law, true);
		}

		{
			Fiber1GoldbergMaterialLaw<T> law;
			//law.f1 = 1234.3;
			//law.f2 = 134.4;
			
			nfail += test_law(law, true);
		}

		{
			Fiber2GoldbergMaterialLaw<T> law;
			//law.f1 = 234.3;
			//law.f2 = 134.4;
			//law.f3 = 34.5;
			
			nfail += test_law(law, true);
		}

		{
			Fiber3GoldbergMaterialLaw<T> law;
			//law.f1 = 234.3;
			//law.f2 = 134.4;
			//law.f3 = 34.5;
			//law.f4 = 47.7;
			
			nfail += test_law(law, true);
		}

		{
			Fiber4GoldbergMaterialLaw<T> law;
			//law.f1 = 234.3;
			//law.f2 = 134.4;
			//law.f3 = 34.5;
			//law.f4 = 47.7;
			
			nfail += test_law(law, true);
		}

		for (int coef = 1; coef < 4; coef++)
		{
			CheckGoldbergMaterialLaw<T> law(coef);
			
			nfail += test_law(law, true);
		}

		// check laminate mixing law
		{
			pRealTensor normals(new RealTensor(1, 1, 1, 3));
			(*normals)[0][0] = 1.0;
			(*normals)[1][0] = 0.0;
			(*normals)[2][0] = 0.0;

			NeoHookeMaterialLaw<T>* law1 = new NeoHookeMaterialLaw<T>();
			NeoHookeMaterialLaw<T>* law2 = new NeoHookeMaterialLaw<T>();
			law1->mu = 3586.94;
			law1->lambda = 3074.52;
			law2->mu = 38.4615;
			law2->lambda = 32.967;

			pPhase p1(new Phase());
			pPhase p2(new Phase());

			p1->name = "matrix";
			p1->init(1, 1, 1, true);
			p1->phi[0] = 0.6;
			p1->law.reset(law1);

			p2->name = "fiber";
			p2->init(1, 1, 1, true);
			p2->phi[0] = 1.0 - p1->phi[0];
			p2->law.reset(law2);

			LaminateMixedMaterialLaw<T, P, 9> law(normals);
			law.add_phase(p1);
			law.add_phase(p2);
			
			nfail += test_law(law, true);
		}

		{
			SaintVenantKirchhoffMaterialLaw<T> law1;
			LinearIsotropicMaterialLaw<T> law2;
			law1.mu = law2.mu = 3;
			law1.lambda = law2.lambda = 5;

			ublas::c_matrix<T,9,9> C;
			Tensor3x3<T> F;
			F.eye();
			
			Tensor3x3<T> G, sigma;
			G.random();
			G[3] = G[6];
			G[4] = G[7];
			G[5] = G[8];
			
			law1.dPK1(0, F, 1, false, G, sigma, 1);
			//LOG_COUT << format(sigma) << std::endl;

			T delta = std::sqrt(std::numeric_limits<T>::epsilon());
			law1.dPK1_fd(0, F, 1, false, G, sigma, 1, 9, delta);
			//LOG_COUT << format(sigma) << std::endl;

			sigma.zero();
			law2.PK1(0, G, 1, false, sigma);
			//LOG_COUT << format(sigma) << std::endl;
		}

		// check material laws
		{
			NeoHooke2MaterialLaw<T> law;
			law.mu = 134.4;
			law.K = 1234.3;
			
			nfail += test_law(law);
		}

		// check material laws
		{
			SaintVenantKirchhoffMaterialLaw<T> law;
			law.lambda = 1234.3;
			law.mu = 134.4;
			
			nfail += test_law(law);
		}

		// checking staggered grid operators for hyperelasticity
		{
			RealTensor tau(F, 9);
			RealTensor tau_org(F, 9);
			pComplexTensor ptau_hat = tau.complex_shadow();
			ComplexTensor& tau_hat = *ptau_hat;

			tau.random();
			epsOperatorStaggeredHyper(ublas::zero_vector<T>(tau.dim), tau, tau);
			tau.copyTo(tau_org);

			calcStressConst(_mu_0, _lambda_0, tau_org, tau);
			divOperatorStaggeredHyper(tau, tau);
			G0OperatorStaggeredHyper(_mu_0, _lambda_0, tau, tau_hat, tau_hat, tau, 1);
			epsOperatorStaggeredHyper(ublas::zero_vector<T>(tau.dim), tau, tau);

			//printTensor("tau", tau, 1);
			//printTensor("tau_org", tau_org, 1);

			tau.xpay(tau, -1, tau_org);
			tau.abs();
			nfail += check_tol("staggered epsG0divHyper identity", ublas::norm_2(tau.max()));

			calcStressConst(_mu_0, _lambda_0, tau_org, tau);
			GammaOperatorStaggeredHyper(ublas::zero_vector<T>(tau.dim), _mu_0, _lambda_0, tau, tau_hat, tau_hat, tau, 1);

			tau.xpay(tau, -1, tau_org);
			tau.abs();
			nfail += check_tol("staggered GammaHyper identity", ublas::norm_2(tau.max()));
		}

		// checking staggered grid dfg operators
		{
			RealTensor c1(F, 9);
			RealTensor c2(F, 9);
			RealTensor f1(*_temp_dfg_1, 9);
			RealTensor f2(*_temp_dfg_1, 9);
		
			c1.random();
			c1.copyTo(c2);
	
			prolongate_to_dfg(c1, f1);
			f1.copyTo(f2);
			restrict_from_dfg(f2, c2);

			//printTensor("c1", c1, 1);
			//printTensor("f1", f1, 1);
			//printTensor("c2", c2, 1);

			//printTensor("c1", c1, 1, 3);
			//printTensor("f1", f1, 1, 3);
			//printTensor("c2", c2, 1, 3);

			c1.xpay(c1, -1, c2);
			c1.abs();
			nfail += check_tol("staggered dfg operator", ublas::norm_2(c1.max()));
		}

		// checking operators for hyperelasticity
		{
			RealTensor u(F, 3);
			pComplexTensor pu_hat = u.complex_shadow();
			ComplexTensor& u_hat = *pu_hat;

			u.random();			// u = random
			//printTensor("u", u);

			RealTensor W(F, 0);
			pComplexTensor pW_hat = W.complex_shadow();
			ComplexTensor& W_hat = *pW_hat;

			fftTensor(u, u_hat);
			//printTensor("u_hat", u);
			GradOperatorFourierHyper(u_hat, W_hat);
			//printTensor("W_hat", W);
			fftInvTensor(W_hat, W);		// W = grad u

			RealTensor W_org(F, 0);
			W.copyTo(W_org);		// W_org = W
			//printTensor("W_org", W_org);

			calcStressConst(_mu_0, _lambda_0, W, W);	// W = C : grad u
			G0DivOperatorHyper(_mu_0, _lambda_0, W, W_hat, u_hat, u, 1);	// u = G0 Div(C : grad u)
			
			//printTensor("u2", u);

			fftTensor(u, u_hat);
			GradOperatorFourierHyper(u_hat, W_hat);
			fftInvTensor(W_hat, W);		// W = -grad G0 Div(C : grad u)

			//printTensor("W", W, 1);		// should be W_org
			//printTensor("W_org", W_org, 1);
			
			W.xpay(W, -1, W_org);
			W.abs();
			nfail += check_tol("G0DivHyper identity", ublas::norm_2(W.max()));
		}

		// checking operators for hyperelasticity
		{
			RealTensor W1(F, 0);
			pComplexTensor pW1_hat = W1.complex_shadow();
			ComplexTensor& W1_hat = *pW1_hat;

			RealTensor W2(F, 0);
			pComplexTensor pW2_hat = W2.complex_shadow();
			ComplexTensor& W2_hat = *pW2_hat;

			W1.random();
			GammaOperatorCollocatedHyper(ublas::zero_vector<T>(F.dim), _mu_0, _lambda_0, W1, W1_hat, W1_hat, W1);	// W1 = -Gamma W1
			W1.copyTo(W2);	// W2 = W1

			calcStressConst(_mu_0, _lambda_0, W2, W2);	// W2 = C : W2
			fftTensor(W2, W2_hat);
			G0DivOperatorFourierHyper(_mu_0, _lambda_0, W2_hat, W2_hat, 1);
			GradOperatorFourierHyper(W2_hat, W2_hat);
			fftInvTensor(W2_hat, W2);	// W2 = -Gamma W2

			//printTensor("W1", W1, 1);
			//printTensor("W2", W2, 1);
			
			W2.xpay(W2, -1, W1);
			W2.abs();
			nfail += check_tol("GammaHyper identity", ublas::norm_2(W2.max()));
		}


		#if 0
		{
			NeoHookeMaterialLaw<T> law1, law2;
			law1.mu = 3586.94;
			law1.lambda = 3074.52;
			law2.mu = 38.4615;
			law2.lambda = 32.967;

			T c1 = 0.987886, c2 = 0.0121138;
			T pn[3] = {6.596e-01,-7.516e-01, 0.000e+00};
			T Fbar[9] = {1.047e+00, 9.844e-01, 1.000e+00, 0.000e+00, 0.000e+00, -1.116e-01, 0.000e+00, 0.000e+00, 3.403e-03}; // det(Fbar) = 1.03131

			boost::shared_ptr< TensorField<T> > normals;
			LaminateMixedMaterialLaw<T,P,9> law(normals);

			Tensor3<T> n(pn);

			omp_set_num_threads(36);
			
			#pragma omp parallel for schedule (static)
			for (std::size_t i = 0; i < 100000000; i++) {
				try {
					Tensor3x3<T> F1, F2;
					law.solve_newton(0, law1, law2, c1, c2, n, Fbar, F1, F2);
				}
				catch(...) {
					#pragma omp critical
					{
					LOG_COUT << omp_get_thread_num() << " " << i << std::endl;
					}
				}
			}

			return;
		}
		#endif

		// newton convergence failure for SVK material example
		#if 0
		{
			SaintVenantKirchhoffMaterialLaw<T> law1, law2;
			law1.mu = 3586.94;
			law1.lambda = 3074.52;
			law2.mu = 38.4615;
			law2.lambda = 32.967;

			T c1 = 0.987886, c2 = 0.0121138;
			T pn[3] = {6.596e-01,-7.516e-01, 0.000e+00};
			T Fbar[9] = {1.047e+00, 9.844e-01, 1.000e+00, 0.000e+00, 0.000e+00, -1.116e-01, 0.000e+00, 0.000e+00, 3.403e-03}; // det(Fbar) = 1.03131

			boost::shared_ptr< TensorField<T> > normals;
			LaminateMixedMaterialLaw<T,P,9> law(normals);

			Tensor3<T> n(pn);
			Tensor3x3<T> F1, F2;
			law.solve_newton(0, law1, law2, c1, c2, n, Fbar, F1, F2);

			/*
			#### newton iter 9
			W 31.27611638006251
			a -0.7233511127312791 1.756484212892713 0
			F1  1.041e+00  1.000e+00  1.000e+00  0.000e+00  0.000e+00 -1.182e-01  0.000e+00  0.000e+00  1.744e-02
			F2  1.518e+00 -3.198e-01  1.000e+00  0.000e+00  0.000e+00  4.255e-01  0.000e+00  0.000e+00 -1.141e+00
			det(F1) 1.043689623197184
			det(F2) 1.229103298583389e-07
			dF1da0:  7.990e-03  0.000e+00 -0.000e+00 -0.000e+00 -0.000e+00  9.105e-03  0.000e+00  0.000e+00  0.000e+00
			dF2da0: -6.516e-01 -0.000e+00  0.000e+00  0.000e+00  0.000e+00 -7.425e-01 -0.000e+00 -0.000e+00 -0.000e+00
			dF1da1:  0.000e+00  9.105e-03 -0.000e+00 -0.000e+00 -0.000e+00  0.000e+00  0.000e+00  0.000e+00  7.990e-03
			dF2da1: -0.000e+00 -7.425e-01  0.000e+00  0.000e+00  0.000e+00 -0.000e+00 -0.000e+00 -0.000e+00 -6.516e-01
			dF1da2:  0.000e+00  0.000e+00 -0.000e+00 -0.000e+00 -0.000e+00  0.000e+00  9.105e-03  7.990e-03  0.000e+00
			dF2da2: -0.000e+00 -0.000e+00  0.000e+00  0.000e+00  0.000e+00 -0.000e+00 -7.425e-01 -6.516e-01 -0.000e+00
			gradient: -1.921e+00  5.518e-01  0.000e+00
			hessian:  4.668e+00  3.702e+00  2.122e+00  0.000e+00  0.000e+00 -1.569e+00
			lambda2: 0.793221013707454
			norm da: 0.4224615542703788
			*/

			return;
		}
		#endif

		#if 0
		// check Neo-Hooke material law
		{
			NeoHookeMaterialLaw<T> law;
			law.lambda = law.mu = 1;
			Tensor3x3<T> E;
			E.eye();
			ublas::c_matrix<T,9,9> C;
			law.dPK1(0, E, 1, false, TensorIdentity<T,9>::Id, C.data(), 9);
			LOG_COUT << format(C) << std::endl;
		}
		#endif

		#if 0
		omp_set_num_threads(1);

		std::size_t nx = atoi(argv[1]);
		std::size_t ny = atoi(argv[2]);
		std::size_t nz = atoi(argv[3]);
		double Lx = atof(argv[4]);
		double Ly = atof(argv[5]);
		double Lz = atof(argv[6]);
		std::size_t nzp = 2*(nz/2+1);
		std::size_t nxyz = nx*ny*nz;
		std::size_t n = nx*ny*nzp;
		boost::shared_ptr< MultiGridLevel<double> > mg(new MultiGridLevel<double>(nx, ny, nz, nzp, Lx, Ly, Lz, false));
		std::vector<double> q(n, 0);
		std::vector<double> b(n, 0);
		std::vector<double> r(n, 0);
		std::vector<double> x(n, 0);

		// init rhs
		double s = 0;
		double R = 0.3*std::sqrt(nx*nx + ny*ny + nz*nz);
		for (std::size_t i = 0; i < nx; i++) {
			for (std::size_t j = 0; j < ny; j++) {
				for (std::size_t k = 0; k < nz; k++) {
					std::size_t kk = i*ny*nzp + j*nzp + k;
					b[kk] = std::sqrt((i+0.5-nx*0.5)*(i+0.5-nx*0.5)
						+ (j+0.5-ny*0.5)*(j+0.5-ny*0.5)
						+ (k+0.5-nz*0.5)*(k+0.5-nz*0.5)) <= R ? 1.0 : -1.0;
					s += b[kk];
				}
			}
		}

		// project out the null space from b
		s /= nxyz;
		for (std::size_t i = 0; i < n; i++) {
			b[i] -= s;
		}


		mg->r = &(r[0]);
		mg->b = &(b[0]);
		mg->x = &(x[0]);

		mg->init_levels(4);

		{
			Timer __t("fft");
			mg->solve_direct_fft(mg->b, mg->x);
		}

		mg->zero(mg->x);
		mg->run_direct(mg->r, mg->b, mg->x, 1e-10);
		#endif

		return nfail;
	}
};



//! fibergen interface (e.g. for interfacing with Python)
class FGI
{
public:
	typedef boost::function<bool()> ConvergenceCallback;
	typedef boost::function<bool()> LoadstepCallback;

	//! see https://stackoverflow.com/questions/827196/virtual-default-destructors-in-c/827205
	virtual ~FGI() {}

	//! reset solver
	virtual void reset() = 0;

	//! run actions in confing path
	virtual int run(const std::string& actions_path) = 0;

	//! cancel a running solver
	virtual void cancel() = 0;

	//! init Lippmann Schwinger solver
	virtual void init_lss() = 0;

	//! init fiber geometry
	virtual void init_fibers() = 0;

	//! init phase fields
	virtual void init_phase() = 0;

	//! get a list of phase names
	virtual std::vector<std::string> get_phase_names() = 0;

	//! get volume fraction for a phase (computed from the phase field)
	virtual double get_volume_fraction(const std::string& field) = 0;

	//! get volume fraction for a phase (computed directly from the geometry)
	virtual double get_real_volume_fraction(const std::string& field) = 0;

	//! get list of residuals for each iteration
	virtual std::vector<double> get_residuals() = 0;

	//! get solution time
	virtual double get_solve_time() = 0;

	//! get number of distance computations for the geometry
	virtual std::size_t get_distance_evals() = 0;

	//! get the effective property (after calc_effective_properties action was performed)
	virtual std::vector< std::vector<double> > get_effective_property() = 0;

	//! get RVE dimensions
	virtual std::vector<double> get_rve_dims() = 0;

	//! get second moment of fiber orientations
	virtual std::vector< std::vector<double> > get_A2() = 0;

	//! get fourth moment of fiber orientations
	virtual std::vector< std::vector< std::vector< std::vector<double> > > > get_A4() = 0;

	//! get mean strain
	virtual std::vector<double> get_mean_strain() = 0;

	//! get mean stress
	virtual std::vector<double> get_mean_stress() = 0;

	//! get mean Cauchy stress
	virtual std::vector<double> get_mean_cauchy_stress() = 0;

	//! get mean energy
	virtual double get_mean_energy() = 0;

	//! set a convergence callback routine (run for testing for convergence)
	virtual void set_convergence_callback(ConvergenceCallback cb) = 0;

	//! set a loadstep callback routine (run each loadstep)
	virtual void set_loadstep_callback(LoadstepCallback cb) = 0;

	//! get a raw data field (phase, strain, stress, pressure, ...)
	virtual void* get_raw_field(const std::string& field, std::vector<void*>& components, size_t& nx, size_t& ny, size_t& nz, size_t& nzp, size_t& elsize) = 0;

	//! free a raw data field component obtained by get_raw_field
	virtual void free_raw_field(void* handle) = 0;

	//! set Python fibergen instance, which can be used (in Python scripts) within a project file as "fg"
	virtual void set_pyfg_instance(PyObject* instance) = 0;

	//! set Python variable, which can be used (in Python scripts/expressions) within a project file
	virtual void set_variable(std::string key, py::object value) = 0;
};


//! Basic implementation of the fibergen interface
template <typename T, typename R, int DIM>
class FG : public FGI
{
protected:
	boost::shared_ptr< ptree::ptree > xml_root;
	boost::shared_ptr< FiberGenerator<T, DIM> > gen;
	boost::shared_ptr< LSSolver<T, T, DIM> > lss;
	bool phase_valid;
	bool solver_valid;
	bool fibers_valid;
	bool raw_phase;
	ublas::matrix<T> Ceff_voigt;
	ConvergenceCallback convergence_callback;
	LoadstepCallback loadstep_callback;
	PyObject* pyfg_instance;

public:
	FG(boost::shared_ptr< ptree::ptree > xml) : xml_root(xml)
	{
		pyfg_instance = NULL;
		reset();
	}

	void set_pyfg_instance(PyObject* instance)
	{
		pyfg_instance = instance;
	}

	void set_variable(std::string key, py::object value)
	{
		PY::instance().add_local(key, value);
	}

	noinline void init_python()
	{
		const ptree::ptree& pt = xml_root->get_child("settings", empty_ptree);
		const ptree::ptree& variables = pt.get_child("variables", empty_ptree);

		// TODO: when to clear locals?
		// commented because set_variable not workin otherwise
		//PY::instance().clear_locals();

		if (pyfg_instance != NULL) {
			py::object fg(py::handle<>(py::borrowed(pyfg_instance)));
			set_variable("fg", fg);
		}

		// set variables
		BOOST_FOREACH(const ptree::ptree::value_type &v, variables)
		{
			// skip comments
			if (v.first == "<xmlcomment>") {
				continue;
			}

			const ptree::ptree& attr = v.second.get_child("<xmlattr>", empty_ptree);

			std::string type = pt_get<std::string>(attr, "type", "str");
			std::string value = pt_get<std::string>(attr, "value", "");
			py::object py_value;

			if (type == "str") {
				py_value = py::object(value);
			}
			else if (type == "int") {
				py_value = py::object(pt_get<long>(attr, "value"));
			}
			else if (type == "float") {
				py_value = py::object(pt_get<double>(attr, "value"));
			}
			else {
				BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown variable type '%s' for %s") % type % v.first).str()));
			}
			
			set_variable(v.first, py_value);

			LOG_COUT << "Variable " << v.first << " = " << value << std::endl;
		}

		// execute all python blocks
		BOOST_FOREACH(const ptree::ptree::value_type &v, pt)
		{
			if (v.first == "python") {
				Timer __t(v.first);
				PY::instance().exec(v.second.data());
			}
		}
	}

	void reset()
	{
		Timer::reset_stats();
		_except.reset();

		lss.reset();
		gen.reset(new FiberGenerator<T, DIM>());
		phase_valid = solver_valid = false;
		fibers_valid = true;
		raw_phase = false;
	}

	std::string get_hash()
	{
		std::stringstream ss;
		write_xml(ss, *xml_root);
		return ss.str();
	}

	bool update_required(std::string& state)
	{
		std::string hash = get_hash();
		if (hash != state) {
			state = hash;
			return true;
		}
		return false;
	}

	bool loadstep_callback_wrap()
	{
		if (!loadstep_callback) {
			return false;
		}

		return loadstep_callback();
	}

	void set_loadstep_callback(LoadstepCallback cb)
	{
		loadstep_callback = cb;
	}


	bool convergence_callback_wrap()
	{
		if (!convergence_callback) {
			return false;
		}

		return convergence_callback();
	}

	void set_convergence_callback(ConvergenceCallback cb)
	{
		convergence_callback = cb;
	}

	void init_lss() 
	{
		if (solver_valid) return;

		// init the solver
		const ptree::ptree& settings = xml_root->get_child("settings", empty_ptree);
		const ptree::ptree& solver = settings.get_child("solver", empty_ptree);
		const ptree::ptree& solver_attr = solver.get_child("<xmlattr>", empty_ptree);

		T dx = pt_get<T>(settings, "dx", 1);
		T dy = pt_get<T>(settings, "dy", 1);
		T dz = pt_get<T>(settings, "dz", 1);

		std::size_t na = pt_get<std::size_t>(solver_attr, "n", 0);
		T mult = pt_get<T>(solver_attr, "mult", 1);
		std::size_t nx = std::max((std::size_t)1, (std::size_t)(pt_get<std::size_t>(solver_attr, "nx", na)*mult));
		std::size_t ny = std::max((std::size_t)1, (std::size_t)(pt_get<std::size_t>(solver_attr, "ny", na)*mult));
		std::size_t nz = std::max((std::size_t)1, (std::size_t)(pt_get<std::size_t>(solver_attr, "nz", na)*mult));

		ublas::c_vector<T, DIM> x0;
		read_vector(settings, x0, "x0", "y0", "z0", (T)0, (T)0, (T)0);

		lss.reset(new LSSolver<T, T, DIM>(nx, ny, nz, dx, dy, dz, x0));
		lss->readSettings(solver);
		lss->setConvergenceCallback(boost::bind(&FG<T,R,DIM>::convergence_callback_wrap, this));
		lss->setLoadstepCallback(boost::bind(&FG<T,R,DIM>::loadstep_callback_wrap, this));
		solver_valid = true;
	}

	void init_fibers() 
	{
		if (fibers_valid) return;
		gen->run();
		fibers_valid = true;
	}

	void init_phase() 
	{
		if (phase_valid) return;
		init_fibers();
		init_lss();
		if (raw_phase) {
			lss->initRawPhi();
		}
		else {
			lss->initPhi(*gen);
		}
		phase_valid = true;
	}

	void* get_raw_field(const std::string& field, std::vector<void*>& components, size_t& nx, size_t& ny, size_t& nz, size_t& nzp, size_t& elsize)
	{
		init_lss();
		return lss->get_raw_field(field, components, nx, ny, nz, nzp, elsize, this->gen);
	}

	void free_raw_field(void* handle)
	{
		init_lss();
		lss->free_raw_field(handle);
	}

	double get_real_volume_fraction(const std::string& field) 
	{
		return gen->getVolumeFraction(lss->getMaterialIndex(field));
	}

	std::vector<std::string> get_phase_names() 
	{
		init_lss();
		return lss->getPhaseNames();
	}

	double get_volume_fraction(const std::string& field) 
	{
		init_lss();
		return lss->getVolumeFraction(field);
	}

	std::vector<double> get_residuals() 
	{
		init_lss();
		return lss->getResiduals();
	}

	double get_solve_time() 
	{
		init_lss();
		return lss->getSolveTime();
	}

	double get_fft_time() 
	{
		init_lss();
		return lss->getFFTTime();
	}

	std::size_t get_distance_evals()
	{
#ifdef TEST_DIST_EVAL
		return g_dist_evals;
#else
		return 0;
#endif
	}
	
	virtual std::vector<double> get_mean_stress()
	{
		init_lss();
		ublas::vector<T> Smean = lss->calcMeanStress();

		std::vector<double> v;
		for (std::size_t i = 0; i < Smean.size(); i++) {
			v.push_back(Smean(i));
		}

		return v;
	}
	
	virtual std::vector<double> get_mean_strain()
	{
		init_lss();
		ublas::vector<T> Smean = lss->calcMeanStrain();

		std::vector<double> v;
		for (std::size_t i = 0; i < Smean.size(); i++) {
			v.push_back(Smean(i));
		}

		return v;
	}
	
	virtual std::vector<double> get_mean_cauchy_stress()
	{
		init_lss();
		ublas::vector<T> Smean = lss->calcMeanCauchyStress();

		std::vector<double> v;
		for (std::size_t i = 0; i < Smean.size(); i++) {
			v.push_back(Smean(i));
		}

		return v;
	}
	
	virtual double get_mean_energy()
	{
		init_lss();
		return lss->calcMeanEnergy();
	}

	template <class ME>
	inline std::vector<std::vector<double> > c_matrix_to_vector(const ublas::matrix_expression<ME>& m)
	{
		std::vector<std::vector<double> > v;
		for (std::size_t i = 0; i < m().size1(); i++) {
			std::vector<double> row;
			for (std::size_t j = 0; j < m().size2(); j++) {
				row.push_back(m()(i,j));
			}
			v.push_back(row);
		}
		return v;
	}

	virtual std::vector<std::vector<double> > get_A2()
	{
		init_fibers();
		ublas::c_matrix<T, DIM, DIM> A2 = gen->getA2();
		return c_matrix_to_vector(A2);
	}

	virtual std::vector<std::vector< std::vector<std::vector<double> > > > get_A4()
	{
		init_fibers();
		ublas::c_matrix<ublas::c_matrix<T, DIM, DIM>, DIM, DIM> A4 = gen->getA4();
		std::vector<std::vector< std::vector<std::vector<double> > > > A4v;

		for (int i = 0; i < DIM; i++) {
			std::vector< std::vector<std::vector<double> > > row;
			for (int j = 0; j < DIM; j++) {
				row.push_back(c_matrix_to_vector(A4(i,j)));
			}
			A4v.push_back(row);
		}

		return A4v;
	}

	virtual std::vector<std::vector<double> > get_effective_property()
	{
		return c_matrix_to_vector(Ceff_voigt);
	}

	virtual std::vector<double> get_rve_dims()
	{
		init_lss();
		return lss->get_rve_dims();
	}

	void cancel()
	{
		set_exception("fibergen canceled");
	}

	int run(const std::string& actions_path)
	{
		reset();

		// init python variables
		init_python();
		struct AfterReturn { ~AfterReturn() { PY::release(); } } ar;

		const ptree::ptree& settings = xml_root->get_child("settings", empty_ptree);

		// set print precision
		int print_precision = pt_get(settings, "print_precision", 4);
		LOG_COUT.precision(print_precision);
		std::cerr.precision(print_precision);

		// init the fiber generator
		gen->readSettings(settings);

		int max_threads = boost::thread::hardware_concurrency();
		//int max_threads = omp_get_max_threads();

		// init OpenMP threads
		int num_threads_omp = pt_get(settings, "num_threads", 1);
		int dynamic_threads_omp = pt_get(settings, "dynamic_threads", 1);

		if (num_threads_omp < 1) {
			num_threads_omp = std::max(1, max_threads + num_threads_omp);
		}
		
		if (num_threads_omp > max_threads) {
#if 0
			BOOST_THROW_EXCEPTION(std::runtime_error(((boost::format("number of threads %d is above system limit of %d threads") % num_threads_omp) % max_threads).str()));
#else
			num_threads_omp = max_threads;
#endif
		}

		omp_set_nested(false);
		omp_set_dynamic(dynamic_threads_omp != 0);
		omp_set_num_threads(num_threads_omp);

		// Init FFTW threads
		int num_threads_fft = pt_get(settings, "fft_threads", (int)-1);

		if (num_threads_fft < 1) {
			num_threads_fft = num_threads_omp;
		}
		if (num_threads_fft > max_threads) {
			BOOST_THROW_EXCEPTION(std::runtime_error(((boost::format("number of threads %d is above system limit of %d threads") % num_threads_fft) % max_threads).str()));
		}
		if (fftw_init_threads() == 0) {
			LOG_CWARN << "could not initialize FFTW threads!" << std::endl;
		}
		fftw_plan_with_nthreads(num_threads_fft);
		
		std::string host = boost::asio::ip::host_name();
		std::string fft_wisdom = pt_get(settings, "fft_wisdom", std::string(getenv("HOME")) + "/.fibergen_fft_wisdom_" + host);
		if (!fft_wisdom.empty()) {
			fftw_import_wisdom_from_filename(fft_wisdom.c_str());
		}

		LOG_COUT << "Current host: " << host << std::endl;
		LOG_COUT << "Current path: " << boost::filesystem::current_path() << std::endl;
		LOG_COUT << "FFTW wisdom: " << fft_wisdom << std::endl;
		LOG_COUT << "Running with dim=" << DIM <<
			", type=" << typeid(T).name() <<
			", result_type=" <<  typeid(R).name() <<
			", num_threads=" << num_threads_omp <<
			", fft_threads=" << num_threads_fft <<
			", max_threads=" << max_threads <<
#ifdef USE_MANY_FFT
			", many_fft" <<
#endif
			std::endl;

		LOG_COUT << "numeric bounds:" <<
			" smallest=" << boost::numeric::bounds<T>::smallest() <<
			" lowest=" << boost::numeric::bounds<T>::lowest() <<
			" highest=" << boost::numeric::bounds<T>::highest() <<
			" eps=" << std::numeric_limits<T>::epsilon() <<
			std::endl;
	
		// perform actions
		int ret = run_actions(settings, actions_path);

		// save fft wisdom
		if (!fft_wisdom.empty()) {
			fftw_export_wisdom_to_filename(fft_wisdom.c_str());
		}

		return ret;
	}

	noinline int run_actions(const ptree::ptree& settings, const std::string& path)
	{
		// read output format
		bool binary = (pt_get<std::string>(settings, "res_format", "binary") == "binary");

		T dx = pt_get<T>(settings, "dx", 1);
		T dy = pt_get<T>(settings, "dy", 1);
		T dz = pt_get<T>(settings, "dz", 1);
		T x0 = pt_get<T>(settings, "x0", 0);
		T y0 = pt_get<T>(settings, "y0", 0);
		T z0 = pt_get<T>(settings, "z0", 0);

		const ptree::ptree& solver = settings.get_child("solver", empty_ptree);
		const ptree::ptree& solver_attr = solver.get_child("<xmlattr>", empty_ptree);
		std::size_t _na = pt_get<std::size_t>(solver_attr, "n", 0);
		T _mult = pt_get<T>(solver_attr, "mult", 1);
		std::size_t _nx = std::max((std::size_t)1, (std::size_t)(pt_get<std::size_t>(solver_attr, "nx", _na)*_mult));
		std::size_t _ny = std::max((std::size_t)1, (std::size_t)(pt_get<std::size_t>(solver_attr, "ny", _na)*_mult));
		std::size_t _nz = std::max((std::size_t)1, (std::size_t)(pt_get<std::size_t>(solver_attr, "nz", _na)*_mult));

		const ptree::ptree& actions = settings.get_child(path, empty_ptree);
		const ptree::ptree& actions_attr = actions.get_child("<xmlattr>", empty_ptree);

		if (pt_get(actions_attr, "skip", 0) != 0) {
			LOG_COUT << "skipping action: " << path << std::endl;
			return 0;
		}

		BOOST_FOREACH(const ptree::ptree::value_type &v, actions)
		{
			// check if last action failed
			if (_except) {
				return EXIT_FAILURE;
			}

			// skip comments
			if (v.first == "<xmlcomment>") {
				continue;
			}

			const ptree::ptree& attr = v.second.get_child("<xmlattr>", empty_ptree);

			if (v.first == "skip" || pt_get(attr, "skip", false)) {
				continue;
			}
			
			Timer __t(v.first, true, false);

			if (boost::starts_with(v.first, "group-"))
			{
				int ret = run_actions(settings, path + "." + v.first);
				if (ret != 0) return ret;
				continue;
			}

			if (v.first == "write_png")
			{
				ublas::c_vector<T, DIM> a0;
				ublas::c_vector<T, DIM> a1;
				ublas::c_vector<T, DIM> a2;

				read_vector(attr, a0, "a0x", "a0y", "a0z", (T)0, (T)0, (T)0);
				read_vector(attr, a1, "a1x", "a1y", "a1z", (T)1, (T)0, (T)0);
				read_vector(attr, a2, "a2x", "a2y", "a2z", (T)0, (T)1, (T)0);

				std::size_t h = pt_get<std::size_t>(attr, "h", _ny);
				std::size_t w = pt_get<std::size_t>(attr, "w", _nx);
				T exponent = pt_get<T>(attr, "exponent", 1);
				T scale = pt_get<T>(attr, "scale", 1);
				T offset = pt_get<T>(attr, "offset", 0);
				std::string filename = pt_get<std::string>(attr, "filename");
				bool fast = pt_get(attr, "fast", false);

				init_fibers();
				LOG_COUT << "writing distance map: " << filename << std::endl;
				gen->writeDistanceMap(filename, a0, a1, a2, h, w, offset, scale, exponent, fast);
			}
			else if (v.first == "write_lss_vtk")
			{
				std::string filename = pt_get<std::string>(attr, "filename");
				init_lss();
				init_phase();
				lss->template writeVTK<R>(filename, binary);
			}
			else if (v.first == "write_vtk")
			{
				std::size_t n = pt_get<std::size_t>(attr, "n", 0);
				std::size_t nx = pt_get<std::size_t>(attr, "nx", (n > 0) ? n : _nx);
				std::size_t ny = pt_get<std::size_t>(attr, "ny", (n > 0) ? n : _ny);
				std::size_t nz = pt_get<std::size_t>(attr, "nz", (n > 0) ? n : _nz);
				std::string filename = pt_get<std::string>(attr, "filename");
				bool fast = pt_get(attr, "fast", false);
				bool distance = pt_get(attr, "distance", true);
				bool normals = pt_get(attr, "normals", true);
				bool orientation = pt_get(attr, "orientation", true);
				bool fiber_id = pt_get(attr, "fiber_id", true);
				bool material_id = pt_get(attr, "material_id", true);

				init_fibers();
				LOG_COUT << "writing vtk file: " << filename << std::endl;
				gen->template writeVTK<R>(filename, nx, ny, nz, fast, distance, normals, orientation,
					fiber_id, material_id, -1, binary);
			}
			else if (v.first == "write_fo_data" || v.first == "write_fiber_data")
			{
				std::string filename = pt_get<std::string>(attr, "filename");

				init_fibers();
				LOG_COUT << "writing fiber data file: " << filename << std::endl;
				gen->template writeData(filename);
			}
			else if (v.first == "write_voxel_data")
			{
				std::string filename = pt_get<std::string>(attr, "filename");

				init_lss();
				init_phase();
				LOG_COUT << "writing voxel data file: " << filename << std::endl;
				lss->template writeData(filename);
			}
			else if (v.first == "write_vtk_phase")
			{
				std::string outfile = pt_get<std::string>(attr, "outfile");
				bool binary = (pt_get<std::string>(settings, "res_format", "binary") == "binary");
				std::string name = pt_get<std::string>(attr, "name");

				init_lss();
				init_phase();

				lss->template writeVTKPhase<R>(outfile, lss->getMaterialId(name), binary);
			}
			else if (v.first == "write_vtk2")
			{
				std::string outfile = pt_get<std::string>(attr, "outfile");
				bool binary = (pt_get<std::string>(settings, "res_format", "binary") == "binary");

				init_lss();
				init_phase();

				// write strains to disk
				lss->template writeVTK<R>(outfile, binary);
			}
			else if (v.first == "write_pvpy")
			{
				std::string filename = pt_get<std::string>(attr, "filename");
				bool bbox = pt_get<bool>(attr, "bbox", true);
				bool fibers = pt_get<bool>(attr, "fibers", true);
				bool clusters = pt_get<bool>(attr, "clusters", true);

				init_fibers();
				LOG_COUT << "writing paraview py file: " << filename << std::endl;
				gen->template writePVPy(filename, bbox, fibers, clusters);
			}
			else if (v.first == "write_raw_data")
			{
				std::string filename = pt_get<std::string>(attr, "filename");
				std::string dtype = pt_get<std::string>(attr, "dtype", "uint8");
				std::string material = pt_get<std::string>(attr, "material", "");
				bool col_order = pt_get<std::string>(attr, "order", "col") == "col";
				bool compressed = boost::algorithm::ends_with(filename, ".gz");
				std::ofstream stream(filename.c_str());

				init_lss();

				std::size_t material_index = lss->getMaterialIndex(material);

				// stream.exceptions(std::ios::failbit);
				if (stream.fail()) {
					BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Error writing file '%s': %s") % filename % strerror(errno)).str()));
				}

				init_phase();

				T scale = 1;

				if (dtype == "uint8") {
					scale = pt_get<T>(attr, "scale", 0.9999 + (T)0xff);
					lss->template writeRawPhase<uint8_t>(filename, material_index, stream, scale, col_order, compressed);
				}
				else if (dtype == "uint16") {
					scale = pt_get<T>(attr, "scale", 0.9999 + (T)0xffff);
					lss->template writeRawPhase<uint16_t>(filename, material_index, stream, scale, col_order, compressed);
				}
				else if (dtype == "uint32") {
					scale = pt_get<T>(attr, "scale", 0.9999 + (T)0xffffffff);
					lss->template writeRawPhase<uint32_t>(filename, material_index, stream, scale, col_order, compressed);
				}
				else if (dtype == "float") {
					scale = pt_get<T>(attr, "scale", (T)1);
					lss->template writeRawPhase<float>(filename, material_index, stream, scale, col_order, compressed);
				}
				else if (dtype == "double") {
					scale = pt_get<T>(attr, "scale", (T)1);
					lss->template writeRawPhase<double>(filename, material_index, stream, scale, col_order, compressed);
				}
				else {
					BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown data type '%s'") % dtype).str()));
				}
			}
			else if (v.first == "read_raw_data")
			{
				std::size_t n = pt_get<std::size_t>(attr, "n", 0);
				std::size_t nx = pt_get<std::size_t>(attr, "nx", (n > 0) ? n : _nx);
				std::size_t ny = pt_get<std::size_t>(attr, "ny", (n > 0) ? n : _ny);
				std::size_t nz = pt_get<std::size_t>(attr, "nz", (n > 0) ? n : _nz);
				std::string filename = pt_get<std::string>(attr, "filename");
				std::string dtype = pt_get<std::string>(attr, "dtype", "uint8");
				T treshold = pt_get<T>(attr, "treshold", (T)-1);
				bool col_order = pt_get<std::string>(attr, "order", "col") == "col";
				bool compressed = boost::algorithm::ends_with(filename, ".gz");
				std::size_t header_bytes = pt_get<std::size_t>(attr, "header_bytes", 0);
				std::ifstream stream(filename.c_str());

				// stream.exceptions(std::ios::failbit);
				if (stream.fail()) {
					BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Error reading file '%s': %s") % filename % strerror(errno)).str()));
				}

				init_lss();

				if (!raw_phase) {
					// reading the first raw material, set all phases to 1
					lss->setPhasesOne();
					raw_phase = true;
				}

				TensorField<T> t(nx, ny, nz, 1);
				T scale = 1;

				if (dtype == "uint8") {
					scale = pt_get<T>(attr, "scale", 1/(T)0xff);
					lss->template readRawPhase<uint8_t>(t, filename, stream, scale, col_order, compressed, header_bytes, treshold);
				}
				else if (dtype == "uint16") {
					scale = pt_get<T>(attr, "scale", 1/(T)0xffff);
					lss->template readRawPhase<uint16_t>(t, filename, stream, scale, col_order, compressed, header_bytes, treshold);
				}
				else if (dtype == "uint32") {
					scale = pt_get<T>(attr, "scale", 1/(T)0xffffffff);
					lss->template readRawPhase<uint32_t>(t, filename, stream, scale, col_order, compressed, header_bytes, treshold);
				}
				else if (dtype == "float") {
					scale = pt_get<T>(attr, "scale", (T)1);
					lss->template readRawPhase<float>(t, filename, stream, scale, col_order, compressed, header_bytes, treshold);
				}
				else if (dtype == "double") {
					scale = pt_get<T>(attr, "scale", (T)1);
					lss->template readRawPhase<double>(t, filename, stream, scale, col_order, compressed, header_bytes, treshold);
				}
				else {
					BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown data type '%s'") % dtype).str()));
				}

				std::map<std::size_t, std::size_t> value_to_material_map;

				BOOST_FOREACH(const ptree::ptree::value_type &a, attr)
				{
					if (boost::starts_with(a.first, "material_")) {
						std::string key = a.first;
						boost::replace_all(key, "material_", "");
						std::size_t value = boost::lexical_cast<std::size_t>(key);
						std::size_t material = lss->getMaterialIndex(a.second.data());
						value_to_material_map.insert(std::pair<std::size_t, std::size_t>(value, material));
					}
					else if (a.first == "material") {
						std::size_t material = lss->getMaterialIndex(a.second.data());
						value_to_material_map.clear();
						lss->initMultiphase(t, 0, value_to_material_map, material);
						break;
					}
					else {
						continue;
					}
				}

				if (value_to_material_map.size() > 0) {
					lss->initMultiphase(t, scale, value_to_material_map, 0);
				}
			}
			else if (v.first == "init_phase")
			{
				bool normals = pt_get<bool>(attr, "normals", false);
				bool orientations = pt_get<bool>(attr, "orientations", false);
				init_lss();
				if (normals) lss->get_normals();
				if (orientations) lss->get_orientation();
				init_phase();
			}
			else if (v.first == "generate_fibers")
			{
				std::string intersecting_materials_str = pt_get<std::string>(attr, "intersecting_materials", "");
				int intersecting_materials = -1; // check all materials for intersection if intersecting = 0

				if (intersecting_materials_str != "") {
					std::vector<std::string> materials;
					boost::split(materials, intersecting_materials_str, boost::is_any_of(","), boost::token_compress_on);
					init_lss();
					std::size_t bits = 0;
					for (std::size_t i = 0; i < materials.size(); i++) {
						std::size_t material_id = lss->getMaterialId(materials[i]);
						std::size_t bit = 1 << material_id;
						if (bit == 0) {
							BOOST_THROW_EXCEPTION(std::runtime_error("maximal number of of material indices for intersection exceeded"));
						}
						bits |= bit;
					}

					intersecting_materials = (int) bits;
				}

				T dmin = pt_get<T>(attr, "dmin", -STD_INFINITY(T));
				T dmax = pt_get<T>(attr, "dmax", STD_INFINITY(T));
				int intersecting = pt_get<int>(attr, "intersecting", -1);
				std::size_t m = pt_get<std::size_t>(attr, "m", 0);
				std::size_t n = pt_get<std::size_t>(attr, "n", 0);
				T v = pt_get<T>(attr, "v", 0);

				gen->run(v, n, m, dmin, dmax, intersecting, intersecting_materials);
				fibers_valid = true;
			}
			else if (v.first == "init_fibers")
			{
				init_fibers();
			}
			else if (v.first == "detect_fibers")
			{
				T threshold = pt_get<T>(attr, "threshold", 0.5);
				T convexity_threshold_high = pt_get<T>(attr, "convexity_threshold_high", 1.0);
				T convexity_threshold_low = pt_get<T>(attr, "convexity_threshold_low", 0.0);
				std::string filename = pt_get<std::string>(attr, "filename", "");
				bool binary = (pt_get<std::string>(settings, "res_format", "binary") == "binary");
				bool old = pt_get<bool>(attr, "old", false);
				bool overwrite_phase = pt_get<bool>(attr, "overwrite_phase", false);
				std::size_t filter_loops = pt_get<std::size_t>(attr, "filter_loops", 0);
				std::size_t convexity_level = pt_get<std::size_t>(attr, "convexity_level", 0);
				std::size_t dir = pt_get<std::size_t>(attr, "dir", 0);
				std::size_t radius = pt_get<std::size_t>(attr, "radius", 0);
				std::size_t max_path_length = pt_get<std::size_t>(attr, "max_path_length", 5);
				T min_segment_volume = pt_get<T>(attr, "min_segment_volume", 1);
				T w_exponent = pt_get<T>(attr, "w_exponent", 2);
				T d_exponent = pt_get<T>(attr, "d_exponent", 1);
				T p_threshold = pt_get<T>(attr, "p_threshold", 0.5);
				std::string fiber_template_str = pt_get<std::string>(attr, "fiber_template", "1 1 1 0.5 0");

				std::vector<T> fiber_template;
				std::vector<std::string> strs;
				boost::split(strs, fiber_template_str, boost::is_any_of("\t; "), boost::token_compress_on);

				for(std::vector<std::string>::iterator it = strs.begin(); it!=strs.end(); ++it) {
					fiber_template.push_back(boost::lexical_cast<T>(*it));
				}

				init_lss();
				init_phase();

				if (!old) {
					lss->detectFibers(threshold, filename, binary, overwrite_phase, filter_loops,
						convexity_threshold_low, convexity_threshold_high, convexity_level, dir,
						radius, min_segment_volume, max_path_length, d_exponent, w_exponent, p_threshold, fiber_template);
				} else {
					lss->detectFibers_old(threshold, filename, binary, overwrite_phase, filter_loops,
						convexity_threshold_low, convexity_threshold_high, convexity_level, dir,
						radius, min_segment_volume, max_path_length);
				}
			}
			else if (v.first == "inv_ellint_rd")
			{
				T tol = pt_get<T>(attr, "tol", std::pow(std::numeric_limits<T>::epsilon(), 2/(T)3));
				//T step = pt_get<T>(attr, "step", 0.5);
				//std::size_t max_iter = pt_get<std::size_t>(attr, "max_iter", 100000);
				std::string filename = pt_get<std::string>(attr, "filename");
				std::size_t r0 = pt_get<std::size_t>(attr, "r0", 1);
				std::size_t nr = pt_get<std::size_t>(attr, "nr", 2);
				std::size_t nt = pt_get<std::size_t>(attr, "nt", 100);
				std::size_t ne = pt_get<std::size_t>(attr, "ne", 1);
				ublas::c_vector<T, DIM> na;
				ublas::c_vector<T, DIM> pa;
				ublas::c_vector<T, DIM> pb;
				std::ofstream fs;
				ProgressBar<T> pbar;

				open_file(fs, filename);

				for (int i = 0; i < DIM; i++) {
					na(i) =  1/(T)DIM;
					pb(i) = (T)1;
				}

				// calculate total number of interations
				std::size_t n = 0;
				for (std::size_t ir = r0; ir < nr; ir++) {
					T r = ir/(T)nr;
					std::size_t ntr = std::max((std::size_t)1, (std::size_t)(r*nt));
					std::size_t ner = (ir > 0) ? ne : 1;
					n += ner*ntr;
					if (ne < 3) n += 1;
				}

				fs << "iter\te\tr\tt\ta0\ta1\ta2\tb0\tb1\tb2\n";

				std::size_t i = 0;
				for (std::size_t ir = r0; ir < nr; ir++)
				{
					T r = ir/(T)nr;
					std::size_t ntr = std::max((std::size_t)1, (std::size_t)(r*nt));
					std::size_t ner = (ir > 0) ? ne : 1;

					for (std::size_t ie = 0; ie < ner; ie++)
					{
						ublas::c_vector<T, DIM> p0 = ublas::unit_vector<T>(DIM, ie);
						ublas::c_vector<T, DIM> p1 = ublas::unit_vector<T>(DIM, (ie + 1) % DIM);

						std::size_t ntrx = ntr;
						if (ne < 3 && ie == ne-1) ntrx += 1;

						for (std::size_t it = 0; it < ntrx; it++)
						{
							T t = it/(T)ntr;
							ublas::c_vector<T, DIM> pt = t*p1 + (1-t)*p0;
							ublas::c_vector<T, DIM> pa = r*pt + (1-r)*na;

							std::size_t iter = compute_B_from_A<T, DIM>(pa, pb, tol);

							fs << iter << "\t" <<
								ie << "\t" << r << "\t" << t << "\t" <<
								pa[0] << "\t" << pa[1] << "\t" << pa[2] << "\t" <<
								pb[0] << "\t" << pb[1] << "\t" << pb[2] << "\n";
							i++;

							if (pbar.update(i*100/n)) {
								pbar.message() << i << " records" << pbar.end();
							}
						}
					}
				}
			}
			else if (v.first == "calc_HS_bounds")
			{
				// TODO: replace by something more general
				T kl, ku, mul, muu;
				Material<T, DIM> m1("", "1");
				Material<T, DIM> m2("", "2");
				m1.readSettings(v.second);
				m2.readSettings(v.second);
				HashinBounds<T>::get(m1.mu, m1.lambda, m1.phi, m2.mu, m2.lambda, m2.phi, kl, mul, ku, muu);

				LOG_COUT << "HS lower bounds: K=" << kl << " mu=" << mul << " lambda=" << (kl - 2.0/3.0*mul) << std::endl;
				LOG_COUT << "HS upper bounds: K=" << ku << " mu=" << muu << " lambda=" << (ku - 2.0/3.0*muu) << std::endl;
			}
			else if (v.first == "python")
			{
				PY::instance().exec(v.second.data());
			}
			else if (v.first == "print_A2")
			{
				init_fibers();
				ublas::c_matrix<T, DIM, DIM> A2 = gen->getA2();
				LOG_COUT << "A2:" << format(A2) << std::endl;
			}
			else if (v.first == "set_radius_distribution")
			{
				boost::shared_ptr< DiscreteDistribution<T, 1> > dist(new CompositeDistribution<T, 1>());
				dist->readSettings(v.second);
				gen->setRadiusDistribution(dist);
				phase_valid = fibers_valid = false;
			}
			else if (v.first == "set_length_distribution")
			{
				boost::shared_ptr< DiscreteDistribution<T, 1> > dist(new CompositeDistribution<T, 1>());
				dist->readSettings(v.second);
				gen->setLengthDistribution(dist);
				phase_valid = fibers_valid = false;
			}
			else if (v.first == "set_fiber_distribution" || v.first == "set_orientation_distribution")
			{
				boost::shared_ptr< DiscreteDistribution<T, DIM> > dist(new CompositeDistribution<T, DIM>());
				dist->readSettings(v.second);
				gen->setOrientationDistribution(dist);
				phase_valid = fibers_valid = false;
			}
			else if (v.first == "tune_num_threads")
			{
				T t_measure = pt_get<T>(attr, "tmeas", 0.05);
				T treshfac = pt_get<T>(attr, "tfac", 1.5);
				init_lss();
				lss->tune_num_threads(t_measure, treshfac);
			}
			else if (v.first == "select_material")
			{
				std::string name = pt_get<std::string>(attr, "name");
				init_lss();
				gen->selectMaterial(lss->getMaterialId(name));
				LOG_COUT << "selected material: " << name << std::endl;
			}
			else if (v.first == "place_fiber")
			{
				ublas::c_vector<T, DIM> c;
				ublas::c_vector<T, DIM> a;
				T L = pt_get<T>(attr, "L", 0.0);
				T r = pt_get<T>(attr, "R", 0.25*dx);
				T V = pt_get<T>(attr, "V", -1.0);
				std::string type = pt_get<std::string>(attr, "type", "capsule");

				if (V >= 0) {
					// convient function to compute R by volume of equivalent sphere
					r = std::pow(V/(4*M_PI/(T)3), 1/(T)3);
				}

				read_vector(attr, c, "cx", "cy", "cz", x0 + 0.5*dx, y0 + 0.5*dy, z0 + 0.5*dz);
				read_vector(attr, a, "ax", "ay", "az", (T)1, (T)0, (T)0);

				LOG_COUT << "placing " << type << " fiber: c=" << format(c) << " a=" << format(a) << " L=" << L << " R=" << r << std::endl;
				boost::shared_ptr< const Fiber<T, DIM> > fiber;
				if (type == "capsule") {
					fiber.reset(new CapsuleFiber<T, DIM>(c, a, L, r));
				}
				else if (type == "cylinder") {
					fiber.reset(new CylindricalFiber<T, DIM>(c, a, L, r));
				}
				else if (type == "halfspace") {
					fiber.reset(new HalfSpaceFiber<T, DIM>(c, a));
				}
				else {
					BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown fiber type '%s'") % type).str()));
				}

				gen->addFiber(fiber);
				phase_valid = false; 
			}
			else if (v.first == "place_triangle")
			{
				ublas::c_vector<T, DIM> p[3];

				read_vector(attr, p[0], "p1x", "p1y", "p1z", (T)0, (T)0, (T)0);
				read_vector(attr, p[1], "p2x", "p2y", "p2z", (T)0, (T)0, (T)0);
				read_vector(attr, p[2], "p3x", "p3y", "p3z", (T)0, (T)0, (T)0);

				LOG_COUT << "placing triangle: p1=" << format(p[0]) << " p2=" << format(p[1]) << " p3=" << format(p[2]) << std::endl;
				
				boost::shared_ptr< const Fiber<T, DIM> > fiber;
				fiber.reset(new TriangleFiber<T, DIM>(p));

				gen->addFiber(fiber);
				phase_valid = false; 
			}
			else if (v.first == "place_tetrahedron")
			{
				ublas::c_vector<T, DIM> p[4];

				read_vector(attr, p[0], "p1x", "p1y", "p1z", (T)0, (T)0, (T)0);
				read_vector(attr, p[1], "p2x", "p2y", "p2z", (T)0, (T)0, (T)0);
				read_vector(attr, p[2], "p3x", "p3y", "p3z", (T)0, (T)0, (T)0);
				read_vector(attr, p[3], "p4x", "p4y", "p4z", (T)0, (T)0, (T)0);

				LOG_COUT << "placing tetrahedron: p1=" << format(p[0]) << " p2=" << format(p[1]) << " p3=" << format(p[2]) << " p4=" << format(p[3]) << std::endl;
				
				boost::shared_ptr< const Fiber<T, DIM> > fiber;
				fiber.reset(new TetrahedronFiber<T, DIM>(p));

				gen->addFiber(fiber);
				phase_valid = false; 
			}
			else if (v.first == "place_tetvtk")
			{
				std::string filename = pt_get<std::string>(attr, "filename");
				std::size_t start = pt_get<std::size_t>(attr, "start", 0);
				std::size_t end = pt_get<std::size_t>(attr, "end", -1);
				bool fill = pt_get<bool>(attr, "fill", true);

				ublas::matrix<T> _a = ublas::identity_matrix<T>(3);
				ublas::c_matrix<T,3,3> a;
				ublas::c_vector<T,3> t;
				read_matrix(attr, _a, "a", false); a = _a;
				read_vector(attr, t, "tx", "ty", "tz", (T)0, (T)0, (T)0);

				LOG_COUT << "placing tetvtk: filename=" << filename << std::endl;
				
				boost::shared_ptr< const Fiber<T, DIM> > fiber;
				fiber.reset(new TetVTKFiber<T, DIM>(filename, start, end, fill, a, t));

				gen->addFiber(fiber);
				phase_valid = false; 
			}
			else if (v.first == "place_tetdolfin")
			{
				std::string filename = pt_get<std::string>(attr, "filename");
				std::size_t start = pt_get<std::size_t>(attr, "start", 0);
				std::size_t end = pt_get<std::size_t>(attr, "end", -1);
				bool fill = pt_get<bool>(attr, "fill", true);

				ublas::matrix<T> _a = ublas::identity_matrix<T>(3);
				ublas::c_matrix<T,3,3> a;
				ublas::c_vector<T,3> t;
				read_matrix(attr, _a, "a", false); a = _a;
				read_vector(attr, t, "tx", "ty", "tz", (T)0, (T)0, (T)0);

				LOG_COUT << "placing tetvtk: filename=" << filename << std::endl;
				
				boost::shared_ptr< const Fiber<T, DIM> > fiber;
				fiber.reset(new TetDolfinXMLFiber<T, DIM>(filename, start, end, fill, a, t));

				gen->addFiber(fiber);
				phase_valid = false; 
			}
			else if (v.first == "place_stl")
			{
				std::string filename = pt_get<std::string>(attr, "filename");
				std::size_t start = pt_get<std::size_t>(attr, "start", 0);
				std::size_t end = pt_get<std::size_t>(attr, "end", -1);
				bool fill = pt_get<bool>(attr, "fill", true);

				ublas::matrix<T> _a = ublas::identity_matrix<T>(3);
				ublas::c_matrix<T,3,3> a;
				ublas::c_vector<T,3> t;
				read_matrix(attr, _a, "a", false); a = _a;
				read_vector(attr, t, "tx", "ty", "tz", (T)0, (T)0, (T)0);

				LOG_COUT << "placing stl: filename=" << filename << std::endl;
				
				boost::shared_ptr< const Fiber<T, DIM> > fiber;
				fiber.reset(new STLFiber<T, DIM>(filename, start, end, fill, a, t));

				gen->addFiber(fiber);
				phase_valid = false; 
			}
			else if (v.first == "run_load_case")
			{
				std::string outfile = pt_get<std::string>(attr, "outfile", "");

				ublas::vector<T> Ep = ublas::zero_vector<T>(6), Sp = ublas::zero_vector<T>(6);
				read_voigt_vector(attr, Ep, "e");
				read_voigt_vector(attr, Sp, "s");

				ublas::matrix<T> BCP = Voigt::Id4<T>(6);
				read_matrix(attr, BCP, "p", true);

				init_lss();
				init_phase();

				if (lss->mode() == "elasticity")
				{
					lss->setBCProjector(BCP);
					lss->setStrain(Ep);
					lss->setStress(Sp);
					lss->run();

					if (_except) return EXIT_FAILURE;

					if (!outfile.empty()) {
						// write strains to disk
						lss->template writeVTK<R>(outfile, binary);
					}
				}
				else if (lss->mode() == "hyperelasticity")
				{
					Ep.resize(9);
					read_voigt_vector(attr, Ep, "e");
					
					Sp.resize(9);
					read_voigt_vector(attr, Sp, "s");

					BCP.resize(9, 9);
					BCP = Voigt::Id4<T>(9);
					read_matrix(attr, BCP, "p", true);

					ublas::vector<T> Id = ublas::zero_vector<T>(9);
					Id(0) = Id(1) = Id(2) = 1;
					Ep += Voigt::dyad4(BCP, Id);

					lss->setBCProjector(BCP);
					lss->setStrain(Ep);
					lss->setStress(Sp);
					lss->run();

					if (_except) return EXIT_FAILURE;

					if (!outfile.empty()) {
						// write strains to disk
						lss->template writeVTK<R>(outfile, binary);
					}
				}
				else if (lss->mode() == "viscosity")
				{
					// check zero trace
					T trEp = Ep(0) + Ep(1) + Ep(2);
					T trSp = Sp(0) + Sp(1) + Sp(2);
					T tol = 100.0*std::numeric_limits<T>::epsilon();
					
					if (std::abs(trEp) > tol) {
						BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Prescibed fluid stress %s has not zero trace!") % Ep).str()));
					}

					if (std::abs(trSp) > tol) {
						BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Prescibed fluid strain %s has not zero trace!") % Sp).str()));
					}

					lss->setBCProjector(BCP);
					lss->setStrain(Ep);
					lss->setStress(Sp);
					lss->run();

					if (_except) return EXIT_FAILURE;

					if (!outfile.empty()) {
						// write strains to disk
						lss->template writeVTK<R>(outfile, binary);
					}
				}
				else if (lss->mode() == "heat" || lss->mode() == "porous")
				{
					Ep.resize(3);
					read_voigt_vector(attr, Ep, "e");
					
					Sp.resize(3);
					read_voigt_vector(attr, Sp, "s");

					BCP.resize(3, 3);
					BCP = Voigt::Id4<T>(3);
					read_matrix(attr, BCP, "p", true);

					lss->setBCProjector(BCP);
					lss->setStrain(Ep);
					lss->setStress(Sp);
					lss->run();

					if (_except) return EXIT_FAILURE;

					if (!outfile.empty()) {
						// write strains to disk
						lss->template writeVTK<R>(outfile, binary);
					}
				}
				else {
					BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown solver mode of operation '%s'") % lss->mode()).str()));
				}
			}
			else if (v.first == "calc_effective_properties")
			{
				std::string outdir = pt_get<std::string>(attr, "outdir", "");
				//bool mask = pt_get<bool>(attr, "mask", false);

				init_lss();
				init_phase();

				if (lss->mode() == "elasticity")
				{
					ublas::c_matrix<T,6,6> E = ublas::identity_matrix<T>(6);	// matrix of experiments
					ublas::c_matrix<T,6,6> S;	// matrix of responses

					// TODO: parallelize this loop?
					for (std::size_t i = 0; i < 6; i++)
					{
						Timer __t("load case");

						ublas::vector<T> Ep = ublas::column(E, i);

						lss->setStrain(Ep);
						bool ret = lss->run();

						if (_except || ret) return EXIT_FAILURE;

						ublas::vector<T> Smean = lss->calcMeanStress();

						if (!outdir.empty()) {
							// write strains to disk
							std::stringstream fn;
							fn << outdir << "/results_" << (i+1) << ".vtk";
							lss->template writeVTK<R>(fn.str(), binary);
						}

						// assign average stress to column in response matrix
						ublas::column(S, i) = Smean;
					}

					// compute inverse of E
					ublas::c_matrix<T,6,6> Einv = ublas::identity_matrix<T>(6);
					ublas::c_matrix<T,6,6> Ecopy(E);
					ublas::c_matrix<T,6,6> Ceff;
					int err = lapack::gesv(Ecopy, Einv);
					if (err == 0) {
						// compute Ceff = S*E^{-1}
						Ceff = ublas::prod(S, Einv);
					}
					else {
						// infinite stiffness
						Ceff = ublas::identity_matrix<T>(6)*STD_INFINITY(T);
					}

					// the last 3 columns are scaled by 1/2 to have Voigt notation
					Ceff_voigt = Ceff;
					for (std::size_t i = 0; i < 6; i++) {
						for (std::size_t j = 3; j < 6; j++) {
							Ceff_voigt(i,j) *= 0.5;
						}
					}

					LOG_COUT << "Effective stiffness matrix (Voigt notation):" << format(Ceff_voigt) << std::endl;

					LOG_COUT << "A least square fit w.r.t. the Frobenian inner product to an isotropic material gives the parameters:" << std::endl;

					T S_1 = Ceff(0,0) + Ceff(0,1) + Ceff(0,2) +
						Ceff(1,0) + Ceff(1,1) + Ceff(1,2) +
						Ceff(2,0) + Ceff(2,1) + Ceff(2,2);
					T S_2 = Ceff(0,0) + Ceff(1,1) + Ceff(2,2) +
						Ceff(3,3) + Ceff(4,4) + Ceff(5,5);
					T lambda_eff = (1/15.0)*(2*S_1 - S_2);
					T mu_eff = (1/30.0)*(3*S_2 - S_1);
					T K_eff = lambda_eff + 2/3.0*mu_eff;
					ublas::c_matrix<T,6,6> Cfit = ublas::zero_matrix<T>(6,6);

					Cfit(0,0) = Cfit(1,1) = Cfit(2,2) = lambda_eff + 2*mu_eff;
					Cfit(3,3) = Cfit(4,4) = Cfit(5,5) = 2*mu_eff;
					Cfit(0,1) = Cfit(0,2) = Cfit(1,2) = Cfit(1,0) = Cfit(2,0) = Cfit(2,1) = lambda_eff;

					T rel_err = ublas::norm_frobenius(Ceff - Cfit) / ublas::norm_frobenius(Ceff);

					LOG_COUT << "  K_eff      = " << K_eff << std::endl;
					LOG_COUT << "  mu_eff     = " << mu_eff << std::endl;
					LOG_COUT << "  lambda_eff = " << lambda_eff << std::endl;
					LOG_COUT << "  relative error of fit = " << rel_err << std::endl;
				}
				else if (lss->mode() == "heat" || lss->mode() == "porous")
				{
					ublas::c_matrix<T,3,3> E = ublas::identity_matrix<T>(3);	// matrix of experiments
					ublas::c_matrix<T,3,3> S;	// matrix of responses

					// TODO: parallelize this loop?
					for (std::size_t i = 0; i < 3; i++)
					{
						Timer __t("load case");

						ublas::vector<T> Ep = ublas::column(E, i);

						lss->setStrain(Ep);
						bool ret = lss->run();

						if (_except || ret) return EXIT_FAILURE;

						ublas::vector<T> Smean = lss->calcMeanStress();

						if (!outdir.empty()) {
							// write strains to disk
							std::stringstream fn;
							fn << outdir << "/results_" << (i+1) << ".vtk";
							lss->template writeVTK<R>(fn.str(), binary);
						}

						// assign average stress to column in response matrix
						ublas::column(S, i) = Smean;
					}

					// compute inverse of E
					ublas::c_matrix<T,3,3> Einv = ublas::identity_matrix<T>(3);
					ublas::c_matrix<T,3,3> Ecopy(E);
					ublas::c_matrix<T,3,3> Ceff;
					int err = lapack::gesv(Ecopy, Einv);
					if (err == 0) {
						// compute Ceff = S*E^{-1}
						Ceff = ublas::prod(S, Einv);
					}
					else {
						// infinite stiffness
						Ceff = ublas::identity_matrix<T>(3)*STD_INFINITY(T);
					}

					Ceff_voigt = Ceff;

					if (lss->mode() == "heat") {
						LOG_COUT << "Effective conductivity matrix:" << format(Ceff) << std::endl;
					} else {
						LOG_COUT << "Effective permeability matrix:" << format(Ceff) << std::endl;
					}
				}
				else if (lss->mode() == "hyperelasticity")
				{
					BOOST_THROW_EXCEPTION(std::runtime_error("not implemented"));
					
					/*
					ublas::c_matrix<T,6,6> E = ublas::identity_matrix<T>(6);	// matrix of experiments
					ublas::c_matrix<T,6,6> S;	// matrix of responses

					// TODO: parallelize this loop?
					for (std::size_t i = 0; i < 6; i++)
					{
						Timer __t("load case");

						ublas::vector<T> Ep = ublas::column(E, i);

						LOG_COUT << "prescribed elastic strain: " << Ep << std::endl;

						lss->setStrain(Ep);
						bool ret = lss->run();

						if (_except || ret) return EXIT_FAILURE;

						ublas::vector<T> Smean = lss->calcMeanStress();
						LOG_COUT << "average elastic strain: " << lss->averageValue() << std::endl;
						LOG_COUT << "average elastic stress: " << Smean << std::endl;

						if (!outdir.empty()) {
							// write strains to disk
							std::stringstream fn;
							fn << outdir << "/results_" << (i+1) << ".vtk";
							lss->template writeVTK<R>(fn.str(), binary);
						}

						// assign average stress to column in response matrix
						ublas::column(S, i) = Smean;
					}

					// compute inverse of E
					ublas::c_matrix<T,6,6> Einv = ublas::identity_matrix<T>(6);
					ublas::c_matrix<T,6,6> Ecopy(E);
					ublas::c_matrix<T,6,6> Ceff;
					int err = lapack::gesv(Ecopy, Einv);
					if (err == 0) {
						// compute Ceff = S*E^{-1}
						Ceff = ublas::prod(S, Einv);
					}
					else {
						// infinite stiffness
						Ceff = ublas::identity_matrix<T>(6)*STD_INFINITY(T);
					}

					// the last 3 columns are scaled by 1/2 to have Voigt notation
					Ceff_voigt = Ceff;
					for (std::size_t i = 0; i < 6; i++) {
						for (std::size_t j = 3; j < 6; j++) {
							Ceff_voigt(i,j) *= 0.5;
						}
					}

					LOG_COUT << "Effective stiffness matrix (Voigt notation):" << format(Ceff_voigt) << std::endl;

					LOG_COUT << "A least square fit w.r.t. the Frobenian inner product to an isotropic material gives the parameters:" << std::endl;

					T S_1 = Ceff(0,0) + Ceff(0,1) + Ceff(0,2) +
						Ceff(1,0) + Ceff(1,1) + Ceff(1,2) +
						Ceff(2,0) + Ceff(2,1) + Ceff(2,2);
					T S_2 = Ceff(0,0) + Ceff(1,1) + Ceff(2,2) +
						Ceff(3,3) + Ceff(4,4) + Ceff(5,5);
					T lambda_eff = (1/15.0)*(2*S_1 - S_2);
					T mu_eff = (1/30.0)*(3*S_2 - S_1);
					T K_eff = lambda_eff + 2/3.0*mu_eff;
					ublas::c_matrix<T,6,6> Cfit = ublas::zero_matrix<T>(6,6);

					Cfit(0,0) = Cfit(1,1) = Cfit(2,2) = lambda_eff + 2*mu_eff;
					Cfit(3,3) = Cfit(4,4) = Cfit(5,5) = 2*mu_eff;
					Cfit(0,1) = Cfit(0,2) = Cfit(1,2) = Cfit(1,0) = Cfit(2,0) = Cfit(2,1) = lambda_eff;

					T rel_err = ublas::norm_frobenius(Ceff - Cfit) / ublas::norm_frobenius(Ceff);

					LOG_COUT << "  K_eff      = " << K_eff << std::endl;
					LOG_COUT << "  mu_eff     = " << mu_eff << std::endl;
					LOG_COUT << "  lambda_eff = " << lambda_eff << std::endl;
					LOG_COUT << "  relative error of fit = " << rel_err << std::endl;
					*/
				}
				else if (lss->mode() == "viscosity")
				{
					// for the effective viscosity we need a traceless prescribed stress
					// and in the incompressible case only 5 experiments
					// NOTE: we will solve the equation shear = viscosity^-1 * prescribed stress,
					ublas::c_matrix<T,6,5> E = ublas::zero_matrix<T>(6,5);	// matrix of experiments
					ublas::c_matrix<T,6,5> S;	// matrix of responses
					E(0,0) = E(1,1) = 1;
					E(1,0) = E(2,1) = -1;
					E(3,2) = E(4,3) = E(5,4) = 1;

					// TODO: parallelize this loop?
					for (std::size_t i = 0; i < 5; i++)
					{
						Timer __t("load case");

						ublas::vector<T> Ep = ublas::column(E, i);

						lss->setStrain(Ep);
						bool ret = lss->run();

						if (_except || ret) return EXIT_FAILURE;

						ublas::vector<T> Smean = lss->calcMeanStress();

						if (!outdir.empty()) {
							// write strains to disk
							std::stringstream fn;
							fn << outdir << "/results_" << (i+1) << ".vtk";
							// TODO: make mask working (problem: modifies stress field)
							//if (mask) lss->maskSolidRegions();
							lss->template writeVTK<R>(fn.str(), binary);
						}

						// assign average shear to column in response matrix
						ublas::column(S, i) = Smean;
					}

					// extract invertible sub matrices
					ublas::c_matrix<T,5,5> E55 = subrange(E, 1, 6, 0, 5);
					ublas::c_matrix<T,5,5> S55 = subrange(S, 1, 6, 0, 5);
					
					// compute inverse of S55
					ublas::c_matrix<T,5,5> S55inv = ublas::identity_matrix<T>(5);
					ublas::c_matrix<T,5,5> S55copy(S55);
					ublas::c_matrix<T,5,5> Ceff55;
					int err = lapack::gesv(S55copy, S55inv);
					if (err == 0) {
						// compute Ceff55 = E55*S55^{-1}
						Ceff55 = ublas::prod(E55, S55inv);
					}
					else {
						// infinite viscosity
						Ceff55 = ublas::identity_matrix<T>(5)*STD_INFINITY(T);
					}

					ublas::c_matrix<T,5,5> Ceff55copy = Ceff55;
					ublas::c_matrix<T,5,5> Feff55 = ublas::identity_matrix<T>(5);
					err = lapack::gesv(Ceff55copy, Feff55);

					LOG_COUT << "Effective fluidity matrix \"0.5*f\" (5x5):" << format(Feff55) << std::endl;
					LOG_COUT << "Effective viscosity matrix \"2*eta\" (5x5):" << format(Ceff55) << std::endl;

					// build 6x6 effecitve viscosity matrix by knowing
					// that Ceff maps from traceless 3x3 to traceless 3x3 matricces
					ublas::c_matrix<T,6,6> Ceff;
					subrange(Ceff, 1, 6, 1, 6) = Ceff55;
					for (std::size_t i = 0; i < 5; i++) {
						if (S(0,i) != 0) {
							for (std::size_t j = 1; j < 6; j++) {
								Ceff(j,0) = E(j,i);
								for (std::size_t k = 1; k < 6; k++) {
									Ceff(j,0) -= Ceff(j,k)*S(k,i);
								}
								Ceff(j,0) /= S(0,i);
							}
							break;
						}
					}
					row(Ceff, 0) = -(row(Ceff, 1) + row(Ceff, 2));

					// for the first 3 columns we may add to each row an arbitarary constant due to the incompressibility
					// we choose this constant to be the minimum value of this row
					for (std::size_t i = 0; i < 6; i++) {
						T min_j = std::min(std::min(Ceff(i,0), Ceff(i,1)), Ceff(i,2));
						for (std::size_t j = 0; j < 3; j++) {
							Ceff(i,j) -= min_j;
						}
					}

					// the last 3 columns are scaled by 1/2 to have Voigt notation
					Ceff_voigt = Ceff;
					for (std::size_t i = 0; i < 6; i++) {
						for (std::size_t j = 3; j < 6; j++) {
							Ceff_voigt(i,j) *= 0.5;
						}
					}

					LOG_COUT << "Effective viscosity matrix \"2*eta\" (Voigt notation):" << format(Ceff_voigt) << std::endl;

					// indices for Voigt notation
					const int v[3][3] = {{0, 5, 4}, {5, 1, 3}, {4, 3, 2}};

					// compute the parameters alpha and beta assuming spheriodal particles as in Nunan & Keller
					// we compute every possible combination 
					std::vector<T> alphas;
					std::vector<T> betas;
					T mu0 = 0.5/lss->mu_matrix();	// need to scale because we scaled mu initially by 0.5 to get the correct fluidity

					// mu_ijkl = mu*(1+beta)/2*(delta_ik*delta_jl + delta_il*delta_jk - 3/2*delta_ij*delta_kl)
					//		+ mu*(alpha-beta)*(delta_ijkl - 1/3*delta_ij*delta_kl)
					//
					// mu_ijij/mu = (1+beta)/2
					// beta = 2*mu_ijij/mu - 1
					// 2*mu_ijij = Ceff_voigt(v[i][j],v[i][j])
					// 
					// we need to substract two entries to get rid of ambiguity
					// mu_iijj = mu*(1+beta)/2*(- 3/2) + mu*(alpha-beta)*(-1/3)
					// mu_iiii = mu*(1+beta)/2*(1/2) + mu*(alpha-beta)*(2/3)
					// alpha = (mu_iiii - mu_iijj)/mu - 1

					for (std::size_t i = 0; i < 3; i++) {
						for (std::size_t j = 0; j < 3; j++) {
							if (i == j) continue;
							// Ceff_voigt(v[i][j],v[i][j]) is scaled by
							T beta = Ceff_voigt(v[i][j],v[i][j])/mu0 - 1.0;
							T alpha = 0.5*Ceff_voigt(v[i][i],v[i][i])/mu0 - 0.5*Ceff_voigt(v[i][i],v[j][j])/mu0 - 1.0;
							alphas.push_back(alpha);
							betas.push_back(beta);
						}
					}

					// compute the parameters alpha and beta assuming spheriodal particles as in Nunan & Keller

					acc::accumulator_set<T, acc::features<acc::tag::mean, acc::tag::min, acc::tag::max, acc::tag::variance> > alpha_acc;
					acc::accumulator_set<T, acc::features<acc::tag::mean, acc::tag::min, acc::tag::max, acc::tag::variance> > beta_acc;
					std::for_each(alphas.begin(), alphas.end(), boost::bind<void>(boost::ref(alpha_acc), _1));
					std::for_each(betas.begin(), betas.end(), boost::bind<void>(boost::ref(beta_acc), _1));

					LOG_COUT << "alpha min: " << acc::min(alpha_acc) << std::endl;
					LOG_COUT << "alpha max: " << acc::max(alpha_acc) << std::endl;
					LOG_COUT << "alpha mean: " << acc::mean(alpha_acc) << std::endl;
					LOG_COUT << "alpha standard devation: " << std::sqrt(acc::variance(alpha_acc)) << std::endl;

					LOG_COUT << "beta min: " << acc::min(beta_acc) << std::endl;
					LOG_COUT << "beta max: " << acc::max(beta_acc) << std::endl;
					LOG_COUT << "beta mean: " << acc::mean(beta_acc) << std::endl;
					LOG_COUT << "beta standard devation: " << std::sqrt(acc::variance(beta_acc)) << std::endl;
				}
				else {
					BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown solver mode of operation '%s'") % lss->mode()).str()));
				}
			}
			else if (v.first == "calc_isotropic_laminate")
			{
				// calculate properties of locally instropic laminate
				// Milton, G.W. - The Theory of Composites, p. 163, Eqn. 9.9
				// NOTE: there is an error in the eqn. for C3333 = 4μ(λ + μ)λ/(λ + 2μ) + ...
				// should be 4μ(λ + μ)/(λ + 2μ) + ...

				ublas::c_matrix<T,6,6> Ceff(ublas::zero_matrix<T>(6));
				T c1, c2, c3, c4, c5, c6;
				
				c1 = c2 = c3 = c4 = c5 = c6 = 0;
				BOOST_FOREACH(const ptree::ptree::value_type &mat, v.second)
				{
					if (mat.first == "<xmlcomment>" || mat.first == "<xmlattr>") {
						continue;
					}

					Material<T, DIM> m;
					m.readSettings(mat.second);
					LOG_COUT << mat.first << ": phi=" << m.phi << " lambda=" << m.lambda << " mu=" << m.mu << std::endl;

					c1 += m.phi/(m.lambda + 2*m.mu);
					c2 += m.phi/m.mu;
					c3 += m.phi*m.mu;
					c4 += m.phi*m.lambda/(m.lambda + 2*m.mu);
					c5 += m.phi*4*m.mu*(m.lambda + m.mu)/(m.lambda + 2*m.mu);
					c6 += m.phi*2*m.mu*m.lambda/(m.lambda + 2*m.mu);
				}

				T C1111 = 1/c1;
				T C1212_C1313 = 1/c2;
				T C2323 = c3;
				T C1122_C1133 = c4/c1;
				T C2222_C3333 = c5 + c4*c4/c1;
				T C2233 = c6 + c4*c4/c1;

				Ceff(0, 0) = C1111;
				Ceff(1, 1) = Ceff(2, 2) = C2222_C3333;
				Ceff(3, 3) = C2323;
				Ceff(4, 4) = Ceff(5, 5) = C1212_C1313;
				Ceff(0, 1) = Ceff(1, 0) = Ceff(0, 2) = Ceff(2, 0) = C1122_C1133;
				Ceff(1, 2) = Ceff(2, 1) = C2233;

/*
				T rotate_z = pt_get<T>(attr, "rotate_z", 0);
				if (rotate_z != 0) {
					rotate_z *= M_PI/180;
					T c = std::cos(rotate_z);
					T s = std::sin(rotate_z);
					ublas::c_matrix<T,6,6> K(ublas::zero_matrix<T>(6));
					K(0,0) = K(1,1) = c*c;
					K(0,1) = K(1,0) = s*s;
					K(2,2) = 1;
					K(3,3) = K(4,4) = c;
					K(3,4) = s;
					K(4,3) = -s;
					K(5,5) = c*c - s*s; // = cos(2*rotate_z)
					K(0,5) = 2*c*s;  // = sin(2*rotate_z)
					K(1,5) = -2*c*s;
					K(5,0) = -c*s;
					K(5,1) = c*s;

					Ceff = ublas::prod(K, Ceff);
					K = ublas::trans(K);
					Ceff = ublas::prod(Ceff, K);
				}
*/

				LOG_COUT << "Effective stiffness matrix (Voigt notation):" << format(Ceff) << std::endl;
			}
			else if (v.first == "print_timings")
			{
				Timer::print_stats();
			}
			else if (v.first == "exit")
			{
				return pt_get<int>(attr, "code", EXIT_SUCCESS);
			}
			else {
				BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("Unknown action: '%s'") % v.first).str()));
			}
		}
		
		return EXIT_SUCCESS;
	}
};


//! Handles stuff before exit of the application
void exit_handler()
{
	LOG_COUT << DEFAULT_TEXT;

	// shut down FFTW
	fftw_cleanup_threads();
	fftw_cleanup();
}


//! Handles signals of the application
void signal_handler(int signum)
{
	LOG_COUT << std::endl;
	LOG_CERR << (boost::format("Program aborted: Signal %d received") % signum).str() << std::endl;
	print_stacktrace(LOG_CERR);
	exit(signum);  
}



//! Class which interfaces the FG interface with a project xml file
class FGProject
{
protected:
	boost::shared_ptr< ptree::ptree > xml_root;
	boost::shared_ptr< FGI > fgi;
	int xml_precision;
	PyObject* pyfg_instance;

public:
	FGProject()
	{
		this->xml_precision = -1;

		// register signal SIGINT handler and exit handler
		atexit(exit_handler);
		signal(SIGINT, signal_handler);
		signal(SIGSEGV, signal_handler);

		reset();
	}

	// reset to initial state
	void reset()
	{
		xml_root.reset(new ptree::ptree());
		fgi.reset();
		pyfg_instance = NULL;
	}

	void init_fgi()
	{
		ptree::ptree& settings = xml_root->get_child("settings", empty_ptree);

		// get type and dimension
		std::size_t dim = pt_get(settings, "dim", 3);
		std::string type = pt_get<std::string>(settings, "datatype", "double");
		std::string rtype = pt_get<std::string>(settings, "restype", "float");

	#define RUN_TYPE_AND_DIM(T, R, DIM) \
		else if (#T == type && #R == rtype && DIM == dim) fgi.reset(new FG<T, R, DIM>(xml_root))

		if (false) {}
		//RUN_TYPE_AND_DIM(double, double, 2);
		//RUN_TYPE_AND_DIM(double, float, 2);
		//RUN_TYPE_AND_DIM(double, double, 3);
		RUN_TYPE_AND_DIM(double, float, 3);
#ifdef FFTWF_ENABLED
		//RUN_TYPE_AND_DIM(float, float, 2);
		//RUN_TYPE_AND_DIM(float, float, 3);
#endif
		else {
			BOOST_THROW_EXCEPTION(std::runtime_error("dimension/datatype not supported"));
		}

		fgi->set_pyfg_instance(pyfg_instance);
	}

	boost::shared_ptr< FGI > fg()
	{
		if (!fgi) init_fgi();
		return fgi;
	}

	int run() { return fg()->run("actions"); }
	int run_path(const std::string& actions_path) { return fg()->run(actions_path); }
	void cancel() { return fg()->cancel(); }
	void init_lss() { fg()->init_lss(); }
	void init_fibers() { fg()->init_fibers(); }
	void init_phase() { fg()->init_phase(); }
	bool get_error() { return (_except.get() != NULL); }
	std::vector<std::string> get_phase_names() { return fg()->get_phase_names(); }
	double get_volume_fraction(const std::string& field) { return fg()->get_volume_fraction(field); }
	double get_real_volume_fraction(const std::string& field) { return fg()->get_real_volume_fraction(field); }
	std::vector<double> get_residuals() { return fg()->get_residuals(); }
	double get_solve_time() { return fg()->get_solve_time(); }
	std::size_t get_distance_evals() { return fg()->get_distance_evals(); }
	std::vector<double> get_mean_stress() { return fg()->get_mean_stress(); }
	std::vector<double> get_mean_strain() { return fg()->get_mean_strain(); }
	std::vector<double> get_mean_cauchy_stress() { return fg()->get_mean_cauchy_stress(); }
	double get_mean_energy() { return fg()->get_mean_energy(); }
	std::vector<std::vector<double> > get_effective_property() { return fg()->get_effective_property(); }
	std::vector<double> get_rve_dims() { return fg()->get_rve_dims(); }
	std::vector<std::vector<double> > get_A2() { return fg()->get_A2(); }
	std::vector<std::vector<std::vector<std::vector<double> > > > get_A4() { return fg()->get_A4(); }
	void set_pyfg_instance(py::object instance) { pyfg_instance = instance.ptr(); }
	void set_variable(std::string key, py::object value) { fg()->set_variable(key, value); }

	/// Enable/Disable Python evaluation of expressions
	void set_py_enabled(bool enabled)
	{
		PY::instance().set_enabled(enabled);
	}

	std::string get_xml()
	{
		std::stringstream ss;
		std::size_t indent = 1;
		char indent_char = '\t';
#if BOOST_VERSION < 105800
		ptree::xml_writer_settings<char> settings(indent_char, indent);
#else
		ptree::xml_writer_settings<std::string> settings = boost::property_tree::xml_writer_make_settings<std::string>(indent_char, indent);
#endif

		#if 1
			write_xml(ss, *xml_root, settings);	// adds <?xml declaration
		#else
			write_xml_element(ss, std::string(), *xml_root, -1, settings);	// omits <?xml declaration
		#endif

		return ss.str();
	}

	typedef boost::property_tree::basic_ptree<std::basic_string<char>, std::basic_string<char> > treetype;

	ptree::ptree* get_path(const std::string& path, int create = 1)
	{
		std::string full_path = "settings." + path;
		boost::replace_all(full_path, "..", ".<xmlattr>.");

		std::vector<std::string> parts;
		boost::split(parts, full_path, boost::is_any_of("."));
		treetype* current = xml_root.get();

		for (std::size_t i = 0; i < parts.size(); i++)
		{
			std::vector<std::string> elements;
			boost::split(elements, parts[i], boost::is_any_of("[]()"));

			std::string name = elements[0];
			std::size_t index = 0;

			if (elements.size() > 1) {
				index = boost::lexical_cast<std::size_t>(elements[1]);
			}

			// find the element with the same name and index
			treetype* next = NULL;
			std::size_t counter = 0;
			for(treetype::iterator iter = current->begin(); iter != current->end(); ++iter) {
			// BOOST_FOREACH(ptree::ptree::value_type &v, *current) {
				if (iter->first == elements[0]) {
					if (counter == index) {
						if (create < 0 && i == (parts.size()-1)) {
							// delete item
							current->erase(iter);
							return NULL;
						}
						next = &(iter->second);
						break;
					}
					counter++;
				}
			}

			if (next == NULL) {
				if (create > 0) {
					// add proper number of elements
					for (std::size_t c = counter; c <= index; c++) {
						treetype::iterator v = current->push_back(treetype::value_type(name, empty_ptree));
						next = &(v->second);
					}
				}
				else if (create < 0) {
					// nothing to remove
					return NULL;
				}
				else {
					BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("XML path '%s' not found") % path).str()));
				}
			}

			current = next;
		}

		return current;
	}
	
	std::string get(const std::string& key)
	{
		return this->get_path(key, 0)->get_value<std::string>("");
	}

	void erase(const std::string& key)
	{
		this->get_path(key, -1);
	}

	void set(const std::string& key)
	{
		this->get_path(key)->put_value("");
	}

	void set(const std::string& key, long value)
	{
		this->get_path(key)->put_value(value);
	}

	void set(const std::string& key, double value)
	{
		if (this->xml_precision >= 0) {
#if 1
			this->set(key, (boost::format("%g") % boost::io::group(std::setprecision(this->xml_precision), value)).str());
			return;
#else
			int exponent;
			double mantissa = std::frexp(value, exponent);
			double scale = std::pow(10.0, this->xml_precision);
			mantissa = std::round(mantissa*scale)/scale;
			value = std::ldexp(mantissa, exponent);
#endif
		}

		this->get_path(key)->put_value(value);
	}

	void set(const std::string& key, const std::string& value)
	{
		this->get_path(key)->put_value(value);
	}

	void set_xml(const std::string& xml)
	{
		std::stringstream ss;
		ss << xml;
		// read settings
		xml_root.reset(new ptree::ptree());
		read_xml(ss, *xml_root, 0*ptree::xml_parser::trim_whitespace);
	}

	void set_log_file(const std::string& logfile)
	{
		Logger::instance().setTeeFilename(logfile);
	}

	void set_xml_precision(int digits)
	{
		this->xml_precision = digits;
	}

	int get_xml_precision()
	{
		return this->xml_precision;
	}

	void load_xml(const std::string& filename)
	{
		// read settings
		xml_root.reset(new ptree::ptree());
		read_xml(filename, *xml_root, 0*ptree::xml_parser::trim_whitespace);
	}

	noinline int exec(const po::variables_map& vm)
	{
		Timer __t("application");

		std::string filename = vm["input-file"].as<std::string>();
		std::string actions_path = vm["actions-path"].as<std::string>();

		// read settings
		load_xml(filename);

		return run_path(actions_path);
	}
};


//! Python interface class for fibergen
class PyFG : public FGProject
{
protected:
	py::object _py_convergence_callback;
	py::object _py_loadstep_callback;

public:
	~PyFG()
	{
		//LOG_COUT << "~PyFG" << std::endl;
		PY::release();
	}

	bool convergence_callback()
	{
		if (_py_convergence_callback) {
			py::object ret = _py_convergence_callback();
			py::extract<bool> bool_ret(ret);
			if (bool_ret.check()) {
				return bool_ret();
			}
		}

		return false;
	}

	void set_convergence_callback(py::object cb)
	{
		_py_convergence_callback = cb;
		fg()->set_convergence_callback(boost::bind(&PyFG::convergence_callback, this));
	}

	bool loadstep_callback()
	{
		if (_py_loadstep_callback) {
			py::object ret = _py_loadstep_callback();
			py::extract<bool> bool_ret(ret);
			if (bool_ret.check()) {
				return bool_ret();
			}
		}

		return false;
	}

	void set_loadstep_callback(py::object cb)
	{
		_py_loadstep_callback = cb;
		fg()->set_loadstep_callback(boost::bind(&PyFG::loadstep_callback, this));
	}

	std::vector<double> get_B_from_A(double a0, double a1, double a2)
	{
		double tol = 1e-10;
		ublas::c_vector<double, 3> a, b;
		b[0] = b[1] = b[2] = 1.0;
		a[0] = a0; a[1] = a1; a[2] = a2;
		compute_B_from_A<double, 3>(a, b, tol);
		std::vector<double> B(3);
		B[0] = b[0]; B[1] = b[1]; B[2] = b[2];
		return B;
	}
};


py::object SetParameters(py::tuple args, py::dict kwargs)
{
	PyFG& self = py::extract<PyFG&>(args[0]);
	std::string key = py::extract<std::string>(args[1]);
	py::list keys = kwargs.keys();
	int nargs = py::len(args) + py::len(kwargs);

	if (nargs == 2) {
		self.set(key);
	}

	for(int i = 2; i < nargs; ++i)
	{
		std::string attr_key;
		py::object curArg;

		if (i < py::len(args)) {
			curArg = args[i];
			attr_key = key;
		}
		else {
			int j = i - py::len(args);
			std::string key_j = py::extract<std::string>(keys[j]);
			curArg = kwargs[keys[j]];
			attr_key = key + "." + key_j;
		}

		py::extract<long> int_arg(curArg);
		if (int_arg.check()) {
			self.set(attr_key, int_arg());
			continue;
		}
		py::extract<double> double_arg(curArg);
		if (double_arg.check()) {
			self.set(attr_key, double_arg());
			continue;
		}
		py::extract<std::string> string_arg(curArg);
		if (string_arg.check()) {
			self.set(attr_key, string_arg());
			continue;
		}

		BOOST_THROW_EXCEPTION(std::runtime_error((boost::format("invalid argument for attribute '%s' specified") % attr_key).str()));
	}

	return py::object();
}


void ExtractRangeArg(const std::string& name, size_t n, std::vector<size_t>& range, py::dict& kwargs)
{
	py::object range_arg = kwargs.get(name, py::list());
	py::list range_list = py::extract<py::list>(range_arg);
	size_t range_list_len = py::len(range_list);
	
	if (range_list_len > 0) {
		range.resize(range_list_len);
		for (size_t i = 0; i < range_list_len; i++) {
			range[i] = py::extract<size_t>(range_list[i]);
			if (range[i] < 0 || range[i] >= n) {
				BOOST_THROW_EXCEPTION(std::out_of_range("index out of range"));
			}
		}
		std::sort(range.begin(), range.end()); 
		std::vector<size_t>::iterator last = std::unique(range.begin(), range.end());
		range.erase(last, range.end());
	}
	else {
		range.resize(n);
		for (size_t i = 0; i < n; i++) {
			range[i] = i;
		}
	}
}


py::object GetField(py::tuple args, py::dict kwargs)
{
	PyFG& self = py::extract<PyFG&>(args[0]);
	std::string field = py::extract<std::string>(args[1]);

	std::vector<void*> components;
	size_t nx, ny, nz, nzp, elsize;
	void* handle = self.fg()->get_raw_field(field, components, nx, ny, nz, nzp, elsize);

	std::vector<size_t> range_x, range_y, range_z, comp_range;
	ExtractRangeArg("range_x", nx, range_x, kwargs);
	ExtractRangeArg("range_y", ny, range_y, kwargs);
	ExtractRangeArg("range_z", nz, range_z, kwargs);
	ExtractRangeArg("components", components.size(), comp_range, kwargs);

	std::vector<int> dims(4);
	dims[0] = comp_range.size();
	dims[1] = range_x.size();
	dims[2] = range_y.size();
	dims[3] = range_z.size();

	int dtype;
	switch (elsize) {
		case 4: dtype = NPY_FLOAT; break;
		case 8: dtype = NPY_DOUBLE; break;
		default:
			BOOST_THROW_EXCEPTION(std::runtime_error("datatype not supported"));
	}

	py::object obj(py::handle<>(PyArray_FromDims(dims.size(), &dims[0], dtype)));
	void* array_data = PyArray_DATA((PyArrayObject*) obj.ptr());

	if (PyArray_ITEMSIZE((PyArrayObject*) obj.ptr()) != (int)elsize) {
		BOOST_THROW_EXCEPTION(std::runtime_error("problem here"));
	}

	std::vector<size_t> iz_end_vec(range_z.size());
	for (std::size_t iz = 0; iz < range_z.size();) {
		size_t k = range_z[iz];
		size_t iz_end = iz + 1;
		while (iz_end < range_z.size() && range_z[iz_end] == k) { iz_end++; }
		iz_end_vec[iz] = iz_end;
		iz = iz_end;
	}

	#pragma omp parallel for schedule (static) collapse(2)
	for (size_t ic = 0; ic < comp_range.size(); ic++) {
		for (size_t ix = 0; ix < range_x.size(); ix++) {
			size_t c = comp_range[ic];
			size_t i = range_x[ix];
			for (std::size_t iy = 0; iy < range_y.size(); iy++) {
				size_t j = range_y[iy];
				size_t dest0 = ((size_t) array_data) + ((ic*range_x.size() + ix)*range_y.size() + iy)*range_z.size()*elsize;
				size_t src0 = ((size_t) components[c]) + (i*ny + j)*nzp*elsize;
				for (std::size_t iz = 0; iz < range_z.size();) {
					size_t k = range_z[iz];
					size_t iz_end = iz_end_vec[iz];
					void* dest = (void*)(dest0 + iz*elsize);
					void* src = (void*)(src0 + k*elsize);
					memcpy(dest, src, elsize*(iz_end-iz));
					iz = iz_end;
				}
			}
		}
	}

	self.fg()->free_raw_field(handle);

	return obj;
}


//! Convert a vector to Python list
template<class T>
struct VecToList
{
	static PyObject* convert(const std::vector<T>& vec)
	{
		py::list* l = new py::list();
		for(std::size_t i = 0; i < vec.size(); i++) {
			(*l).append(vec[i]);
		}
		return l->ptr();
	}
};


//! Convert a nested vector (rank 2 tensor) to Python list
template<class T>
struct VecVecToList
{
	static PyObject* convert(const std::vector<std::vector<T> >& vec)
	{
		py::list* l = new py::list();
		for(std::size_t i = 0; i < vec.size(); i++) {
			py::list* l2 = new py::list();
			for(std::size_t j = 0; j < vec[i].size(); j++) {
				l2->append(vec[i][j]);
			}
			(*l).append(*l2);
		}
		return l->ptr();
	}
};


//! Convert a nested vector (rank 4 tensor) to Python list
template<class T>
struct VecVecVecVecToList
{
	static PyObject* convert(const std::vector<std::vector< std::vector<std::vector<T> > > >& vec)
	{
		py::list* l = new py::list();
		for(std::size_t i = 0; i < vec.size(); i++) {
			py::list* l2 = new py::list();
			for(std::size_t j = 0; j < vec[i].size(); j++) {
				py::list* l3 = new py::list();
				for(std::size_t m = 0; m < vec[i][j].size(); m++) {
					py::list* l4 = new py::list();
					for(std::size_t n = 0; n < vec[i][j][m].size(); n++) {
						l4->append(vec[i][j][m][n]);
					}
					(*l3).append(*l4);
				}
				(*l2).append(*l3);
			}
			(*l).append(*l2);
		}
		return l->ptr();
	}
};


void translate1(boost::exception const& e)
{
	// Use the Python 'C' API to set up an exception object
	PyErr_SetString(PyExc_RuntimeError, boost::diagnostic_information(e).c_str());
}


void translate2(std::runtime_error const& e)
{
	// Use the Python 'C' API to set up an exception object
	PyErr_SetString(PyExc_RuntimeError, e.what());
}


void translate3(py::error_already_set const& e)
{
	// Use the Python 'C' API to set up an exception object
	//PyErr_SetString(PyExc_RuntimeError, "There was a Python error inside fibergen!");
}


#if PY_VERSION_HEX >= 0x03000000
void* init_numpy() { import_array(); return NULL; }
#else
void init_numpy() { import_array(); }
#endif


py::object PyFGInit(py::tuple args, py::dict kwargs)
{
	py::object pyfg = args[0];
	PyFG& fg = py::extract<PyFG&>(pyfg);
	fg.set_pyfg_instance(pyfg);
	return pyfg;
}


//! Python module for fibergen
class PyFGModule
{
public:
	py::object FG;

	PyFGModule()
	{
		// this is required to return py::numeric::array as numpy array
		init_numpy();

		#if BOOST_VERSION < 106500
		py::numeric::array::set_module_and_type("numpy", "ndarray");
		#endif

		py::register_exception_translator<boost::exception>(&translate1);
		py::register_exception_translator<std::runtime_error>(&translate2);
		py::register_exception_translator<py::error_already_set>(&translate3);

		py::to_python_converter<std::vector<std::string>, VecToList<std::string> >();
		py::to_python_converter<std::vector<double>, VecToList<double> >();
		py::to_python_converter<std::vector<std::vector<double> >, VecVecToList<double> >();
		py::to_python_converter<std::vector<std::vector<std::vector<std::vector<double> > > >, VecVecVecVecToList<double> >();

		void (PyFG::*PyFG_set_string)(const std::string& key, const std::string& value) = &PyFG::set;
		void (PyFG::*PyFG_set_double)(const std::string& key, double value) = &PyFG::set;
		void (PyFG::*PyFG_set_int)(const std::string& key, long value) = &PyFG::set;
		void (PyFG::*PyFG_set)(const std::string& key) = &PyFG::set;

		this->FG = py::class_<PyFG, boost::noncopyable>("FG", "The fibergen solver class")
			.def("init", py::raw_function(&PyFGInit, 0), "Initialize the object with a Python fibergen reference to it self, which can be used (in Python scripts) within a project file as 'fg'. If not called 'fg' will not be available. This step is necessary to expose the correctly wrapped Python object to C++, e.g. fg = fibergen.FG(); fg.init(fg)")
			.def("init_lss", &PyFG::init_lss, "Initialize the Lippmann-Schwinger solver (this is usually done automatically)", py::args("self"))
			.def("init_fibers", &PyFG::init_fibers, "Generate the random geometry (this is usually done automatically)", py::args("self"))
			.def("init_phase", &PyFG::init_phase, "Discretize the geometry (this is usually done automatically)", py::args("self"))
			.def("run", &PyFG::run, "Runs the solver (i.e. the actions in the <actions> section)", py::args("self"))
			.def("run", &PyFG::run_path, "Run actions from a specified path in the XML tree", py::args("self", "path"))
			.def("cancel", &PyFG::cancel, "Cancel a running solver. This can be called in a callback routine for instance.", py::args("self"))
			.def("reset", &PyFG::reset, "Resets the solver to its initial state and unloads any loaded XML file.", py::args("self"))
			.def("get_xml", &PyFG::get_xml, "Get the current project configuration as XML string", py::args("self"))
			.def("set_xml", &PyFG::set_xml, "Load the current project configuration from a XML string", py::args("self"))
			.def("set_xml_precision", &PyFG::set_xml_precision, "Set the precision (number of digits) for representing floating point numbers as XML string attributes", py::args("self", "digits"))
			.def("get_xml_precision", &PyFG::get_xml_precision, "Return the precision (number of digits) for representing floating point numbers as XML string attributes", py::args("self"))
			.def("load_xml", &PyFG::load_xml, "Load a project from a XML file", py::args("self", "filename"))
			.def("set", PyFG_set_string, "Set XML attribute or value of an element. Use set('element-path..attribute', value) to set an attribute value.", py::args("self", "path", "value"))
			.def("set", PyFG_set_double, "Set a floating point property", py::args("self", "path", "value"))
			.def("set", PyFG_set_int, "Set an integer property", py::args("self", "path", "value"))
			.def("set", PyFG_set, "Set a property to an empty value", py::args("self", "path"))
			.def("set", py::raw_function(&SetParameters, 1), "Set a property in the XML tree using a path and multiple arguments or keyword arguments, i.e. set('path', x=1, y=2, z=0) is equivalent to set('path.x', 1), set('path.y', 2), set('path.z', 0)")
			.def("get", &PyFG::get, "Get XML attribute or value of an element. Use get('element-path..attribute') to get an attribute value. If the emelent does not exists returns an empty string", py::args("self", "path"))
			.def("erase", &PyFG::erase, "Remove a path from the XML tree", py::args("self", "path"))
			.def("get_phase_names", &PyFG::get_phase_names, "Return a list of the phase names (materials)", py::args("self"))
			.def("get_volume_fraction", &PyFG::get_volume_fraction, "Get the volume fraction of a phase (material) from the discretized geometry (voxels)", py::args("self", "name"))
			.def("get_real_volume_fraction", &PyFG::get_real_volume_fraction, "Get the volume fraction of a phase (material) using the exact geometry (including multiple overlaps)", py::args("self", "name"))
			.def("get_solve_time", &PyFG::get_solve_time, "Get the total runtime of the solver", py::args("self"))
			.def("get_distance_evals", &PyFG::get_distance_evals, "Get number of distance evaluations for discretizing the geometry", py::args("self"))
			.def("get_residuals", &PyFG::get_residuals, "Get the residual for each iteration", py::args("self"))
			.def("get_effective_property", &PyFG::get_effective_property, "Get the effective property, computet by the action <calc_effective_properties>", py::args("self"))
			.def("get_rve_dims", &PyFG::get_rve_dims, "Get origin (first 3 components) and size of the RVE (last 3 components) as tuple", py::args("self"))
			.def("get_A2", &PyFG::get_A2, "Get second order moment of fiber orientations", py::args("self"))
			.def("get_A4", &PyFG::get_A4, "Get fourth order moment of fiber orientations", py::args("self"))
			.def("get_error", &PyFG::get_error, "Returns true, if there was an error running the solver", py::args("self"))
			.def("get_mean_stress", &PyFG::get_mean_stress, "Returns the mean stress of the current solution", py::args("self"))
			.def("get_mean_strain", &PyFG::get_mean_strain, "Returns the mean strain of the current solution", py::args("self"))
			.def("get_mean_cauchy_stress", &PyFG::get_mean_cauchy_stress, "Returns the mean Cauchy stress of the current solution", py::args("self"))
			.def("get_mean_energy", &PyFG::get_mean_energy, "Returns the mean energy of the current solution", py::args("self"))
			.def("get_field", py::raw_function(&GetField, 1), "Returns solution data as numpy ndarray. Available solution fields are: 'epsilon' (strain), 'sigma' (stress), 'u' (displacement), 'p' (pressure), 'orientation' (fiber orientation vector), 'material_id' (material id), 'fiber_id' (id of the closest fiber), 'distance' (distance to closest interface), 'phi' (phases as listed in the <materials> section), any material name. The first dimension of the returned array denotes the component (3 for vectors, 6 (11,22,33,23,13,12) for symmetric 3x3 matrices, 9 (11,22,33,23,13,12,32,31,21) for 3x3 matrices), the last 3 dimensions address the spatial coordinates.")
			.def("get_B_from_A", &PyFG::get_B_from_A, "Get angular central Gaussian covariance matrix from moment matrix", py::args("self", "A"))
			.def("set_convergence_callback", &PyFG::set_convergence_callback, "Set a callback function to be called each iteration of the solver. If the callback returns True, the solver is canceled.", py::args("self", "func"))
			.def("set_loadstep_callback", &PyFG::set_loadstep_callback, "Set a callback function to be called each loadstep of the solver. If the callback returns True, the solver is canceled.", py::args("self", "func"))
			.def("set_variable", &PyFG::set_variable, "Set a Python variable, which can be later used in XML attributes as Python expressions", py::args("self", "name", "value"))
			.def("set_log_file", &PyFG::set_log_file, "Set filename for capturing the console output", py::args("self", "filename"))
			.def("set_py_enabled", &PyFG::set_py_enabled, "Enable/Disable Python evaluation of XML attributes as Python expressions requested by the solver", py::args("self"))
		;
	}
};


#include FG_PYTHON_HEADER_NAME
// BOOST_PYTHON_MODULE(fibergen)
{
	PyFGModule module;
}


//! exception handling routine for std::set_terminate
void exception_handler()
{
	static bool tried_throw = false;

	#pragma omp critical
	{

	LOG_CERR << "exception handler called" << std::endl;

	try {
		if (!tried_throw) throw;
		LOG_CERR << "no active exception" << std::endl;
	}
	catch (boost::exception& e) {
		LOG_CERR << "boost::exception: " << boost::diagnostic_information(e) << std::endl;
	}
	catch (std::exception& e) {
		LOG_CERR << "std::exception: " << e.what() << std::endl;
	}
	catch (std::string& e) {
		LOG_CERR << e << std::endl;
	}
	catch(const char* e) {
		LOG_CERR << e << std::endl;
	}
	catch(py::error_already_set& e) {
		//PyObject *ptype, *pvalue, *ptraceback;
		//PyErr_Fetch(&ptype, &pvalue, &ptraceback);
		//char* pStrErrorMessage = PyString_AsString(pvalue);
		//LOG_CERR << "Python error: " << pStrErrorMessage << std::endl;
		LOG_CERR << "Python error" << std::endl;
	}
	catch(...) {
		LOG_CERR << "Unknown error" << std::endl;
	}

	// print date/time
	LOG_CERR << "Local timestamp: " << boost::posix_time::second_clock::local_time() << std::endl;

	print_stacktrace(LOG_CERR);

	// restore teminal colors
	LOG_COUT << DEFAULT_TEXT << std::endl;

	} // critical

	// exit app with failure code
	exit(EXIT_FAILURE);
}


//! Run test routines
template<typename T, typename P, int DIM>
int run_tests()
{
	int nfail = 0;

	LOG_COUT << "Running tests for T=" << typeid(T).name() << " P=" << typeid(P).name() << " DIM=" << DIM << "..." << std::endl;

	{
		LOG_COUT << "\n# Test 1" << std::endl;
		LSSolver<T, P, DIM> lss(2, 1, 1, 1, 1, 1);
		nfail += lss.run_tests();
	}
	{
		LOG_COUT << "\n# Test 2" << std::endl;
		LSSolver<T, P, DIM> lss(41, 33, 11, 1, 1, 1);
		nfail += lss.run_tests();
	}
	{
		LOG_COUT << "\n# Test 3" << std::endl;
		LSSolver<T, P, DIM> lss(41, 33, 11, 41, 33, 11);
		nfail += lss.run_tests();
	}
#if 0
	{
		LOG_COUT << "\n# Test 4" << std::endl;
		LSSolver<T, P, DIM> lss(42, 33, 11, 1.1, 10.4, 2.23);
		nfail += lss.run_tests();
	}
#endif

	if (nfail == 0) {
		LOG_COUT << GREEN_TEXT << "ALL TESTS PASSED" << DEFAULT_TEXT << std::endl;
	}
	else if (nfail == 1) {
		LOG_COUT << RED_TEXT << "1 TEST FAILED" << DEFAULT_TEXT << std::endl;
	}
	else {
		LOG_COUT << RED_TEXT << nfail << " TESTS FAILED" << DEFAULT_TEXT << std::endl;
	}

	return nfail;
}


//#include "checkcpu.h"


//! main entry point of application
int main(int argc, char* argv[])
{
	// read program arguments
	po::options_description desc("Allowed options");
	desc.add_options()
	    ("help", "produce help message")
	    ("test", "run tests")
	    ("disable-python-ref", "disable Python FG object reference in project files")
	    ("disable-python-eval", "disable Python code evaluation in project files")
	    ("input-file", po::value< std::string >()->default_value("project.xml"), "input file")
	    ("actions-path", po::value< std::string >()->default_value("actions"), "actions xpath to run in input file")
	;

	po::positional_options_description p;
	p.add("input-file", -1);

	po::variables_map vm;
	po::store(po::command_line_parser(argc, argv).
		  options(desc).positional(p).run(), vm);
	po::notify(vm);

	if (vm.count("help")) {
		// print help
		LOG_COUT << desc << "\n";
		return 1;
	}

	// set exception handler
	std::set_terminate(exception_handler);

	//if (!vm.count("disable-python-ref") || !vm.count("disable-python-eval")) {
		// init python
		Py_Initialize();
	//}

	// run some small problems for checking correctness
	if (vm.count("test")) {
		return run_tests<double, double, 3>();
	}

#if 0
	// check CPU features (only informative)
	if (can_use_intel_core_4th_gen_features()) {
		LOG_COUT << GREEN_TEXT << BOLD_TEXT << "Info: This CPU supports ISA extensions introduced in Haswell!" << std::endl;
	}
#endif

	// run the app
	int ret;
	if (vm.count("disable-python-ref"))
	{
		FGProject fgp;
		fgp.set_py_enabled(vm.count("disable-python-eval") == 0);
		ret = fgp.exec(vm);
		fgp.reset();
	}
	else
	{
		PyFGModule module;
		py::object pyfg = module.FG();
		pyfg.attr("init")();
		PyFG& fgp = py::extract<PyFG&>(pyfg);
		fgp.set_py_enabled(vm.count("disable-python-eval") == 0);
		ret = fgp.exec(vm);
		fgp.reset();
	}

	// return exit code
	return ret;
}

