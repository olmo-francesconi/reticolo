########################################### autoconf and automake initialization
AC_PREREQ([2.69])
AC_INIT([reticolo], [0.1], [olmo.francesconi@glasgow.ac.uk], [reticolo])
AM_INIT_AUTOMAKE([foreign subdir-objects])
AC_CONFIG_SRCDIR([src/reticolo.hpp])

AC_PROG_RANLIB

############################################################ Check c++ compiler
AC_PROG_CXX

################################################# Check Language specifications
AC_LANG([C++])

#################################################### Check C++ standard version
AX_CXX_COMPILE_STDCXX(20, noext, mandatory)

######################################################################## OpenMP
AC_OPENMP
if test "${ac_cv_prog_cxx_openmp}X" != "unsupportedX"; then
  AM_CXXFLAGS="${AM_CXXFLAGS} ${OPENMP_CXXFLAGS}"
  AM_LDFLAGS="${AM_LDFLAGS} ${OPENMP_CXXFLAGS}"
  AC_DEFINE([OPENMP_SUPPORTED], [], [OpenMP support])
fi

# ########################################################################## HDF5
AC_ARG_WITH([hdf5],
    [AS_HELP_STRING([--with-hdf5=prefix],
    [try this for a non-standard install prefix of the HDF5 library])],
    [AM_CXXFLAGS="-I$with_hdf5/include $AM_CXXFLAGS"]
    [AM_LDFLAGS="-L$with_hdf5/lib $AM_LDFLAGS"])

############################################################### Check libraries
CXXFLAGS_CPY=$CXXFLAGS
LDFLAGS_CPY=$LDFLAGS
CXXFLAGS="$AM_CXXFLAGS $CXXFLAGS"
LDFLAGS="$AM_LDFLAGS $LDFLAGS"

AC_SEARCH_LIBS([H5Fopen], [hdf5_cpp],
               [AC_DEFINE([HAVE_HDF5], [1], [Define to 1 if you have the `HDF5' library])]
               [have_hdf5=true]
               [LIBS="${LIBS} -lhdf5"], [], [-lhdf5])
AM_CONDITIONAL(BUILD_HDF5, [ test "${have_hdf5}X" == "trueX" ])

CXXFLAGS=$CXXFLAGS_CPY
LDFLAGS=$LDFLAGS_CPY
  
########################################################## Set output variables
AC_SUBST([AM_CXXFLAGS])
AC_SUBST([AM_LDFLAGS])

######################################################## Set config.h file path
AC_CONFIG_HEADERS([config.h])

############################################################ Set Makefile paths
AC_CONFIG_FILES([
  Makefile
  src/Makefile
  test/Makefile
])

############################################# Set silent output for compilation
AM_SILENT_RULES([yes]) 

############################################################### Write everthing
AC_OUTPUT