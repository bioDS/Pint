#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([FULL-PACKAGE-NAME], [VERSION], [BUG-REPORT-ADDRESS])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])
AC_CONFIG_SRCDIR([config.h.in])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIRS([m4])
m4_ifdef([AM_PROG_AR], [AM_PROG_AR])
LT_INIT

# Checks for programs.
AC_PROG_CC
AC_PROG_CXX
AC_PROG_AWK
AC_PROG_INSTALL
AC_PROG_MKDIR_P

# Checks for libraries.
PKG_CHECK_MODULES(GLIB, glib-2.0)
PKG_CHECK_MODULES(GSL, gsl)
PKG_CHECK_MODULES(NCURSES, ncurses)
# FIXME: Replace `main' with a function in `-largeArrayDims':
AC_CHECK_LIB([argeArrayDims], [main])
# FIXME: Replace `main' with a function in `-lgomp':
AC_CHECK_LIB([gomp], [main])

# Checks for header files.
AC_CHECK_HEADERS([locale.h memory.h stdlib.h string.h sys/time.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_SIZE_T

# Checks for library functions.
AC_FUNC_ERROR_AT_LINE
AC_FUNC_MALLOC
AC_FUNC_REALLOC
AC_FUNC_STRTOD
AC_CHECK_FUNCS([clock_gettime memset pow setlocale sqrt])

AC_CONFIG_FILES([Makefile
				 src/Makefile])
AC_OUTPUT
