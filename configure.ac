AC_INIT([<package>],[<version>])

PACKAGE_NAME="Pint"
PACKAGE_VERSION="0.2"

# Find the compiler and compiler flags used by R.
: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
  echo "could not determine R_HOME"
  exit 1
fi
CXX=`"${R_HOME}/bin/R" CMD config CXX`
CXXFLAGS=`"${R_HOME}/bin/R" CMD config CXXFLAGS`
CPPFLAGS=`"${R_HOME}/bin/R" CMD config CPPFLAGS`
AC_LANG(C++)
AC_PROG_CPP
AC_OPENMP

have_xxhash=no
XXH_CXXFLAGS=""
XXH_LIBS=""
XXH_MIN_VERSION=0.8

PKG_PROG_PKG_CONFIG

if test [ -n "$PKG_CONFIG" ] ; then
  AC_MSG_CHECKING([pkg-config for xxhash])
  if $PKG_CONFIG libxxhash --atleast-version=${XXH_MIN_VERSION}; then
    have_xxhash=yes
    XXH_CXXFLAGS=`"${PKG_CONFIG}" --cflags libxxhash`
    XXH_LIBS=`"${PKG_CONFIG}" --libs libxxhash`
  fi
  AC_MSG_RESULT([${have_xxhash}])
fi

#if test "${have_xxhash}" = no; then
#  AC_SEARCH_LIBS(XXH3_state_t, xxhash, [], [AC_ERROR([The xxhash library (>= 0.8) is required.])])
#  AC_CHECK_HEADERS(xxhash.h, [have_xxhash=yes], [AC_ERROR([The xxhash library headers (>= 0.8) are required.])])
#fi

if test "${have_xxhash}" = no; then
    try_bundled=true
fi

if test "${try_bundled}" = true; then
    AC_MSG_WARN("Couldn't find xxhash >= 0.8, attempting to use bundled version")
    make -C xxHash-0.8.1
    PKG_LIBS="$(pwd)/xxHash-0.8.1/libxxhash.a"
fi

AC_SUBST([PKG_LIBS], ["${LIBS} ${PKG_LIBS} ${XXH_LIBS}"])


# Write the flags into the src/Makevars file.
AC_SUBST([PKG_CPPFLAGS], ["${PKG_CPPFLAGS}"])
AC_SUBST([PKG_LIBS], ["${LIBS} ${PKG_LIBS} ${ac_cv_search_deflate}"])
AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT

echo "
  --------------------------------------------------
  Configuration for ${PACKAGE_NAME} ${PACKAGE_VERSION}

    cppflags: ${CPPFLAGS} ${PKG_CPPFLAGS}
    libs:     ${PKG_LIBS}

  --------------------------------------------------
"

