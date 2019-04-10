lasso_lib_deps = [
	dependency('gsl'),
]

lasso_lib = shared_library('lasso_lib', 'lasso_lib.c', 'lasso_lib.h',
	dependencies: lasso_lib_deps,
)


lasso_exe = executable('lasso_exe', 'lasso_exe.c', link_with: lasso_lib,
	dependencies: dependency('gsl')
)
x2fx_exe = executable('X2_from_X', 'X2_from_X.c', link_with: lasso_lib)
