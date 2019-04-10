test_lasso_sources = [
	'test_lasso.c', 'test_lasso.h',
]

test_lasso_deps = [
	dependency('gsl'),
]

lasso_exe = executable('test_lasso',
	test_lasso_sources,
	dependencies : dependency('gsl'),
	install: true,
)

x2fx_exe = executable('X2_from_X', 'X2_from_X.c')
