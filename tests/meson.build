cc = meson.get_compiler('c')

vp4c = cc.find_library('vp4c', dirs: ['/home/kieran/work/lasso_testing/TurboPFor'])
vp4d = cc.find_library('vp4d', dirs: ['/home/kieran/work/lasso_testing/TurboPFor'])
bitpack = cc.find_library('bitpack', dirs: ['/home/kieran/work/lasso_testing/TurboPFor'])
bitunpack = cc.find_library('bitunpack', dirs: ['/home/kieran/work/lasso_testing/TurboPFor'])
bitutil = cc.find_library('bitutil', dirs: ['/home/kieran/work/lasso_testing/TurboPFor'])
vint = cc.find_library('vint', dirs: ['/home/kieran/work/lasso_testing/TurboPFor'])
bitunpack_sse = cc.find_library('bitunpack_sse', dirs: ['/home/kieran/work/lasso_testing/TurboPFor'])

turbopfor_deps = [
	vp4c,
	vp4d,
	bitpack,
	bitunpack,
	bitunpack_sse,
	bitutil,
	vint,
]

misc_tests = executable('misc-tests', 'misc-tests.c', dependencies: dependency('glib-2.0'), link_with: liblasso)
func_tests = executable('func-tests', 'func-tests.c', dependencies: [dependency('glib-2.0'), dependency('gsl'), turbopfor_deps], link_with: liblasso)

test('misc tests', misc_tests, args: ['-p', '/misc'],)
test('read_y', func_tests, args: ['-p', '/func/test-read-y-csv'],)
test('read_x', func_tests, args: ['-p', '/func/test-read-x-csv'],)
test('soft_threshold', func_tests, args: ['-p', '/func/test-soft-threshol'],)
test('update_beta_cyclic', func_tests, args: ['-p', '/func/test-update-beta-cyclic'],)
test('update_intercept_cyclic', func_tests, args: ['-p', '/func/test-update-intercept-cyclic'],)
test('X2_from_X', func_tests, args: ['-p', '/func/test-X2_from_X'],)
test('cyclic coordinate descent - interactions', func_tests, args: ['-p', '/func/test-simple-coordinate-descent-int'],)
test('cyclic coordinate descent - shuffled interactions', func_tests, args: ['-p', '/func/test-simple-coordinate-descent-int-shuffle'],)
test('compare_n', func_tests, args: ['-p', '/func/test-check-n'],)
test('merge_n', func_tests, args: ['-p', '/func/test-merge-n'],)
test('X2 encoding', func_tests, args: ['-p', '/func/test-X2-encoding'],)
