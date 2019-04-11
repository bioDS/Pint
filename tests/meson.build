misc_tests = executable('misc-tests', 'misc-tests.c', dependencies: dependency('glib-2.0'), link_with: lasso_lib)
func_tests = executable('func-tests', 'func-tests.c', dependencies: dependency('glib-2.0'), link_with: lasso_lib)

test('misc tests', misc_tests, args: ['misc'],)
test('func tests', func_tests, args: ['func'],)
