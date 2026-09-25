"""shared helpers for the test suite"""

import doctest


def doctests_for(*modules):
    """a load_tests hook adding the doctests of the given modules"""
    def load_tests(loader, tests, ignore):
        for module in modules:
            tests.addTests(doctest.DocTestSuite(module))
        return tests
    return load_tests
