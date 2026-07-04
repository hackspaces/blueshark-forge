# -*- coding: utf-8 -*- 

import unittest
from forge.__main__ import cmd_run

def run_test(task, expected_output):
    # Mock dependencies for testing purposes
    class MockBackend:
        def __init__(self, name):
            self.name = name
        def send(self, text):
            return f'Response to {text}'
    backend = MockBackend('test_model')
session = ... # Mock session object
cmd_run(task=task, model='test_model', dir='/tmp', max_steps=10)
assert expected_output in output

class TestMain(unittest.TestCase):
    def test_simple_task(self):
        task = 'Hello world'
        expected_output = 'Response to Hello world'
        task = 'Hello world'
    my_task = 'Hello world'
    run_test(my_task, expected_output)

    # Add more tests for different scenarios here
if __name__ == '__main__':
    unittest.main()