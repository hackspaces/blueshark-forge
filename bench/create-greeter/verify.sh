#!/usr/bin/env bash
# Verdict: exit 0 iff greet.py defines greet(name) correctly.
python3 -c "from greet import greet; assert greet('Ada') == 'Hello, Ada!'; assert greet('Bo') == 'Hello, Bo!'; print('ok')"
