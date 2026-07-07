from mathx import add, double

assert add(2, 3) == 5, "add(2, 3) should be 5"
assert add(0, 0) == 0, "add(0, 0) should be 0"
assert add(-1, 1) == 0, "add(-1, 1) should be 0"
assert double(4) == 8, "double(4) should be 8"
print("ok")
