from strutil import reverse_words

assert reverse_words("hello world foo") == "foo world hello"
assert reverse_words("single") == "single"
assert reverse_words("") == ""
assert reverse_words("  padded  words  ") == "words padded"
print("ok")
