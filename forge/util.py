"""Small shared helpers used across modules."""


def slurp(path):
    """Read a whole file as text, tolerating odd bytes. Closes the handle."""
    with open(path, errors="replace") as f:
        return f.read()


def dump(path, text):
    """Write text to a file, closing the handle."""
    with open(path, "w") as f:
        f.write(text)
