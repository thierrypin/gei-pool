import os
import errno




# Creates a new directory, ignoring if it already exists
# Similar to `mkdir -p` shell command
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python > 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

