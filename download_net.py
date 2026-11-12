#!/usr/bin/env python3

import urllib.request
import hashlib
import os

def main():
    name = "goliath-sb670"
    hash = "b04afc7560d0e147c4a311efddf1a4a490e78d83edb3dc28ad2a73c50215abe6"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nets", "goliath.nnue")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        if hashlib.sha256(open(path, "rb").read()).digest().hex() == hash:
            print("Net already exists!")
            return
    except OSError:
        pass

    print(f"Downloading net {name} to {path}")
    net = urllib.request.urlopen(
        f"https://github.com/Silverrzz/sable-nets/releases/download/{name}/{name}.nnue"
    ).read()
    if hashlib.sha256(net).digest().hex() != hash:
        print("Invalid hash!")
        exit(1)
    open(path, "wb").write(net)

if __name__ == "__main__":
    main()
