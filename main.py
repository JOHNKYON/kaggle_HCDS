from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from preprocess.preprocessor import Preprocessor


def main():
    print("Main function is running")
    preprocesspor = Preprocessor()
    preprocesspor.read_data("train")


if __name__ == "__main__":
    main()
