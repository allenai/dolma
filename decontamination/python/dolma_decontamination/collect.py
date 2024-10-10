from .tasks import *


def main():
    i = 0
    for doc in cb().docs():
        print(doc)
        i += 1
        if i > 100:
            break


if __name__ == "__main__":
    main()
