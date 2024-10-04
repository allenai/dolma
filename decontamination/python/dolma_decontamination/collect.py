from .tasks import squad, coqa


def main():
    i = 0
    for doc in coqa().docs():
        print(doc)
        i += 1
        if i > 10:
            break


if __name__ == "__main__":
    main()
