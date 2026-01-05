import json


def import_and_change (path, dso_name, az, ra):

    with open(path, 'r') as f:
        data = json.load(f)

    # Access the data
    print(data)


if __name__ == '__main__':
    main()