import json
from iris_astronomy import astro_dso_visibility


def read_base ()->json:

    with open('base_sequence.json', 'r') as f:
        sequence = json.load(f)
    return sequence


def update_dso(sequence:json, dso_json:json)->json:
    pass

def save_updated (sequence):
    with open('updated_sequence.json', 'w') as f:
        f.writelines(json.dumps(sequence, indent=4))




def search_in_json(data, key, replace_with, just_find):
    """
    Recursively search for a key in the given JSON data.
    :param data: The JSON data (parsed as a dictionary or list)
    :param key: The string key to search for
    :return: List of found key-value pairs
    """
    results = []

    if isinstance(data, dict):  # If the JSON object is a dictionary
        for k, v in data.items():
            if k == key:
                results.append({k: v})
                if not just_find:
                   data[key] = replace_with
            if isinstance(v, (dict, list)):  # Recurse if the value is also a dict or list
                results.extend(search_in_json(v, key, replace_with, just_find))

    elif isinstance(data, list):  # If the JSON object is a list
        for item in data:
            if isinstance(item, (dict, list)):
                results.extend(search_in_json(item, key, replace_with, just_find))

    return results

def ra_degrees_to_hms(ra_degrees):
    """
    Convert Right Ascension (RA) in degrees to hours, minutes, and seconds.
    :param ra_degrees: RA in degrees (float)
    :return: Tuple of hours, minutes, seconds
    """
    total_hours = ra_degrees / 15.0  # Convert degrees to hours
    hours = int(total_hours)  # Extract the whole hours
    total_minutes = (total_hours - hours) * 60  # Convert fraction of hours to minutes
    minutes = int(total_minutes)  # Extract the whole minutes
    seconds = (total_minutes - minutes) * 60  # Convert fraction of minutes to seconds

    return hours, minutes, seconds


def dec_degrees_to_dms(dec_degrees):
    """
    Converts a Declination (in degrees) to hours, minutes, and seconds format.

    :param dec_degrees: Declination (or angle) in decimal degrees (float).
    :return: Tuple of (hours, minutes, seconds) retaining the sign.
    """
    # Handle negative declination
    sign = -1 if dec_degrees < 0 else 1
    dec_degrees = abs(dec_degrees)

    # Convert degrees to total hours
    total_hours = dec_degrees - int(dec_degrees)

    # Extract hours
    hours = int(total_hours)

    # Extract minutes
    minutes_float = (total_hours - hours) * 60
    minutes = int(minutes_float)

    # Extract seconds
    seconds = (minutes_float - minutes) * 60

    # Return hours, minutes, and seconds with the correct sign for hours
    return (int(dec_degrees), minutes, seconds, sign)

if __name__ == '__main__':
    dso_name = "m27"
    object = astro_dso_visibility.is_a_dso_object(dso_name)
    print (object.coord)
    ra = ra_degrees_to_hms(object.coord.ra.degree)
    dec = dec_degrees_to_dms(object.coord.dec.degree)
    print ()
    sequence = read_base()
    value = search_in_json(sequence, 'TargetName', 'foo', False)
    search_in_json(sequence, 'RAHOURS', ra[0], False)
    search_in_json(sequence, 'RAMinutes', ra[1], False)
    search_in_json(sequence, 'RASeconds', ra[2], False)
    if dec[3] == -1:
        ndec = "true"
    else:
        ndec = "false"
    search_in_json(sequence, 'NegativeDec', ndec, False)
    search_in_json(sequence, 'DecDegrees', dec[0], False)
    search_in_json(sequence, 'RAMinutes', dec[1], False)
    search_in_json(sequence, 'DecSeconds', dec[2], False)
    # "RAHours": 19,
    # "RAMinutes": 59,
    # "RASeconds": 33.56299,
    # "NegativeDec": false,
    # "DecDegrees": 22,
    # "DecMinutes": 43,
    # "DecSeconds": 19.49152
    print (value)
    save_updated(sequence)