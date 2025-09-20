import functools
import json
import logging
from datetime import datetime

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from astropy.time import Time

import astro_dso_visibility
import social_server
import utils

status_dict = {"in process": 3, "waiting": 2, "completed": 1}


def delete_instruction_db(hash_value):
    utils.set_install_dir()
    with open('my_instructions.json', 'r') as f:
        instructions = json.load(f)
    for instruction in instructions:
        if instruction["hash"] == hash_value:
            instructions.remove(instruction)
    with open('my_instructions.json', 'w') as f:
        f.writelines(json.dumps(instructions, indent=4))


def set_completed_instruction_db(hash_value):
    logger = logging.getLogger(__name__)
    logger.info("completing", hash_value)

    utils.set_install_dir()
    with open('my_instructions.json', 'r') as f:
        instructions = json.load(f)
    for instruction in instructions:
        if instruction["hash"] == hash_value:
            instruction["status"] = "completed"
    with open('my_instructions.json', 'w') as f:
        f.writelines(json.dumps(instructions, indent=4))


def remove_hash():
    with open('my_instructions.json', 'r') as f:
        instructions = json.load(f)
    for instruction in instructions:
        if 'hash' in instruction.keys():
            del instruction["hash"]

    with open('my_instructions.json', 'w') as f:
        f.writelines(json.dumps(instructions, indent=4))


def rehash_db():
    remove_hash()
    next_hash = 0
    hash_set = {-1}
    utils.set_install_dir()
    instructions = get_sorted_instructions()

    for instruction in instructions:
        instruction["hash"] = str(next_hash)
        next_hash = next_hash + 1

    with open('my_instructions.json', 'w') as f:
        f.writelines(json.dumps(instructions, indent=4))


def calc_and_store_hours_above_horizon():
    utils.set_install_dir()
    with open('my_instructions.json', 'r') as f:
        instructions = json.load(f)
    for instruction in instructions:
        dso = text = instruction["dso"]
        obj = astro_dso_visibility.is_a_dso_object(dso)
        if obj is not None:
            above, max_altitude  = astro_dso_visibility.get_above_horizon_time(obj, Time.now())
            instruction["above_horizon"] = str(above)
            instruction["air_mass"] = "{:.2f}".format(astro_dso_visibility.air_mass (max_altitude))
        else:
            instruction["above_horizon"] = '0'
            instruction["air_mass"] = '0'
        try:
            value = instruction['best']
        except KeyError:
            best_date, best_time, max_altitude = astro_dso_visibility.best_day_for_dso(object)
            formatted_date = best_date.strftime("%Y-%m-%d")
            instruction['best']=formatted_date + "\n"+str(best_time)


    with open('my_instructions.json', 'w') as f:
        f.writelines(json.dumps(instructions, indent=4))


def time_to_seconds(time_str):
    """Converts a time string (HH:MM:SS or MM:SS or SS) to seconds."""
    if time_str is None:
        return 0
    if time_str == "None":
        return 0

    parts = time_str.split(":")
    parts.reverse()
    seconds = 0
    multiplier = 1
    for part in parts:
        seconds += int(part) * multiplier
        multiplier *= 60
    return seconds


def compare(r1, r2):
    s1 = r1["status"]
    s2 = r2["status"]
    s1p = status_dict.get(s1, 0)
    s2p = status_dict.get(s2, 0)

    if s1p < s2p:
        return 1
    if s1p > s2p:
        return -1

    p1 = r1.get("priority", 5)
    p2 = r2.get("priority", 5)

    if p1 < p2:
        return 1
    if p1 > p2:
        return -1

    oh1 = time_to_seconds(r1.get("above_horizon","0"))
    oh2 = time_to_seconds(r2.get("above_horizon","0"))

    if oh1 < oh2:
        return 1
    if oh1 > oh2:
        return -1

    n1 = r1["dso"]
    n2 = r2["dso"]

    if n1 < n2:
        return 1
    if n1 > n2:
        return -1

    return 0


def create_instructions_table():
    rehash_db()
    calc_and_store_hours_above_horizon()
    sorted_l = get_sorted_instructions()
    logger = logging.getLogger(__name__)
    logger.info("in create")


    per_page = 20
    idx = per_page

    fig, ax = plt.subplots(figsize=(10, per_page))
    ax.axis('off')

    # Set the title
    # ax.set_title("Image Requests", fontsize=20, pad=20)

    # Weekday labels
    headers = ['DSO', 'Requestor', 'State', 'Best Date', 'Tonight', 'Air Mass','ID']
    for i, header in enumerate(headers):
        ax.text(i + 1.5, per_page, header, ha='center', fontsize=12, weight='bold')

    # Generate the  grid

    for instruction in sorted_l[:per_page]:
        # Get color and text for the days

        if instruction["status"] == 'in process':
            color = 'lightgreen'
        if instruction["status"] == 'waiting':
            color = 'lightblue'
        if instruction["status"] == 'completed':
            color = 'pink'
        for col_idx in range(7):
            text = ''

            if col_idx == 0:
                text = instruction["dso"]
            elif col_idx == 1:
                text = instruction["requestor"]
                string = instruction["request_time"]
                if string != "":
                    datetime_object = datetime.strptime(string, '%Y-%m-%d')
                    formatted_date = datetime_object.strftime("%m-%d\n%Y")
                    text += "\n" + formatted_date

            elif col_idx == 2:
                text = instruction["status"]
            elif col_idx == 3:
                text = instruction["best"]
            elif col_idx == 4:
                text = instruction["above_horizon"]
            elif col_idx == 5:
                text = instruction["air_mass"]
            elif col_idx == 6:
                text = instruction["hash"]

            rect = mpatches.Rectangle((col_idx + 1, idx - 1), 1, 1, edgecolor="black", facecolor=color)
            ax.add_patch(rect)
            # Add text
            ax.text(col_idx + 1.5, idx - 0.5, text, ha='center', va='center', fontsize=10)

        idx = idx - 1
    # break


    # Set limits and aspect
    ax.set_xlim(0, 8)
    ax.set_ylim(0, per_page)
    # ax.set_aspect('equal')
    fig.savefig('instructions.png')
    social_server.post_social_message("", "instructions.png")


def add_dso_object_instruction(dso_name, recipe, requestor, priority=0):
    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d")
    with open('my_instructions.json', 'r') as f:
        instructions = json.load(f)
    new_instruction = {
        "dso": dso_name,
        "uuid": "1",
        "recipe": recipe,
        "requestor": requestor,
        "request_time": formatted_date,
        "status": "waiting",
        "priority": priority
    }
    utils.set_install_dir()
    instructions.append(new_instruction)
    with open('my_instructions.json', 'w') as f:
        f.writelines(json.dumps(instructions, indent=4))


def get_sorted_instructions():
    utils.set_install_dir()

    with open('my_instructions.json', 'r') as f:
        instructions = json.load(f)

    sorted_l = sorted(instructions, key=functools.cmp_to_key(compare))
    return sorted_l


def get_dso_object_tonight():
    sorted_l = get_sorted_instructions()

    best = sorted_l[0]

    return best


if __name__ == "__main__":
    rehash_db()
    create_instructions_table()
    #set_completed_instruction_db(0)
    # delete_instruction_db(4)
