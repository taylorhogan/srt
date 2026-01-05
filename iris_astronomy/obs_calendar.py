import calendar
import json
import os,sys
from datetime import date
from astropy.time import Time

from astroplan import moon_illumination

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),  '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from configs import config


def generate_custom_calendar(year, month, cal_data, cfg):
    # Generate the calendar for the given month
    cal = calendar.Calendar()
    days = cal.itermonthdays2(year, month)

    # Initialize the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')

    # Set the title
    ax.set_title(calendar.month_name[month] + f" {year}", fontsize=20, pad=20)

    # Weekday labels
    weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for i, weekday in enumerate(weekdays):
        ax.text(i + 0.5, 6.5, weekday, ha='center', fontsize=12, weight='bold')

    # Generate the calendar grid
    row = 5
    col = 7
    for (day, weekday) in days:
        if day == 0:
            continue  # Skip days not in the month
        week_row = row
        col_pos = weekday
        specific_date = Time(str(year) + '-' + str(month) + '-' + str(day) + 'T00:00:00',format='isot', scale='utc')
        illumination = int (float(moon_illumination(specific_date) * 100))
        # Get color and text for the day
        s = get_day_stats(cal_data, str(year), str(month), str(day))
        if s is None:
            color, text = 'white', str(day)
        else:
            state = s.get("state")
            color = cfg["Calendar"][state]
            text = str(day)
            if state == 'image':
                text = text + "\n" + s.get("dso")
            elif state == 'weather':
                text = text + "\nweather"
            elif state == 'service':
                text = text + "\nservice"
        text = text + "\n" + str(illumination) + "%"

        # Draw day cell
        rect = mpatches.Rectangle((col_pos, week_row), 1, 1, edgecolor="black", facecolor=color)
        ax.add_patch(rect)

        # Add text
        ax.text(col_pos + 0.5, week_row + 0.5, text, ha='center', va='center', fontsize=12)

        if weekday == 6:
            row = row - 1

    # Set limits and aspect
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')

    # plt.show()
    fig.savefig('cal.png')


def read_cal():
    with open('../my_calendar.json', 'r') as f:
        cal = json.load(f)

    return cal


def get_day_stats(cal, y, m, d):
    years = cal['years']
    year = years.get(y)
    if year is None:
        return None
    months = year['months']
    month = months.get(m)
    if month is None:
        return None
    days = month['days']
    day = days.get(d)
    if day is None:
        return None
    return day


def print_month(y, m, cfg):
    generate_custom_calendar(y, m, read_cal(), cfg)


def set_today_stat(state, dso):
    cal = read_cal()
    today = date.today()
    this_year = str(2026)
    this_month = str(today.month)
    this_day = str(today.day)

    years = cal['years']
    year = years.get(this_year)
    if year is None:
        years[this_year] = {}
        year = years.get(this_year)
    months = year.get('months')
    month = months.get(this_month)
    if month is None:
        months[this_month] = {'days': {}}
        month = months.get(this_month)
    days = month.get('days')
    day = days.get(this_day)
    if day is None:
        days[this_day] = {}
        day = days.get(this_day)
    print(day)
    print(state)
    print(dso)
    day['state'] = state
    day['dso'] = dso
    with open("../my_calendar.json", "w") as outfile:
        json.dump(cal, outfile, indent=4)

    return cal


if __name__ == "__main__":
    cfg = config.data()
    path = os.path.join(cfg["Install"], 'iris.log')
    print_month(2026, 2, cfg)
