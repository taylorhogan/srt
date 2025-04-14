import datetime
import sqlite3
import sys
import os
import shutil
import config


def db_full_path():
    cfg = config.data()
    install = cfg['install_location']
    full = os.path.join(cfg['install_location'], 'db/dso.db')
    print (full)
    return full

def db_path():
    cfg = config.data()
    install = cfg['install_location']
    full = os.path.join(cfg['install_location'], 'db')
    print (full)
    return full

class DB:

    def create(self):
        conn = sqlite3.connect(db_full_path())
        c = conn.cursor()
        c.execute(
            "CREATE TABLE if not exists dso(key integer primary key autoincrement, job integer, dso_name TEXT, requested_by TEXT, recipe TEXT, dir TEXT, timestamp DATE, action TEXT, minutes integer, snr float, priority integer, notes TEXT)")
        conn.commit()
        conn.close()

    def exist(self, name):
        conn = sqlite3.connect(db_full_path())
        c = conn.cursor()
        for row in c.execute("SELECT job, dso_name FROM dso ORDER BY dso_name DESC LIMIT 1"):
            if row[1] == name:
                conn.close()
                return True
        conn.close()
        return False

    def add_request(self, dso_name, requester, recipe):
        now = datetime.datetime.now()
        conn = sqlite3.connect(db_full_path())
        job = 1
        c = conn.cursor()
        # s = """INSERT INTO 'dso' ('dso_name', 'job', requested_by','recipe',  'timestamp', 'snr', 'priority', notes') VALUES ( ?, ?, ?,?,?,?,?, ?);"""
        # data_tuple = (
        #     dso_name,
        #     job,
        #     requester,
        #     recipe,
        #     now.strftime("%m/%d/%Y  %H:%M:%S"),
        #     1.0,
        #     0,
        #     "new db")
        # print (data_tuple)
        print("adding " + dso_name)
        s = """INSERT INTO 'dso' ('dso_name', 'job', 'requested_by', 'recipe', 'timestamp') VALUES ( ?,?,?,?,?);"""
        data_tuple = (dso_name, job, requester, recipe, now.strftime("%m/%d/%Y  %H:%M:%S"))

        c.execute(s, data_tuple)
        conn.commit()
        conn.close()

    def add_expo_time(self, name, seconds):
        conn = sqlite3.connect(db_full_path())
        c = conn.cursor()
        for row in c.execute("SELECT job, name, requested_by, expo_time, FROM dso ORDER BY name"):
            if row[1] == name:
                cur_time = row[3]
                cur_time = cur_time + seconds
                print(cur_time)
                conn.close()
                return

        conn.close()

    # def add_expo_time(self):

    def print_table(self):
        conn = sqlite3.connect(db_full_path())
        c = conn.cursor()

        for row in c.execute("SELECT dso_name, requested_by , recipe FROM dso ORDER BY dso_name"):
            print(row)
        conn.close()

    def do_stats(self):
        conn = sqlite3.connect(db_full_path())
        c = conn.cursor()
        expo_time = 0
        rows = 0
        requesters = set()
        dso_objects = set()

        for row in c.execute("SELECT imaged, expo_time, user, name FROM dso ORDER BY expo_time"):
            rows = rows + 1
            expo_time = expo_time + row[1]
            requesters.add(row[2])
            dso_objects.add(row[3])
        return rows, expo_time, requesters, dso_objects

    def make_new_table(self):
        now = datetime.datetime.now()
        db = DB()
        db.create()
        db.add_request("m31", 'Thogan', 'galaxy')
        db.print_table()


def main(argv):
    db = DB()
    print(argv)
    db.print_table()
    print(db.exist("m31"))
    print(db.exist("foo"))
    db.add_expo_time("m31", 60)
    db.print_table()
    now = datetime.datetime.now()
    if not db.exist("m37"):
        db.add("m37", now, now, 0, "testing", "default", "mastodon", 0, "")
    db.print_table()


def get_next_back_name():
    exist = True
    index = 0
    while exist:

        new_db_name = 'dso' + str(index) + ".db"
        filename = os.path.join(db_path(), new_db_name)
        if os.path.isfile(filename):
            index += 1
        else:
            return filename


if __name__ == '__main__':
    db = DB()

    print(sys.argv[1])

    if sys.argv[1] == 'backup':

        cfg = config.data()
        back = get_next_back_name()
        shutil.copyfile('./db/dso.db', back)
        print("Created " + back)

    if sys.argv[1] == "reset":
        db.make_new_table()

    if sys.argv[1] == "print":
        db.print_table()

    if sys.argv[1] == "add":
        if not db.exist(sys.argv[2]):
            db.add_request(sys.argv[2], sys.argv[3], sys.argv[4])


