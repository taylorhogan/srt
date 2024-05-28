import datetime
import sqlite3


class DB:
    def create(self):
        conn = sqlite3.connect('db/dso.db')
        c = conn.cursor()
        c.execute(
            "CREATE TABLE if not exists dso(key integer primary key autoincrement, name TEXT, requested, finished, imaged, user, recipe, notes, expo_time, file)")
        conn.commit()
        conn.close()

    def exist(self, name):
        conn = sqlite3.connect('db/dso.db')
        c = conn.cursor()
        for row in c.execute("SELECT key, name FROM dso ORDER BY name"):
            if row[1] == name:
                conn.close()
                return True
        conn.close()
        return False

    def add(self, name, requested, finished, imaged, user, recipe, notes, expo_time, file):
        conn = sqlite3.connect('db/dso.db')

        c = conn.cursor()
        s = """INSERT INTO 'dso' ('name', 'requested', 'finished','imaged','user','recipe', 'notes', 'expo_time', 'file') VALUES ( ?, ?, ?,?,?,?,?,?,?);"""
        data_tuple = (
            name, requested.strftime("%m/%d/%Y  %H:%M:%S"), finished.strftime("%m/%d/%Y %H:%M:%S"), imaged, user,
            recipe,
            notes, expo_time, file)

        c.execute(s, data_tuple)
        conn.commit()
        conn.close()

    def add_expo_time(self, name, seconds):
        conn = sqlite3.connect('db/dso.db')
        c = conn.cursor()
        for row in c.execute("SELECT key, name, user, expo_time  FROM dso ORDER BY name"):
            if row[1] == name:
                cur_time = row[3]
                cur_time = cur_time + seconds
                print(cur_time)
                conn.close()
                return

        conn.close()

    # def add_expo_time(self):

    def print_table(self):
        conn = sqlite3.connect('db/dso.db')
        c = conn.cursor()
        for row in c.execute("SELECT key, name, user, expo_time requested FROM dso ORDER BY name"):
            print(row)
        conn.close()

    def do_stats(self):
        conn = sqlite3.connect('db/dso.db')
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
        db.add("m31", now, now, 0, "Thogan", "lrgb", "test", 200, "m31.jpg")
        db.printtable()

    def unit_test(self):
        db = DB()
        db.print_table()
        print(db.exist("m31"))
        print(db.exist("foo"))
        db.add_expo_time("m31", 60)
        db.print_table()
        now = datetime.datetime.now()
        if not db.exist("m37"):
            db.add("m37", now, now, 0, "testing", "default", "mastodon", 0, "")
        db.print_table()


