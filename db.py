import sqlite3


class DB:
    def create(self):
        conn = sqlite3.connect('./db/dso.db')
        c = conn.cursor()
        c.execute(
            "CREATE TABLE if not exists dso(key integer primary key autoincrement, name TEXT, requested, finished, imaged, user, recipe, notes, expo_time, file)")
        conn.commit()
        conn.close()

    def add(self, name, requested, finished, imaged, user, recipe, notes, expo_time, file):
        conn = sqlite3.connect('./db/dso.db')

        c = conn.cursor()
        s = """INSERT INTO 'dso' ('name', 'requested', 'finished','imaged','user','recipe', 'notes', 'expo_time', 'file') VALUES ( ?, ?, ?,?,?,?,?,?,?);"""
        data_tuple = (
            name, requested.strftime("%m/%d/%Y  %H:%M:%S"), finished.strftime("%m/%d/%Y %H:%M:%S"), imaged, user,
            recipe,
            notes, expo_time, file)

        c.execute(s, data_tuple)
        conn.commit()
        conn.close()

    def printtable(self):
        conn = sqlite3.connect('./db/dso.db')
        c = conn.cursor()
        for row in c.execute("SELECT key, name, user, requested FROM dso ORDER BY name"):
            print(row)
        conn.close()

    def do_stats(self):
        conn = sqlite3.connect('./db/dso.db')
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

# now = currentDateTime = datetime.datetime.now()
# db = DB()
# db.create()
#
# db.add("m31", now, now, 0, "thogan", "lrgb", "test", 200, "m31.jpg")
# db.add("m32", now, now, 0, "thogan", "HaSI", "test", 300, "m32.jpg")
# db.printtable()
# db.total_rows()
# #
# db.total_expo_time()
