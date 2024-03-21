import datetime
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

    def total_expo_time(self):
        conn = sqlite3.connect('./db/dso.db')
        c = conn.cursor()
        sum = 0
        for row in c.execute("SELECT imaged, expo_time FROM dso ORDER BY expo_time"):
            sum = sum + row[1]
        print("sum: " + str(sum))


now = currentDateTime = datetime.datetime.now()
db = DB()
db.create()

db.add("m31", now, now, 0, "thogan", "lrgb", "test", 200, "m31.jpg")
db.add("m32", now, now, 0, "thogan", "HaSI", "test", 300, "m32.jpg")
db.printtable()
db.total_rows()
#
db.total_expo_time()

