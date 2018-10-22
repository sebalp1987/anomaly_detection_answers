import STRING
import pickle
import csv
import sqlite3
import glob
import os


def directory(dirname, db):
    for filename in glob.glob(os.path.join(dirname, '*.csv')):
        file(filename, db)


def file(filename, db):
    with open(filename) as f:
        with db:
            data = csv.DictReader(f, delimiter=';')

            # we get the cols
            cols = data.fieldnames

            # we return the final component of the filename
            table = os.path.splitext(os.path.basename(filename))[0]

            # Drop if table exists
            sql = 'DROP TABLE IF EXISTS "{}"'.format(table)
            db.execute(sql)

            # We create the table with the CSV data
            sql = 'CREATE TABLE "{table}" ({cols})'.format(table=table, cols=','.join('"{}"'.format(col) for col in cols))
            db.execute(sql)

            # We insert the data
            sql = 'INSERT INTO "{table}" VALUES ({vals})'.format(table=table, vals=','.join('?' for col in cols))
            db.executemany(sql, (list(map(row.get, cols)) for row in data))

if __name__ == '__main__':
    parent_dir = os.path.dirname(os.getcwd())
    os.chdir(parent_dir)
    connection = sqlite3.connect('{}.db'.format(parent_dir + STRING.path_test_db))
    directory(parent_dir + STRING.path_test, connection)
