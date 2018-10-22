import sqlite3
import pickle
import STRING
import os
import json


def sql_to_pickle(folder_in, db_file):

    # We bring the files we want to transform from SQL to Pickle
    files_name = [f for f in os.listdir(folder_in) if f.endswith('.csv')]

    # Connect to DB
    connection = sqlite3.connect('{}.db'.format(db_file))

    # Cursor to fetch the results
    c = connection.cursor()

    for file in files_name:
        data_list = []
        # SQL query
        table = file.split('.')[0]
        sql = 'SELECT * FROM {}'.format(table)
        c.execute(sql)

        # We fetch columns names
        data_list.append([description[0] for description in c.description])

        # Fetch the data
        data_list.append(c.fetchall())

        # data to array
        # data = np.array(data_list)

        # save the pickle file
        pickle_file_name = '\\' + file.split('.')[0] + '.pickle'

        with open(folder_in + pickle_file_name, 'wb') as f:
            pickle.dump(data_list, f)
            f.close()


def sql_to_json(folder_in, db_file):

    # We bring the files we want to transform from SQL to Pickle
    files_name = [f for f in os.listdir(folder_in) if f.endswith('.csv')]

    # Connect to DB
    connection = sqlite3.connect('{}.db'.format(db_file))

    # Cursor to fetch the results
    c = connection.cursor()

    for file in files_name:
        data_list = []

        # SQL query
        table = file.split('.')[0]
        sql = 'SELECT * FROM {}'.format(table)
        c.execute(sql)

        # We fetch columns names
        data_list.append([description[0] for description in c.description])

        # Fetch the data
        data_list.append(c.fetchall())

        # save the json file
        json_file_name = '\\' + file.split('.')[0] + '.txt'

        with open(folder_in + json_file_name, 'w', encoding='utf8') as f:
            json.dump(data_list, f)


if __name__ == '__main__':
    parent_dir = os.path.dirname(os.getcwd())
    sql_to_pickle(parent_dir + STRING.path_test, parent_dir + STRING.path_test_db)
    # sql_to_json(parent_dir + STRING.path_test, parent_dir + STRING.path_test_db)
