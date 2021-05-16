import sqlite3
from sqlite3 import Error


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn


def select_all_tasks(conn):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("""select name from sqlite_master where type = 'table' """)

    rows = cur.fetchall()

    for row in rows:
        print(row)


def select_task_by_priority(conn, priority):
    """
    Query tasks by priority
    :param conn: the Connection object
    :param priority:
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM sampleSetOne")

    rows = cur.fetchall()

    for row in rows:
        print(row)


import pandas as pd


def main():
    database = r"C:\Users\vpavl\test\tkteach\storage.db"

    # create a database connection
    conn = create_connection(database)
    print(conn)
    data = pd.DataFrame(columns=['LabelName','ImageName', 'LabelId']
    )
    with conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT categories.categoryName, imageName, categories.id from categories "
                "inner join  "
                    "(SELECT images.id, images.imageName, labels.category_id  FROM images "
                        "inner join "
                        "labels on images.id = labels.image_id) as tmp "
                "on tmp.category_id = categories.id"
        )

        # cur.execute("PRAGMA table_info(categories)")
        rows = cur.fetchall()

        for row in rows:
            data = data.append({'LabelName': row[0], 'ImageName': row[1], 'LabelId': row[2]}, ignore_index=True)
            # print(row)
    return data


if __name__ == '__main__':
    annotation = main()
    annotation.to_csv('letters_val.csv')
    annotation = annotation.drop([i for i in range(1223)], axis=0)
    annotation.to_csv('letters2_val.csv', index=False)

