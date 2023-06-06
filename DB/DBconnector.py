import sqlite3
import os.path

data = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "acupuncture_point.db")


# dict[name]:[rel_position, offset_x, offset_y, offset_z, customize]

def get_sql_data(data, table_name):
    """
    fill data to dict
    :param data: data to store db data
    :param table_name: ActionTable、AcupuncturePoint、ESymptomTable、
    ObjectTable、PersonTable、PositionTable、SymptomTable
    :return: none
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    print("SUCCESS TO CONNECT DATA BASE", end=", ")

    cursor = conn.execute("SELECT * from " + table_name)
    conn.commit()
    for row in cursor:
        if table_name == "New_acupuncture_point":
            # print(row[12])
            #  name: str
            #  rel_point: int
            #  offset_x: int
            #  offset_y: int
            #  handness: int
            #  type str
            data[row[1]] = [int(row[8]), int(row[9]), int(row[10]), int(row[11]), int(row[12]), str(row[6]).split(",")]

    print("ALL OPERATION IS SUCCESS")
    conn.close()


def get_acupuncture_point_data(acupuncture_name, table_name):
    """
        return acupuncture data: way、summary、 type、message、content、name
        :param acupuncture_name: name of acupuncture
        :param table_name: ActionTable、AcupuncturePoint、ESymptomTable、
        ObjectTable、PersonTable、PositionTable、SymptomTable
        :return: one acupuncture data
        """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    print("SUCCESS TO CONNECT DATA BASE")

    cursor = conn.execute("SELECT * from " + table_name + " WHERE title='" + acupuncture_name + "'")
    conn.commit()
    for row in cursor:
        if table_name == "New_acupuncture_point":
            # way、summary、 type、message、content、name
            return [str(row[2]), str(row[4]), str(row[6]), str(row[7]), str(row[5]), str(row[3])]
    conn.close()


def print_disease(da):
    for k, v in da.items():
        print(str(k) + ": " + v)


def type_to_disease(dist, str):
    """
    use num to find the disease in disease dist
    :param dist: disease dist
    :param num: type
    :return: acupuncture_point
    """
    for k, v in dist.items():
        if k == str:
            return v[0]


if __name__ == '__main__':
    get_sql_data(data, "New_acupuncture_point")
    ss = []
    for k, v in data.items():
        print(str(k) + ": " + str(v))
        for i in v[5]:
            ss.append(i)
    print(set(ss))

    # print(get_acupuncture_point_data("四白穴", "New_acupuncture_point"))
    # way, summary, types, massage, content, name = get_acupuncture_point_data("少商穴", "New_acupuncture_point")
    # print(way)
    # print(way[0:len(way)])
    # print(way[16:20])
    # print(len(way))

    # content_list = content.split("；")
    # content_len = len(content_list)
    # print(content_list)
    # print(content_len)

    # usr_input = ' '
    #
    # while usr_input != '-1':
    #     usr_input = input("enter: ")
    #     print(type_to_disease(data, usr_input))
