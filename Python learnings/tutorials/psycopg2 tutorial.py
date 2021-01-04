
import psycopg2  
con = psycopg2.connect(database='dvdrental - test', user='postgres', password='nico')   #this is where we connect to a specific database and specific user
print('Database open successfully')

con.autocommit = False                          #with such wording, no actions here in Python will be committed into the SQL database unless I commit manually
cur = con.cursor()                              #we create a cursor to establish the connection and run basic ops
cur.execute('SELECT * from customer')           #we use cur.execute to run PostgreSQL commands. NB. write the text inside here directly to avoid using variables messing up...
data = cur.fetchone()                           #we run fetchone, fetchmany(n), fetchall to show what's inside tables with the SELECT function
con.commit() / cxon.rollback()                  #such commands are to commit changes, or to go basck to previous version
cur.close()
con.close()                                     #we close the connection with the database (every change will be lost unless we commit)
print('Database closed')





# ---------------------------------------------------
# ---------------------------------------------------
# ---------------------------------------------------

# General tutorial - SEE BELOW for classes
# ## Problem: 
# ### Your data is in a SQL database, but your machine learning tools are in Python.
# 
# ## Solution:
# ### Run SQL queries from Python
# 
# * Very useful for scaled data pipelines, pre-cleaning, data exploration
# * Allows for dynamic query generation
# 
# ---------
# ### Psycopg2 is a library that allows Python to connect to an existing PostgreSQL database to utilize SQL functionality.
# 
# basic usage when I want to run queries....but watch out that it is better to write the text directly into execute as suggested above...
query = "SELECT * FROM some_table;"
cursor.execute(query)
results = cursor.fetchall()
#
query1 = '''
        CREATE TABLE logins (
            userid integer
            , tmstmp timestamp
            , type varchar(10)
        );
        '''
#
cur.execute(query1)
#
query2 = '''
        COPY logins 
        FROM '/Users/brad/Dropbox/Galvanize/sql-python/data/lecture-example/logins01.csv' 
        DELIMITER ',' 
        CSV;
        '''
#
#
cur.execute(query2)
#
query3 = '''
        SELECT *
        FROM logins
        LIMIT 20;
        '''
#
#
cur.execute(query3)

# ---------------------------------------------------
# Another good example: create a bigger table...

import psycopg2

cars = (
    (1, 'Audi', 52642),
    (2, 'Mercedes', 57127),
    (3, 'Skoda', 9000),
    (4, 'Volvo', 29000),
    (5, 'Bentley', 350000),
    (6, 'Citroen', 21000),
    (7, 'Hummer', 41400),
    (8, 'Volkswagen', 21600)
)

con = psycopg2.connect(database='testdb', user='postgres',
                    password='s$cret')

with con:                               #with seems to be used to handle issues, so that in case of errors there is no commit to ruin the database: research more

    cur = con.cursor()

    cur.execute("DROP TABLE IF EXISTS cars")
    cur.execute("CREATE TABLE cars(
                id SERIAL PRIMARY KEY, 
                name VARCHAR(255), 
                price INT)"
                )

    query = "INSERT INTO cars (id, name, price) VALUES (%s, %s, %s)"
    cur.executemany(query, cars)

    con.commit()



# ---------------------------------------------------
# ---------------------------------------------------
# ---------------------------------------------------
# BEST PRACTICE
# Better than the above: let's set up classes to manage database connections, open and close connections and run queries
# we call connection conn instead of con here
# watch out for the code below as some lines seem not to be from python (such as line 165 below without print and parenthesis)

import sys
import logging
import psycopg2
import psycopg2.extras

class Database:
    #PostgreSQL Database class

    def __init__(self, config):
        self.host = config.db_host
        self.username = config.db_user
        self.password = config.db_password
        self.port = config.db_port
        self.dbname = config.db_name
        self.conn = None

    def open_connection(self):
        #Connect to a Postgres database
        try:
            if(self.conn is None):
                self.conn = psycopg2.connect(host=self.host,
                                             user=self.username,
                                             password=self.password,
                                             port=self.port,
                                             dbname=self.dbname)
        except psycopg2.DatabaseError as e:
            logging.error(e)
            sys.exit()
        finally:
            logging.info('Connection opened successfully.')

# The above open_connection() function is safe enough to be called at any time in our application because we're checking to see if self.conn is open first 
# (we could really call this function open_database_connection_if_not_already_open(), but thats a bit wordy).

    def run_query(self, query):
    	#function to run queries
        try:
            self.open_connection()
            with self.conn.cursor() as cur:
                if 'SELECT' in query:
                    records = []
                    cur.execute(query)
                    result = cur.fetchall()
                    for row in result:
                        records.append(row)
                    cur.close()
                    return records
                else:
                    result = cur.execute(query)
                    self.conn.commit()
                    affected = f"{cur.rowcount} rows affected."
                    cur.close()
                    return affected
        except psycopg2.DatabaseError as e:
            print(e)
        finally:
            if self.conn:
                self.conn.close()
                logging.info('Database connection closed.')

# EXTRAS
# EXTRAS
# There are many extras, but in detail...
# Extras: it seems particularly useful to apply DictCursor (as psycopg2.extras) which renders the rows being returned by our query as Python dictionaries as opposed to lists. 
# When using a DictCursor, the key is always the column name, and the value is the value of that column in that particular row.
#TO USE THE EXTRA: import psycopg2.extras and go to the line 153 above [self.conn.cursor() as cur:] and change it with [conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:]
# the oputput will be much more readable
# MORE EXTRAS
# check for the command 'copy_expert' in psycopg2.extras as it allows to copy/paste CSV files to PostgreSQL tables



# ---------------------------------------------------

# # ADVANCED - cannot understand it as Jan 2020
# Dynamic Queries
# 
# We have 8 login csv files that we need to insert into the logins table. 
# Instead of doing a COPY FROM query 8 times, we should utilize Python to make this more efficient.  This is possible due to tokenized strings.
#
# os is needed because we want to dynamically identify the files (os is used to create an interface with the operating system) 
# #we need to insert.
import os
#
query4 = '''
        COPY logins 
        FROM %(file_path)s
        DELIMITER ',' 
        CSV;
        '''
#
folder_path = '/Users/brad/Dropbox/Galvanize/sql-python/data/lecture-example/'
#
fnames = os.listdir(folder_path)
#
for fname in fnames:
    path = os.path.join(folder_path, fname)
    cur.execute(query4, {'file_path': path})
#
# # WARNING: BEWARE OF SQL INJECTION
# ## NEVER use + or % to reformat strings to be used with .execute
num = 579
terribly_unsafe = "SELECT * FROM logins WHERE userid = " + str(num)
print terribly_unsafe
#
#
date_cut = "2014-08-01"
horribly_risky = "SELECT * FROM logins WHERE tmstmp > %s" % date_cut
print horribly_risky
## Python is happy, but if num or date_cut included something malicious
## your data could be at risk
# ### Don't forget to commit your changes


cur.commit()
cur.close()
conn.close()


# # Key Things to Know
# 
# * Connections must be established using an existing database, username, database IP/URL, and maybe passwords
# * If you have no existing databases, you can connect to Postgres using the dbname 'postgres' to initialize one
# * Data changes are not actually stored until you choose to commit. This can be done either through commit() or setting autocommit = True.  
#   Until commited, transactions are only stored temporarily
#     - Autocommit = True is necessary to do database commands like CREATE DATABASE.  This is because Postgres does not have temporary transactions at the database level.
#     - Use .rollback() on the connection if your .execute() command results in an error. (Only works if change has not yet been committed) 
# * SQL connection databases utilize cursors for data traversal and retrieval.  This is kind of like an iterator in Python.
# * Cursor operations typically go like the following:
#     - execute a query
#     - fetch rows from query result if it is a SELECT query
#     - because it is iterative, previously fetched rows can only be fetched again by rerunning the query
#     - close cursor through .close()
# * Cursors and Connections must be closed using .close() or else Postgres will lock certain operations on the database/tables until the connection is severed. 
# 
# ## And don't leave yourself vulnerable to SQL injection!
# http://xkcd.com/327/




