import pandas as pd
import sqlite3

def merge_and_load_to_new_sqlite(old_sqlite_file, csv_file, new_sqlite_file, table_name, column_mapping):
    """
    Merges data from an existing SQLite database and a CSV file into a new SQLite database.

    Args:
        old_sqlite_file (str): Path to the existing SQLite database file.
        csv_file (str): Path of the CSV file to process.
        new_sqlite_file (str): Path to the new SQLite database file.
        table_name (str): Name of the table in the SQLite database.
        column_mapping (dict): Hardcoded mapping of CSV headers to SQLite table columns.
    """
    try:
        # Connect to the old SQLite database
        old_conn = sqlite3.connect(old_sqlite_file)
        old_cursor = old_conn.cursor()

        # Fetch data from the old SQLite database
        old_cursor.execute(f"SELECT * FROM {table_name}")
        old_data = pd.DataFrame(old_cursor.fetchall(), columns=[col[0] for col in old_cursor.description])

        # Close the old database connection
        old_conn.close()

        # Load CSV data into a DataFrame
        csv_data = pd.read_csv(csv_file)

        # Ensure each CSV header maps to an existing database column without creating new columns
        valid_columns = {csv_col: db_col for csv_col, db_col in column_mapping.items() if db_col in old_data.columns}

        # Combine data from CSV into existing database columns
        for csv_col, db_col in valid_columns.items():
            if db_col in old_data.columns:
                # Append values from the CSV column to the existing database column
                old_data[db_col] = old_data[db_col].astype(str) + ' ' + csv_data[csv_col].astype(str).fillna('')

        # Drop duplicates
        merged_data = old_data.drop_duplicates()

        # Connect to the new SQLite database
        new_conn = sqlite3.connect(new_sqlite_file)

        # Write merged data to the new database
        merged_data.to_sql(table_name, new_conn, if_exists='replace', index=False)
        print(f"Merged data loaded into new database table '{table_name}' successfully.")

        # Close the new database connection
        new_conn.close()

    except Exception as e:
        print(f"An error occurred: {e}")

# Define file paths
old_sqlite_file = "../data/db/courses.sqlite"
csv_file = "../data/courses/course-catalog.csv"
new_sqlite_file = "../data/db/new_courses.sqlite"
table_name = "zqm_module_en"

# Define hardcoded column mapping
column_mapping = {
    "Year": "remarks",
    "Term": "remarks",
    "YearTerm": "remarks",
    "Subject": "course_type",
    "Number": "credits",
    "Name": "title",
    "Description": "description",
    "Credit Hours": "credits",
    "Section Info": "course_contents",
    "Degree Attributes": "applicability",
    "Schedule Information": "schedule_info",
    "CRN": "file_loc",
    "Section": "title",
    "Status Code": "remarks",
    "Part of Term": "remarks",
    "Section Title": "title",
    "Section Credit Hours": "credits",
    "Section Status": "remarks",
    "Enrollment Status": "remarks",
    "Type": "course_type",
    "Type Code": "remarks",
    "Start Time": "time",
    "End Time": "time",
    "Days of Week": "schedule_info",
    "Room": "remarks",
    "Building": "remarks",
    "Instructors": "instructor"
}

# Run the function
merge_and_load_to_new_sqlite(old_sqlite_file, csv_file, new_sqlite_file, table_name, column_mapping)
