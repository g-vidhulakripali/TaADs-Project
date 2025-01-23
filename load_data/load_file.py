import pandas as pd
import sqlite3

def merge_and_load_to_new_sqlite(old_sqlite_file, csv_file, new_sqlite_file, column_mapping):
    """
    Merges data from an existing SQLite database and a CSV file into a new SQLite database.

    Args:
        old_sqlite_file (str): Path to the existing SQLite database file.
        csv_file (str): Path of the CSV file to process.
        new_sqlite_file (str): Path to the new SQLite database file.
        column_mapping (dict): Mapping of CSV headers to SQLite table columns.
    """
    try:
        # Connect to the old SQLite database
        old_conn = sqlite3.connect(old_sqlite_file)
        old_cursor = old_conn.cursor()

        # Fetch data from the old SQLite database
        table_name = "zqm_module_en"
        old_cursor.execute(f"SELECT * FROM {table_name}")
        old_data = pd.DataFrame(old_cursor.fetchall(), columns=[col[0] for col in old_cursor.description])

        # Close the old database connection
        old_conn.close()

        # Load CSV data into a DataFrame
        csv_data = pd.read_csv(csv_file)

        # Rename columns based on the mapping
        csv_data.rename(columns=column_mapping, inplace=True)

        # Drop duplicate rows from CSV data
        csv_data_cleaned = csv_data.drop_duplicates()

        # Merge old data and new data
        merged_data = pd.concat([old_data, csv_data_cleaned], ignore_index=True).drop_duplicates()

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

# Define column mapping for course-catalog
column_mapping = {
    "Catalog ID": "catalog_id",
    "Title": "title",
    "Course Name": "title",  # Map 'Course Name' to 'title' as well
    "Description": "description",
    "Duration": "duration"
}

# Run the function
merge_and_load_to_new_sqlite(old_sqlite_file, csv_file, new_sqlite_file, column_mapping)
