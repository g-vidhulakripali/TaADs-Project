import pandas as pd
import sqlite3

def merge_and_load_to_new_sqlite(old_sqlite_file, csv_file, new_sqlite_file, table_name):
    """
    Adds data from a CSV file to an SQLite database by mapping specific CSV columns to SQLite columns.

    Args:
        old_sqlite_file (str): Path to the existing SQLite database file.
        csv_file (str): Path of the CSV file to process.
        new_sqlite_file (str): Path to the new SQLite database file.
        table_name (str): Name of the table in the new SQLite database.
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

        # Define the mapping from CSV columns to SQLite columns
        column_mapping = {
            "Name": "title",
            "Description": "course_contents",
            "Credit Hours": "credits",
            "Section Info": "prerequisites",
            "Type": "teaching-methods",
            "Instructors": "instructor"
        }

        # Create a DataFrame to collect new rows with all mapped columns
        new_rows = pd.DataFrame(columns=old_data.columns)

        for _, csv_row in csv_data.iterrows():
            temp_row = {}
            for csv_col, sqlite_col in column_mapping.items():
                if csv_col in csv_data.columns and sqlite_col in old_data.columns:
                    temp_row[sqlite_col] = csv_row[csv_col] if not pd.isna(csv_row[csv_col]) else None

            # Ensure all other columns are set to None for consistency
            for col in old_data.columns:
                if col not in temp_row:
                    temp_row[col] = None

            # Append the row to new_rows
            new_rows = pd.concat([new_rows, pd.DataFrame([temp_row])], ignore_index=True)

        # Combine old data and new rows
        combined_data = pd.concat([old_data, new_rows], ignore_index=True).drop_duplicates()

        # Connect to the new SQLite database
        new_conn = sqlite3.connect(new_sqlite_file)

        # Write combined data to the new database
        combined_data.to_sql(table_name, new_conn, if_exists='replace', index=False)
        print(f"Data from CSV mapped to respective columns and loaded into new SQLite file '{new_sqlite_file}' successfully.")

        # Close the new database connection
        new_conn.close()

    except Exception as e:
        print(f"An error occurred: {e}")

# Define file paths
old_sqlite_file = "../data/db/courses.sqlite"
csv_file = "../data/courses/course-catalog.csv"
new_sqlite_file = "../data/db/new_courses.sqlite"
table_name = "zqm_module_en"

# Run the function
merge_and_load_to_new_sqlite(old_sqlite_file, csv_file, new_sqlite_file, table_name)
