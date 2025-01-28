import sqlite3
import logging
from cachetools import cached, TTLCache

cache = TTLCache(maxsize=100, ttl=300)
def fetch_records_in_batches(db_file, batch_size=5000):
    """
    Fetch records from the database in batches.

    Args:
        db_file (str): Path to the SQLite database file.
        batch_size (int): Number of records to fetch per batch.

    Yields:
        List[str]: A batch of concatenated text records.
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT COUNT(*) FROM zqm_module_en")
        total_records = cursor.fetchone()[0]
        logging.info(f"Total records to process: {total_records}")

        for offset in range(0, total_records, batch_size):
            logging.info(f"Fetching records {offset} to {offset + batch_size}")
            query = f"""
            SELECT learning_obj, course_contents, prerequisites 
            FROM zqm_module_en
            LIMIT {batch_size} OFFSET {offset}
            """
            cursor.execute(query)
            records = cursor.fetchall()

            combined_records = [
                " ".join(filter(None, [
                    row[0].strip() if isinstance(row[0], str) else None,
                    row[1].strip() if isinstance(row[1], str) else None,
                    row[2].strip() if isinstance(row[2], str) else None,
                ])) for row in records
            ]
            yield [record for record in combined_records if record.strip()]
    except Exception as e:
        logging.error(f"Error fetching records from the database: {e}")
    finally:
        conn.close()


@cached(cache)
def get_learning_obj_en(db_file):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        statement = 'SELECT title, instructor, learning_obj, course_contents, prerequisites, credits, evaluation, time, frequency, duration, course_type, platform FROM zqm_module_en'
        cursor.execute(statement)
        records = cursor.fetchall()
        conn.close()
        return records
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return []
