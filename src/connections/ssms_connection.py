import pyodbc
import pandas as pd
import json
import os

def main(config_path = "config.json"):
    """Fetches data from a SQL server table and returns it as a DataFrame.
    Args:
        config_path (str): Path to the configuration JSON file.
        
    Returns:
        pd.DataFrame: DataFrame containing the table data."""

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script path:{script_dir}")
    config_file = os.path.join(script_dir,config_path)
    print(f"Config path : {config_file}")

    # Load configuration from JSON
    with open(config_file,'r') as file:
        config = json.load(file)

    # Read SQL Server connection details
    server = config['sql_server']['server']
    database = config['sql_server']['database']
    table = config['sql_server']['table']

    print(f"Server : {server}, database : {database},table:{table}")

    # Define connection string for Windows Authentication
    connection_string = (
        f"DRIVER = {{SQL Server}};"
        f"SERVER = {server};"
        f"DATABASE = {database};"
        f"Trusted_Connection = yes;"

    )
    print(f"{connection_string}")
    try:
        # Establish connection
        conn = pyodbc.connect(connection_string)
        if conn:
            print("Connection to SQL Server successful!")

        else:
            print("Counld not connect t SSMS")

        # Fetch data from teh specified table
        query = f"SELECT * FROM {table}"
        df = pd.read_sql(query,conn)
        conn.close()
        print(f"Data Fetched successfully from table '{table}'.")
        return df
    
    except Exception as e:
        print(f"Error connecting to SQL Server or fetching data:{e}")
        return None