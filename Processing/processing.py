import pandas as pd
import glob
import os
import pprint
from rapidfuzz import process
from datetime import timedelta

### PREPROCESSING WORK

# Get all column names
files = glob.glob('Processing\\States\\**\\*.csv')  
cols = set()   

for file in files:
    df = pd.read_csv(file, nrows=0)                            
    df.columns = [col.strip().lower() for col in df.columns]    
    cols.update(df.columns)

county = ['countynam', 'countyname', 'county', 'counties', 'area', 'name', 'area_name']
c_affected = ['out', 'customersaffected', '# out', 'n_out', 'aff', 'cust_a', 'customersoutnow', 'numoutages']
c_served = ['served', 'customersserved', '# served', 'premisecount', 'cust_s', 'total', 'customercount', 'count']
timestamp = ['timestamp']
col_map = {}

for name in county:
    col_map[name] = "county"
for caff in c_affected:
    col_map[caff] = "customers_affected"
for cser in c_served:
    col_map[cser] = "customers_served"
for time in timestamp:
    col_map[time] = "timestamp"



# Get the raw county set for each state
raw_county_dict = {}
base = r"Processing\\States"
key_val_set = set()

for state in glob.glob(os.path.join(base, '*')):
    state_code = state[-2:]
    county_set = set()

    for file in glob.glob(os.path.join(state, '*.csv')):
        df = pd.read_csv(file)
        df.columns = [col.strip().lower() for col in df.columns]
        col_name = [col for col in df.columns if col in county]
        key_col = [col for col in df.columns if col == 'key']

        # Some providers also include municipalities, this filters those out
        if 'key' in df.columns:
            mask = df['key'].str.lower() != 'muni'
            county_set.update(c.strip().lower() for c in df.loc[mask, col_name[0]].dropna())
        else:
            county_set.update(c.strip().lower() for c in df[col_name[0]].dropna())
        
    county_set = sorted(county_set)
    raw_county_dict[state_code] = county_set



# Compile the "master" county list we will use
master_county_dict = {}
base = "Processing\\CountyList\\*.txt"

for state in glob.glob(base):
    with open(state, 'r', encoding="utf-8") as file:
        state_code = state[-6:-4]
        lines = file.readlines()
        county_list = []
        for line in lines:
            county_list.append(line.lower().replace("county", "").strip())
        master_county_dict[state_code] = county_list

# Check for different names being used for the same county
dupe_dict = {}  # Stores {State : {original name : final name}}
dupe_list = {}  # Stores {State : [original names]} 

for code in raw_county_dict:
    dupe_entry = {}
    dupe_names = []

    for raw in raw_county_dict[code]:
        match, score, _ = process.extractOne(raw, master_county_dict[code])
        if score >= 85: 
            dupe_entry[raw] = match
            # dupe_entry.append({raw: {match: score}})
            dupe_names.append(raw)
    dupe_dict[code] = dupe_entry
    dupe_list[code] = dupe_names

# TODO: manually go through and filter the list. Most entries seem good, but there are some weird things.
# e.g. in PerCounty MidAmerica for Illinois, they have things like Bremer, Polk , Mahaska county, which seem 
# to either just not exist, be some unincorporated place, or something else entirely.
# This seems to occur at around score = 80, but results may vary.


### PROCESSING 
STATE = 'AL'

# Initialize a list of dictionaries, each entry representing a county and its data 
schema = ["ID", "county", "customers_affected", "customers_served", "start_time", "end_time", "duration"]
county_dfs = {c: pd.DataFrame(columns=schema) for c in master_county_dict[STATE]}
files = glob.glob(f'Processing\\States\\{STATE}\\*.csv')  

for file in files:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()

    # Remove duplicate customers affected columns, keeping the one with the highest sum
    dupes = [c for c in df.columns if c.strip().lower() in c_affected]

    if len(dupes) > 1:
        keep = df[dupes].sum(skipna=True).idxmax()
        df.drop(columns=[c for c in df.columns if c != keep and c in dupes], inplace=True)

    # Standardize column names and remove unnecessary columns
    for col in df.columns:
        lower = col.strip().lower()

        if lower in col_map:
            df.rename(columns={
                col: col_map[lower]
            }, inplace=True)
        else:
            df.drop(columns=[col], inplace=True)

    # Check for if any columns are still missing. If so, skip this provider
    std_names = ['county', 'customers_affected', 'customers_served', 'timestamp']
    skip = False
    missing = []

    for name in std_names:
        if name not in df.columns:
            missing.append(name)
            skip = True
        
    if skip:
        print(f"Missing: {missing}\nFrom: {file}\n")
        continue

    # Remove invalid rows in county col
    df.dropna(subset=['county'])
    df = df[~df['county'].isin(['unknown', 'Unknown', 'UNKNOWN',  ''])]

    # Standardize county names
    county_list = dupe_list[STATE]
    county_dict = dupe_dict[STATE]

    df['county'] = df['county'].astype(str).str.strip().str.lower().map(county_dict)
    df = df[df['county'].map(type) == str]
    
    # Filter by date
    date = "2024-05-23"

    # Standardize timestamp data type
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df[df['timestamp'].dt.date == pd.to_datetime(date).date()]

    # Sort by county name then by date and reorder columns 
    for county in county_list:
        county_df = df[df['county'] == county]
        
        if county_df.size != 0:
            county_df = county_df[['county', 'customers_affected', 'customers_served', 'timestamp']]
            county_df.sort_values('timestamp')

            # Group by timestamp and give IDs to each new outage
            last_id = county_dfs[county]['ID'].max() if not county_dfs[county].empty else 0
            threshold = timedelta(hours=1, minutes=14)

            county_df['diff'] = county_df['timestamp'].diff()
            mask = county_df['diff'] > threshold
            county_df['new_outage'] = (county_df['diff'].isna() | mask)
            county_df['ID'] = county_df['new_outage'].cumsum() + last_id

            result = (
                county_df.groupby('ID')
                        .agg(
                            county=('county', 'first'),
                            customers_affected=('customers_affected', 'max'),
                            customers_served=('customers_served', 'max'),
                            start_time=('timestamp', 'min'),
                            end_time=('timestamp', 'max')
                        )
                        .reset_index()
            )

            result['duration'] = result['end_time'] - result['start_time']

            if county_dfs[county].empty:
                county_dfs[county] = result
            else:
                county_dfs[county] = pd.concat([county_dfs[county], result], ignore_index=True)

# Within each county, sort by chronological starting time
for county in county_dfs:
    county_dfs[county] = county_dfs[county].sort_values('start_time').reset_index(drop=True)
    county_dfs[county]['ID'] = range(1, len(county_dfs[county]) + 1)  

# Create filler dataframes for counties with no reported outages
for county in master_county_dict[STATE]:
    if county_dfs[county].empty:
            result = pd.DataFrame([{
                "ID": 1,
                "county": county,
                "customers_affected": 0,
                "customers_served": 0,
                "start_time": pd.NaT,
                "end_time": pd.NaT,
                "duration": pd.Timedelta(0)
            }], columns=schema) 
            if county_dfs[county].empty:
                county_dfs[county] = result
            else:
                county_dfs[county] = pd.concat([county_dfs[county], result], ignore_index=True)

# Uncomment below to print county level data 
# pprint.pprint(county_dfs)

# Uncomment below to write data to corresponding csv file
# combined = pd.concat(county_dfs.values(), ignore_index=True)
# combined.to_csv(f"Processing\\Processed_Data\\{STATE}_all_counties_{date}.csv", index=False)