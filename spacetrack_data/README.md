# Space-Track Data Fetcher

This folder contains scripts for fetching, storing, and managing space debris data from Space-Track.org. These scripts are designed to work independently from the main project, allowing you to test and validate the data before importing it into the main application.

## Files

- `space_track_data.py`: Main module containing the SpaceTrackClient and SpaceDebrisDatabase classes
- `test_fetch.py`: Script to test fetching data from Space-Track.org and storing it in a test database
- `import_to_main.py`: Utility to import data from the test database to the main project database
- `space_debris_test.db`: SQLite database for storing test data (created automatically)
- `space_debris_export.csv`: CSV export of the test data (created automatically)
- `space_debris_export.json`: JSON export of the test data (created automatically)
- `export_for_main.json`: JSON export formatted for the main project (created automatically)

## Setup

1. Ensure you have the required Python packages installed:
   ```
   pip install requests numpy pandas python-dotenv
   ```

2. Make sure your `.env` file in the project root contains your Space-Track.org credentials:
   ```
   SPACETRACK_USERNAME=your_username
   SPACETRACK_PASSWORD=your_password
   ```

## Usage

### Fetching Test Data

To fetch data from Space-Track.org and store it in the test database:

```bash
python test_fetch.py
```

This will:
1. Connect to Space-Track.org using your credentials
2. Fetch space debris data (limited to 100 objects for testing)
3. Store the data in `space_debris_test.db`
4. Display a sample of the retrieved data
5. Export the data to CSV and JSON files

### Importing to Main Project

After successfully fetching and validating the test data, you can import it into the main project:

```bash
python import_to_main.py
```

This will:
1. Retrieve data from the test database
2. Export it to `export_for_main.json` for backup
3. Import the data into the main project's database

## Database Schema

The test database includes the following tables:

### space_debris

Stores information about space debris objects:

| Column | Type | Description |
|--------|------|-------------|
| id | TEXT | Primary key (format: ST-NORAD_ID) |
| altitude | REAL | Average altitude in km |
| latitude | REAL | Latitude in degrees |
| longitude | REAL | Longitude in degrees |
| x | REAL | X coordinate in Earth-centered frame |
| y | REAL | Y coordinate in Earth-centered frame |
| z | REAL | Z coordinate in Earth-centered frame |
| size | REAL | Estimated size in meters |
| velocity | REAL | Orbital velocity in km/s |
| inclination | REAL | Orbital inclination in degrees |
| risk_score | REAL | Calculated collision risk (0-1) |
| last_updated | TEXT | Timestamp of last update |
| object_name | TEXT | Name of the object |
| object_type | TEXT | Type of object (PAYLOAD, ROCKET BODY, DEBRIS) |
| apogee | REAL | Apogee altitude in km |
| perigee | REAL | Perigee altitude in km |
| period | REAL | Orbital period in minutes |

### data_fetches

Tracks when data was fetched:

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key (auto-increment) |
| fetch_time | TEXT | Timestamp of the fetch |
| num_objects | INTEGER | Number of objects fetched |
| source | TEXT | Data source (e.g., "Space-Track.org") |

## Troubleshooting

If you encounter issues with Space-Track.org:

1. Verify your credentials in the `.env` file
2. Check if you have reached your API rate limit
3. Ensure you have an active Space-Track.org account
4. Try reducing the `limit` parameter in `test_fetch.py`

If the import to the main project fails:

1. Check if the main database exists
2. Verify the database schema compatibility
3. Use the exported JSON file for manual import if needed 