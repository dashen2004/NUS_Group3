import pandas as pd

file_path = '../../docs/sorted_athlete_sprint_data_with_event_ages.csv'
data = pd.read_csv(file_path)

# generate the relations between string and ID
unique_competitors = data['Competitor'].unique()
unique_venue = data['Venue'].unique()
competitor_to_int = {name: idx for idx, name in enumerate(unique_competitors)}
venue_to_int = {name: idx for idx, name in enumerate(unique_venue)}

# create a new dataframe
mapping_df1 = pd.DataFrame(list(competitor_to_int.items()), columns=['Competitor', 'ID'])
mapping_df2 = pd.DataFrame(list(venue_to_int.items()), columns=['Venue', 'ID'])

# Replace string to int
data['Competitor'] = data['Competitor'].map(competitor_to_int)
data['Venue'] = data['Venue'].map(venue_to_int)

# save to new CSV file
processed_file_path = '../../docs/new_data_s2i.csv'
mapping_file_path1 = '../../docs/mapping_competitor.csv'
mapping_file_path2 = '../../docs/mapping_venue.csv'

data.to_csv(processed_file_path, index=False)
mapping_df1.to_csv(mapping_file_path1, index=False)
mapping_df2.to_csv(mapping_file_path2, index=False)

print("Complete")