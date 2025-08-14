import pandas as pd

def fetch_fbref_table(url, feature):
    df = pd.read_html(url, attrs = {'id': feature})[0]
    return df


def scrape_multiple_tables(url, table_ids, output_prefix = 'fbref'):

    results = {}
    
    for id in table_ids:
        print(f"Scraping table: {id}")
        data = fetch_fbref_table(url, id)
        
        if data is not None:
            filename = f"team_data/{output_prefix}_{id}.csv"
            data.to_csv(filename)
            print(f"Saved to {filename}")
            results[id] = 'Success'
        else:
            print(f"Failed to fetch table: {id}")
            results[id] = "Failed, mate"
    return results

if __name__ == "__main__":
    url = 'https://fbref.com/en/comps/9/2024-2025/2024-2025-Premier-League-Stats'
    tables_to_scrape = ['stats_squads_defense_for',
                 'stats_squads_gca_for',
                 'stats_squads_shooting_for',
                 'stats_squads_passing_for',
                 'stats_squads_possession_for',
                 'stats_squads_standard_for',
                 'stats_squads_defense_against',
                 'stats_squads_gca_against',
                 'stats_squads_shooting_against',
                 'stats_squads_passing_against',
                 'stats_squads_possession_against',
                 'stats_squads_standard_against']
    results = scrape_multiple_tables(url, tables_to_scrape, output_prefix="EPL_2425")
    
    print("\nScraping Results:")
    for table_id, status in results.items():
        print(f"  {table_id}: {status}")