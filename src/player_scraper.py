from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import os

def scrape_fbref_table(url, stat_type):
    """
    Generic function to scrape player stats tables from FBref
    
    Args:
        url: The URL to scrape
        stat_type: Type of statistic (for naming the output file)
        
    Returns:
        DataFrame of the scraped table or None if failed
    """
    # Setup Chrome options
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")
    
    # Setup the webdriver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    try:
        # Navigate to the page
        driver.get(url)
        print(f"Accessing {stat_type} data...")
        
        # Wait for page to load fully
        time.sleep(5)
        
        # Get the page HTML after JavaScript execution
        html_content = driver.page_source
        
        # Remove HTML comments (where the tables may be hidden)
        html_content = html_content.replace('<!--', '').replace('-->', '')
        
        # Parse all tables with pandas
        tables = pd.read_html(html_content)
        
        # Identify the main stats table - generally has Player, Nation, Pos columns
        for i, table in enumerate(tables):
            # Convert column names to strings for checking
            columns = [str(col) for col in table.columns]
            
            # Check for player table indicators
            has_player_col = any('Player' in col for col in columns)
            has_position_col = any('Pos' in col for col in columns)
            
            if has_player_col and has_position_col and len(table) > 10:
                print(f"Found {stat_type} table with {len(table)} rows")
                return table
        
        print(f"No suitable {stat_type} table found")
        return None
    
    except Exception as e:
        print(f"Error scraping {stat_type}: {e}")
        return None
    
    finally:
        # Always close the browser
        driver.quit()

def save_data(df, stat_type):
    """Save the dataframe to a CSV file"""
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Save to CSV
    file_path = f'data/player_{stat_type}_2025_raw.csv'
    df.to_csv(file_path, index=False)
    print(f"Saved to {file_path}")

# Dictionary of stat URLs and their types
stat_urls = {
    'gca': 'https://fbref.com/en/comps/9/gca/Premier-League-Stats',
    'defense': 'https://fbref.com/en/comps/9/defense/Premier-League-Stats',
    'possession': 'https://fbref.com/en/comps/9/possession/Premier-League-Stats',
    'passing_types': 'https://fbref.com/en/comps/9/passing_types/Premier-League-Stats',
    'passing': 'https://fbref.com/en/comps/9/passing/Premier-League-Stats',
    'shooting': 'https://fbref.com/en/comps/9/shooting/Premier-League-Stats',
    'playingtime': 'https://fbref.com/en/comps/9/playingtime/Premier-League-Stats',
    'misc': 'https://fbref.com/en/comps/9/misc/Premier-League-Stats'
}

# Scrape each stat type
for stat_type, url in stat_urls.items():
    print(f"\nScraping {stat_type} statistics...")
    df = scrape_fbref_table(url, stat_type)
    
    if df is not None:
        # Basic cleaning - rename multi-level column headers to be flat
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [f"{'' if col[0] == '' else col[0]}_{col[1]}" if isinstance(col, tuple) else col 
                         for col in df.columns]
        
        # Preview the data
        print(df.head(2))
        
        # Save the data
        save_data(df, stat_type)
    
    
    time.sleep(3)

print("\nAll data scraping completed!")