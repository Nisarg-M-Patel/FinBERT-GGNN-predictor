import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pickle
import json
import logging

# Configure logger
logger = logging.getLogger(__name__)

# Create directory for data
DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

def get_last_update_date():
    """
    Get the date of the last data update from the timestamp file
    """
    timestamp_file = f"{DATA_DIR}/last_update.txt"
    
    if os.path.exists(timestamp_file):
        with open(timestamp_file, "r") as f:
            last_update_str = f.read().strip()
            logger.debug(f"Found last update timestamp: {last_update_str}")
            return datetime.strptime(last_update_str, "%Y-%m-%d %H:%M:%S")
    
    # If no timestamp file exists, return a date one year ago
    default_date = datetime.now() - timedelta(days=365)
    logger.debug(f"No previous timestamp found, defaulting to {default_date}")
    return default_date

def get_sp500_data(force_full_download=False, timeframe_days=365):
    """
    Get S&P 500 OHLC price data and sector information
    
    Args:
        force_full_download: If True, download entire year of data regardless of last update
        timeframe_days: Number of days to maintain in the dataset (default: 365)
    """
    logger.info("Fetching S&P 500 components...")
    try:
        # Read the S&P 500 table from Wikipedia
        logger.debug("Reading S&P 500 table from Wikipedia")
        sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        
        # Clean ticker symbols (remove any trailing .X or similar)
        sp500_table['Symbol'] = sp500_table['Symbol'].str.replace(r'\n', '')
        
        # Create a clean dataframe with exactly what we need
        sp500_info = pd.DataFrame({
            'Symbol': sp500_table['Symbol'],
            'Name': sp500_table['Security'],
            'Sector': sp500_table['GICS Sector'],
            'Sub_Industry': sp500_table['GICS Sub-Industry']
        })
        
        # Save the sector information
        logger.debug(f"Saving sector information for {len(sp500_info)} companies")
        sp500_info.to_csv(f"{DATA_DIR}/sp500_sectors.csv", index=False)
        
        # Extract ticker list for downloading price data
        tickers = sp500_info['Symbol'].tolist()
        
        logger.info(f"Successfully extracted {len(tickers)} S&P 500 components")
        
        # Get sector counts for logging
        sector_counts = sp500_info['Sector'].value_counts()
        logger.info(f"Found {len(sector_counts)} sectors")
        for sector, count in sector_counts.items():
            logger.debug(f"  - {sector}: {count} companies")
        
    except Exception as e:
        logger.error(f"Error fetching S&P 500 components: {e}", exc_info=True)
        return None
    
    # Check if we need to do a full download or just update
    if force_full_download or not os.path.exists(f"{DATA_DIR}/sp500_ohlc.pkl"):
        logger.info("Performing full download (forced or no existing data)")
        return do_full_download(tickers, timeframe_days)
    else:
        logger.info("Updating existing data")
        return update_existing_data(tickers, timeframe_days)

def do_full_download(tickers, timeframe_days=365):
    """
    Download complete OHLC data for all tickers for the specified timeframe
    
    Args:
        tickers: List of ticker symbols to download
        timeframe_days: Number of days to download (default: 365)
    """
    logger.info(f"Performing FULL download of OHLC data for {len(tickers)} stocks for the past {timeframe_days} days")
    try:
        # Calculate start date based on the specified timeframe
        start_date = (datetime.now() - timedelta(days=timeframe_days)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.debug(f"Download period: {start_date} to {end_date}")
        
        # Download data for the specific date range
        logger.debug("Starting yfinance download (this may take a while)...")
        data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            group_by='ticker',
            auto_adjust=True,
            threads=True
        )
        
        logger.info(f"Data downloaded successfully. Date range: {data.index[0]} to {data.index[-1]}")
        logger.info(f"Total trading days: {len(data.index)}")
        
        # Check for tickers with missing data
        available_tickers = set(data.columns.levels[0]) if isinstance(data.columns, pd.MultiIndex) else set()
        missing_tickers = set(tickers) - available_tickers
        
        if missing_tickers:
            logger.warning(f"Missing data for {len(missing_tickers)} tickers")
            logger.debug(f"Tickers with missing data: {missing_tickers}")
        
        # Save the data
        logger.info("Saving data...")
        data.to_pickle(f"{DATA_DIR}/sp500_ohlc.pkl")
        
        # Save timestamp of download
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(f"{DATA_DIR}/last_update.txt", "w") as f:
            f.write(current_time)
        
        logger.info(f"Data saved to {DATA_DIR}/ with timestamp {current_time}")
        
        return {
            "price_data": data,
            "sector_data": pd.read_csv(f"{DATA_DIR}/sp500_sectors.csv")
        }
        
    except Exception as e:
        logger.error(f"Error downloading data: {e}", exc_info=True)
        return None

def update_existing_data(tickers, timeframe_days=365):
    """
    Update existing OHLC data with only the new data since last update
    and maintain a consistent timeframe by clipping older data
    
    Args:
        tickers: List of ticker symbols to update
        timeframe_days: Number of days to keep in the dataset (default: 365)
    """
    try:
        # Load existing data
        logger.debug("Loading existing price data")
        existing_data = pd.read_pickle(f"{DATA_DIR}/sp500_ohlc.pkl")
        logger.info(f"Loaded existing data with date range: {existing_data.index[0]} to {existing_data.index[-1]}")
        
        # Get the last update date
        last_update = get_last_update_date()
        logger.debug(f"Last update timestamp: {last_update}")
        
        # Calculate the start date for new data (last trading day in existing data)
        if existing_data.index[-1].date() >= datetime.now().date():
            logger.info("Data is already up to date. No new data to download.")
            
            # Still clip older data to maintain timeframe
            cutoff_date = datetime.now() - timedelta(days=timeframe_days)
            logger.debug(f"Clipping data older than {cutoff_date}")
            clipped_data = existing_data[existing_data.index >= cutoff_date]
            
            if len(clipped_data) < len(existing_data):
                logger.info(f"Clipped {len(existing_data) - len(clipped_data)} older trading days to maintain {timeframe_days}-day timeframe")
                # Save the clipped data
                clipped_data.to_pickle(f"{DATA_DIR}/sp500_ohlc.pkl")
            
            # Return the existing data and sector information
            return {
                "price_data": clipped_data,
                "sector_data": pd.read_csv(f"{DATA_DIR}/sp500_sectors.csv")
            }
        
        # We need to get data from the day after the last day in the existing data
        start_date = existing_data.index[-1] + timedelta(days=1)
        
        logger.info(f"Downloading incremental data from {start_date.date()} to today")
        
        # Download only the new data
        logger.debug("Starting yfinance download for new data")
        new_data = yf.download(
            tickers=tickers,
            start=start_date.strftime('%Y-%m-%d'),
            end=datetime.now().strftime('%Y-%m-%d'),
            group_by='ticker',
            auto_adjust=True,
            threads=True
        )
        
        if new_data.empty:
            logger.info("No new data available to download.")
            return {
                "price_data": existing_data,
                "sector_data": pd.read_csv(f"{DATA_DIR}/sp500_sectors.csv")
            }
        
        logger.info(f"Downloaded new data from {new_data.index[0]} to {new_data.index[-1]}")
        logger.info(f"New data contains {len(new_data.index)} trading days")
        
        # Combine existing and new data
        logger.debug("Combining existing and new data")
        updated_data = pd.concat([existing_data, new_data])
        
        # Remove any duplicates that might have occurred
        duplicates = updated_data.index.duplicated(keep='last')
        if any(duplicates):
            logger.warning(f"Found {sum(duplicates)} duplicate dates, keeping most recent data")
            updated_data = updated_data[~duplicates]
        
        # Sort by date
        updated_data = updated_data.sort_index()
        
        # Clip older data to maintain consistent timeframe
        cutoff_date = datetime.now() - timedelta(days=timeframe_days)
        logger.debug(f"Clipping data older than {cutoff_date}")
        clipped_data = updated_data[updated_data.index >= cutoff_date]
        
        logger.info(f"Combined data ranges from {updated_data.index[0]} to {updated_data.index[-1]}")
        logger.info(f"After clipping older data, keeping {len(clipped_data)} trading days from {clipped_data.index[0]} to {clipped_data.index[-1]}")
        
        # Save the updated and clipped data
        logger.debug("Saving updated data")
        clipped_data.to_pickle(f"{DATA_DIR}/sp500_ohlc.pkl")
        
        # Update timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(f"{DATA_DIR}/last_update.txt", "w") as f:
            f.write(current_time)
        
        logger.info(f"Updated data saved to {DATA_DIR}/ with timestamp {current_time}")
        
        return {
            "price_data": clipped_data,
            "sector_data": pd.read_csv(f"{DATA_DIR}/sp500_sectors.csv")
        }
        
    except Exception as e:
        logger.error(f"Error updating data: {e}", exc_info=True)
        return None

def check_for_sp500_changes():
    """
    Check for changes in S&P 500 composition by comparing current list with our saved list
    """
    logger.info("Checking for changes in S&P 500 composition")
    
    if not os.path.exists(f"{DATA_DIR}/sp500_sectors.csv"):
        logger.info("No existing sector data found. Will perform full download.")
        return True
    
    try:
        # Load existing sector data
        logger.debug("Loading existing sector data")
        existing_sectors = pd.read_csv(f"{DATA_DIR}/sp500_sectors.csv")
        existing_tickers = set(existing_sectors['Symbol'])
        logger.debug(f"Found {len(existing_tickers)} existing tickers")
        
        # Get current S&P 500 components
        logger.debug("Getting current S&P 500 components from Wikipedia")
        current_sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        current_sp500['Symbol'] = current_sp500['Symbol'].str.replace(r'\n', '')
        current_tickers = set(current_sp500['Symbol'])
        logger.debug(f"Found {len(current_tickers)} current tickers")
        
        # Check for differences
        added = current_tickers - existing_tickers
        removed = existing_tickers - current_tickers
        
        if added or removed:
            logger.info(f"Detected changes in S&P 500 composition:")
            if added:
                logger.info(f"Added tickers ({len(added)}): {sorted(added)}")
            if removed:
                logger.info(f"Removed tickers ({len(removed)}): {sorted(removed)}")
            return True
        else:
            logger.info("No changes detected in S&P 500 composition.")
            return False
            
    except Exception as e:
        logger.error(f"Error checking for S&P 500 changes: {e}", exc_info=True)
        # If there's an error, we'll do a full download to be safe
        logger.warning("Defaulting to full download due to error")
        return True

def get_data_summary():
    """
    Get a summary of the current data without visualization
    """
    logger.info("Generating data summary")
    
    # Check if data files exist
    if not os.path.exists(f"{DATA_DIR}/sp500_ohlc.pkl"):
        logger.error("Price data file not found.")
        return None
        
    if not os.path.exists(f"{DATA_DIR}/sp500_sectors.csv"):
        logger.error("Sector data file not found.")
        return None
    
    # Load sector data
    logger.debug("Loading sector data")
    sectors = pd.read_csv(f"{DATA_DIR}/sp500_sectors.csv")
    logger.info(f"Sector data: {len(sectors)} companies across {len(sectors['Sector'].unique())} sectors")
    
    # Load price data
    logger.debug("Loading price data")
    price_data = pd.read_pickle(f"{DATA_DIR}/sp500_ohlc.pkl")
    logger.info(f"Price data: {len(price_data.index)} trading days from {price_data.index[0].date()} to {price_data.index[-1].date()}")
    
    # Verify we have both sector and price data for stocks
    tickers_with_sectors = set(sectors['Symbol'])
    tickers_with_prices = set(price_data.columns.levels[0])
    overlap = tickers_with_sectors.intersection(tickers_with_prices)
    
    logger.info(f"Tickers with both sector and price data: {len(overlap)} of {len(tickers_with_sectors)} total")
    
    # Calculate missing tickers
    missing_prices = tickers_with_sectors - tickers_with_prices
    missing_sectors = tickers_with_prices - tickers_with_sectors
    
    if missing_prices:
        logger.warning(f"{len(missing_prices)} tickers have sector data but no price data")
        logger.debug(f"Missing price data for: {sorted(missing_prices)}")
    
    if missing_sectors:
        logger.warning(f"{len(missing_sectors)} tickers have price data but no sector data")
        logger.debug(f"Missing sector data for: {sorted(missing_sectors)}")
    
    return {
        "price_data": price_data,
        "sector_data": sectors
    }

def update_data(timeframe_days=365):
    """
    Main function to update data, checking for both S&P 500 changes and new price data
    
    Args:
        timeframe_days: Number of days to maintain in the dataset (default: 365)
    """
    logger.info(f"Starting data update with {timeframe_days}-day timeframe")
    
    # Check if we need to do a full download due to S&P 500 composition changes
    force_full_download = check_for_sp500_changes()
    
    if force_full_download:
        logger.info("Will perform full data download due to S&P 500 composition changes")
    else:
        logger.info("Will perform incremental data update (no S&P 500 composition changes)")
    
    # Get the data (either full download or update)
    result = get_sp500_data(force_full_download=force_full_download, timeframe_days=timeframe_days)
    
    if result:
        # Add additional metadata about the timeframe
        logger.debug("Saving data information to JSON")
        with open(f"{DATA_DIR}/data_info.json", "w") as f:
            info = {
                "timeframe_days": timeframe_days,
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "full_download": force_full_download,
                "tickers_count": len(result["sector_data"]),
                "trading_days": len(result["price_data"].index),
                "date_range": {
                    "start": result["price_data"].index[0].strftime("%Y-%m-%d"),
                    "end": result["price_data"].index[-1].strftime("%Y-%m-%d")
                }
            }
            json.dump(info, f, indent=4)
        
        logger.info("Data update completed successfully.")
        return result
    else:
        logger.error("Data update failed.")
        return None

if __name__ == "__main__":
    # Set up logging when script is run directly
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data_collector.log'),
            logging.StreamHandler()
        ]
    )
    
    # Update the data
    result = update_data()
    
    if result:
        # Display a summary of the data
        get_data_summary()