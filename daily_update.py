#!/usr/bin/env python
"""
Daily update script for S&P 500 data.
This script can be scheduled to run daily using cron or Task Scheduler.
"""

import os
import sys
import logging
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_update.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('data_updater')

if __name__ == "__main__":
    try:
        logger.info("Starting daily data update...")
        
        # Import here to ensure paths are set up correctly
        from scripts.common.data_collector import update_data, get_data_summary
        
        # Update data with 365-day window
        result = update_data(timeframe_days=365)
        
        if result:
            logger.info("Data update completed successfully")
            
            # Get a summary of the updated data
            get_data_summary()
        else:
            logger.error("Data update failed")
        
    except Exception as e:
        logger.exception(f"An error occurred during the update: {str(e)}")
    
    logger.info("Update process finished")