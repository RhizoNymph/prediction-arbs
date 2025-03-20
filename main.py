import asyncio
from datetime import datetime
import logging
from pathlib import Path

from data.polymarket import PolymarketClient
from data.kalshi import KalshiClient
import utils

# Set up logging
def setup_logging():        
    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    console_handler.setFormatter(console_formatter)
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Add the handlers to the logger    
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

async def fetch_polymarket():
    logger.info("Starting Polymarket data fetch")
    start_time = datetime.now()
    
    async with PolymarketClient(requests_per_second=20.0) as client:
        try:
            # Fetch active markets
            logger.info("Initiating Polymarket fetch...")
            markets = await client.fetch_markets(
                active_only=True,
                max_concurrent=20,
                limit=500
            )
            
            fetch_time = datetime.now() - start_time
            logger.info(f"Successfully fetched {len(markets)} Polymarket markets in {fetch_time}")
            
            if markets:
                # Calculate some basic stats
                total_volume = sum(market.get('volumeNum', 0) for market in markets)
                categories = set(market.get('category', 'Unknown') for market in markets)
                
                logger.info(f"Polymarket total volume: ${total_volume:,.2f}")
                logger.info(f"Polymarket categories: {', '.join(sorted(categories))}")
            else:
                logger.info("No new Polymarket markets found")
                
        except Exception as e:
            logger.error(f"Error fetching Polymarket markets: {str(e)}", exc_info=True)
            raise

async def fetch_kalshi():
    logger.info("Starting Kalshi data fetch")
    start_time = datetime.now()
    
    async with KalshiClient(requests_per_second=20.0) as client:
        try:
            # Fetch open markets
            logger.info("Initiating Kalshi fetch...")
            markets = await client.fetch_markets(
                status="open",
                max_concurrent=20,
                limit=1000
            )
            
            fetch_time = datetime.now() - start_time
            logger.info(f"Successfully fetched {len(markets)} Kalshi markets in {fetch_time}")
            
            if markets:
                # Calculate some basic stats
                total_volume = sum(market.get('volume', 0) for market in markets)
                total_liquidity = sum(market.get('liquidity', 0) for market in markets)
                
                logger.info(f"Kalshi total volume: ${total_volume:,.2f}")
                logger.info(f"Kalshi total liquidity: ${total_liquidity:,.2f}")                
            else:
                logger.info("No new Kalshi markets found")
                
        except Exception as e:
            logger.error(f"Error fetching Kalshi markets: {str(e)}", exc_info=True)
            raise

async def find_arbitrage_opportunities():
    logger.info("Starting arbitrage analysis")
    start_time = datetime.now()
    
    try:
        # Load market data
        kalshi_markets, poly_markets = utils.load_market_data()
        logger.info(f"Loaded {len(kalshi_markets)} Kalshi markets and {len(poly_markets)} Polymarket markets")
        
        # Find similar markets
        similar_pairs = utils.find_similar_markets(kalshi_markets, poly_markets)
        logger.info(f"Found {len(similar_pairs)} potentially similar market pairs")
        
        # Calculate arbitrage opportunities
        opportunities = utils.calculate_arbitrage_opportunities(similar_pairs)
        logger.info(f"Found {len(opportunities)} potential arbitrage opportunities")
        
        # Generate report
        df = utils.generate_arbitrage_report(opportunities)
        
        if not df.is_empty():
            # Save to CSV
            output_path = "./data_files/arbitrage_opportunities.csv"
            df.write_csv(output_path)
            logger.info(f"Saved arbitrage report to {output_path}")
        else:
            logger.info("No arbitrage opportunities found")
            
    except Exception as e:
        logger.error(f"Error in arbitrage analysis: {str(e)}", exc_info=True)
    finally:
        total_time = datetime.now() - start_time
        logger.info(f"Arbitrage analysis completed in {total_time}")

async def main():
    total_start_time = datetime.now()
    logger.info("Starting market data fetch")
    
    try:
        # Fetch from both sources concurrently
        await asyncio.gather(
            fetch_polymarket(),
            fetch_kalshi()
        )
        
        # Run arbitrage analysis
        await find_arbitrage_opportunities()
        
    except Exception as e:
        logger.error(f"Error in market fetch: {str(e)}", exc_info=True)
    finally:
        total_time = datetime.now() - total_start_time
        logger.info(f"Total execution time: {total_time}")

if __name__ == "__main__":
    asyncio.run(main())
