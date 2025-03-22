import aiohttp
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

import polars as pl
from aiohttp import ClientSession
from aiolimiter import AsyncLimiter

logger = logging.getLogger(__name__)

class KalshiClient:
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    MARKETS_ENDPOINT = "/markets"
    DATA_DIR = "./data_files"
    MARKETS_FILE = "kalshi_markets.json"
    
    def __init__(
        self,
        requests_per_second: float = 10.0,
        session: Optional[ClientSession] = None
    ):
        self.rate_limiter = AsyncLimiter(requests_per_second)
        self.session = session
        Path(self.DATA_DIR).mkdir(parents=True, exist_ok=True)
        
    async def __aenter__(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    def _get_markets_file_path(self) -> Path:
        return Path(self.DATA_DIR) / self.MARKETS_FILE
        
    def _save_markets(self, markets: List[Dict]):
        file_path = self._get_markets_file_path()
        existing_markets = self._load_existing_markets()
        
        
        all_markets = existing_markets + markets
        logger.info(f"Combining {len(existing_markets)} existing markets with {len(markets)} new markets")
        
        
        df = pl.DataFrame(all_markets)
        if len(df) > 0:
            original_count = len(all_markets)
            df = df.unique(subset=['ticker'], maintain_order=True)
            all_markets = df.to_dicts()
            logger.info(f"Removed {original_count - len(all_markets)} duplicate markets")
            
        logger.info(f"Saving {len(all_markets)} total markets to {file_path}")
        with open(file_path, 'w') as f:
            json.dump(all_markets, f, indent=2)
            
    async def _fetch_markets_page(
        self,
        cursor: Optional[str] = None,
        limit: int = 1000,
        status: str = "open",
        min_close_ts: Optional[int] = None,
        event_ticker: Optional[str] = None,
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        params = {
            "limit": limit,
            "status": status
        }
        
        if cursor:
            params["cursor"] = cursor
        if min_close_ts:
            params["min_close_ts"] = min_close_ts
        if event_ticker:
            params["event_ticker"] = event_ticker
            
        logger.debug(f"Fetching markets with params: {params}")
        async with self.rate_limiter:
            async with self.session.get(
                f"{self.BASE_URL}{self.MARKETS_ENDPOINT}",
                params=params
            ) as response:
                response.raise_for_status()
                data = await response.json()
                markets = data.get("markets", [])
                next_cursor = data.get("cursor")
                logger.info(f"Fetched {len(markets)} markets" + (f" with cursor: {cursor}" if cursor else ""))
                return markets, next_cursor
                
    async def fetch_markets(
        self,
        status: str = "open",
        min_close_ts: Optional[int] = None,
        event_ticker: Optional[str] = None,
        limit: int = 1000,
        max_concurrent: int = 5
    ) -> List[Dict]:
        all_markets = []
        cursors = [None]  
        has_more = True
        
        logger.info("Starting market fetch with concurrent requests")
        
        while has_more and cursors:            
            current_cursors = cursors[:max_concurrent]
            cursors = cursors[max_concurrent:]
                        
            tasks = [
                self._fetch_markets_page(
                    cursor=cursor,
                    limit=limit,
                    status=status,
                    min_close_ts=min_close_ts,
                    event_ticker=event_ticker
                )
                for cursor in current_cursors
            ]
                        
            results = await asyncio.gather(*tasks, return_exceptions=True)
                        
            new_markets = []
            for result in results:
                if isinstance(result, tuple):
                    markets, next_cursor = result
                    new_markets.extend(markets)
                    if next_cursor:
                        cursors.append(next_cursor)
                elif isinstance(result, Exception):
                    logger.error(f"Error in fetch: {str(result)}")
            
            if not new_markets:
                has_more = False
            else:
                all_markets.extend(new_markets)
                logger.info(f"Fetched batch of {len(new_markets)} markets, total so far: {len(all_markets)}")
            
            
            if not cursors:
                has_more = False
                
        if all_markets:
            self._save_markets(all_markets)
            
        return all_markets
