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

class PolymarketClient:
    BASE_URL = "https://gamma-api.polymarket.com"
    MARKETS_ENDPOINT = "/markets"
    DATA_DIR = "./data_files"
    MARKETS_FILE = "polymarket_markets.json"
    
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
        existing_markets, _ = self._load_existing_markets()        
        
        all_markets = existing_markets + markets
        logger.info(f"Combining {len(existing_markets)} existing markets with {len(markets)} new markets")        
        
        df = pl.DataFrame(all_markets)
        if len(df) > 0:
            original_count = len(all_markets)
            df = df.unique(maintain_order=True)
            all_markets = df.to_dicts()
            logger.info(f"Removed {original_count - len(all_markets)} duplicate markets")
            
        logger.info(f"Saving {len(all_markets)} total markets to {file_path}")
        with open(file_path, 'w') as f:
            json.dump(all_markets, f, indent=2)
            
    async def _fetch_markets_page(
        self,
        limit: int,
        offset: int,
        active_only: bool = True,
        closed: bool = False,
        start_date_min: Optional[str] = None,
        start_date_max: Optional[str] = None,
        volume_num_min: Optional[float] = None,
        liquidity_num_min: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        params = {
            "limit": limit,
            "offset": offset
        }
        
        if active_only:
            params["active"] = "true"
        if not closed:
            params["closed"] = "false"
        if start_date_min:
            params["start_date_min"] = start_date_min
        if start_date_max:
            params["start_date_max"] = start_date_max
        if volume_num_min is not None:
            params["volume_num_min"] = volume_num_min
        if liquidity_num_min is not None:
            params["liquidity_num_min"] = liquidity_num_min
            
        logger.debug(f"Fetching markets with params: {params}")
        async with self.rate_limiter:
            async with self.session.get(
                f"{self.BASE_URL}{self.MARKETS_ENDPOINT}",
                params=params
            ) as response:
                response.raise_for_status()
                markets = await response.json()
                logger.info(f"Fetched {len(markets)} markets (offset: {offset})")
                return markets
                
    async def fetch_markets(
        self,
        active_only: bool = True,
        closed: bool = False,
        start_date_min: Optional[str] = None,
        start_date_max: Optional[str] = None,
        volume_num_min: Optional[float] = None,
        liquidity_num_min: Optional[float] = None,
        limit: int = 500,
        max_concurrent: int = 5
    ) -> List[Dict]:
        
        existing_markets, max_start_date = self._load_existing_markets()                
        
        if not start_date_min and max_start_date:
            start_date_min = max_start_date
        
        all_markets = []
        current_offset = 0
        has_more = True
        
        logger.info("Starting market fetch with concurrent requests")
        
        while has_more:            
            tasks = []
            for i in range(max_concurrent):
                offset = current_offset + (i * limit)
                tasks.append(
                    self._fetch_markets_page(
                        limit=limit,
                        offset=offset,
                        active_only=active_only,
                        closed=closed,
                        start_date_min=start_date_min,
                        start_date_max=start_date_max,
                        volume_num_min=volume_num_min,
                        liquidity_num_min=liquidity_num_min
                    )
                )            
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
                        
            new_markets = []
            for result in results:
                if isinstance(result, list) and result:
                    new_markets.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Error in fetch: {str(result)}")
            
            if not new_markets:
                has_more = False
            else:
                all_markets.extend(new_markets)
                logger.info(f"Fetched batch of {len(new_markets)} markets, total so far: {len(all_markets)}")
                current_offset += len(tasks) * limit                
                
                if any(isinstance(r, list) and len(r) < limit for r in results):
                    has_more = False
                
        if all_markets:
            self._save_markets(all_markets)
        
        return all_markets
