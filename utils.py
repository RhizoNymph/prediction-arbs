import math
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime
import pytz
import logging

logger = logging.getLogger(__name__)

def kalshi_fee(price, num_contracts=1):
    """
    Computes the trading fee for contracts on Kalshi.
    
    Args:
        price (float): Price of contract in decimal form (e.g. 0.5 for 50 cents)
        num_contracts (int): Number of contracts being traded
        
    Returns:
        float: Fee in dollars, rounded up to next cent
        
    Formula:
        fees = ceil(0.07 * C * P * (1-P))
        where:
        - C = number of contracts
        - P = price in decimal form
    """
    return math.ceil(0.07 * num_contracts * price * (1 - price) * 100) / 100.0

def load_market_data():
    """Load market data from JSON files"""
    with open('./data_files/kalshi_markets.json', 'r') as f:
        kalshi_markets = json.load(f)
    
    with open('./data_files/polymarket_markets.json', 'r') as f:
        poly_markets = json.load(f)
        
    return kalshi_markets, poly_markets

def prepare_market_text(market, platform):
    """Prepare market text for TF-IDF vectorization"""
    if platform == 'kalshi':
        text = f"{market['title']} {market.get('rules_primary', '')} {market.get('rules_secondary', '')}"
    else:  
        text = f"{market['question']} {market.get('description', '')}"
    return text

def find_similar_markets(kalshi_markets, poly_markets, similarity_threshold=0.8):
    """Find similar markets between Kalshi and Polymarket using TF-IDF and cosine similarity"""
    
    
    kalshi_texts = [prepare_market_text(m, 'kalshi') for m in kalshi_markets]
    poly_texts = [prepare_market_text(m, 'polymarket') for m in poly_markets]
    
    
    vectorizer = TfidfVectorizer(stop_words='english')
    all_texts = kalshi_texts + poly_texts
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    
    kalshi_vectors = tfidf_matrix[:len(kalshi_texts)]
    poly_vectors = tfidf_matrix[len(kalshi_texts):]
    similarities = cosine_similarity(kalshi_vectors, poly_vectors)
    
    
    similar_pairs = []
    for i, kalshi_market in enumerate(kalshi_markets):
        for j, poly_market in enumerate(poly_markets):
            if similarities[i, j] >= similarity_threshold:
                similar_pairs.append((kalshi_market, poly_market, similarities[i, j]))
    
    return similar_pairs

def calculate_arbitrage_opportunities(similar_pairs):
    """Calculate arbitrage opportunities from similar market pairs"""
    opportunities = []
    
    for kalshi_market, poly_market, similarity in similar_pairs:
        try:
            
            if not all([
                kalshi_market.get('expiration_time'),
                poly_market.get('endDate'),
                poly_market.get('outcomePrices')
            ]):
                continue
                
            
            kalshi_yes_price = kalshi_market.get('yes_ask', 0) / 100  
            kalshi_no_price = kalshi_market.get('no_ask', 0) / 100
                        
            poly_prices = json.loads(poly_market['outcomePrices'])
            poly_yes_price = float(poly_prices[0])
            poly_no_price = float(poly_prices[1])
                        
            kalshi_yes_fee = kalshi_fee(kalshi_yes_price, 100) / 100
            kalshi_no_fee = kalshi_fee(kalshi_no_price, 100) / 100
                                
            strat1_cost = kalshi_yes_price + kalshi_yes_fee + poly_no_price
            strat1_arb = 1 - strat1_cost if strat1_cost < 1 else 0
                        
            strat2_cost = kalshi_no_price + kalshi_no_fee + poly_yes_price
            strat2_arb = 1 - strat2_cost if strat2_cost < 1 else 0
                        
            kalshi_exp = datetime.fromisoformat(kalshi_market['expiration_time'].replace('Z', '+00:00'))
            poly_exp = datetime.fromisoformat(poly_market['endDate'].replace('Z', '+00:00'))
            nearest_exp = min(kalshi_exp, poly_exp)
                        
            if strat1_arb > 0 or strat2_arb > 0:
                opportunities.append({
                    'kalshi_market': kalshi_market['title'],
                    'poly_market': poly_market['question'],
                    'similarity_score': similarity,
                    'kalshi_yes_poly_no_arb': strat1_arb,  
                    'kalshi_no_poly_yes_arb': strat2_arb,  
                    'kalshi_yes_price': kalshi_yes_price,
                    'kalshi_no_price': kalshi_no_price,
                    'poly_yes_price': poly_yes_price,
                    'poly_no_price': poly_no_price,
                    'kalshi_yes_fee': kalshi_yes_fee,
                    'kalshi_no_fee': kalshi_no_fee,
                    'expiration_date': nearest_exp,
                    'kalshi_id': kalshi_market['ticker'],
                    'poly_id': poly_market['id']
                })
                
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:            
            logger.warning(f"Error processing market pair: {str(e)}")
            continue
    
    return opportunities

def generate_arbitrage_report(opportunities):
    """Generate a polars DataFrame of arbitrage opportunities sorted by expiration date"""
    if not opportunities:
        return pl.DataFrame()
        
    df = pl.DataFrame(opportunities)
        
    df = df.sort(
        by=['expiration_date', 'kalshi_yes_poly_no_arb', 'kalshi_no_poly_yes_arb'],
        descending=[False, True, True]
    )
    
    return df
