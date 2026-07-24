# src/matcher.py
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz, token_set_ratio
from datetime import datetime

class TransactionMatcher:
    """
    Two-stage matching engine:
    1. Exact matching (fast path)
    2. Fuzzy matching for remaining transactions
    """
    
    def __init__(self, amount_tolerance=0.01, date_tolerance_days=3):
        self.amount_tolerance = amount_tolerance
        self.date_tolerance_days = date_tolerance_days
        self.weights = {
            'date': 0.25,      # 25% weight
            'amount': 0.40,    # 40% weight (most important)
            'description': 0.35 # 35% weight
        }
    
    def match(self, source_a, source_b):
        """
        Main matching pipeline.
        Returns: matched, unmatched_a, unmatched_b
        """
        # Make copies to avoid modifying originals
        source_a = source_a.copy()
        source_b = source_b.copy()
        
        matched = []
        
        # Stage 1: Exact matching
        exact_matches, remaining_a, remaining_b = self._exact_match(source_a, source_b)
        matched.extend(exact_matches)
        
        # Stage 2: Fuzzy matching on remaining transactions
        fuzzy_matches, still_remaining_a, still_remaining_b = self._fuzzy_match(
            remaining_a, remaining_b
        )
        matched.extend(fuzzy_matches)
        
        return matched, still_remaining_a, still_remaining_b
    
    def _exact_match(self, source_a, source_b):
        """Try to match on amount + date (since transaction IDs differ between systems)"""
        matched = []
        remaining_a = []
        remaining_b = source_b.copy()
        
        # Try matching by (amount + date) first - this is the most reliable
        for idx, a_row in source_a.iterrows():
            # Find potential matches in B
            potential_matches = source_b[
                (abs(source_b['amount'] - a_row['amount']) <= self.amount_tolerance) &
                (abs((source_b['date'] - a_row['date']).dt.days) <= self.date_tolerance_days)
            ]
            
            if len(potential_matches) == 1:  # Unique match
                b_idx = potential_matches.index[0]
                b_row = potential_matches.loc[b_idx]
                
                matched.append({
                    'source_a': a_row.to_dict(),
                    'source_b': b_row.to_dict(),
                    'match_type': 'exact',
                    'confidence': 95.0  # High confidence
                })
                
                # Remove matched row from B
                remaining_b = remaining_b.drop(b_idx)
            else:
                remaining_a.append(a_row.to_dict())
        
        return matched, pd.DataFrame(remaining_a), remaining_b
    
    def _fuzzy_match(self, remaining_a, remaining_b):
        """Use fuzzy matching for remaining transactions"""
        matched = []
        unmatched_a = []
        
        if len(remaining_a) == 0 or len(remaining_b) == 0:
            return matched, remaining_a, remaining_b
        
        remaining_b_copy = remaining_b.copy()
        
        for _, a_row in remaining_a.iterrows():
            best_match = None
            best_score = 0
            best_idx = None
            
            for idx, b_row in remaining_b_copy.iterrows():
                # Calculate composite score
                score = self._calculate_match_score(a_row, b_row)
                
                if score > best_score:
                    best_score = score
                    best_match = b_row
                    best_idx = idx
            
            # Minimum threshold: 60%
            if best_score >= 60 and best_match is not None:
                matched.append({
                    'source_a': a_row.to_dict(),
                    'source_b': best_match.to_dict(),
                    'match_type': 'fuzzy',
                    'confidence': best_score
                })
                remaining_b_copy = remaining_b_copy.drop(best_idx)
            else:
                unmatched_a.append(a_row.to_dict())
        
        return matched, pd.DataFrame(unmatched_a), remaining_b_copy
    
    def _calculate_match_score(self, a_row, b_row):
        """
        Multi-dimensional scoring:
        - Amount match (40% weight)
        - Date proximity (25% weight)  
        - Description/Merchant similarity (35% weight)
        """
        score = 0
        
        # 1. Amount score (exact match = 100 points)
        amount_diff = abs(a_row['amount'] - b_row['amount'])
        if amount_diff <= self.amount_tolerance:
            amount_score = 100
        elif amount_diff <= 1.0:  # Within $1
            amount_score = 80 - (amount_diff * 20)  # Gradually reduce
        else:
            amount_score = max(0, 50 - (amount_diff * 10))
        score += amount_score * self.weights['amount']
        
        # 2. Date score (closer = better) 
        date_diff = abs((a_row['date'] - b_row['date']).days)
        if date_diff == 0:
            date_score = 100
        elif date_diff <= self.date_tolerance_days:
            date_score = 90 - ((date_diff - 1) * (40 / self.date_tolerance_days))
        else:
            date_score = max(0, 50 - ((date_diff - self.date_tolerance_days) * 5))
        score += date_score * self.weights['date']
        
        # 3. Description/merchant score using fuzzy matching
        # Try different fields that might contain merchant info
        desc_a = str(a_row.get('description', a_row.get('merchant_desc', ''))).upper()
        desc_b = str(b_row.get('merchant', b_row.get('vendor_name', ''))).upper()
        
        # If we have counterparty info, use that too
        counterparty_a = str(a_row.get('counterparty', '')).upper()
        counterparty_b = str(b_row.get('counterparty', '')).upper()
        
        # Combine description with counterparty for better matching
        combined_a = desc_a + ' ' + counterparty_a
        combined_b = desc_b + ' ' + counterparty_b
        
        # Use token_set_ratio for better partial matching
        desc_score = token_set_ratio(combined_a, combined_b)
        score += desc_score * self.weights['description']
        
        return min(score, 100)  # Cap at 100
