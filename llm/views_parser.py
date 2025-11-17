"""
LLM Views Parser
Parses LLM output and extracts investment views
"""

import json
import re
from typing import Dict, List, Optional


class ViewsParser:
    """Parse LLM output to extract investment views"""
    
    def __init__(self, tickers=None):
        self.tickers = tickers or ['XLK', 'XLY', 'ITA', 'XLE', 'XLV', 'XLF']
    
    def parse_json_response(self, response_text: str) -> Optional[Dict]:
        """
        Parse JSON response from LLM
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            Parsed views dictionary or None if parsing fails
        """
        try:
            # Try direct JSON parsing
            views_data = json.loads(response_text)
            return self.validate_views(views_data)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                try:
                    views_data = json.loads(json_match.group(1))
                    return self.validate_views(views_data)
                except json.JSONDecodeError:
                    pass
            
            # Try to extract JSON without code blocks
            json_match = re.search(r'\{[^{}]*"views"[^{}]*\[.*?\][^{}]*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    views_data = json.loads(json_match.group(0))
                    return self.validate_views(views_data)
                except json.JSONDecodeError:
                    pass
        
        print("Warning: Could not parse JSON from LLM response")
        return None
    
    def validate_views(self, views_data: Dict) -> Dict:
        """
        Validate and clean views data
        
        Args:
            views_data: Parsed views dictionary
            
        Returns:
            Validated views dictionary
        """
        if 'views' not in views_data:
            print("Warning: 'views' key not found in response")
            return views_data
        
        validated_views = []
        for view in views_data['views']:
            # Check required fields
            if 'ticker' not in view or 'expected_return' not in view or 'confidence' not in view:
                print(f"Warning: Missing required fields in view: {view}")
                continue
            
            # Validate ticker
            if view['ticker'] not in self.tickers:
                print(f"Warning: Unknown ticker: {view['ticker']}")
                continue
            
            # Validate and bound values
            expected_return = float(view['expected_return'])
            confidence = float(view['confidence'])
            
            # Bound expected returns to reasonable range (-50% to +50%)
            expected_return = max(-0.5, min(0.5, expected_return))
            
            # Bound confidence to 0-100
            confidence = max(0, min(100, confidence))
            
            validated_view = {
                'ticker': view['ticker'],
                'expected_return': expected_return,
                'confidence': confidence,
                'reasoning': view.get('reasoning', 'No reasoning provided')
            }
            
            validated_views.append(validated_view)
        
        views_data['views'] = validated_views
        return views_data
    
    def create_default_views(self, quarter: str) -> Dict:
        """
        Create conservative default views if LLM fails
        
        Args:
            quarter: Quarter string
            
        Returns:
            Default views dictionary
        """
        print("Creating default neutral views as fallback")
        
        views = []
        for ticker in self.tickers:
            views.append({
                'ticker': ticker,
                'expected_return': 0.02,  # Modest 2% expected return
                'confidence': 30,  # Low confidence
                'reasoning': 'Default neutral view (LLM parsing failed)'
            })
        
        return {
            'quarter': quarter,
            'views': views
        }
    
    def ensure_all_tickers(self, views_data: Dict) -> Dict:
        """
        Ensure all tickers have views, add defaults if missing
        
        Args:
            views_data: Views dictionary
            
        Returns:
            Complete views dictionary with all tickers
        """
        existing_tickers = {view['ticker'] for view in views_data['views']}
        
        for ticker in self.tickers:
            if ticker not in existing_tickers:
                print(f"Warning: Adding default view for missing ticker: {ticker}")
                views_data['views'].append({
                    'ticker': ticker,
                    'expected_return': 0.02,
                    'confidence': 30,
                    'reasoning': 'Default view (ticker was missing from LLM response)'
                })
        
        # Sort views by ticker order
        ticker_order = {ticker: i for i, ticker in enumerate(self.tickers)}
        views_data['views'].sort(key=lambda x: ticker_order.get(x['ticker'], 99))
        
        return views_data
    
    def parse_and_validate(self, response_text: str, quarter: str) -> Dict:
        """
        Parse and validate LLM response, with fallback to defaults
        
        Args:
            response_text: Raw LLM response
            quarter: Quarter string
            
        Returns:
            Validated views dictionary
        """
        views_data = self.parse_json_response(response_text)
        
        if views_data is None:
            return self.create_default_views(quarter)
        
        # Ensure all tickers are present
        views_data = self.ensure_all_tickers(views_data)
        
        return views_data


def main():
    """Test views parser"""
    parser = ViewsParser()
    
    # Test with sample JSON
    sample_response = """
    ```json
    {
      "quarter": "Q1_2024",
      "views": [
        {
          "ticker": "XLK",
          "expected_return": 0.04,
          "confidence": 70,
          "reasoning": "Strong tech sector momentum"
        },
        {
          "ticker": "XLY",
          "expected_return": 0.02,
          "confidence": 60,
          "reasoning": "Consumer spending remains stable"
        }
      ]
    }
    ```
    """
    
    views = parser.parse_and_validate(sample_response, 'Q1_2024')
    print(json.dumps(views, indent=2))


if __name__ == "__main__":
    main()

