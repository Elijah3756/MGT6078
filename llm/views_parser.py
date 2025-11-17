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
        Validate and normalize views (supports both old and new formats)
        
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
            # Determine view type (backward compatibility)
            view_type = view.get('type', 'absolute')
            if 'type' not in view:
                if 'ticker1' in view and 'ticker2' in view:
                    view_type = 'relative'
                else:
                    view_type = 'absolute'
            
            if view_type == 'relative':
                # Validate relative view
                if 'ticker1' not in view or 'ticker2' not in view:
                    print(f"Warning: Relative view missing ticker1 or ticker2: {view}")
                    continue
                if 'expected_outperformance' not in view:
                    print(f"Warning: Relative view missing expected_outperformance: {view}")
                    continue
                
                ticker1 = view['ticker1']
                ticker2 = view['ticker2']
                if ticker1 not in self.tickers:
                    print(f"Warning: Unknown ticker1: {ticker1}")
                    continue
                if ticker2 not in self.tickers:
                    print(f"Warning: Unknown ticker2: {ticker2}")
                    continue
                
                expected_outperformance = float(view['expected_outperformance'])
                confidence = float(view.get('confidence', 50))
                
                # Bound values
                expected_outperformance = max(-0.5, min(0.5, expected_outperformance))
                confidence = max(0, min(100, confidence))
                
                validated_view = {
                    'type': 'relative',
                    'ticker1': ticker1,
                    'ticker2': ticker2,
                    'expected_outperformance': expected_outperformance,
                    'confidence': confidence,
                    'reasoning': view.get('reasoning', 'No reasoning provided')
                }
                validated_views.append(validated_view)
            else:
                # Validate absolute view (backward compatible)
                if 'ticker' not in view or 'expected_return' not in view:
                    print(f"Warning: Missing required fields in view: {view}")
                    continue
                
                ticker = view['ticker']
                if ticker not in self.tickers:
                    print(f"Warning: Unknown ticker: {ticker}")
                    continue
                
                expected_return = float(view['expected_return'])
                confidence = float(view.get('confidence', 50))
                
                # Bound values
                expected_return = max(-0.5, min(0.5, expected_return))
                confidence = max(0, min(100, confidence))
                
                validated_view = {
                    'type': 'absolute',
                    'ticker': ticker,
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
        Note: For relative views, we don't require all tickers to be covered
        
        Args:
            views_data: Views dictionary
            
        Returns:
            Complete views dictionary (may include relative views)
        """
        # Collect all tickers mentioned in views
        mentioned_tickers = set()
        for view in views_data['views']:
            view_type = view.get('type', 'absolute')
            if view_type == 'relative':
                mentioned_tickers.add(view.get('ticker1'))
                mentioned_tickers.add(view.get('ticker2'))
            else:
                mentioned_tickers.add(view.get('ticker'))
        
        # Add default absolute views for missing tickers (only if we have mostly absolute views)
        absolute_views = [v for v in views_data['views'] if v.get('type', 'absolute') == 'absolute']
        if len(absolute_views) > len(views_data['views']) / 2:
            # If mostly absolute views, ensure all tickers covered
            for ticker in self.tickers:
                if ticker not in mentioned_tickers:
                    print(f"Warning: Adding default view for missing ticker: {ticker}")
                    views_data['views'].append({
                        'type': 'absolute',
                        'ticker': ticker,
                        'expected_return': 0.02,
                        'confidence': 30,
                        'reasoning': 'Default view (ticker was missing from LLM response)'
                    })
        
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

