"""
Views Converter
Converts LLM-generated views to Black-Litterman P, Q, and Omega matrices
"""

import numpy as np
from typing import Dict, Tuple, List


class ViewsConverter:
    """Convert LLM views to Black-Litterman matrices"""
    
    def __init__(self, tickers=None):
        self.tickers = tickers or ['XLK', 'XLY', 'ITA', 'XLE', 'XLV', 'XLF']
        self.n_assets = len(self.tickers)
    
    def convert_to_matrices(self, views_data: Dict, 
                           base_variance: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert LLM views to P, Q, and Omega matrices
        Supports both absolute and relative views
        
        Args:
            views_data: Dictionary with LLM views (can include absolute and relative views)
            base_variance: Base variance for uncertainty scaling
            
        Returns:
            Tuple of (P_matrix, Q_vector, Omega_matrix)
        """
        views = views_data['views']
        n_views = len(views)
        
        # Initialize matrices
        P = np.zeros((n_views, self.n_assets))
        Q = np.zeros((n_views, 1))
        Omega = np.zeros((n_views, n_views))
        
        # Build matrices from views
        for i, view in enumerate(views):
            view_type = view.get('type', 'absolute')  # Default to absolute for backward compatibility
            confidence = view.get('confidence', 50)
            
            if view_type == 'relative':
                # Relative view: ticker1 will outperform ticker2 by expected_outperformance
                ticker1 = view['ticker1']
                ticker2 = view['ticker2']
                expected_outperformance = view['expected_outperformance']
                
                # Get ticker indices
                ticker1_idx = self.tickers.index(ticker1)
                ticker2_idx = self.tickers.index(ticker2)
                
                # P matrix: relative view (1 for ticker1, -1 for ticker2, 0 for others)
                P[i, ticker1_idx] = 1.0
                P[i, ticker2_idx] = -1.0
                
                # Q vector: expected outperformance
                Q[i, 0] = expected_outperformance
                
            else:
                # Absolute view (default or explicit)
                ticker = view.get('ticker')
                if ticker is None:
                    # Backward compatibility: try to infer from old format
                    raise ValueError(f"View {i} missing 'ticker' field for absolute view")
                
                expected_return = view.get('expected_return', 0.0)
                
                # Get ticker index
                ticker_idx = self.tickers.index(ticker)
                
                # P matrix: absolute view (1 for the ticker, 0 for others)
                P[i, ticker_idx] = 1.0
                
                # Q vector: expected return
                Q[i, 0] = expected_return
            
            # Omega matrix: diagonal with uncertainty (same for both view types)
            # Higher confidence = lower uncertainty
            # Convert confidence (0-100) to uncertainty
            # confidence 100 -> uncertainty near 0
            # confidence 0 -> uncertainty high
            uncertainty = base_variance * (1 - confidence / 100) * 2
            uncertainty = max(0.0001, uncertainty)  # Avoid zero uncertainty
            Omega[i, i] = uncertainty
        
        return P, Q, Omega
    
    def create_relative_view(self, ticker1: str, ticker2: str, 
                            expected_outperformance: float) -> np.ndarray:
        """
        Create a relative view: ticker1 will outperform ticker2 by X%
        
        Args:
            ticker1: First ticker
            ticker2: Second ticker
            expected_outperformance: Expected relative return difference
            
        Returns:
            P matrix row for this view
        """
        P_row = np.zeros(self.n_assets)
        P_row[self.tickers.index(ticker1)] = 1.0
        P_row[self.tickers.index(ticker2)] = -1.0
        return P_row
    
    def validate_matrices(self, P: np.ndarray, Q: np.ndarray, 
                         Omega: np.ndarray) -> bool:
        """
        Validate the P, Q, Omega matrices
        
        Args:
            P: Views matrix
            Q: Expected returns vector
            Omega: Uncertainty matrix
            
        Returns:
            True if valid, raises exception otherwise
        """
        # Check dimensions
        n_views = P.shape[0]
        
        if P.shape[1] != self.n_assets:
            raise ValueError(f"P matrix should have {self.n_assets} columns, got {P.shape[1]}")
        
        if Q.shape != (n_views, 1):
            raise ValueError(f"Q should be ({n_views}, 1), got {Q.shape}")
        
        if Omega.shape != (n_views, n_views):
            raise ValueError(f"Omega should be ({n_views}, {n_views}), got {Omega.shape}")
        
        # Check Omega is positive definite (or at least positive diagonal)
        if not np.all(np.diag(Omega) > 0):
            raise ValueError("Omega diagonal must be positive")
        
        return True
    
    def print_matrices(self, P: np.ndarray, Q: np.ndarray, Omega: np.ndarray):
        """
        Print formatted matrices for inspection
        
        Args:
            P: Views matrix
            Q: Expected returns vector
            Omega: Uncertainty matrix
        """
        print("\n" + "="*80)
        print("BLACK-LITTERMAN VIEWS MATRICES")
        print("="*80)
        
        print("\nP Matrix (Views Specification):")
        print("Tickers:", self.tickers)
        print(P)
        
        print("\nQ Vector (Expected Returns/Outperformance):")
        for i in range(len(Q)):
            # Check if this is a relative view (sum of P row is 0)
            if np.sum(P[i]) == 0:
                # Relative view: find ticker1 (positive) and ticker2 (negative)
                ticker1_idx = np.where(P[i] > 0)[0][0]
                ticker2_idx = np.where(P[i] < 0)[0][0]
                ticker1 = self.tickers[ticker1_idx]
                ticker2 = self.tickers[ticker2_idx]
                print(f"  {ticker1} vs {ticker2}: {Q[i][0]:+.4f} ({Q[i][0]*100:+.2f}% outperformance)")
            else:
                # Absolute view
                ticker = self.tickers[np.argmax(P[i])]
                print(f"  {ticker}: {Q[i][0]:+.4f} ({Q[i][0]*100:+.2f}%)")
        
        print("\nOmega Matrix (View Uncertainties):")
        print("Diagonal:", np.diag(Omega))
        print(f"Mean uncertainty: {np.mean(np.diag(Omega)):.6f}")
    
    def save_matrices(self, P: np.ndarray, Q: np.ndarray, Omega: np.ndarray, 
                     output_prefix: str):
        """
        Save matrices to files
        
        Args:
            P: Views matrix
            Q: Expected returns vector
            Omega: Uncertainty matrix
            output_prefix: Prefix for output files
        """
        np.save(f'{output_prefix}_P.npy', P)
        np.save(f'{output_prefix}_Q.npy', Q)
        np.save(f'{output_prefix}_Omega.npy', Omega)
        
        print(f"Saved matrices to {output_prefix}_*.npy")


def main():
    """Test views converter"""
    import json
    
    # Load sample views
    with open('output/views/Q1_2024_views.json', 'r') as f:
        views_data = json.load(f)
    
    # Convert to matrices
    converter = ViewsConverter()
    P, Q, Omega = converter.convert_to_matrices(views_data)
    
    # Validate
    converter.validate_matrices(P, Q, Omega)
    
    # Print
    converter.print_matrices(P, Q, Omega)
    
    # Save
    converter.save_matrices(P, Q, Omega, 'output/Q1_2024_matrices')


if __name__ == "__main__":
    main()

