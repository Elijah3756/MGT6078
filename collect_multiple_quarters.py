"""
Script to collect political news for multiple quarters
Based on project timeline: 2024 Q3, Q4 and 2025 Q1, Q2
Supports parallel processing of quarters
"""

from political_news_collector import PoliticalNewsCollector, RateLimitError
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import os

# Lock for thread-safe printing
print_lock = Lock()

def safe_print(*args, **kwargs):
    """Thread-safe print function"""
    with print_lock:
        print(*args, **kwargs)

def process_quarter(year, quarter):
    """
    Process a single quarter - collects news, saves files, and returns results
    
    Args:
        year: Year (e.g., 2024)
        quarter: Quarter number (1, 2, 3, or 4)
    
    Returns:
        Tuple of (period_string, results_dict, success_flag, error_message, is_rate_limit)
    """
    period_str = f"Q{quarter}_{year}"
    
    try:
        safe_print(f"\n{'='*80}")
        safe_print(f"Processing Q{quarter} {year}")
        safe_print(f"{'='*80}")
        
        # Create a new collector instance for this quarter to avoid shared state
        collector = PoliticalNewsCollector()
        
        # Collect news for this quarter
        results = collector.collect_quarterly_news(year, quarter)
        
        # Save results
        collector.save_results(results, year, quarter)
        
        # Format for LLM
        llm_text = collector.format_for_llm(results)
        
        # Save LLM-formatted text
        os.makedirs("political_news_data", exist_ok=True)
        llm_filename = f"political_news_data/Q{quarter}_{year}_llm_input.txt"
        with open(llm_filename, 'w', encoding='utf-8') as f:
            f.write(llm_text)
        
        safe_print(f"\nCompleted Q{quarter} {year}")
        safe_print(f"  LLM input saved to: {llm_filename}")
        
        return (period_str, results, True, None, False)
        
    except RateLimitError as e:
        error_msg = str(e)
        safe_print(f"\n{'!'*80}")
        safe_print(f"RATE LIMIT ERROR processing Q{quarter} {year}")
        safe_print(f"   {error_msg}")
        safe_print(f"{'!'*80}\n")
        return (period_str, {}, False, error_msg, True)
    except Exception as e:
        error_msg = str(e)
        safe_print(f"\nError processing Q{quarter} {year}: {error_msg}")
        return (period_str, {}, False, error_msg, False)

def collect_quarters(quarters_config, parallel=True, max_workers=None):
    """
    Collect political news for multiple quarters
    
    Args:
        quarters_config: List of tuples (year, quarter)
        parallel: If True, process quarters in parallel. If False, process sequentially.
        max_workers: Maximum number of parallel workers. If None, uses default (typically CPU count).
    
    Returns:
        Dictionary mapping period string to results
    """
    all_results = {}
    
    if parallel:
        safe_print(f"\n{'='*80}")
        safe_print("PARALLEL PROCESSING MODE")
        safe_print(f"Processing {len(quarters_config)} quarters in parallel")
        safe_print(f"{'='*80}\n")
        
        # Use ThreadPoolExecutor for I/O-bound tasks (API calls)
        # Default max_workers is typically min(32, (os.cpu_count() or 1) + 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all quarters for processing
            future_to_quarter = {
                executor.submit(process_quarter, year, quarter): (year, quarter)
                for year, quarter in quarters_config
            }
            
            # Process completed tasks as they finish
            rate_limit_quarters = []
            for future in as_completed(future_to_quarter):
                year, quarter = future_to_quarter[future]
                try:
                    period_str, results, success, error, is_rate_limit = future.result()
                    if success:
                        all_results[period_str] = results
                    elif is_rate_limit:
                        rate_limit_quarters.append((year, quarter))
                except Exception as e:
                    safe_print(f"\nUnexpected error processing Q{quarter} {year}: {e}")
            
            # Report rate limit summary
            if rate_limit_quarters:
                safe_print(f"\n{'!'*80}")
                safe_print("RATE LIMIT SUMMARY")
                safe_print(f"{'!'*80}")
                safe_print(f"The following {len(rate_limit_quarters)} quarter(s) hit rate limits:")
                for year, quarter in rate_limit_quarters:
                    safe_print(f"  - Q{quarter} {year}")
                safe_print(f"\nRecommendations:")
                safe_print(f"  1. Reduce max_workers (currently: {max_workers or 'default'})")
                safe_print(f"  2. Switch to sequential mode (set parallel=False)")
                safe_print(f"  3. Wait a few minutes and retry the failed quarters")
                safe_print(f"{'!'*80}\n")
    else:
        safe_print(f"\n{'='*80}")
        safe_print("SEQUENTIAL PROCESSING MODE")
        safe_print(f"Processing {len(quarters_config)} quarters sequentially")
        safe_print(f"{'='*80}\n")
        
        collector = PoliticalNewsCollector()
        
        for year, quarter in quarters_config:
            period_str = f"Q{quarter}_{year}"
            
            try:
                safe_print(f"\n{'='*80}")
                safe_print(f"Processing Q{quarter} {year}")
                safe_print(f"{'='*80}")
                
                # Collect news for this quarter
                results = collector.collect_quarterly_news(year, quarter)
                
                # Save results
                collector.save_results(results, year, quarter)
                
                # Format for LLM
                llm_text = collector.format_for_llm(results)
                
                # Save LLM-formatted text
                os.makedirs("political_news_data", exist_ok=True)
                llm_filename = f"political_news_data/Q{quarter}_{year}_llm_input.txt"
                with open(llm_filename, 'w', encoding='utf-8') as f:
                    f.write(llm_text)
                
                safe_print(f"\nCompleted Q{quarter} {year}")
                safe_print(f"  LLM input saved to: {llm_filename}")
                
                all_results[period_str] = results
                
                # Add delay to avoid rate limiting
                if quarter != quarters_config[-1][1] or year != quarters_config[-1][0]:
                    safe_print("\nWaiting 5 seconds before next quarter...")
                    time.sleep(5)
                
            except Exception as e:
                safe_print(f"\nError processing Q{quarter} {year}: {e}")
                continue
    
    # Print final summary
    safe_print("\n" + "="*80)
    safe_print("FINAL SUMMARY")
    safe_print("="*80)
    
    for period, results in sorted(all_results.items()):
        safe_print(f"\n{period}:")
        for ticker, df in results.items():
            safe_print(f"  {ticker}: {len(df)} articles")
    
    return all_results


def main():
    """
    Main function - collects data for project quarters
    """
    # Project quarters as specified: 2024 Q3, Q4 and 2025 Q1, Q2
    quarters_to_collect = [
        (2024, 1),
        (2024, 2),
        (2024, 3),
        (2024, 4),
        (2025, 1),
        (2025, 2)
    ]
    
    # Set parallel=True to process quarters in parallel, False for sequential
    # max_workers=None uses default (typically CPU count), or specify a number
    parallel_mode = True
    max_workers = None  # Set to a number (e.g., 4) to limit parallel workers
    
    print("Political News Collection - Multiple Quarters")
    print("="*80)
    print(f"Collecting data for {len(quarters_to_collect)} quarters:")
    for year, quarter in quarters_to_collect:
        print(f"  - Q{quarter} {year}")
    print(f"\nMode: {'PARALLEL' if parallel_mode else 'SEQUENTIAL'}")
    if parallel_mode and max_workers:
        print(f"Max workers: {max_workers}")
    
    results = collect_quarters(quarters_to_collect, parallel=parallel_mode, max_workers=max_workers)
    
    print(f"\nCollection complete! Processed {len(results)} quarters.")
    print(f"  All data saved in 'political_news_data/' directory")


if __name__ == "__main__":
    main()

