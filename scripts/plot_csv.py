import pandas as pd
# import matplotlib.pyplot as plt # No longer directly used for plotting
import mplfinance as mpf # For financial plotting
import argparse
import os

def find_file_in_dir(filename_to_find, search_dir):
    """
    Searches for a file within the specified directory and its subdirectories.
    Returns the full path to the first match found, or None if not found.
    """
    for root, dirs, files in os.walk(search_dir):
        if filename_to_find in files:
            return os.path.join(root, filename_to_find)
    return None

def visualize_csv(csv_filename):
    """
    Finds a CSV file by its name within the 'data/processed/' directory (and subdirectories)
    and plots OHLC candles and volume bars.
    """
    search_path = os.path.join('data', 'processed')
    full_file_path = find_file_in_dir(csv_filename, search_path)

    if full_file_path is None:
        print(f"Error: File '{csv_filename}' not found within '{search_path}' or its subdirectories.")
        return

    try:
        # Load the CSV file
        df = pd.read_csv(full_file_path)
        print(f"Successfully loaded: {full_file_path}")

        file_name_for_title = csv_filename.replace('.csv', '')

        # --- Data Preparation for mplfinance ---
        required_ohlc_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_ohlc_cols):
            missing_cols = [col for col in required_ohlc_cols if col not in df.columns]
            print(f"Error: Missing one or more required OHLC columns ({', '.join(missing_cols)}) in {full_file_path}. Cannot plot candlestick chart.")
            return

        # mplfinance expects the index to be DatetimeIndex for time series plots
        if 'window_start' in df.columns:
            try:
                df['window_start'] = pd.to_datetime(df['window_start'])
                df.set_index('window_start', inplace=True)
            except Exception as e:
                print(f"Warning: Could not convert 'window_start' to DatetimeIndex or set it as index: {e}. Plotting may not work as expected.")
        elif not isinstance(df.index, pd.DatetimeIndex):
            print("Warning: DataFrame index is not a DatetimeIndex and 'window_start' column is not available or couldn't be converted. Time axis may not be displayed correctly.")
        
        # --- Plotting with mplfinance ---
        plot_kwargs = {
            'type': 'candle',
            'title': f'{file_name_for_title} - OHLCV Chart',
            'ylabel': 'Price',
            'style': 'yahoo',  # Common style, others include: 'charles', 'mike', 'nightclouds'
            # 'mav': (5, 10),  # Example: Add 5 and 10 period moving averages, uncomment to use
            'figsize': (16, 8) # Adjusted figure size
        }

        if 'volume' in df.columns:
            plot_kwargs['volume'] = True
            plot_kwargs['ylabel_lower'] = 'Volume'
        else:
            print(f"Warning: 'volume' column not found in {full_file_path}. Plotting without volume.")

        mpf.plot(df, **plot_kwargs)

    except FileNotFoundError: # Should be caught by the check above, but good for robustness
        print(f"Error: File not found at {full_file_path}")
    except pd.errors.EmptyDataError:
        print(f"Error: The file {full_file_path} is empty.")
    except Exception as e:
        print(f"An error occurred while processing {full_file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize OHLCV data (candlestick and volume) from a CSV file found within the 'data/processed/' directory and its subdirectories.")
    parser.add_argument("filename",
                        type=str,
                        help="Filename of the CSV (e.g., '2021-05-31_ETC-USD.csv'). The script will search for it in 'data/processed/' and its subdirectories.")

    args = parser.parse_args()
    visualize_csv(args.filename)
