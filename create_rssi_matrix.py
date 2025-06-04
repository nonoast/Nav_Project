import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the matrix dimensions
ROWS = 7
COLS = 15
ROUTERS = 6

# Read the CSV file
def read_rssi_data(filepath):
    df = pd.read_csv(filepath, sep=';', encoding='utf-8')
    # Clean up the dataframe - drop empty rows at the end and handle column names
    df = df.dropna(how='all')
    # Get router columns (skip the first column which is the room number)
    router_cols = df.columns[1:7]  # Taking only the 6 router columns
    return df, router_cols

# Create a mapping between room numbers and matrix positions
# Based on the provided floor plan image
def create_room_to_position_mapping():
    # Initialize the mapping as {room_number: (row, col)}
    room_map = {}
    
    # First row (from top to bottom in the image)
    room_map[91] = (0, 0)
    room_map[84] = (0, 1)
    room_map[81] = (0, 2)
    room_map[74] = (0, 3)
    room_map[58] = (0, 4)
    room_map[57] = (0, 5)
    room_map[56] = (0, 6)
    room_map[41] = (0, 7)
    room_map[42] = (0, 8)
    room_map[43] = (0, 9)
    room_map[22] = (0, 10)
    room_map[23] = (0, 11)
    room_map[24] = (0, 12)
    
    # Second row
    room_map[90] = (1, 0)
    room_map[83] = (1, 1)
    room_map[80] = (1, 2)
    room_map[73] = (1, 3)
    room_map[53] = (1, 4)
    room_map[54] = (1, 5)
    room_map[55] = (1, 6)
    room_map[40] = (1, 7)
    room_map[39] = (1, 8)
    room_map[38] = (1, 9)
    room_map[21] = (1, 10)
    room_map[20] = (1, 11)
    room_map[19] = (1, 12)
    
    # Third row
    room_map[89] = (2, 0)
    room_map[82] = (2, 1)
    room_map[79] = (2, 2)
    room_map[72] = (2, 3)
    room_map[52] = (2, 4)
    room_map[51] = (2, 5)
    room_map[50] = (2, 6)
    room_map[35] = (2, 7)
    room_map[36] = (2, 8)
    room_map[37] = (2, 9)
    room_map[16] = (2, 10)
    room_map[17] = (2, 11)
    room_map[18] = (2, 12)
    
    # Fourth row (green row in image)
    room_map[85] = (3, 0)
    room_map[75] = (3, 1)
    room_map[68] = (3, 2)
    room_map[49] = (3, 3)
    room_map[48] = (3, 4)
    room_map[47] = (3, 5)
    room_map[46] = (3, 6)
    room_map[45] = (3, 7)
    room_map[44] = (3, 8)
    room_map[6] = (3, 9)
    room_map[5] = (3, 10)
    room_map[4] = (3, 11)
    room_map[3] = (3, 12)
    room_map[2] = (3, 13)
    room_map[1] = (3, 14)
    
    # Fifth row
    room_map[86] = (4, 0)
    room_map[76] = (4, 1)
    room_map[69] = (4, 2)
    room_map[61] = (4, 3)
    room_map[60] = (4, 4)
    room_map[59] = (4, 5)
    room_map[27] = (4, 6)
    room_map[26] = (4, 7)
    room_map[25] = (4, 8)
    room_map[7] = (4, 9)
    room_map[8] = (4, 10)
    room_map[9] = (4, 11)
    
    # Sixth row
    room_map[87] = (5, 0)
    room_map[77] = (5, 1)
    room_map[70] = (5, 2)
    room_map[62] = (5, 3)
    room_map[63] = (5, 4)
    room_map[64] = (5, 5)
    room_map[28] = (5, 6)
    room_map[29] = (5, 7)
    room_map[30] = (5, 8)
    room_map[12] = (5, 9)
    room_map[11] = (5, 10)
    room_map[10] = (5, 11)
    
    # Seventh row
    room_map[88] = (6, 0)
    room_map[78] = (6, 1)
    room_map[71] = (6, 2)
    room_map[67] = (6, 3)
    room_map[66] = (6, 4)
    room_map[65] = (6, 5)
    room_map[33] = (6, 6)
    room_map[32] = (6, 7)
    room_map[31] = (6, 8)
    room_map[13] = (6, 9)
    room_map[14] = (6, 10)
    room_map[15] = (6, 11)
    
    return room_map

# Create the matrix and fill it with RSSI values
def create_rssi_matrix(df, router_cols, room_map):
    # Initialize matrix with NaN values
    matrix = np.full((ROWS, COLS, ROUTERS), np.nan)
    
    # Fill the matrix with RSSI values from the CSV
    for index, row in df.iterrows():
        room_num = int(row['nom'])
        
        # Check if this room is in our mapping
        if room_num in room_map:
            matrix_row, matrix_col = room_map[room_num]
            
            # Add RSSI values for each router
            for router_idx, router_col in enumerate(router_cols):
                if pd.notnull(row[router_col]) and row[router_col] != '':
                    # Handle special case like 'AF'
                    if isinstance(row[router_col], str) and not row[router_col].startswith('-'):
                        continue
                    
                    try:
                        # Convert string values to float/int as needed
                        rssi_value = float(row[router_col])
                        matrix[matrix_row, matrix_col, router_idx] = rssi_value
                    except (ValueError, TypeError):
                        # Skip values that can't be converted
                        continue
    
    return matrix

# Create heatmaps for each router
def create_router_heatmaps(matrix, router_cols, room_map, show=True, save_path=None):
    router_count = matrix.shape[2]
    
    # Create a figure with subplots for each router
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Set a consistent color scale for all heatmaps
    vmin = np.nanmin(matrix)
    vmax = np.nanmax(matrix)
    
    # Create reverse mapping (position to room number)
    pos_to_room = {}
    for room, (row, col) in room_map.items():
        pos_to_room[(row, col)] = room
    
    for i in range(router_count):
        ax = axes[i]
        router_data = matrix[:, :, i]
        
        # Create the heatmap
        sns.heatmap(router_data, 
                   ax=ax, 
                   cmap="coolwarm_r",  # Reversed so stronger signals (less negative) are red
                   annot=True, 
                   fmt=".0f", 
                   mask=np.isnan(router_data),
                   vmin=vmin,
                   vmax=vmax)
        
        ax.set_title(f"Router {i+1}: {router_cols[i]}")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        
        # Add room numbers as text annotations
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                if (row, col) in pos_to_room:
                    room_num = pos_to_room[(row, col)]
                    ax.text(col + 0.5, row + 0.15, f"R{room_num}", 
                            horizontalalignment='center', 
                            verticalalignment='center',
                            color='black', 
                            fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig

# Create a combined heatmap showing the best router for each location
def create_best_router_heatmap(matrix, router_cols, room_map, show=True, save_path=None):
    # Find the router with the strongest signal at each position
    # (less negative RSSI value is stronger signal)
    best_router = np.nanargmax(matrix, axis=2)
    
    # Create a mask for positions without data
    all_nan_mask = np.all(np.isnan(matrix), axis=2)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Create a custom colormap for the routers
    cmap = plt.cm.get_cmap('tab10', len(router_cols))
    
    # Create the heatmap for best router
    masked_best_router = np.ma.masked_where(all_nan_mask, best_router)
    sns.heatmap(masked_best_router, 
               ax=ax, 
               cmap=cmap,
               annot=True, 
               fmt=".0f", 
               cbar=False)
    
    # Create a custom colorbar with router names
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), 
                        ax=ax, 
                        ticks=np.arange(len(router_cols)))
    cbar.set_ticklabels([f"Router {i+1}: {router}" for i, router in enumerate(router_cols)])
    
    ax.set_title("Best Router by Location")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    
    # Create reverse mapping (position to room number)
    pos_to_room = {}
    for room, (row, col) in room_map.items():
        pos_to_room[(row, col)] = room
    
    # Add room numbers
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if (row, col) in pos_to_room:
                room_num = pos_to_room[(row, col)]
                ax.text(col + 0.5, row + 0.15, f"R{room_num}", 
                        horizontalalignment='center', 
                        verticalalignment='center',
                        color='white', 
                        fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig

# Function to get the matrix directly in Python without saving to file
def get_rssi_matrix(csv_file_path=None):
    """
    Generate and return the RSSI matrix directly without saving to file.
    
    Args:
        csv_file_path: Path to the CSV file. If None, uses default path.
    
    Returns:
        tuple: (rssi_matrix, router_cols, room_map)
            - rssi_matrix: 3D NumPy array with RSSI values
            - router_cols: List of router column names
            - room_map: Dictionary mapping room numbers to (row, col) positions
    """
    if csv_file_path is None:
        csv_file_path = '/home/who/Téléchargements/RSSI/RSSI.csv'
    
    # Read the data
    df, router_cols = read_rssi_data(csv_file_path)
    
    # Create room mapping
    room_map = create_room_to_position_mapping()
    
    # Create the matrix
    rssi_matrix = create_rssi_matrix(df, router_cols, room_map)
    
    return rssi_matrix, router_cols, room_map

# Function that combines all steps: create matrix and generate heatmaps
def analyze_rssi_data(csv_file_path=None, output_dir=None, show_plots=True):
    """
    Complete analysis function: create matrix, generate visualizations
    
    Args:
        csv_file_path: Path to the CSV file
        output_dir: Directory to save output files (None = don't save)
        show_plots: Whether to display plots
    
    Returns:
        tuple: (matrix, router_cols, room_map, router_fig, best_router_fig)
    """
    # Get the matrix
    matrix, router_cols, room_map = get_rssi_matrix(csv_file_path)
    
    # Create router heatmaps
    router_fig = None
    best_router_fig = None
    
    # Create router heatmaps
    if output_dir:
        save_router_path = f"{output_dir}/router_heatmaps.png"
        save_best_router_path = f"{output_dir}/best_router_heatmap.png"
    else:
        save_router_path = None
        save_best_router_path = None
    
    router_fig = create_router_heatmaps(
        matrix, router_cols, room_map, 
        show=show_plots,
        save_path=save_router_path
    )
    
    best_router_fig = create_best_router_heatmap(
        matrix, router_cols, room_map, 
        show=show_plots,
        save_path=save_best_router_path
    )
    
    # Print some stats
    print("\nMatrix Statistics:")
    print(f"Shape: {matrix.shape}")
    print(f"Number of non-NaN values: {np.count_nonzero(~np.isnan(matrix))}")
    print(f"Percentage filled: {np.count_nonzero(~np.isnan(matrix)) / matrix.size * 100:.2f}%")
    
    return matrix, router_cols, room_map, router_fig, best_router_fig

# Example of direct array usage
def demonstrate_array_usage(matrix, router_cols, room_map):
    """Show examples of working with the RSSI data as NumPy arrays"""
    
    print("\n===== Array Usage Examples =====")
    
    # Example 1: Find average signal strength for each router
    print("\n1. Average signal strength by router:")
    for i, router in enumerate(router_cols):
        avg_signal = np.nanmean(matrix[:,:,i])
        print(f"  Router {router}: {avg_signal:.2f} dBm")
    
    # Example 2: Find the strongest signal for each room
    print("\n2. Strongest signal by room:")
    for room, (row, col) in list(room_map.items())[:5]:  # Show first 5 rooms only
        room_signals = matrix[row, col, :]
        if not np.all(np.isnan(room_signals)):
            max_idx = np.nanargmax(room_signals)
            max_signal = np.nanmax(room_signals)
            print(f"  Room {room}: Best router is {router_cols[max_idx]} with signal {max_signal:.1f} dBm")
    print("  ...and more rooms")
    
    # Example 3: Create a matrix mask for good signal strength
    print("\n3. Good signal coverage matrix (signals stronger than -65 dBm):")
    good_signal_threshold = -65
    good_signal_mask = matrix > good_signal_threshold
    
    # Count good signals per router
    print("  Good signals per router:")
    for i, router in enumerate(router_cols):
        good_count = np.sum(good_signal_mask[:,:,i])
        print(f"  Router {router}: {good_count} locations with good signal")
    
    # Example 4: Find rooms with no coverage
    print("\n4. Rooms with poor or no coverage:")
    for room, (row, col) in room_map.items():
        room_signals = matrix[row, col, :]
        if np.all(np.isnan(room_signals)):
            print(f"  Room {room}: No coverage")
            continue
        
        if not np.any(room_signals > -75):  # -75 dBm threshold for poor signal
            print(f"  Room {room}: Poor coverage (all signals below -75 dBm)")

# Main function
def main():
    input_file = '/home/who/Téléchargements/RSSI/RSSI.csv'
    output_dir = '/home/who/Téléchargements/RSSI'
    
    print("Analyzing RSSI data...")
    matrix, router_cols, room_map, _, _ = analyze_rssi_data(
        csv_file_path=input_file,
        output_dir=output_dir,
        show_plots=True
    )
    
    # Show examples of working with the array directly
    demonstrate_array_usage(matrix, router_cols, room_map)
    
    # Return the matrix for interactive use
    print("\nMatrix is available for interactive use")
    return matrix, router_cols, room_map

if __name__ == '__main__':
    rssi_matrix, routers, room_mapping = main()
