"""
Example integration of matplotlib hooks with your plotting module.

This demonstrates how to use the matplotlib hooks in your existing plotting workflow.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib_hooks import install_all_hooks, get_captured_data, clear_captured_data, CAPTURED_DATA

# Install hooks at module level - this affects all subsequent matplotlib calls
install_all_hooks()

def create_sample_plots():
    """Create some sample plots to demonstrate data capture."""
    # Create some sample data
    x = np.linspace(0, 10, 50)
    y1 = np.sin(x)
    y2 = np.cos(x)

    # Create plots that will have their data captured
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # These calls will be captured by the hooks
    ax1.plot(x, y1, 'b-', label='sin(x)', linewidth=2)
    ax1.scatter([2, 4, 6], [0.5, -0.5, 1.0], color='red', s=50)
    ax1.set_title('Plot 1')

    ax2.plot(x, y2, 'g--', label='cos(x)', linewidth=2)
    ax2.fill_between(x, y2, 0, alpha=0.3)
    ax2.set_title('Plot 2')

    plt.tight_layout()
    return fig

def analyze_captured_data():
    """Analyze the data captured by the hooks."""
    print("=== Analysis of Captured Data ===")

    for func_name, calls in CAPTURED_DATA.items():
        if not calls:
            continue

        print(f"\n{func_name}: {len(calls)} calls")

        for i, call in enumerate(calls):
            print(f"  Call {i+1}:")
            print(f"    Args: {len(call['args'])} arguments")

            # Extract useful information based on the function
            if func_name in ['plt.plot', 'ax.plot']:
                # For ax.plot, first arg is the axes object, so data starts at index 1
                data_start = 1 if func_name.startswith('ax.') else 0
                if len(call['args']) > data_start + 1:
                    x_data = call['args'][data_start]
                    y_data = call['args'][data_start + 1]
                    if hasattr(x_data, '__len__') and hasattr(y_data, '__len__'):
                        print(f"    Data shape: x={len(x_data)}, y={len(y_data)}")
                        print(f"    X range: [{np.min(x_data):.2f}, {np.max(x_data):.2f}]")
                        print(f"    Y range: [{np.min(y_data):.2f}, {np.max(y_data):.2f}]")

            elif func_name in ['plt.scatter', 'ax.scatter']:
                # For ax.scatter, first arg is the axes object, so data starts at index 1
                data_start = 1 if func_name.startswith('ax.') else 0
                if len(call['args']) > data_start + 1:
                    x_data = call['args'][data_start]
                    y_data = call['args'][data_start + 1]
                    if hasattr(x_data, '__len__') and hasattr(y_data, '__len__'):
                        print(f"    Scatter points: {len(x_data)}")
                        print(f"    X points: {x_data}")
                        print(f"    Y points: {y_data}")

            # Show kwargs
            if call['kwargs']:
                print(f"    Kwargs: {call['kwargs']}")

def extract_all_plot_data():
    """Extract all plot data into a structured format."""
    extracted_data = {}

    for func_name, calls in CAPTURED_DATA.items():
        if not calls:
            continue

        func_data = []
        for call in calls:
            call_info = {
                'function': func_name,
                'args': call['args'],
                'kwargs': call['kwargs'],
                'call_number': call['call_count']
            }

            # Add specific data extraction for common plot types
            if func_name in ['plt.plot', 'ax.plot'] and len(call['args']) >= 2:
                call_info['x_data'] = call['args'][0]
                call_info['y_data'] = call['args'][1]
                call_info['plot_type'] = 'line'

            elif func_name in ['plt.scatter', 'ax.scatter'] and len(call['args']) >= 2:
                call_info['x_data'] = call['args'][0]
                call_info['y_data'] = call['args'][1]
                call_info['plot_type'] = 'scatter'

            elif func_name in ['plt.fill_between', 'ax.fill_between'] and len(call['args']) >= 3:
                call_info['x_data'] = call['args'][0]
                call_info['y1_data'] = call['args'][1]
                call_info['y2_data'] = call['args'][2]
                call_info['plot_type'] = 'fill'

            func_data.append(call_info)

        extracted_data[func_name] = func_data

    return extracted_data

if __name__ == "__main__":
    print("Creating plots with hooks enabled...")

    # Clear any previous data
    clear_captured_data()

    # Create plots
    fig = create_sample_plots()

    # Analyze captured data
    analyze_captured_data()

    # Extract structured data
    structured_data = extract_all_plot_data()
    print(f"\n=== Extracted Data Keys ===")
    for key in structured_data:
        print(f"- {key}: {len(structured_data[key])} calls")

    plt.show()