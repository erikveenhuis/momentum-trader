import time
from typing import Any, Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")  # Use TkAgg backend which is better for real-time updates


class TradingVisualizer:
    def __init__(self, market_data, window_size: int = 400):
        """Initialize the trading visualizer.

        Args:
            market_data: Market data object containing price information
            window_size: Number of steps to show in the price chart
        """
        self.market_data = market_data
        self.window_size = window_size
        self.start_time = time.time()
        self.total_steps = 1440  # Total episode steps

        # Set dark theme
        plt.style.use("dark_background")

        # Create figure with subplots
        plt.ion()  # Enable interactive mode
        self.fig = plt.figure(figsize=(15, 12), facecolor="#1a1a1a")

        # Create subplot grid but leave space for info panel
        gs = self.fig.add_gridspec(6, 1, height_ratios=[2, 1, 1, 1, 1, 1], hspace=0.3, top=0.95, bottom=0.08)
        self.ax1 = self.fig.add_subplot(gs[0])  # Price
        self.ax2 = self.fig.add_subplot(gs[1])  # Position
        self.ax3 = self.fig.add_subplot(gs[2])  # Portfolio Value
        self.ax4 = self.fig.add_subplot(gs[3])  # Returns
        self.ax5 = self.fig.add_subplot(gs[4])  # Balance
        self.ax6 = self.fig.add_subplot(gs[5])  # Transaction Cost

        # Set dark background for all axes
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]:
            ax.set_facecolor("#1a1a1a")
            ax.grid(True, color="#333333", linestyle="--", alpha=0.3)
            ax.tick_params(colors="#cccccc")
            ax.spines["bottom"].set_color("#333333")
            ax.spines["top"].set_color("#333333")
            ax.spines["left"].set_color("#333333")
            ax.spines["right"].set_color("#333333")

        self.fig.suptitle("Trading Environment Visualization", fontsize=14, color="white")

        # Initialize price plot
        (self.price_line,) = self.ax1.plot([], [], "#00ffff", label="Price", linewidth=1)  # Cyan
        self.ax1.set_ylabel("Price ($)", fontsize=10, color="#00ffff")
        self.ax1.tick_params(axis="y", labelcolor="#00ffff")
        self.ax1.legend(loc="upper left", facecolor="#1a1a1a", edgecolor="#333333", labelcolor="white")

        # Add buy/sell markers to price plot
        self.buy_scatter = self.ax1.scatter([], [], color="#00ff00", marker="^", s=50, label="Buy")  # Green
        self.sell_scatter = self.ax1.scatter([], [], color="#ff0000", marker="v", s=50, label="Sell")  # Red

        # Initialize position plot
        (self.position_line,) = self.ax2.plot([], [], "#00ff00", label="Position", linewidth=1)  # Green
        self.ax2.set_ylabel("Position (BTC)", fontsize=10, color="#00ff00")
        self.ax2.tick_params(axis="y", labelcolor="#00ff00")
        self.ax2.legend(loc="upper left", facecolor="#1a1a1a", edgecolor="#333333", labelcolor="white")

        # Initialize portfolio value plot
        (self.portfolio_line,) = self.ax3.plot([], [], "#ff0000", label="Portfolio Value", linewidth=2)  # Red
        self.ax3.set_ylabel("Portfolio Value ($)", fontsize=10, color="#ff0000")
        self.ax3.legend(loc="upper left", facecolor="#1a1a1a", edgecolor="#333333", labelcolor="white")

        # Initialize returns plot
        (self.returns_line,) = self.ax4.plot([], [], "#ff00ff", label="Returns %", linewidth=1)  # Magenta
        self.ax4.set_ylabel("Returns (%)", fontsize=10, color="#ff00ff")
        self.ax4.legend(loc="upper left", facecolor="#1a1a1a", edgecolor="#333333", labelcolor="white")

        # Initialize balance plot
        (self.balance_line,) = self.ax5.plot([], [], "#ffaa00", label="Balance", linewidth=1)  # Orange
        self.ax5.set_ylabel("Balance ($)", fontsize=10, color="#ffaa00")
        self.ax5.legend(loc="upper left", facecolor="#1a1a1a", edgecolor="#333333", labelcolor="white")

        # Initialize transaction cost plot
        (self.transaction_cost_line,) = self.ax6.plot([], [], "#aaaaff", label="Transaction Cost", linewidth=1)  # Light blue
        self.ax6.set_ylabel("Transaction Cost ($)", fontsize=10, color="#aaaaff")
        self.ax6.legend(loc="upper left", facecolor="#1a1a1a", edgecolor="#333333", labelcolor="white")

        # Store historical data
        self.steps: List[int] = []
        self.prices: List[float] = []
        self.positions: List[float] = []
        self.portfolio_values: List[float] = []
        self.returns: List[float] = []
        self.balances: List[float] = []
        self.transaction_costs: List[float] = []
        self.buy_points: List[tuple] = []  # (step, price)
        self.sell_points: List[tuple] = []  # (step, price)

        # Info text - create a separate axis for it
        self.info_ax = self.fig.add_axes([0.02, 0.95, 0.35, 0.04], facecolor="#1a1a1a", alpha=0.8)
        self.info_ax.axis("off")  # Hide axis
        self.info_text = self.info_ax.text(0.05, 0.5, "", fontsize=9, family="monospace", color="white", va="center", ha="left")

        # Show the plot
        plt.show(block=False)
        plt.pause(0.1)  # Small pause to ensure window is shown

    def update(self, info: Dict[str, Any]) -> None:
        """Update the visualization with new data.

        Args:
            info: Dictionary containing current step information
        """
        step = info["step"]
        price = info["price"]
        position = info["position"]
        portfolio_value = info["portfolio_value"]
        balance = info.get("balance", 0)
        transaction_cost = info.get("transaction_cost", 0)

        # Calculate step length (for future use)
        # current_time = time.time()
        # elapsed_time = current_time - self.start_time
        # step_length = elapsed_time / (step + 1) if step > 0 else 0

        # Update historical data
        self.steps.append(step)
        self.prices.append(price)
        self.positions.append(position)
        self.portfolio_values.append(portfolio_value)
        self.balances.append(balance)
        self.transaction_costs.append(transaction_cost)

        # Calculate returns
        if len(self.portfolio_values) > 1:
            ret = ((portfolio_value / self.portfolio_values[0]) - 1) * 100
            self.returns.append(ret)
        else:
            self.returns.append(0)

        # Track trades
        if len(self.positions) > 1:
            prev_pos = self.positions[-2]
            if position > prev_pos:  # Buy
                self.buy_points.append((step, price))
            elif position < prev_pos:  # Sell
                self.sell_points.append((step, price))

        # Keep only window_size steps
        if len(self.steps) > self.window_size:
            self.steps = self.steps[-self.window_size :]
            self.prices = self.prices[-self.window_size :]
            self.positions = self.positions[-self.window_size :]
            self.portfolio_values = self.portfolio_values[-self.window_size :]
            self.returns = self.returns[-self.window_size :]
            self.balances = self.balances[-self.window_size :]
            self.transaction_costs = self.transaction_costs[-self.window_size :]

            # Filter trade points
            self.buy_points = [(s, p) for s, p in self.buy_points if s >= self.steps[0]]
            self.sell_points = [(s, p) for s, p in self.sell_points if s >= self.steps[0]]

        # Update plots
        self.price_line.set_data(self.steps, self.prices)
        self.position_line.set_data(self.steps, self.positions)
        self.portfolio_line.set_data(self.steps, self.portfolio_values)
        self.returns_line.set_data(self.steps, self.returns)
        self.balance_line.set_data(self.steps, self.balances)
        self.transaction_cost_line.set_data(self.steps, self.transaction_costs)

        # Update trade markers
        if self.buy_points:
            buy_steps, buy_prices = zip(*self.buy_points)
            self.buy_scatter.set_offsets(np.c_[buy_steps, buy_prices])
        if self.sell_points:
            sell_steps, sell_prices = zip(*self.sell_points)
            self.sell_scatter.set_offsets(np.c_[sell_steps, sell_prices])

        # Create a formatted information panel
        progress = min(1.0, step / self.total_steps)
        bar_width = 20  # Width of the progress bar in characters
        filled_width = int(bar_width * progress)
        bar = "█" * filled_width + "░" * (bar_width - filled_width)
        progress_percent = progress * 100

        # Create a semi-transparent background for the info panel
        self.info_ax.clear()
        self.info_ax.set_facecolor("#1a1a1a")
        self.info_ax.patch.set_alpha(0.7)
        self.info_ax.axis("off")

        # Format the text to only show progress
        info_str = f"Progress: [{bar}] {step}/{self.total_steps} ({progress_percent:.1f}%)"

        # Position text in the info box with proper positioning
        self.info_text = self.info_ax.text(
            0.02,
            0.5,
            info_str,
            fontsize=9,
            family="monospace",
            color="white",
            va="center",
            ha="left",
            transform=self.info_ax.transAxes,
            bbox=dict(facecolor="#1a1a1a", alpha=0.8, edgecolor="#333333", boxstyle="round,pad=0.3"),
        )

        # Update axes limits
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]:
            ax.relim()
            ax.autoscale_view()

        # Redraw with minimal pause
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.000001)  # Minimal pause to allow UI to update

    def close(self) -> None:
        """Close the visualization."""
        plt.close(self.fig)
        plt.ioff()  # Disable interactive mode
