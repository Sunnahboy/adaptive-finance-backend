


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Optional

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

def plot_comprehensive_training_analysis(
    arm_history: List[int],
    reward_history: List[float],
    arm_selection_history: Dict[int, List[int]],
    arm_reward_history: Dict[int, List[float]],
    output_dir: Union[str, Path, None] = None
):
    """
    Creates 6 comprehensive visualizations for CMAB training analysis.
    Saves the output as 'cmab_comprehensive_analysis.png'.
    """
    if output_dir is None:
        output_dir = Path(".").resolve()
    elif isinstance(output_dir, str):
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not arm_history:
        print(" No training history to visualize.")
        return

    # Create 2x3 subplot grid
    fig = plt.figure(figsize=(18, 12))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # ========== PLOT 1: Arm Selection Distribution ==========
    ax1 = plt.subplot(2, 3, 1)
    arms = list(arm_selection_history.keys())
    # Total times each arm was pulled
    selections = [len(arm_selection_history[arm]) for arm in arms]
    
    bar_colors = colors[:len(arms)] if len(arms) <= len(colors) else None
    
    bars = ax1.bar([f'Arm {i}' for i in arms], selections, color=bar_colors)
    ax1.set_title('Action Selection Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Times Selected', fontsize=11)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    # ========== PLOT 2: Per-Arm Success Rate (FIXED) ==========
    ax2 = plt.subplot(2, 3, 2)
    success_rates = []
    
    for i, arm in enumerate(arms):
        n_selected = selections[i] # Total pulls
        
        # In train.py, we only appended to arm_reward_history if reward > 0.
        # So sum(rewards) is the total number of successes.
        # BUT we must divide by TOTAL SELECTIONS (n_selected), not len(rewards).
        
        rewards = arm_reward_history.get(arm, [])
        n_success = len(rewards) # Assuming rewards are binary 1.0
        # If rewards are float (non-binary), use sum(rewards)
        # For safety, let's use sum(rewards)
        total_reward_val = sum(rewards)

        if n_selected > 0:
            success_rate = total_reward_val / n_selected
        else:
            success_rate = 0.0
        success_rates.append(success_rate)
    
    bars = ax2.bar([f'Arm {i}' for i in arms], success_rates, color=bar_colors)
    ax2.set_title('Per-Arm Success Rate', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1.05])
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% Baseline')
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom', fontsize=10)
    
    # ========== PLOT 3: Learning Curve ==========
    ax3 = plt.subplot(2, 3, 3)
    window_size = max(50, len(reward_history) // 50)
    moving_avg = pd.Series(reward_history).rolling(window=window_size).mean()
    
    ax3.plot(reward_history, alpha=0.15, color='blue', label='Raw Reward')
    ax3.plot(moving_avg, color='green', linewidth=2, label=f'Avg (n={window_size})')
    ax3.set_title('Learning Curve', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Reward')
    ax3.legend(loc='lower right')
    
    # ========== PLOT 4: Cumulative Reward ==========
    ax4 = plt.subplot(2, 3, 4)
    cumulative_reward = np.cumsum(reward_history)
    ax4.plot(cumulative_reward, color='purple', linewidth=2)
    ax4.fill_between(range(len(cumulative_reward)), cumulative_reward, alpha=0.1, color='purple')
    ax4.set_title('Cumulative Reward', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Iteration')
    
    # ========== PLOT 5: Selection Trend ==========
    ax5 = plt.subplot(2, 3, 5)
    window = max(100, len(arm_history) // 20)
    
    if len(arm_history) > window:
        series = pd.Series(arm_history)
        dummies = pd.get_dummies(series).reindex(columns=arms, fill_value=0)
        rolling = dummies.rolling(window=window).mean()
        
        for i, arm in enumerate(arms):
            if arm in rolling.columns:
                ax5.plot(rolling[arm], label=f'Arm {arm}', linewidth=2, color=colors[i%len(colors)])
        
        ax5.legend(loc='upper left', fontsize='small')
    
    ax5.set_title(f'Selection Trend (Rolling {window})', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Selection Probability')
    
    # ========== PLOT 6: Heatmap (FIXED) ==========
    ax6 = plt.subplot(2, 3, 6)
    n_bins = 10
    bin_size = len(reward_history) // n_bins
    
    if bin_size > 0:
        heatmap_data = np.zeros((len(arms), n_bins))
        
        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size
            
            bin_arms = arm_history[start:end]
            bin_rewards = reward_history[start:end]
            
            for idx, arm in enumerate(arms):
                # Count total pulls for this arm in this bin
                indices = [k for k, x in enumerate(bin_arms) if x == arm]
                n_pulls = len(indices)
                
                if n_pulls > 0:
                    # Sum rewards for these specific pulls
                    total_r = sum([bin_rewards[k] for k in indices])
                    heatmap_data[idx, i] = total_r / n_pulls
                else:
                    heatmap_data[idx, i] = 0.0
                    
        sns.heatmap(heatmap_data, cmap='RdYlGn', vmin=0, vmax=1, 
                    yticklabels=[f'Arm {i}' for i in arms], 
                    xticklabels=False, 
                    ax=ax6, cbar_kws={'label': 'Avg Reward'})
        
        ax6.set_title('Performance Heatmap (Time)', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Training Progression →')

    plt.tight_layout()
    output_path = output_dir / "cmab_comprehensive_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" Graphs saved to {output_path}")
    plt.close()