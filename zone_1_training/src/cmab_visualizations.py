import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Optional

# Set style
sns.set_style("whitegrid")

def plot_comprehensive_training_analysis(
    arm_history: List[int],
    reward_history: List[float],
    arm_selection_history: Dict[int, List[int]],
    arm_reward_history: Dict[int, List[float]],
    optimal_reward_history: Optional[List[float]] = None,
    uncertainty_history: Optional[List[float]] = None,
    context_history: Optional[List[Dict]] = None,
    output_dir: Union[str, Path, None] = None
):
    
    """
    Creates 9 comprehensive visualizations for CMAB training analysis using the Object-Oriented API.
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
    

    #  Object-Oriented API (Safe for Big Data & Web Servers)
    fig, axes = plt.subplots(3, 3, figsize=(22, 18))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    arms = list(arm_selection_history.keys())
    
    # ========== PLOT 1: Arm Selection Distribution ==========
    ax1 = axes[0, 0]
    selections = [len(arm_selection_history[arm]) for arm in arms]
    bar_colors = colors[:len(arms)] if len(arms) <= len(colors) else None
    bars = ax1.bar([f'Arm {i}' for i in arms], selections, color=bar_colors)
    ax1.set_title('Action Selection Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Times Selected')


    
    # ========== PLOT 2: Per-Arm Success Rate ==========
    ax2 = axes[0, 1]
    success_rates = [
        (sum(arm_reward_history.get(arm, [])) / selections[i]) if selections[i] > 0 else 0.0 
        for i, arm in enumerate(arms)
    ]
    bars = ax2.bar([f'Arm {i}' for i in arms], success_rates, color=bar_colors)
    ax2.set_title('Per-Arm Success Rate', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1.05])
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)



    
    # ========== PLOT 3: Learning Curve ==========
    ax3 = axes[0, 2]
    window_size = max(50, len(reward_history) // 50)
    moving_avg = pd.Series(reward_history).rolling(window=window_size).mean()
    ax3.plot(reward_history, alpha=0.15, color='blue', label='Raw Reward')
    ax3.plot(moving_avg, color='green', linewidth=2, label=f'Avg (n={window_size})')
    ax3.set_title('Learning Curve', fontsize=14, fontweight='bold')
    ax3.legend(loc='lower right')


    
    # ========== PLOT 4: Cumulative Reward ==========
    ax4 = axes[1, 0]
    cumulative_reward = np.cumsum(reward_history)
    ax4.plot(cumulative_reward, color='purple', linewidth=2)
    ax4.fill_between(range(len(cumulative_reward)), cumulative_reward, alpha=0.1, color='purple')
    ax4.set_title('Cumulative Reward (Business KPI)', fontsize=14, fontweight='bold')



    
    # ========== PLOT 5: Selection Trend ==========
    ax5 = axes[1, 1]
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


    
    # ========== PLOT 6: Heatmap Pandas Vectorized  ==========
    ax6 = axes[1, 2]
    if len(reward_history) > 0:
        df = pd.DataFrame({'arm': arm_history, 'reward': reward_history})
        #Cut data into 10 chronological bins instantly
        df['time_bin'] = pd.cut(df.index, bins=10, labels=False)
        # Vectorized groupby 
        heatmap_data = df.groupby(['arm', 'time_bin'])['reward'].mean().unstack(fill_value=0.0)
        
        #Ensure all arms are in the index
        for a in arms:
            if a not in heatmap_data.index:
                heatmap_data.loc[a] = 0.0
                
        heatmap_data = heatmap_data.sort_index()
        sns.heatmap(heatmap_data, cmap='RdYlGn', vmin=0, vmax=1, 
                    yticklabels=[f'Arm {i}' for i in heatmap_data.index], xticklabels=False, ax=ax6)
        ax6.set_title('Performance Heatmap (Time)', fontsize=14, fontweight='bold')



    # ========== PLOT 7: Cumulative Regret (The Holy Grail) ==========
    ax7 = axes[2, 0]
    if optimal_reward_history and len(optimal_reward_history) == len(reward_history):
        regret_history = [opt - act for opt, act in zip(optimal_reward_history, reward_history)]
        cumulative_regret = np.cumsum(regret_history)
        ax7.plot(cumulative_regret, color='red', linewidth=2)
        ax7.fill_between(range(len(cumulative_regret)), cumulative_regret, alpha=0.1, color='red')
        ax7.set_title('Cumulative Regret (Lower is Better)', fontsize=14, fontweight='bold')
    else:
        ax7.text(0.5, 0.5, "Optimal Rewards Not Tracked", ha='center', va='center', fontsize=12)
        ax7.set_title('Cumulative Regret', fontsize=14, fontweight='bold')



    # ========== PLOT 8: Uncertainty Shrinkage ==========
    ax8 = axes[2, 1]
    if uncertainty_history:
        unc_series = pd.Series(uncertainty_history)
        unc_avg = unc_series.rolling(window=window_size).mean()
        ax8.plot(unc_series, alpha=0.2, color='orange')
        ax8.plot(unc_avg, color='darkorange', linewidth=2)
        ax8.set_title('LinUCB Confidence Bound Shrinkage', fontsize=14, fontweight='bold')
        ax8.set_ylabel('Uncertainty Score')
    else:
        ax8.text(0.5, 0.5, "Uncertainty Data Not Tracked", ha='center', va='center', fontsize=12)
        ax8.set_title('Uncertainty Shrinkage', fontsize=14, fontweight='bold')



    # ========== PLOT 9: Explainable AI (Context vs Action) ==========
    ax9 = axes[2, 2]
    if context_history and len(context_history) == len(arm_history):
        #SUBSAMPLING TO PREVENT BLOB (Max 2000 points)
        max_points = 2000
        step = max(1, len(context_history) // max_points)
        
        sample_vols = [ctx.get('spending_volatility', 0) for ctx in context_history[::step]]
        sample_rets = [ctx.get('return_rate', 0) for ctx in context_history[::step]]
        sample_arms = arm_history[::step]
        
        scatter = ax9.scatter(sample_vols, sample_rets, c=sample_arms, cmap='tab10', alpha=0.6, edgecolors='w', s=50)
        ax9.set_title('Context Mapping (XAI - Subsampled)', fontsize=14, fontweight='bold')
        ax9.set_xlabel('Spending Volatility')
        ax9.set_ylabel('Return Rate')


        
        handles, _ = scatter.legend_elements()
        if handles:
            labels = [f"Arm {i}" for i in sorted(list(set(sample_arms)))]
            ax9.legend(handles, labels, loc="upper right", title="Actions")


    else:
        ax9.text(0.5, 0.5, "Context Data Not Tracked", ha='center', va='center', fontsize=12)
        ax9.set_title('Context Mapping (XAI)', fontsize=14, fontweight='bold')

        

    plt.tight_layout()
    output_path = output_dir / "cmab_comprehensive_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" 9-Panel Dashboard saved to {output_path}")
    plt.close()