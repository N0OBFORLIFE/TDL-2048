import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import matplotlib.gridspec as gridspec
import os

class MetricsAnalyzer:
    def __init__(self, metrics_file_path: str, output_dir: str = "graphs"):
        self.metrics_path = Path(metrics_file_path)
        self.output_dir = Path(output_dir)
        self.metrics_data = self._load_metrics()
        self.df = self._create_dataframe()
        
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _load_metrics(self) -> Dict:
        with open(self.metrics_path, 'r') as f:
            return json.load(f)
            
    def _create_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.metrics_data['metrics_history'])
    
    def save_learning_progress(self, window_size: int = 50) -> None:
        plt.figure(figsize=(10, 6))
        rolling_score = self.df['score'].rolling(window=window_size).mean()
        plt.plot(self.df.index, self.df['score'], alpha=0.3, color='blue', label='Raw')
        plt.plot(self.df.index, rolling_score, color='red', label=f'{window_size}-game average')
        plt.title('Score Progression Over Time')
        plt.ylabel('Score')
        plt.xlabel('Game Number')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'score_progression.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        rolling_win_rate = self.df['2048_rate'].rolling(window=window_size).mean()
        plt.plot(self.df.index, self.df['2048_rate'], alpha=0.3, color='blue', label='Raw')
        plt.plot(self.df.index, rolling_win_rate, color='red', label=f'{window_size}-game average')
        plt.title('2048 Achievement Rate Over Time')
        plt.ylabel('Win Rate')
        plt.xlabel('Game Number')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'win_rate.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        rolling_efficiency = self.df['merge_efficiency'].rolling(window=window_size).mean()
        plt.plot(self.df.index, self.df['merge_efficiency'], alpha=0.3, color='blue', label='Raw')
        plt.plot(self.df.index, rolling_efficiency, color='red', label=f'{window_size}-game average')
        plt.title('Merge Efficiency Over Time')
        plt.xlabel('Game Number')
        plt.ylabel('Efficiency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'merge_efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_score_distribution(self) -> None:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x='score', bins=50)
        plt.title('Score Distribution')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'score_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        
        Q1 = self.df['score'].quantile(0.25)
        Q3 = self.df['score'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        sns.boxplot(data=self.df, x='score', whis=1.5)
        
        sns.stripplot(data=self.df, x='score', color='blue', 
                     alpha=0.3, size=4, jitter=0.2)
        
        outliers = self.df[
            (self.df['score'] < lower_bound) | 
            (self.df['score'] > upper_bound)
        ]['score']
        
        if not outliers.empty:
            plt.plot(outliers, [0] * len(outliers), 'o', 
                    color='red', alpha=0.5, markersize=4)
        
        plt.title('Score Distribution (Box Plot)')
        plt.xlabel('Score')
        plt.ylabel('')
        
        stats_text = f'Median: {self.df["score"].median():,.0f}\n'
        stats_text += f'Q1: {Q1:,.0f}\n'
        stats_text += f'Q3: {Q3:,.0f}\n'
        stats_text += f'IQR: {IQR:,.0f}'
        
        plt.text(0.95, 0.95, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'score_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_tile_achievement_rates(self) -> None:
        tile_columns = ['128_rate', '256_rate', '512_rate', '1024_rate', '2048_rate']
        
        plt.figure(figsize=(12, 6))
        for col in tile_columns:
            plt.plot(self.df.index, self.df[col].rolling(window=50).mean(), 
                    label=f'Tile {col.split("_")[0]}')
            
        plt.title('Tile Achievement Rates Over Time')
        plt.xlabel('Game Number')
        plt.ylabel('Achievement Rate')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tile_achievement_rates.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_correlation_analysis(self) -> None:
        cols = ['score', 'moves_per_game', 'merge_efficiency', '2048_rate']
        correlation_matrix = self.df[cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix of Key Metrics')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df, x='moves_per_game', y='score', alpha=0.5)
        plt.title('Score vs Moves per Game')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'score_vs_moves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df, x='merge_efficiency', y='score', alpha=0.5)
        plt.title('Score vs Merge Efficiency')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'score_vs_efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df, x='merge_efficiency', y='2048_rate', alpha=0.5)
        plt.title('Win Rate vs Merge Efficiency')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'winrate_vs_efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df, x='moves_per_game', y='2048_rate', alpha=0.5)
        plt.title('Win Rate vs Moves per Game')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'winrate_vs_moves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_performance_summary(self) -> Dict:
        latest_games = self.df.tail(100)
        
        summary = {
            'overall_stats': {
                'total_games': len(self.df),
                'average_score': self.df['score'].mean(),
                'max_score': self.df['score'].max(),
                'average_win_rate': self.df['2048_rate'].mean() * 100,
                'average_moves_per_game': self.df['moves_per_game'].mean()
            },
            'current_performance': {
                'last_100_average_score': latest_games['score'].mean(),
                'last_100_win_rate': latest_games['2048_rate'].mean() * 100,
                'last_100_merge_efficiency': latest_games['merge_efficiency'].mean()
            },
            'learning_progress': {
                'score_improvement': (latest_games['score'].mean() - 
                                   self.df['score'].head(100).mean()),
                'win_rate_improvement': ((latest_games['2048_rate'].mean() - 
                                      self.df['2048_rate'].head(100).mean()) * 100)
            }
        }
        
        return summary
    
    def save_performance_summary(self) -> None:
        summary = self.generate_performance_summary()
        
        with open(self.output_dir / 'performance_summary.txt', 'w') as f:
            f.write("=== 2048 AI Performance Summary ===\n\n")
            f.write("Overall Statistics:\n")
            f.write(f"Total Games Played: {summary['overall_stats']['total_games']:,}\n")
            f.write(f"Average Score: {summary['overall_stats']['average_score']:,.1f}\n")
            f.write(f"Maximum Score: {summary['overall_stats']['max_score']:,}\n")
            f.write(f"Overall Win Rate: {summary['overall_stats']['average_win_rate']:.1f}%\n")
            f.write(f"Average Moves per Game: {summary['overall_stats']['average_moves_per_game']:.1f}\n\n")
            
            f.write("Current Performance (Last 100 Games):\n")
            f.write(f"Average Score: {summary['current_performance']['last_100_average_score']:,.1f}\n")
            f.write(f"Win Rate: {summary['current_performance']['last_100_win_rate']:.1f}%\n")
            f.write(f"Merge Efficiency: {summary['current_performance']['last_100_merge_efficiency']:.2f}\n\n")
            
            f.write("Learning Progress:\n")
            f.write(f"Score Improvement: {summary['learning_progress']['score_improvement']:,.1f}\n")
            f.write(f"Win Rate Improvement: {summary['learning_progress']['win_rate_improvement']:.1f}%\n")

def analyze_metrics(metrics_file_path: str, output_dir: str = "graphs") -> None:
    analyzer = MetricsAnalyzer(metrics_file_path, output_dir)
    
    analyzer.save_performance_summary
    analyzer.save_learning_progress()
    analyzer.save_score_distribution()
    analyzer.save_tile_achievement_rates()
    analyzer.save_correlation_analysis()
    
    print(f"Analysis complete. All graphs and summary have been saved to: {output_dir}/")

if __name__ == "__main__":
    metrics_file = "metrics/ntuplenewrok_20250208_235335_302a5fe0_metrics.json"
    analyze_metrics(metrics_file, "analysis_graphs")