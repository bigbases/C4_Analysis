import os
import pandas as pd
import json
import numpy as np
import math
import random
import glob
from datetime import datetime
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class BalancedSamplingGenerator:
    def __init__(self):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.extreme_datasets_dir = os.path.join(self.current_dir, './data/extreme_datasets')
        self.analysis_results_dir = os.path.join(self.current_dir, 'analysis_results')
        self.balanced_datasets_dir = os.path.join(self.current_dir, './data/balanced_datasets')
        
        # Create directories if they don't exist
        os.makedirs(self.analysis_results_dir, exist_ok=True)
        os.makedirs(self.balanced_datasets_dir, exist_ok=True)
        
        # Model-persona combinations (for reference)
        self.models = ['claude-sonnet-4-20250514', 'gpt-4.1']
        self.personas = ['opp_left', 'opp_right', 'sup_left', 'sup_right']
        self.model_persona_combinations = []
        
        for model in self.models:
            for persona in self.personas:
                self.model_persona_combinations.append(f"{model}_{persona}")
        
        print("Balanced Sampling Generator (Extreme Left 50% + Extreme Right 50%)")
        print(f"Extreme datasets directory: {self.extreme_datasets_dir}")
        print(f"Model-persona combinations: {len(self.model_persona_combinations)}")

    def find_latest_extreme_datasets(self):
        """Find the latest extreme datasets (left and right)"""
        print("\n=== Finding latest extreme datasets ===")
        
        if not os.path.exists(self.extreme_datasets_dir):
            print(f"  ❌ Extreme datasets directory not found: {self.extreme_datasets_dir}")
            return None, None
        
        # Find extreme_left_extreme_*.csv files
        left_pattern = os.path.join(self.extreme_datasets_dir, "extreme_left_extreme_*.csv")
        left_files = glob.glob(left_pattern)
        
        # Find extreme_right_extreme_*.csv files  
        right_pattern = os.path.join(self.extreme_datasets_dir, "extreme_right_extreme_*.csv")
        right_files = glob.glob(right_pattern)
        
        print(f"  Found {len(left_files)} left extreme files")
        print(f"  Found {len(right_files)} right extreme files")
        
        if not left_files or not right_files:
            print(f"  ❌ Missing extreme datasets. Please run extreme_sampling.py first.")
            return None, None
        
        # Get the latest files (by modification time)
        latest_left = max(left_files, key=os.path.getmtime)
        latest_right = max(right_files, key=os.path.getmtime)
        
        print(f"  ✓ Latest left extreme: {os.path.basename(latest_left)}")
        print(f"  ✓ Latest right extreme: {os.path.basename(latest_right)}")
        
        return latest_left, latest_right

    def load_extreme_dataset(self, filepath, dataset_type):
        """Load extreme dataset from CSV file"""
        try:
            df = pd.read_csv(filepath)
            print(f"  ✓ Loaded {dataset_type}: {os.path.basename(filepath)} (shape: {df.shape})")
            
            # Extract samples with metadata
            samples = []
            for idx, row in df.iterrows():
                sample_info = {
                    'dataset_type': dataset_type,
                    'original_index': idx,
                    'political_score': row.get('achieved_political_score', None),
                    'stance_score': row.get('achieved_stance_score', None),
                    'source_topic': row.get('source_topic', 'Unknown'),
                    'source_topic_display': row.get('source_topic_display', 'Unknown'),
                    'bias_category': row.get('bias_category', dataset_type.replace('_extreme', '')),
                    'row_data': row.to_dict()
                }
                samples.append(sample_info)
            
            return samples, df
            
        except Exception as e:
            print(f"  ❌ Error loading {dataset_type}: {e}")
            return None, None

    def load_extreme_datasets(self):
        """Load both extreme left and right datasets"""
        print("\n=== Loading extreme datasets ===")
        
        # Find latest extreme datasets
        left_file, right_file = self.find_latest_extreme_datasets()
        
        if not left_file or not right_file:
            return None, None, None, None
        
        # Load datasets
        left_samples, left_df = self.load_extreme_dataset(left_file, 'left_extreme')
        right_samples, right_df = self.load_extreme_dataset(right_file, 'right_extreme')
        
        if not left_samples or not right_samples:
            print("  ❌ Failed to load extreme datasets")
            return None, None, None, None
        
        print(f"\n=== Loading completed ===")
        print(f"Left extreme samples: {len(left_samples)}")
        print(f"Right extreme samples: {len(right_samples)}")
        
        # Print score statistics
        if left_samples:
            left_scores = [s['political_score'] for s in left_samples if s['political_score'] is not None]
            if left_scores:
                print(f"Left scores - Mean: {np.mean(left_scores):.3f}, Range: {min(left_scores):.3f} ~ {max(left_scores):.3f}")
        
        if right_samples:
            right_scores = [s['political_score'] for s in right_samples if s['political_score'] is not None]
            if right_scores:
                print(f"Right scores - Mean: {np.mean(right_scores):.3f}, Range: {min(right_scores):.3f} ~ {max(right_scores):.3f}")
        
        return left_samples, right_samples, left_df, right_df

    def determine_balanced_sample_size(self, left_samples, right_samples):
        """Determine sample size for balanced sampling (50% left extreme, 50% right extreme)"""
        print("\n📏 Determining balanced sample size...")
        
        print(f"  Left extreme samples available: {len(left_samples)}")
        print(f"  Right extreme samples available: {len(right_samples)}")
        
        # Take half from each dataset
        left_half = len(left_samples) // 2
        right_half = len(right_samples) // 2
        balanced_sample_size = left_half + right_half
        
        print(f"  Left extreme half: {left_half}")
        print(f"  Right extreme half: {right_half}")
        print(f"  Total balanced sample size: {balanced_sample_size}")
        
        return balanced_sample_size, left_half, right_half

    def perform_balanced_sampling(self, left_samples, right_samples, left_half_size, right_half_size):
        """Perform balanced sampling: half from left extreme + half from right extreme"""
        print(f"\n🎯 Performing balanced sampling from extreme datasets")
        print(f"  Selecting {left_half_size} from left extreme, {right_half_size} from right extreme")
        
        # Random sampling from each extreme category (half from each)
        selected_left = random.sample(left_samples, left_half_size)
        selected_right = random.sample(right_samples, right_half_size)
        
        # Combine and shuffle
        balanced_samples = selected_left + selected_right
        random.shuffle(balanced_samples)
        
        # Calculate statistics
        total_count = len(balanced_samples)
        left_count = len(selected_left)
        right_count = len(selected_right)
        
        # Calculate political scores
        all_political_scores = [s['political_score'] for s in balanced_samples if s['political_score'] is not None]
        left_political_scores = [s['political_score'] for s in selected_left if s['political_score'] is not None]
        right_political_scores = [s['political_score'] for s in selected_right if s['political_score'] is not None]
        
        achieved_stats = {
            'total_samples': total_count,
            'left_extreme_count': left_count,
            'right_extreme_count': right_count,
            'left_extreme_ratio': left_count / total_count,
            'right_extreme_ratio': right_count / total_count,
            'overall_mean_political_score': np.mean(all_political_scores) if all_political_scores else 0,
            'overall_std_political_score': np.std(all_political_scores, ddof=1) if len(all_political_scores) > 1 else 0,
            'left_mean_political_score': np.mean(left_political_scores) if left_political_scores else 0,
            'right_mean_political_score': np.mean(right_political_scores) if right_political_scores else 0,
            'min_political_score': min(all_political_scores) if all_political_scores else 0,
            'max_political_score': max(all_political_scores) if all_political_scores else 0
        }
        
        print(f"  ✅ Balanced sampling completed:")
        print(f"    Total samples: {total_count}")
        print(f"    Left extreme: {left_count} ({left_count/total_count*100:.1f}%)")
        print(f"    Right extreme: {right_count} ({right_count/total_count*100:.1f}%)")
        print(f"    Overall mean score: {achieved_stats['overall_mean_political_score']:.3f}")
        print(f"    Left extreme mean: {achieved_stats['left_mean_political_score']:.3f}")
        print(f"    Right extreme mean: {achieved_stats['right_mean_political_score']:.3f}")
        
        return balanced_samples, achieved_stats

    def save_balanced_dataset(self, balanced_samples, achieved_stats):
        """Save the balanced extreme dataset to CSV file"""
        if not balanced_samples:
            return None
        
        # Create DataFrame from balanced samples
        rows = []
        for sample in balanced_samples:
            row_data = sample['row_data'].copy()
            # Add metadata
            row_data['sampling_type'] = 'balanced_extreme_50_50'
            row_data['extreme_dataset_type'] = sample['dataset_type']
            row_data['source_topic'] = sample['source_topic']
            row_data['source_topic_display'] = sample['source_topic_display']
            row_data['achieved_political_score'] = sample['political_score']
            row_data['achieved_stance_score'] = sample['stance_score']
            row_data['bias_category'] = sample['bias_category']
            
            rows.append(row_data)
        
        balanced_df = pd.DataFrame(rows)
        
        # Add statistics as columns
        for key, value in achieved_stats.items():
            if isinstance(value, (int, float, str)):
                balanced_df[f'balanced_{key}'] = value
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"balanced_extreme_sampling_50_50_{timestamp}.csv"
        filepath = os.path.join(self.balanced_datasets_dir, filename)
        balanced_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        print(f"  💾 Balanced extreme dataset saved: {filepath}")
        print(f"  📊 Saved columns: {len(balanced_df.columns)}")
        
        return filepath

    def save_statistics(self, balanced_samples, achieved_stats):
        """Save sampling statistics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save balanced sampling results
        results_data = {
            'sampling_type': 'balanced_extreme_50_50',
            'timestamp': timestamp,
            **achieved_stats
        }
        
        # Add topic distribution
        topic_distribution = {}
        extreme_type_distribution = {}
        for sample in balanced_samples:
            topic = sample['source_topic']
            extreme_type = sample['dataset_type']
            topic_distribution[topic] = topic_distribution.get(topic, 0) + 1
            extreme_type_distribution[extreme_type] = extreme_type_distribution.get(extreme_type, 0) + 1
        
        results_data['topic_distribution'] = str(topic_distribution)
        results_data['extreme_type_distribution'] = str(extreme_type_distribution)
        
        results_df = pd.DataFrame([results_data])
        results_path = os.path.join(self.analysis_results_dir, 
                                   f'balanced_extreme_sampling_results_{timestamp}.csv')
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        
        print(f"\n📊 Balanced extreme sampling results saved:")
        print(f"  Detailed results: {results_path}")
        
        return results_path

    def print_report(self, balanced_samples, achieved_stats):
        """Print comprehensive balanced extreme sampling report"""
        print(f"\n{'='*80}")
        print("Balanced Extreme Sampling Results Report (Left Extreme 50% + Right Extreme 50%)")
        print(f"{'='*80}")
        
        print(f"\n1. Sampling Results:")
        print(f"{'-'*50}")
        print(f"  Total samples: {achieved_stats['total_samples']}")
        print(f"  Left extreme samples: {achieved_stats['left_extreme_count']} ({achieved_stats['left_extreme_ratio']*100:.1f}%)")
        print(f"  Right extreme samples: {achieved_stats['right_extreme_count']} ({achieved_stats['right_extreme_ratio']*100:.1f}%)")
        print(f"  Overall mean political score: {achieved_stats['overall_mean_political_score']:.3f}")
        print(f"  Overall standard deviation: {achieved_stats['overall_std_political_score']:.3f}")
        print(f"  Score range: {achieved_stats['min_political_score']:.3f} ~ {achieved_stats['max_political_score']:.3f}")
        
        print(f"\n2. Extreme Categories Breakdown:")
        print(f"{'-'*50}")
        print(f"  Left extreme mean score: {achieved_stats['left_mean_political_score']:.3f}")
        print(f"  Right extreme mean score: {achieved_stats['right_mean_political_score']:.3f}")
        
        print(f"\n3. Topic Distribution:")
        print(f"{'-'*50}")
        topic_distribution = {}
        for sample in balanced_samples:
            topic = sample['source_topic_display']
            topic_distribution[topic] = topic_distribution.get(topic, 0) + 1
        
        for topic, count in sorted(topic_distribution.items()):
            percentage = count / len(balanced_samples) * 100
            print(f"  {topic}: {count} ({percentage:.1f}%)")
        
        print(f"\n4. Extreme Dataset Type Distribution:")
        print(f"{'-'*50}")
        extreme_type_distribution = {}
        for sample in balanced_samples:
            extreme_type = sample['dataset_type']
            extreme_type_distribution[extreme_type] = extreme_type_distribution.get(extreme_type, 0) + 1
        
        for extreme_type, count in sorted(extreme_type_distribution.items()):
            percentage = count / len(balanced_samples) * 100
            print(f"  {extreme_type}: {count} ({percentage:.1f}%)")

    def run_analysis(self):
        """Run the complete balanced extreme sampling analysis"""
        print(f"Starting balanced extreme sampling analysis (Left Extreme 50% + Right Extreme 50%)...")
        
        # Step 1: Load extreme datasets
        left_samples, right_samples, left_df, right_df = self.load_extreme_datasets()
        
        if not left_samples or not right_samples:
            print("❌ Failed to load extreme datasets")
            return None, None
        
        # Step 2: Determine balanced sample size
        total_sample_size, left_half_size, right_half_size = self.determine_balanced_sample_size(left_samples, right_samples)
        
        # Step 3: Perform balanced sampling
        balanced_samples, achieved_stats = self.perform_balanced_sampling(
            left_samples, right_samples, left_half_size, right_half_size
        )
        
        if not balanced_samples:
            print("❌ Balanced extreme sampling failed")
            return None, None
        
        # Step 4: Save dataset
        dataset_path = self.save_balanced_dataset(balanced_samples, achieved_stats)
        
        # Step 5: Save statistics
        results_path = self.save_statistics(balanced_samples, achieved_stats)
        
        # Step 6: Print report
        self.print_report(balanced_samples, achieved_stats)
        
        print(f"\n✅ Balanced extreme sampling completed!")
        return balanced_samples, results_path


if __name__ == '__main__':
    # Initialize and run analysis
    generator = BalancedSamplingGenerator()
    
    try:
        balanced_samples, results_path = generator.run_analysis()
        print(f"\n🎯 Balanced extreme sampling analysis completed!")
        print(f"📊 Detailed results: {results_path}")
        print(f"📁 Balanced extreme dataset: {generator.balanced_datasets_dir}")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc() 