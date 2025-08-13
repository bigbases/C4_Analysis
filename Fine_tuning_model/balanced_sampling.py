import os
import pandas as pd
import json
import numpy as np
import math
import random
from datetime import datetime
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class BalancedSamplingGenerator:
    def __init__(self):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.annotation_dir = os.path.join(self.current_dir, '../annotation_datasets')
        self.analysis_results_dir = os.path.join(self.current_dir, 'analysis_results')
        self.balanced_datasets_dir = os.path.join(self.current_dir, './data/balanced_datasets')
        
        # Create directories if they don't exist
        os.makedirs(self.analysis_results_dir, exist_ok=True)
        os.makedirs(self.balanced_datasets_dir, exist_ok=True)
        
        # All topics for integration
        self.topics = {
            'Tax': {
                'filename': 'annotated_tax_policy_20250802_142311',
                'display_name': 'Tax Increase'
            },
            'Trade': {
                'filename': 'annotated_trade_policy_20250802_162816',
                'display_name': 'Trade Increase'
            },
            'Free Market': {
                'filename': 'annotated_free-market_20250803_234743',
                'display_name': 'Free Market Economy'
            },
            'Civil Liberties': {
                'filename': 'annotated_civil_liberties_20250804_181052',
                'display_name': 'Civil Liberties'
            },
            'Gun Control': {
                'filename': 'annotated_gun_control_20250731_231907',
                'display_name': 'Gun Control'
            },
            'Death Penalty': {
                'filename': 'annotated_death_penalty_20250805_024453',
                'display_name': 'Death Penalty'
            },
            'Abortion': {
                'filename': 'annotated_abortion_20250801_141418',
                'display_name': 'Abortion Rights'
            },
            'LGBTQ': {
                'filename': 'annotated_LGBTQ_20250802_005914',
                'display_name': 'LGBTQ Rights'
            },
            'Drug Policy': {
                'filename': 'annotated_drug_policy_20250802_022252',
                'display_name': 'Drug Legalization'
            },
            'Immigration': {
                'filename': 'annotated_immigration_20250801_173425',
                'display_name': 'Immigration Policy'
            },
            'Gender Equality': {
                'filename': 'annotated_gender_equality_20250805_074411',
                'display_name': 'Gender Equality'
            },
            'Bioethics': {
                'filename': 'annotated_bioethics_20250805_194425',
                'display_name': 'Bioethics'
            },
            'Nationalism': {
                'filename': 'annotated_nationalism_20250805_233926',
                'display_name': 'Nationalism'
            },
            'Multiculturalism': {
                'filename': 'annotated_multiculturalism_20250806_122302',
                'display_name': 'Multiculturalism'
            },
            'Climate Change': {
                'filename': 'annotated_climate_change_20250806_124733',
                'display_name': 'Environmental Protection'
            }
        }
        
        # Model-persona combinations
        self.models = ['claude-sonnet-4-20250514', 'gpt-4.1']
        self.personas = ['opp_left', 'opp_right', 'sup_left', 'sup_right']
        self.model_persona_combinations = []
        
        for model in self.models:
            for persona in self.personas:
                self.model_persona_combinations.append(f"{model}_{persona}")
        
        # Bias category thresholds
        self.left_threshold = -0.2
        self.right_threshold = 0.2
        
        print("Balanced Sampling Generator (Left 50% + Right 50%)")
        print(f"Integrated topics: {len(self.topics)}")
        print(f"Model-persona combinations: {len(self.model_persona_combinations)}")

    def parse_json_response(self, json_str):
        """Parse JSON response and extract Political and Stance scores"""
        try:
            if pd.isna(json_str) or json_str == "":
                return None, None
                
            data = json.loads(json_str)
            political_score = data.get('Political', {}).get('score', None)
            stance_score = data.get('Stance', {}).get('score', None)
            
            return float(political_score) if political_score is not None else None, \
                   float(stance_score) if stance_score is not None else None
            
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None, None

    def categorize_bias(self, political_score):
        """Categorize political score into left/neutral/right"""
        if political_score <= self.left_threshold:
            return 'left'
        elif political_score >= self.right_threshold:
            return 'right'
        else:
            return 'neutral'

    def load_and_integrate_all_topics(self):
        """Load and integrate data from all topics into a single dataset"""
        print("\n=== Integrating all topic data ===")
        
        integrated_samples = []
        topic_stats = {}
        
        for topic_key, topic_info in self.topics.items():
            print(f"\n📂 Loading {topic_key}...")
            
            filename = f"{topic_info['filename']}.csv"
            filepath = os.path.join(self.annotation_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"  ⚠️ File not found: {filepath}")
                continue
            
            try:
                df = pd.read_csv(filepath)
                print(f"  ✓ Loaded: {filename} (shape: {df.shape})")
                
                # Process each row
                valid_samples = 0
                for idx, row in df.iterrows():
                    row_political_scores = []
                    row_stance_scores = []
                    
                    # Extract scores from all model-persona combinations
                    for combo in self.model_persona_combinations:
                        if combo in df.columns:
                            pol_score, stance_score = self.parse_json_response(row[combo])
                            
                            if pol_score is not None:
                                row_political_scores.append(pol_score)
                            if stance_score is not None:
                                row_stance_scores.append(stance_score)
                    
                    # Only include samples with complete data
                    if (len(row_political_scores) == len(self.model_persona_combinations) and 
                        len(row_stance_scores) == len(self.model_persona_combinations)):
                        
                        political_avg = np.mean(row_political_scores)
                        stance_avg = np.mean(row_stance_scores)
                        
                        # Categorize based on political score
                        category = self.categorize_bias(political_avg)
                        
                        sample_info = {
                            'topic_key': topic_key,
                            'topic_display_name': topic_info['display_name'],
                            'original_index': idx,
                            'political_score': political_avg,
                            'stance_score': stance_avg,
                            'category': category,
                            'row_data': row.to_dict()
                        }
                        
                        integrated_samples.append(sample_info)
                        valid_samples += 1
                
                topic_stats[topic_key] = {
                    'total_rows': len(df),
                    'valid_samples': valid_samples,
                    'validity_rate': valid_samples / len(df) if len(df) > 0 else 0
                }
                
                print(f"  📊 Valid samples: {valid_samples}/{len(df)} ({valid_samples/len(df)*100:.1f}%)")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue
        
        print(f"\n=== Integration completed ===")
        print(f"Total integrated samples: {len(integrated_samples)}")
        
        # Calculate category distribution
        category_counts = {}
        for sample in integrated_samples:
            cat = sample['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        total = len(integrated_samples)
        print(f"Category distribution:")
        print(f"  Left-biased: {category_counts.get('left', 0)} ({category_counts.get('left', 0)/total*100:.1f}%)")
        print(f"  Neutral: {category_counts.get('neutral', 0)} ({category_counts.get('neutral', 0)/total*100:.1f}%)")
        print(f"  Right-biased: {category_counts.get('right', 0)} ({category_counts.get('right', 0)/total*100:.1f}%)")
        
        return integrated_samples, topic_stats

    def determine_balanced_sample_size(self, integrated_samples):
        """Determine sample size for balanced sampling (50% left, 50% right)"""
        print("\n📏 Determining balanced sample size...")
        
        # Count left and right samples
        left_samples = [s for s in integrated_samples if s['category'] == 'left']
        right_samples = [s for s in integrated_samples if s['category'] == 'right']
        
        print(f"  Left-biased samples: {len(left_samples)}")
        print(f"  Right-biased samples: {len(right_samples)}")
        
        # Use the smaller count as the limiting factor for 50:50 balance
        half_size = min(len(left_samples), len(right_samples))
        balanced_sample_size = half_size * 2  # 50% left + 50% right
        
        # Apply minimum limit for safety
        if half_size < 10:
            print(f"  ⚠️ Too few samples per category: {half_size}")
            half_size = max(half_size, 10)  # Minimum 10 samples per category
            balanced_sample_size = half_size * 2
        
        print(f"  Determined balanced sample size: {balanced_sample_size} ({half_size} per category)")
        return balanced_sample_size, half_size

    def perform_balanced_sampling(self, integrated_samples, sample_size_per_category):
        """Perform balanced sampling: 50% left + 50% right"""
        print(f"\n🎯 Performing balanced sampling")
        print(f"  Selecting {sample_size_per_category} samples per category")
        
        # Separate samples by category
        left_samples = [s for s in integrated_samples if s['category'] == 'left']
        right_samples = [s for s in integrated_samples if s['category'] == 'right']
        
        # Random sampling from each category
        selected_left = random.sample(left_samples, sample_size_per_category)
        selected_right = random.sample(right_samples, sample_size_per_category)
        
        # Combine and shuffle
        balanced_samples = selected_left + selected_right
        random.shuffle(balanced_samples)
        
        # Calculate statistics
        total_count = len(balanced_samples)
        left_count = len(selected_left)
        right_count = len(selected_right)
        
        actual_mean = np.mean([s['political_score'] for s in balanced_samples])
        actual_std = np.std([s['political_score'] for s in balanced_samples], ddof=1)
        
        achieved_stats = {
            'total_samples': total_count,
            'left_count': left_count,
            'right_count': right_count,
            'left_ratio': left_count / total_count,
            'right_ratio': right_count / total_count,
            'mean_political_score': actual_mean,
            'std_political_score': actual_std,
            'min_political_score': min([s['political_score'] for s in balanced_samples]),
            'max_political_score': max([s['political_score'] for s in balanced_samples])
        }
        
        print(f"  ✅ Balanced sampling completed:")
        print(f"    Total samples: {total_count}")
        print(f"    Left-biased: {left_count} ({left_count/total_count*100:.1f}%)")
        print(f"    Right-biased: {right_count} ({right_count/total_count*100:.1f}%)")
        print(f"    Mean score: {actual_mean:.3f}")
        print(f"    Standard deviation: {actual_std:.3f}")
        
        return balanced_samples, achieved_stats

    def save_balanced_dataset(self, balanced_samples, achieved_stats):
        """Save the balanced dataset to CSV file"""
        if not balanced_samples:
            return None
        
        # Create DataFrame from balanced samples
        rows = []
        for sample in balanced_samples:
            row_data = sample['row_data'].copy()
            # Add metadata
            row_data['sampling_type'] = 'balanced_50_50'
            row_data['source_topic'] = sample['topic_key']
            row_data['source_topic_display'] = sample['topic_display_name']
            row_data['achieved_political_score'] = sample['political_score']
            row_data['achieved_stance_score'] = sample['stance_score']
            row_data['bias_category'] = sample['category']
            
            rows.append(row_data)
        
        balanced_df = pd.DataFrame(rows)
        
        # Add statistics as columns
        for key, value in achieved_stats.items():
            if isinstance(value, (int, float, str)):
                balanced_df[f'balanced_{key}'] = value
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"balanced_sampling_50_50_{timestamp}.csv"
        filepath = os.path.join(self.balanced_datasets_dir, filename)
        balanced_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        print(f"  💾 Balanced dataset saved: {filepath}")
        print(f"  📊 Saved columns: {len(balanced_df.columns)}")
        
        return filepath

    def save_statistics(self, balanced_samples, achieved_stats, topic_stats):
        """Save sampling statistics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save balanced sampling results
        results_data = {
            'sampling_type': 'balanced_50_50',
            'timestamp': timestamp,
            **achieved_stats
        }
        
        # Add topic distribution
        topic_distribution = {}
        for sample in balanced_samples:
            topic = sample['topic_key']
            topic_distribution[topic] = topic_distribution.get(topic, 0) + 1
        
        results_data['topic_distribution'] = str(topic_distribution)
        
        results_df = pd.DataFrame([results_data])
        results_path = os.path.join(self.analysis_results_dir, 
                                   f'balanced_sampling_results_{timestamp}.csv')
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        
        # Save integration statistics
        integration_stats = []
        for topic_key, stats in topic_stats.items():
            stats['topic_key'] = topic_key
            stats['display_name'] = self.topics[topic_key]['display_name']
            integration_stats.append(stats)
        
        integration_df = pd.DataFrame(integration_stats)
        integration_path = os.path.join(self.analysis_results_dir,
                                       f'topic_integration_stats_{timestamp}.csv')
        integration_df.to_csv(integration_path, index=False, encoding='utf-8-sig')
        
        print(f"\n📊 Balanced sampling results saved:")
        print(f"  Detailed results: {results_path}")
        print(f"  Integration statistics: {integration_path}")
        
        return results_path, integration_path

    def print_report(self, balanced_samples, achieved_stats, integrated_samples):
        """Print comprehensive balanced sampling report"""
        print(f"\n{'='*80}")
        print("Balanced Sampling Results Report (Left 50% + Right 50%)")
        print(f"{'='*80}")
        
        print(f"\n1. Sampling Results:")
        print(f"{'-'*50}")
        print(f"  Total samples: {achieved_stats['total_samples']}")
        print(f"  Left-biased samples: {achieved_stats['left_count']} ({achieved_stats['left_ratio']*100:.1f}%)")
        print(f"  Right-biased samples: {achieved_stats['right_count']} ({achieved_stats['right_ratio']*100:.1f}%)")
        print(f"  Mean political score: {achieved_stats['mean_political_score']:.3f}")
        print(f"  Standard deviation: {achieved_stats['std_political_score']:.3f}")
        print(f"  Score range: {achieved_stats['min_political_score']:.3f} ~ {achieved_stats['max_political_score']:.3f}")
        
        print(f"\n2. Topic Distribution:")
        print(f"{'-'*50}")
        topic_distribution = {}
        for sample in balanced_samples:
            topic = sample['topic_display_name']
            topic_distribution[topic] = topic_distribution.get(topic, 0) + 1
        
        for topic, count in sorted(topic_distribution.items()):
            percentage = count / len(balanced_samples) * 100
            print(f"  {topic}: {count} ({percentage:.1f}%)")
        
        print(f"\n3. Compared to Overall Data:")
        print(f"{'-'*50}")
        
        # Overall category distribution
        category_counts = {}
        for sample in integrated_samples:
            cat = sample['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        total = len(integrated_samples)
        print(f"  Overall integrated data:")
        print(f"    Left-biased: {category_counts.get('left', 0)} ({category_counts.get('left', 0)/total*100:.1f}%)")
        print(f"    Neutral: {category_counts.get('neutral', 0)} ({category_counts.get('neutral', 0)/total*100:.1f}%)")
        print(f"    Right-biased: {category_counts.get('right', 0)} ({category_counts.get('right', 0)/total*100:.1f}%)")
        
        print(f"  Balanced sampling data:")
        print(f"    Left-biased: {achieved_stats['left_count']} ({achieved_stats['left_ratio']*100:.1f}%)")
        print(f"    Right-biased: {achieved_stats['right_count']} ({achieved_stats['right_ratio']*100:.1f}%)")

    def run_analysis(self):
        """Run the complete balanced sampling analysis"""
        print(f"Starting balanced sampling analysis (Left 50% + Right 50%)...")
        
        # Step 1: Load and integrate all topic data
        integrated_samples, topic_stats = self.load_and_integrate_all_topics()
        
        if len(integrated_samples) == 0:
            print("❌ No integrated samples")
            return None, None, None
        
        # Step 2: Determine balanced sample size
        total_sample_size, sample_size_per_category = self.determine_balanced_sample_size(integrated_samples)
        
        # Step 3: Perform balanced sampling
        balanced_samples, achieved_stats = self.perform_balanced_sampling(
            integrated_samples, sample_size_per_category
        )
        
        if not balanced_samples:
            print("❌ Balanced sampling failed")
            return None, None, None
        
        # Step 4: Save dataset
        dataset_path = self.save_balanced_dataset(balanced_samples, achieved_stats)
        
        # Step 5: Save statistics
        results_path, integration_path = self.save_statistics(balanced_samples, achieved_stats, topic_stats)
        
        # Step 6: Print report
        self.print_report(balanced_samples, achieved_stats, integrated_samples)
        
        print(f"\n✅ Balanced sampling completed!")
        return balanced_samples, results_path, integration_path


if __name__ == '__main__':
    # Initialize and run analysis
    generator = BalancedSamplingGenerator()
    
    try:
        balanced_samples, results_path, integration_path = generator.run_analysis()
        print(f"\n🎯 Balanced sampling analysis completed!")
        print(f"📊 Detailed results: {results_path}")
        print(f"📈 Integration statistics: {integration_path}")
        print(f"📁 Balanced dataset: {generator.balanced_datasets_dir}")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc() 