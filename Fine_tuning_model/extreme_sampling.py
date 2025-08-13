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

class ExtremeSamplingGenerator:
    def __init__(self):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.annotation_dir = os.path.join(self.current_dir, '../annotation_datasets')
        self.analysis_results_dir = os.path.join(self.current_dir, 'analysis_results')
        self.extreme_datasets_dir = os.path.join(self.current_dir, './data/extreme_datasets')
        
        # Create directories if they don't exist
        os.makedirs(self.analysis_results_dir, exist_ok=True)
        os.makedirs(self.extreme_datasets_dir, exist_ok=True)
        
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
        
        # Extreme sampling scenarios
        self.extreme_scenarios = {
            'left_extreme': {
                'description': '극좌편향 (Left 100%, 극단적 좌편향만)',
                'category': 'left',
                'score_threshold': -0.3,  # -0.3보다 작은 점수만
                'target_mean_score': -0.7,  # 목표 평균 점수
                'extremity_preference': 'most_extreme'  # 가장 극단적인 것 선호
            },
            'right_extreme': {
                'description': '극우편향 (Right 100%, 극단적 우편향만)',
                'category': 'right',
                'score_threshold': 0.3,   # 0.3보다 큰 점수만
                'target_mean_score': 0.7,  # 목표 평균 점수
                'extremity_preference': 'most_extreme'  # 가장 극단적인 것 선호
            }
        }
        
        # Bias category thresholds (same as proportion sampling)
        self.left_threshold = -0.2
        self.right_threshold = 0.2
        
        print("Extreme Bias Sampling Generator")
        print(f"Integrated topics: {len(self.topics)}")
        print(f"Extreme scenarios: {list(self.extreme_scenarios.keys())}")
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

    def filter_extreme_samples(self, integrated_samples, scenario_config):
        """Filter samples for extreme scenarios based on score thresholds"""
        scenario_name = scenario_config.get('description', 'Unknown')
        print(f"\n🔍 Filtering extreme samples: {scenario_name}")
        
        if scenario_config['category'] == 'left':
            # Left extreme: only very left-leaning samples
            threshold = scenario_config['score_threshold']
            extreme_samples = [s for s in integrated_samples 
                             if s['political_score'] <= threshold]
            print(f"  Left extreme criterion: score <= {threshold}")
            
        elif scenario_config['category'] == 'right':
            # Right extreme: only very right-leaning samples  
            threshold = scenario_config['score_threshold']
            extreme_samples = [s for s in integrated_samples 
                             if s['political_score'] >= threshold]
            print(f"  Right extreme criterion: score >= {threshold}")
            
        else:
            extreme_samples = integrated_samples
        
        print(f"  Filtering result: {len(extreme_samples)}/{len(integrated_samples)} samples")
        
        if len(extreme_samples) > 0:
            scores = [s['political_score'] for s in extreme_samples]
            print(f"  Score range: {min(scores):.3f} ~ {max(scores):.3f}")
            print(f"  Mean score: {np.mean(scores):.3f}")
        
        return extreme_samples

    def determine_fair_sample_size(self, integrated_samples):
        """Determine fair sample size based on the limiting factor (usually right extremes)"""
        print("\n📏 Determining fair sample size...")
        
        # Count extreme samples for each direction
        left_extreme_count = len([s for s in integrated_samples if s['political_score'] <= -0.3])
        right_extreme_count = len([s for s in integrated_samples if s['political_score'] >= 0.3])
        
        print(f"  Extreme left samples (≤ -0.3): {left_extreme_count}")
        print(f"  Extreme right samples (≥ 0.3): {right_extreme_count}")
        
        # Use the smaller count as the limiting factor (no artificial limits)
        fair_sample_size = min(left_extreme_count, right_extreme_count)
        
        # Only apply minimum limit for safety
        if fair_sample_size < 20:
            print(f"  ⚠️ Sample size too small: {fair_sample_size}")
            fair_sample_size = max(fair_sample_size, 20)  # Minimum 20 samples
        
        print(f"  Determined fair sample size: {fair_sample_size} (based on right extreme sample count)")
        return fair_sample_size

    def select_samples_for_target_mean(self, candidate_samples, target_mean, sample_size, preference='most_extreme'):
        """Select samples to achieve target mean score while preferring extremity"""
        if len(candidate_samples) <= sample_size:
            return candidate_samples
        
        print(f"    Selecting {sample_size} samples for target mean {target_mean:.3f}...")
        
        # Sort by score (for extremity preference)
        if preference == 'most_extreme':
            if target_mean < 0:
                # For left extreme, prefer most negative scores
                candidate_samples.sort(key=lambda x: x['political_score'])
            else:
                # For right extreme, prefer most positive scores
                candidate_samples.sort(key=lambda x: x['political_score'], reverse=True)
        
        # Try different combinations to find best match for target mean
        best_combination = None
        best_mean_diff = float('inf')
        
        # For efficiency, try multiple random samples and pick the best
        num_trials = min(1000, len(candidate_samples) * 2)
        
        for trial in range(num_trials):
            if trial == 0:
                # First trial: use the most extreme samples
                selected = candidate_samples[:sample_size]
            else:
                # Random sampling for other trials
                selected = random.sample(candidate_samples, sample_size)
            
            mean_score = np.mean([s['political_score'] for s in selected])
            mean_diff = abs(mean_score - target_mean)
            
            if mean_diff < best_mean_diff:
                best_mean_diff = mean_diff
                best_combination = selected.copy()
        
        actual_mean = np.mean([s['political_score'] for s in best_combination])
        print(f"    Achieved mean: {actual_mean:.3f} (target: {target_mean:.3f}, diff: {best_mean_diff:.3f})")
        
        return best_combination

    def perform_extreme_sampling(self, integrated_samples, scenario_name, scenario_config, sample_size):
        """Perform extreme sampling for a specific scenario"""
        print(f"\n🎯 Extreme sampling: {scenario_name}")
        print(f"  Description: {scenario_config['description']}")
        
        # Filter extreme samples
        extreme_samples = self.filter_extreme_samples(integrated_samples, scenario_config)
        
        if len(extreme_samples) == 0:
            print(f"  ❌ No extreme samples found")
            return None, None
        
        sampled_combination = []
        achieved_stats = {}
        
        if scenario_config['category'] in ['left', 'right']:
            # Single-sided extreme
            target_mean = scenario_config['target_mean_score']
            preference = scenario_config['extremity_preference']
            
            selected_samples = self.select_samples_for_target_mean(
                extreme_samples, target_mean, sample_size, preference
            )
            
            if selected_samples:
                sampled_combination = selected_samples
                actual_mean = np.mean([s['political_score'] for s in selected_samples])
                actual_std = np.std([s['political_score'] for s in selected_samples], ddof=1)
                
                achieved_stats = {
                    'total_samples': len(selected_samples),
                    'mean_political_score': actual_mean,
                    'std_political_score': actual_std,
                    'min_political_score': min([s['political_score'] for s in selected_samples]),
                    'max_political_score': max([s['political_score'] for s in selected_samples]),
                    'category_distribution': {scenario_config['category']: 1.0}
                }
        

        
        if sampled_combination:
            print(f"  ✅ Sampling completed: {len(sampled_combination)} samples")
            print(f"  Mean score: {achieved_stats['mean_political_score']:.3f}")
            print(f"  Standard deviation: {achieved_stats['std_political_score']:.3f}")
        
        return sampled_combination, achieved_stats

    def save_extreme_dataset(self, scenario_name, sampled_combination, achieved_stats):
        """Save the extreme sampled dataset to CSV file"""
        if not sampled_combination:
            return None
        
        # Create DataFrame from sampled combination
        rows = []
        for sample in sampled_combination:
            row_data = sample['row_data'].copy()
            # Add metadata
            row_data['extreme_scenario'] = scenario_name
            row_data['source_topic'] = sample['topic_key']
            row_data['source_topic_display'] = sample['topic_display_name']
            row_data['achieved_political_score'] = sample['political_score']
            row_data['achieved_stance_score'] = sample['stance_score']
            row_data['bias_category'] = sample['category']
            
            rows.append(row_data)
        
        sampled_df = pd.DataFrame(rows)
        
        # Add scenario statistics as columns
        for key, value in achieved_stats.items():
            if isinstance(value, (int, float, str)):
                sampled_df[f'scenario_{key}'] = value
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"extreme_{scenario_name}_{timestamp}.csv"
        filepath = os.path.join(self.extreme_datasets_dir, filename)
        sampled_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        print(f"  💾 Extreme dataset saved: {filepath}")
        print(f"  📊 Saved columns: {len(sampled_df.columns)}")
        
        return filepath

    def run_extreme_sampling(self):
        """Run extreme sampling for all scenarios"""
        print("\n=== Extreme Bias Sampling Started ===")
        
        # Step 1: Load and integrate all topic data
        integrated_samples, topic_stats = self.load_and_integrate_all_topics()
        
        if len(integrated_samples) == 0:
            print("❌ No integrated samples")
            return None
        
        # Step 2: Determine fair sample size
        fair_sample_size = self.determine_fair_sample_size(integrated_samples)
        
        # Step 3: Perform sampling for each extreme scenario
        sampling_results = []
        
        for scenario_name, scenario_config in self.extreme_scenarios.items():
            print(f"\n{'='*60}")
            print(f"🔥 Extreme scenario: {scenario_name}")
            print(f"{'='*60}")
            
            # Perform extreme sampling
            sampled_combination, achieved_stats = self.perform_extreme_sampling(
                integrated_samples, scenario_name, scenario_config, fair_sample_size
            )
            
            if sampled_combination:
                # Save dataset
                saved_path = self.save_extreme_dataset(scenario_name, sampled_combination, achieved_stats)
                
                # Store results
                result_record = {
                    'scenario_name': scenario_name,
                    'description': scenario_config['description'],
                    'sample_size': len(sampled_combination),
                    'fair_sample_size': fair_sample_size,
                    'saved_file_path': saved_path,
                    **achieved_stats
                }
                
                # Add topic distribution
                topic_distribution = {}
                for sample in sampled_combination:
                    topic = sample['topic_key']
                    topic_distribution[topic] = topic_distribution.get(topic, 0) + 1
                
                result_record['topic_distribution'] = topic_distribution
                sampling_results.append(result_record)
                
            else:
                print(f"  ❌ {scenario_name} sampling failed")
        
        return pd.DataFrame(sampling_results), integrated_samples, topic_stats

    def save_extreme_statistics(self, results_df, integrated_samples, topic_stats):
        """Save extreme sampling statistics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_path = os.path.join(self.analysis_results_dir, 
                                   f'extreme_sampling_results_{timestamp}.csv')
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
        
        print(f"\n📊 Extreme sampling results saved:")
        print(f"  Detailed results: {results_path}")
        print(f"  Integration statistics: {integration_path}")
        
        return results_path, integration_path

    def print_extreme_report(self, results_df, integrated_samples):
        """Print comprehensive extreme sampling report"""
        print(f"\n{'='*80}")
        print("Extreme Bias Sampling Results Report")
        print(f"{'='*80}")
        
        print(f"\n1. Overall Summary:")
        print(f"{'-'*50}")
        print(f"  Total integrated samples: {len(integrated_samples)}")
        print(f"  Successful extreme scenarios: {len(results_df)}")
        print(f"  Total scenarios: {len(self.extreme_scenarios)}")
        
        if len(results_df) > 0:
            total_extreme_samples = results_df['sample_size'].sum()
            print(f"  Generated extreme samples total: {total_extreme_samples}")
        
        print(f"\n2. Scenario Results:")
        print(f"{'-'*50}")
        
        for _, row in results_df.iterrows():
            print(f"\n  🔥 {row['scenario_name']}:")
            print(f"    Description: {row['description']}")
            print(f"    Sample size: {row['sample_size']}")
            print(f"    Mean score: {row['mean_political_score']:.3f}")
            print(f"    Score range: {row['min_political_score']:.3f} ~ {row['max_political_score']:.3f}")
            print(f"    Save path: {row['saved_file_path']}")
        
        print(f"\n3. Data Distribution:")
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

    def run_analysis(self):
        """Run the complete extreme sampling analysis"""
        print(f"Starting extreme bias sampling analysis...")
        
        # Run extreme sampling
        results_df, integrated_samples, topic_stats = self.run_extreme_sampling()
        
        if results_df is None or len(results_df) == 0:
            print("❌ Extreme sampling failed")
            return None, None, None
        
        # Save results
        results_path, integration_path = self.save_extreme_statistics(results_df, integrated_samples, topic_stats)
        
        # Print report
        self.print_extreme_report(results_df, integrated_samples)
        
        print(f"\n✅ Extreme bias sampling completed!")
        return results_df, results_path, integration_path


if __name__ == '__main__':
    # Initialize and run analysis
    generator = ExtremeSamplingGenerator()
    
    try:
        results, results_path, integration_path = generator.run_analysis()
        print(f"\n🎯 Extreme analysis completed!")
        print(f"📊 Detailed results: {results_path}")
        print(f"📈 Integration statistics: {integration_path}")
        print(f"📁 Extreme datasets: {generator.extreme_datasets_dir}")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc() 