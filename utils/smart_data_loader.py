"""
Smart Data Loader for M2S Experiments
- Random source selection from available sources
- Row tracking to avoid duplicates
- Original jailbreak prompt preservation
"""

import json
import pandas as pd
import numpy as np
import random
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import ast


class SmartDataLoader:
    """
    Enhanced data loader with intelligent sampling and tracking
    """
    
    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.df = None
        self.source_groups = {}
        self.used_rows = set()  # Track used row indices
        self.available_sources = []
        self.load_data()
    
    def load_data(self):
        """Load Excel data and organize by 4 balanced source groups"""
        print(f"Loading dataset: {self.excel_path}")
        self.df = pd.read_excel(self.excel_path)
        
        # Define the 4 balanced groups
        self.source_group_mapping = {
            "SafeMT": ["SafeMTData_Attack600", "SafeMTData_1K"],
            "MHJ": ["MHJ_local"],
            "HH-RLHF": [
                "hh-rlhf_harmless-base_chosen", "hh-rlhf_harmless-base_rejected",
                "hh-rlhf_helpful-base_chosen", "hh-rlhf_helpful-base_rejected",
                "hh-rlhf_helpful-online_chosen", "hh-rlhf_helpful-online_rejected",
                "hh-rlhf_helpful-rejection-sampled_chosen", "hh-rlhf_helpful-rejection-sampled_rejected"
            ],
            "Guard": ["Guard-Train"]
        }
        
        # Group rows by the 4 balanced groups
        self.balanced_groups = {}
        for group_name, source_list in self.source_group_mapping.items():
            group_rows = []
            for source in source_list:
                if source in self.df['source'].values:
                    source_rows = self.df[self.df['source'] == source].index.tolist()
                    # Filter out rows without valid jailbreak_turns
                    for idx in source_rows:
                        if pd.notna(self.df.loc[idx, 'jailbreak_turns']):
                            group_rows.append(idx)
            
            if group_rows:
                self.balanced_groups[group_name] = group_rows
                print(f"Debug: {group_name} rows sample: {group_rows[:3]}")
        
        self.available_groups = list(self.balanced_groups.keys())
        
        print(f"Dataset loaded: {len(self.df)} total rows")
        print(f"Available balanced groups: {len(self.available_groups)}")
        for group_name, rows in self.balanced_groups.items():
            print(f"  {group_name}: {len(rows)} valid rows")
            # Show which sources are in this group
            sources_in_group = set()
            for row_idx in rows[:5]:  # Sample first 5 to show sources
                source = self.df.loc[row_idx, 'source']
                if pd.notna(source):
                    sources_in_group.add(source)
            print(f"    Sources: {list(sources_in_group)}")
        
        # Keep original source_groups for backward compatibility
        self.source_groups = {}
        for source in self.df['source'].unique():
            if pd.notna(source):
                source_rows = self.df[self.df['source'] == source].index.tolist()
                valid_rows = []
                for idx in source_rows:
                    if pd.notna(self.df.loc[idx, 'jailbreak_turns']):
                        valid_rows.append(idx)
                if valid_rows:
                    self.source_groups[source] = valid_rows
        self.available_sources = list(self.source_groups.keys())
    
    def get_balanced_conversations(self, max_conversations: int) -> List[Dict]:
        """
        Get balanced conversations from 4 groups:
        1. SafeMT: SafeMTData_Attack600, SafeMTData_1K
        2. MHJ: MHJ_local  
        3. HH-RLHF: hh-rlhf_* variants
        4. Guard: Guard-Train
        """
        conversations = []
        
        print(f"\nExtracting {max_conversations} conversations with balanced group sampling...")
        
        # Calculate samples per group (as evenly as possible)
        samples_per_group = max_conversations // len(self.available_groups)
        extra_samples = max_conversations % len(self.available_groups)
        
        group_targets = {}
        for i, group_name in enumerate(self.available_groups):
            target = samples_per_group
            if i < extra_samples:  # Distribute extra samples to first few groups
                target += 1
            group_targets[group_name] = target
        
        print(f"Target distribution:")
        for group_name, target in group_targets.items():
            print(f"  {group_name}: {target} samples")
        
        # Sample from each group
        group_usage = {}
        for group_name in self.available_groups:
            target_count = group_targets[group_name]
            if target_count == 0:
                continue
                
            # Get unused rows from this group
            unused_rows = [row for row in self.balanced_groups[group_name] if row not in self.used_rows]
            
            if not unused_rows:
                print(f"Warning: No unused rows in group {group_name}")
                continue
            
            # Sample the target number (or all available if less)
            actual_samples = min(target_count, len(unused_rows))
            selected_rows = random.sample(unused_rows, actual_samples)
            
            group_conversations = []
            for row_idx in selected_rows:
                self.used_rows.add(row_idx)
                try:
                    row_data = self.df.loc[row_idx]
                    conversation = self.extract_conversation_from_row(row_data, row_idx)
                    if conversation:
                        conversation['dataset_group'] = group_name
                        group_conversations.append(conversation)
                except Exception as e:
                    print(f"Error processing row {row_idx} from group {group_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            conversations.extend(group_conversations)
            group_usage[group_name] = len(group_conversations)
            print(f"  {group_name}: extracted {len(group_conversations)}/{target_count} conversations")
        
        print(f"\nSuccessfully extracted {len(conversations)} conversations")
        print(f"Final distribution: {group_usage}")
        
        return conversations
    
    def get_random_conversations(self, max_conversations: int) -> List[Dict]:
        """
        Legacy method - now uses balanced sampling
        """
        return self.get_balanced_conversations(max_conversations)
    
    def extract_conversation_from_row(self, row_data: pd.Series, row_idx: int) -> Optional[Dict]:
        """Extract conversation from a single row"""
        try:
            # Parse jailbreak_turns
            jailbreak_turns_raw = row_data['jailbreak_turns']
            if pd.isna(jailbreak_turns_raw):
                return None
            
            # Parse the jailbreak_turns JSON
            if isinstance(jailbreak_turns_raw, str):
                jailbreak_data = ast.literal_eval(jailbreak_turns_raw)
            else:
                jailbreak_data = jailbreak_turns_raw
            
            if not isinstance(jailbreak_data, dict):
                return None
            
            # Extract turns in order
            conversation_turns = []
            original_jailbreak_prompts = {}  # Store original prompts before M2S conversion
            
            for turn_key in sorted(jailbreak_data.keys()):
                turn_content = jailbreak_data[turn_key]
                if turn_content and str(turn_content).strip():
                    conversation_turns.append({
                        "attacker": str(turn_content).strip(),
                        "target": f"Response to {turn_key}"
                    })
                    # Store original jailbreak prompt
                    original_jailbreak_prompts[turn_key] = str(turn_content).strip()
            
            if not conversation_turns:
                return None
            
            # Create conversation object with original prompts preserved
            conversation = {
                "id": f"row_{row_idx}",
                "source": str(row_data.get('source', 'unknown')),
                "base_prompt": str(row_data.get('base_prompt', '')),
                "conversation": conversation_turns,
                "num_turns": len(conversation_turns),
                "row_index": int(row_idx),  # Convert numpy int64 to Python int
                # CRITICAL: Store original jailbreak prompts before M2S conversion
                "original_jailbreak_prompts": original_jailbreak_prompts,
                "extraction_metadata": {
                    "extraction_method": "smart_random_sampling",
                    "source_selected": str(row_data.get('source', 'unknown')),
                    "total_available_turns": int(row_data.get('num_turns', len(conversation_turns)))
                }
            }
            
            return conversation
            
        except Exception as e:
            print(f"Error extracting conversation from row {row_idx}: {e}")
            return None
    
    def get_source_usage_stats(self) -> Dict[str, int]:
        """Get statistics of how many rows were used from each source"""
        source_usage = defaultdict(int)
        for row_idx in self.used_rows:
            if row_idx < len(self.df):
                source = self.df.loc[row_idx, 'source']
                source_usage[source] += 1
        return dict(source_usage)
    
    def get_remaining_capacity(self) -> Dict[str, int]:
        """Get remaining unused rows per source"""
        remaining = {}
        for source, rows in self.source_groups.items():
            unused_count = len([r for r in rows if r not in self.used_rows])
            remaining[source] = unused_count
        return remaining
    
    def reset_usage_tracking(self):
        """Reset used rows tracking for fresh sampling"""
        self.used_rows.clear()
        print("Usage tracking reset. All rows available for sampling again.")
    
    def get_sampling_report(self) -> Dict:
        """Generate comprehensive sampling report"""
        return {
            "total_rows_in_dataset": len(self.df),
            "total_sources_available": len(self.available_sources),
            "rows_used": len(self.used_rows),
            "source_usage_stats": self.get_source_usage_stats(),
            "remaining_capacity_per_source": self.get_remaining_capacity(),
            "available_sources": self.available_sources
        }


# Wrapper function for backward compatibility
def load_excel_dataset_smart(file_path: str, max_samples: int = 50) -> List[Dict]:
    """
    Smart data loading with enhanced sampling
    """
    loader = SmartDataLoader(file_path)
    conversations = loader.get_random_conversations(max_samples)
    
    # Print sampling report
    report = loader.get_sampling_report()
    print(f"\n--- SMART SAMPLING REPORT ---")
    print(f"Total conversations extracted: {len(conversations)}")
    print(f"Sources used: {len(report['source_usage_stats'])}")
    print(f"Diversity achieved: {report['source_usage_stats']}")
    
    return conversations


# Test function
if __name__ == "__main__":
    # Test smart data loading
    file_path = "../multi-turn_integrated_data.xlsx"
    
    print("Testing Smart Data Loader...")
    loader = SmartDataLoader(file_path)
    
    # Test with small sample
    conversations = loader.get_random_conversations(10)
    
    print(f"\n--- SAMPLE CONVERSATIONS ---")
    for i, conv in enumerate(conversations[:3], 1):
        print(f"Conversation {i}:")
        print(f"  ID: {conv['id']}")
        print(f"  Source: {conv['source']}")
        print(f"  Turns: {conv['num_turns']}")
        print(f"  Original prompts: {list(conv['original_jailbreak_prompts'].keys())}")
        print(f"  First prompt: {conv['original_jailbreak_prompts'].get('turn_1', 'N/A')[:100]}...")
        print()
    
    # Print final report
    report = loader.get_sampling_report()
    print(f"\n--- FINAL REPORT ---")
    print(json.dumps(report, indent=2))