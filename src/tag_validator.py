"""
Tag validation and correction utilities for LLM output.

This module provides functions to validate and fix mismatched speaker tags
in the output from local LLMs.
"""

import re
from typing import List, Tuple, Optional, Dict
from difflib import SequenceMatcher


def validate_and_fix_tags(text: str) -> Tuple[str, List[Dict]]:
    """
    Validate and fix mismatched speaker tags in text.
    
    Args:
        text: Input text with speaker tags
        
    Returns:
        Tuple of (fixed_text, list of corrections made)
    """
    corrections = []
    
    # Find all opening and closing tags
    opening_tags = list(re.finditer(r'\[([a-zA-Z0-9_\-]+)\]([^[]*)', text))
    closing_tags = list(re.finditer(r'\[/([a-zA-Z0-9_\-]+)\]', text))
    
    # Build list of tag positions and types
    tag_positions = []
    for match in opening_tags:
        tag_positions.append({
            'type': 'open',
            'name': match.group(1),
            'start': match.start(),
            'end': match.end(),
            'content_start': match.end()
        })
    for match in closing_tags:
        tag_positions.append({
            'type': 'close',
            'name': match.group(1),
            'start': match.start(),
            'end': match.end()
        })
    
    # Sort by position
    tag_positions.sort(key=lambda x: x['start'])
    
    # Validate tag pairs
    stack = []  # Stack of (tag_name, position)
    fixed_segments = []
    last_end = 0
    
    for tag in tag_positions:
        if tag['type'] == 'open':
            stack.append((tag['name'], tag['start'], tag.get('content_start', tag['end'])))
        else:  # closing tag
            if stack:
                open_tag, open_pos, content_start = stack.pop()
                if open_tag == tag['name']:
                    # Matching pair - good
                    pass
                else:
                    # Mismatched tags - fix it
                    corrections.append({
                        'type': 'mismatch',
                        'expected': f'[/{open_tag}]',
                        'found': f'[/{tag["name"]}]',
                        'position': tag['start'],
                        'fix': f'[/{open_tag}]'
                    })
                    # Replace the mismatched closing tag
                    text = text[:tag['start']] + f'[/{open_tag}]' + text[tag['end']:]
                    # Adjust offset for the replacement
                    tag['end'] = tag['start'] + len(f'[/{open_tag}]')
    
    # Check for unclosed tags
    for open_tag, open_pos, content_start in stack:
        corrections.append({
            'type': 'unclosed',
            'tag': f'[{open_tag}]',
            'position': open_pos,
            'fix': f'[/{open_tag}]'
        })
        # Add missing closing tag
        text = text + f'[/{open_tag}]'
    
    # Check for orphaned closing tags (already handled by position sorting)
    
    return text, corrections


def find_similar_speakers(speakers: List[str], threshold: float = 0.8) -> List[Tuple[str, str, float]]:
    """
    Find speakers that might be the same person with different name formats.
    
    Args:
        speakers: List of speaker names
        threshold: Similarity threshold (0-1), default 0.8
        
    Returns:
        List of tuples (speaker1, speaker2, similarity_score)
    """
    similar_pairs = []
    
    for i, sp1 in enumerate(speakers):
        for sp2 in speakers[i+1:]:
            score = SequenceMatcher(None, sp1.lower(), sp2.lower()).ratio()
            if score >= threshold:
                similar_pairs.append((sp1, sp2, round(score, 2)))
    
    return similar_pairs


def normalize_speaker_name(name: str) -> str:
    """
    Normalize speaker name for comparison.
    
    - Convert to lowercase
    - Replace spaces with hyphens
    - Remove special characters
    """
    return re.sub(r'[^a-z0-9\-]', '', name.lower().replace(' ', '-'))


def suggest_speaker_mapping(speakers: List[str], known_speakers: List[str]) -> Dict[str, str]:
    """
    Suggest mappings from detected speakers to known speakers.
    
    Args:
        speakers: List of speakers found in current chunk
        known_speakers: List of known speakers from previous chunks
        
    Returns:
        Dict mapping detected speaker -> suggested known speaker
    """
    mapping = {}
    
    for speaker in speakers:
        if speaker in known_speakers:
            continue  # Already known
        
        normalized = normalize_speaker_name(speaker)
        
        # Try exact match with normalized
        for known in known_speakers:
            known_normalized = normalize_speaker_name(known)
            if normalized == known_normalized:
                mapping[speaker] = known
                break
        
        # Try fuzzy match if no exact match
        if speaker not in mapping:
            best_match = None
            best_score = 0
            
            for known in known_speakers:
                known_normalized = normalize_speaker_name(known)
                score = SequenceMatcher(None, normalized, known_normalized).ratio()
                if score > best_score and score >= 0.7:
                    best_score = score
                    best_match = known
            
            if best_match:
                mapping[speaker] = best_match
    
    return mapping


def apply_speaker_mapping(text: str, speaker_map: Dict[str, str]) -> str:
    """
    Apply speaker name mapping to text.
    
    Args:
        text: Text with speaker tags
        speaker_map: Dict mapping old speaker names to new ones
        
    Returns:
        Text with updated speaker names
    """
    for old_name, new_name in speaker_map.items():
        # Replace opening tags
        text = re.sub(
            rf'\[{re.escape(old_name)}\]',
            f'[{new_name}]',
            text
        )
        # Replace closing tags
        text = re.sub(
            rf'\[/{re.escape(old_name)}\]',
            f'[/{new_name}]',
            text
        )
    
    return text


def validate_tags_strict(text: str) -> Tuple[bool, List[str]]:
    """
    Strictly validate that all tags are properly matched.
    
    Args:
        text: Text with speaker tags
        
    Returns:
        Tuple of (is_valid, list of errors)
    """
    errors = []
    
    # Find all tags
    opening = set(re.findall(r'\[([a-zA-Z0-9_\-]+)\]([^[]+)', text))
    closing = set(re.findall(r'\[/([a-zA-Z0-9_\-]+)\]', text))
    
    # Check each opening tag has a closing
    for tag, content in opening:
        if f'[/{tag}]' not in text:
            errors.append(f"Tag [{tag}] is missing closing tag")
    
    # Check for unmatched closing tags
    for tag in closing:
        if f'[{tag}]' not in text:
            errors.append(f"Closing tag [/{tag}] has no opening")
    
    return len(errors) == 0, errors


if __name__ == "__main__":
    # Test the validation
    test_text = """
[narrator]Chapter One[/narrator]
[mrs-bennet-female]My dear Mr. Bennet[/mrs-bennet-female]
[mr-bennet-male]I have not[/mr-bennet-male]
[mrs-bennet-female]But it is; for Mrs. Long has just been here[/mr-bennet-male]
[narrator]This was invitation enough[/narrator]
"""
    
    fixed, corrections = validate_and_fix_tags(test_text)
    print("Original:")
    print(test_text)
    print("\nCorrected:")
    print(fixed)
    print("\nCorrections:")
    for c in corrections:
        print(f"  {c}")
