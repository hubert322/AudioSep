import json
import os
import argparse

def fix_paths(input_json, output_json, base_path):
    print(f"Reading {input_json}...")
    with open(input_json, 'r') as f:
        data_dict = json.load(f)
    
    orig_data = data_dict['data']
    new_data = []
    
    print(f"Fixing {len(orig_data)} paths using base directory: {base_path}")
    
    for item in orig_data:
        # Prepend the base path to the filename
        # Handles cases where the path might already have a slash
        new_wav_path = os.path.join(base_path, item['wav'])
        
        new_item = {
            'wav': new_wav_path,
            'caption': item['caption']
        }
        new_data.append(new_item)
    
    print(f"Writing fixed JSON to {output_json}...")
    with open(output_json, 'w') as f:
        json.dump({'data': new_data}, f, indent=4)
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepend a base directory to all 'wav' paths in an AudioSep JSON file.")
    parser.add_argument("--input", type=str, required=True, help="Path to the original JSON file.")
    parser.add_argument("--output", type=str, help="Optional: Path where the fixed JSON should be saved. If omitted, uses input filename + suffix.")
    parser.add_argument("--suffix", type=str, default="_abs", help="Suffix to add to the filename if output is not specified (default: _abs).")
    parser.add_argument("--base_dir", type=str, required=True, help="The absolute path to the folder containing the .wav files.")
    
    args = parser.parse_args()
    
    output_path = args.output
    if not output_path:
        base, ext = os.path.splitext(os.path.basename(args.input))
        output_path = os.path.join("datafiles", f"{base}{args.suffix}{ext}")
    
    fix_paths(args.input, output_path, args.base_dir)
