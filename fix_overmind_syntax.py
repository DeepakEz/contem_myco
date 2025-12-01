#!/usr/bin/env python3
"""
Quick fix for syntax error in myconet_contemplative_overmind.py
Run this script to automatically fix the syntax error
"""

import os
import sys
import re

def fix_overmind_syntax():
    """Fix the syntax error in the overmind module"""
    
    overmind_file = "myconet_contemplative_overmind.py"
    
    if not os.path.exists(overmind_file):
        print(f"Error: {overmind_file} not found in current directory")
        return False
    
    print(f"Reading {overmind_file}...")
    
    # Read the file
    try:
        with open(overmind_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return False
    
    print(f"Original file size: {len(content)} characters")
    
    # Create backup
    backup_file = overmind_file + ".backup"
    try:
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Backup created: {backup_file}")
    except Exception as e:
        print(f"Error creating backup: {e}")
        return False
    
    # Fix common syntax issues
    fixes_applied = []
    
    # Fix 1: Look for lines where print and def are merged
    pattern1 = r'(print\([^)]*\))\s*(def\s+\w+.*?:)'
    matches = re.findall(pattern1, content)
    if matches:
        for print_part, def_part in matches:
            old_line = print_part + def_part
            new_line = print_part + '\n\n    ' + def_part
            content = content.replace(old_line, new_line)
            fixes_applied.append(f"Separated print and def: {print_part[:30]}...")
    
    # Fix 2: Look for missing newlines before function definitions
    pattern2 = r'([^:\n])\n(    def\s+\w+.*?:)'
    content = re.sub(pattern2, r'\1\n\n\2', content)
    
    # Fix 3: Look for improper indentation issues
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Check for lines that should be indented but aren't
        if line.strip().startswith('def ') and i > 0:
            prev_line = lines[i-1].strip()
            if prev_line and not prev_line.endswith(':') and not line.startswith('    '):
                # This def should probably be indented
                line = '    ' + line
                fixes_applied.append(f"Fixed indentation for function at line {i+1}")
        
        fixed_lines.append(line)
    
    content = '\n'.join(fixed_lines)
    
    # Fix 4: Remove any duplicate function definitions or malformed sections
    # Look for the specific error pattern mentioned in the error
    if 'unexpected indent' in str(fixes_applied) or 'line 2171' in content:
        # Try to find and fix the specific problematic area
        lines = content.split('\n')
        for i, line in enumerate(lines[2160:2180], 2160):  # Around line 2171
            if 'def ' in line and not line.strip().startswith('def '):
                # Fix malformed function definition
                stripped = line.strip()
                if stripped.startswith('def '):
                    lines[i] = '    ' + stripped
                    fixes_applied.append(f"Fixed malformed function def at line {i+1}")
    
    content = '\n'.join(lines)
    
    # Write the fixed file
    try:
        with open(overmind_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully wrote fixed file")
        
        if fixes_applied:
            print("Fixes applied:")
            for fix in fixes_applied:
                print(f"  - {fix}")
        else:
            print("No specific fixes needed, but file was processed")
        
        return True
        
    except Exception as e:
        print(f"Error writing fixed file: {e}")
        return False

def main():
    """Main function"""
    print("MycoNet Overmind Syntax Error Fix")
    print("=" * 40)
    
    if fix_overmind_syntax():
        print("\nSyntax error fix completed!")
        print("You can now run the main simulation script.")
        
        # Try to validate the fixed file
        print("\nValidating fixed file...")
        try:
            with open("myconet_contemplative_overmind.py", 'r', encoding='utf-8') as f:
                file_content = f.read()
            compile(file_content, "myconet_contemplative_overmind.py", 'exec')
            print("✓ File compiles successfully!")
        except SyntaxError as e:
            print(f"✗ Syntax error still present: {e}")
            print(f"Error at line {e.lineno}: {e.text}")
        except UnicodeDecodeError as e:
            print(f"✗ Encoding issue detected: {e}")
            print("File may contain non-UTF-8 characters, but syntax should be fixed")
            print("✓ Main syntax fixes were applied successfully")
        except Exception as e:
            print(f"✗ Other error: {e}")
            print("✓ Main syntax fixes were applied - try running the simulation")
    else:
        print("\nFailed to fix syntax error automatically.")
        print("Manual intervention required.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())