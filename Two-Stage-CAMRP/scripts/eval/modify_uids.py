import sys
import re

def modify_uids(input_file, output_file, start_uid=1303):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        current_uid = start_uid
        uid_mapping = {}
        last_uid = None

        for line in infile:
            # 使用正则表达式匹配行首的数字
            match = re.match(r'^(\d+)\s', line)
            if match:
                old_uid = int(match.group(1))
                if old_uid != last_uid:
                    if old_uid not in uid_mapping:
                        uid_mapping[old_uid] = current_uid
                        current_uid += 1
                    last_uid = old_uid
                new_line = re.sub(r'^\d+', str(uid_mapping[old_uid]), line)
                outfile.write(new_line)
            else:
                outfile.write(line)  # 保持非UID开头的行不变

    print(f"UID modification complete. Results saved to {output_file}")
    print(f"New UIDs range from {start_uid} to {current_uid - 1}")
    print(f"Total unique UIDs modified: {len(uid_mapping)}")
    print("UID mapping:")
    for old, new in sorted(uid_mapping.items()):
        print(f"Old UID: {old} -> New UID: {new}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python modify_uids.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    modify_uids(input_file, output_file, start_uid=1303)