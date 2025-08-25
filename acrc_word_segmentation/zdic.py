import pandas as pd
import requests
from bs4 import BeautifulSoup
import re

def get_zdic_meaning(character):
    """Fetches the meaning of a character or word from zdic.net."""
    url = f'https://www.zdic.net/hans/{character}'
    response = requests.get(url)
    response.encoding = 'utf-8'
    
    if response.status_code != 200:
        return None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Determine parsing strategy based on character length
    if len(character) == 1:
        # For single characters
        meaning_section = soup.find('div', class_='content')
        if not meaning_section:
            return None
        
        # Extract text before colons
        meanings = meaning_section.get_text(separator='\n').strip()
        extracted_meanings = []
        for line in meanings.split('\n'):
            match = re.match(r'^([^：]+)：', line)
            if match:
                extracted_meanings.append(match.group(1).strip())
        return '\n'.join(extracted_meanings)
    else:
        # For words
        meanings = []
        specific_div = soup.find('div', class_='jnr')
        if specific_div:
            for p in specific_div.find_all('p'):
                chinese_meaning = ''.join([text for text in p.stripped_strings if re.search(r'[\u4e00-\u9fff]', text)])
                if chinese_meaning:
                    meanings.append(chinese_meaning.strip())
        return '\n'.join(meanings) if meanings else None

# Load the thesaurus CSV file
thesaurus = pd.read_csv('train_thesaurus.csv')

# Filter words with frequency higher than 30
high_freq_words = thesaurus[thesaurus['Frequency'] > 30]['Word'].tolist()

# Initialize list for results
results = []

# Query meaning and record
for word in high_freq_words:
    meaning = get_zdic_meaning(word)
    if meaning:
        results.append({'word': word, 'annotation': meaning})
    else:
        print(f"Error: No annotation found for '{word}'")

# Save results to a new CSV file
output_df = pd.DataFrame(results, columns=['word', 'annotation'])
output_df.to_csv('high_freq_annotations.csv', index=False)

print("Processing complete. Results saved to 'high_freq_annotations.csv'.")
