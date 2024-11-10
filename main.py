import os
import re
import json
import pandas as pd
import requests
import matplotlib.pyplot as plt
from matplotlib.table import Table
from dotenv import load_dotenv
from PIL import Image
import subprocess  # Added for opening files
import platform  # Added to determine the OS

# Load environment variables from .env file
load_dotenv()

# Load configuration from config.json
CONFIG_FILE_PATH = 'config.json'
with open(CONFIG_FILE_PATH, 'r') as config_file:
    config = json.load(config_file)

# Load configuration parameters
CSV_PATH = config.get('csv_path', '')
OUTPUT_PATH = config.get('output_path', '')
BING_API_KEY = os.getenv('BING_API_KEY', config.get('bing_api_key'))
TOP_ITEMS_COUNT = config.get('top_items_count', 3)
NUMBER_OF_TOP_STOCKS = config.get('number_of_top_stocks', 10)
LINKEDIN_API_ENDPOINT = config.get('linkedin_api_endpoint', '')
LINKEDIN_MODEL = config.get('linkedin_model', 'mistral-small')
TABLE_IMAGE_WIDTH = config.get('table_image_width', 8)
TABLE_IMAGE_HEIGHT_PER_ROW = config.get('table_image_height_per_row', 0.6)
LINKEDIN_DPI = config.get('linkedin_dpi', 300)
PROMPT_TEMPLATE_PATH = config.get('prompt_template_path', 'prompt_template.txt')


# Main function to execute the entire process
def main():
    # Validate and load API key
    validate_bing_api_key()
    
    # Load DataFrame from CSV file
    df = load_dataframe(CSV_PATH)
    if df is None:
        return  # Exit if DataFrame couldn't be loaded

    # Get top items from the DataFrame
    top_items = get_top_items(df, top_n=TOP_ITEMS_COUNT)
    print(f"Top Items: {top_items}")

    # Search web for top items and gather search results
    search_results = search_top_items(top_items)

    # Generate a LinkedIn post using Ollama API
    post = create_linkedin_post(top_items, search_results)
    if post:
        save_linkedin_post(post, OUTPUT_PATH)

    # Create a table image from the DataFrame
    image_path = create_stock_table_image(df, num_items=NUMBER_OF_TOP_STOCKS, output_path=OUTPUT_PATH + 'Top_Stocks_Table.png')

    # Open the resulting text file and image if they exist
    #if image_path:
    #    open_file(image_path)
    open_file(OUTPUT_PATH + 'Generated_LinkedIn_Post.txt')

# Utility Functions
def validate_bing_api_key():
    """
    Validates the Bing API key from environment variables.
    """
    if not BING_API_KEY:
        raise ValueError("Bing API key not found. Set BING_API_KEY in your environment.")

def load_dataframe(file_path):
    """
    Loads the CSV into a DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame or None: Returns the DataFrame if successful, otherwise None.
    """
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip', encoding='utf-8')
        print(df.head())  # Verify the structure of the DataFrame
        return df
    except pd.errors.ParserError as e:
        print(f"ParserError: {e}")
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None

def get_top_items(df, top_n):
    """
    Retrieves the top items from the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing stock data.
        top_n (int): The number of top items to retrieve.

    Returns:
        list: List of top items.
    """
    return df.head(top_n)['Ticker'].tolist()

def search_web(query):
    """
    Searches the web for a given query using Bing API.

    Args:
        query (str): The search query.

    Returns:
        dict: The search result as a JSON dictionary.
    """
    search_url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
    params = {
        "q": query,
        "textDecorations": True,
        "textFormat": "HTML",
        "count": 5
    }
    try:
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the search: {e}")
        return {}

def search_top_items(top_items):
    """
    Searches for market news for the top stock items.

    Args:
        top_items (list): List of top stock items.

    Returns:
        dict: A dictionary of search results keyed by stock item.
    """
    search_results = {}
    for item in top_items:
        query = f"Stock and market news, including price performance from this week about {item} stock"
        search_results[item] = search_web(query)
    return search_results

def extract_insightful_snippet(search_results, item):
    """
    Extracts insightful snippets from search results.

    Args:
        search_results (dict): Dictionary of search results.
        item (str): The item to extract information for.

    Returns:
        str: The insightful snippet or default description.
    """
    web_pages = search_results.get(item, {}).get('webPages', {}).get('value', [])
    if not web_pages:
        return f"No snippet found for {item}. This is a high-ranking item worth exploring."
    
    relevant_snippets = [
        page.get('snippet', '') for page in web_pages
        if re.search(r'\d+%|\$\d+|weekly price performance|earnings results|IBD News|quarter|contract|revenue|acquisition|merger|partnership', 
                     page.get('snippet', ''), re.IGNORECASE)
    ]
    
    return relevant_snippets[0] if relevant_snippets else web_pages[0].get('snippet', '')

def generate_post(prompt):
    """
    Generates a social media post using the Ollama API.

    Args:
        prompt (str): The prompt for the LLM.

    Returns:
        str: The generated post text or None.
    """
    try:
        payload = {
            "model": LINKEDIN_MODEL,
            "prompt": prompt,
            "stream": False
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(LINKEDIN_API_ENDPOINT, data=json.dumps(payload), headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get('response', '').strip()
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while interacting with Ollama API: {e}")
    return None

def load_prompt_template():
    """
    Loads the prompt template from an external file.

    Returns:
        str: The prompt template.
    """
    with open(PROMPT_TEMPLATE_PATH, 'r') as file:
        return file.read()

def create_linkedin_post(top_items, search_results):
    """
    Creates a LinkedIn post for the top items.

    Args:
        top_items (list): List of top items.
        search_results (dict): Dictionary of search results for each item.

    Returns:
        str: The generated LinkedIn post.
    """
    snippets = [
        f"📊 ${item}: {extract_insightful_snippet(search_results, item)}"
        for item in top_items
    ]
    
    prompt_template = load_prompt_template()
    prompt = prompt_template.replace("{snippets}", "\n\n".join(snippets))

    return generate_post(prompt)

def save_linkedin_post(post, output_path):
    """
    Saves the generated LinkedIn post to a text file.

    Args:
        post (str): The LinkedIn post text.
        output_path (str): Directory path to save the text file.
    """
    file_path = os.path.join(output_path, 'Generated_LinkedIn_Post.txt')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(post + "\n\nCheck out the leaders below ⤵️\n\n#investing #stocks #stockmarket #tech #AI #growth $QQQ $SPY $IWM $DIA #stocks")
    print(f"\nGenerated LinkedIn post saved to: {file_path}\n")

def create_stock_table_image(df, num_items, output_path):
    """
    Create an image of a table containing stock data from a DataFrame with columns resized based on content length.

    Args:
        df (pd.DataFrame): The DataFrame containing stock data.
        num_items (int): The number of items to display in the table.
        output_path (str): Path to save the generated image.

    Returns:
        str: Path to the saved image if successful, otherwise None.
    """
    try:
        filtered_df = df.drop(columns=['Date/Time', 'GICS ID'], errors='ignore')
        top_stocks = filtered_df.head(num_items)

        column_lengths = [max(top_stocks[col].astype(str).map(len).max(), len(col)) for col in top_stocks.columns]
        total_length = sum(column_lengths)
        column_widths = [length / total_length for length in column_lengths]

        fig, ax = plt.subplots(figsize=(TABLE_IMAGE_WIDTH, len(top_stocks) * TABLE_IMAGE_HEIGHT_PER_ROW))
        ax.axis('off')

        table = Table(ax, bbox=[0, 0, 1, 1])

        for j, column in enumerate(top_stocks.columns):
            table.add_cell(0, j, width=column_widths[j], height=0.08, loc='center', 
                           text=column, facecolor='#40466e', edgecolor='white')
            header_cell = table[(0, j)]
            header_cell.get_text().set_color('white')
            header_cell.get_text().set_weight('bold')

        for i, row in enumerate(top_stocks.values):
            for j, cell_value in enumerate(row):
                table.add_cell(i + 1, j, width=column_widths[j], height=0.08, loc='center', 
                               text=cell_value, facecolor='#f1f3f4' if i % 2 == 0 else '#e1e4e8', edgecolor='white')
                row_cell = table[(i + 1, j)]
                row_cell.get_text().set_fontsize(10)

        ax.add_table(table)

        temp_path = output_path.replace('.png', '_temp.png')
        plt.savefig(temp_path, bbox_inches='tight', dpi=LINKEDIN_DPI)

        with Image.open(temp_path) as img:
            bbox = img.getbbox()
            cropped_img = img.crop(bbox)
            cropped_img.save(output_path)

        os.remove(temp_path)

        print(f"\nTable image saved to: {output_path}\n")
        return output_path

    except Exception as e:
        print(f"An error occurred while creating the table image: {e}")
        return None

def open_file(file_path):
    """
    Opens a file using the default application on Windows.

    Args:
        file_path (str): Path to the file to be opened.
    """
    if platform.system() == 'Windows':
        try:
            subprocess.Popen(['start', file_path], shell=True)
        except Exception as e:
            print(f"An error occurred while trying to open the file: {e}")

# Execute the main function
if __name__ == "__main__":
    main()