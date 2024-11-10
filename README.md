# DMTribe Social Media Poster

![Stock Analysis Banner](banner.png)

Welcome to the **Stock Market Analysis & LinkedIn Content Automation** project! This Python-based tool is designed to simplify the process of collecting, analyzing, and sharing key insights on leading stocks via LinkedIn, making it perfect for content creators, investors, and data enthusiasts.

## Overview

This project:

- Analyzes leading stocks from a provided CSV watchlist.
- Fetches relevant market information using the Bing API.
- Generates insightful LinkedIn posts using an LLM API (Ollama).
- Creates visually appealing tables of stock data for LinkedIn.

The goal is to automate data processing, content creation, and sharing in an engaging and professional way, driving more visibility and engagement with stock market insights.

## Features

- **Automated Stock Analysis**: Extracts key stock performance metrics and generates insightful snippets.
- **LinkedIn Post Generator**: Utilizes AI to create professional posts ready for LinkedIn, with emojis for enhanced engagement.
- **Customizable Table Images**: Generates tables of the top stock picks in a visually appealing manner suitable for social sharing.
- **Fully Configurable**: Easily customize settings via a configuration file.

## Table of Contents

- [Getting Started](#getting-started)
- [Setup Instructions](#setup-instructions)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Setup Environment Variables](#2-setup-environment-variables)
  - [3. Configure Project Settings](#3-configure-project-settings)
  - [4. Install Dependencies](#4-install-dependencies)
  - [5. Run the Project](#5-run-the-project)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

To get started, you will need Python installed on your local machine. This project uses several API keys and settings, which need to be configured using `.env` and `config.json` files.

Follow the instructions below to set up everything you need!

## Setup Instructions

### 1. Clone the Repository

Start by cloning the repository from GitHub and navigating to the directory:

```bash
$ git clone https://github.com/rocketpoweryul/DMTribeSocialPoster
$ cd stock-market-analysis
```

### 2. Setup Environment Variables

You will need a `.env` file to store your Bing API key and any other sensitive data. You can use the provided `.env.example` template:

```bash
$ cp .env.example .env
```

Then edit the `.env` file to add your Bing API key:

```dotenv
BING_API_KEY=YOUR_BING_API_KEY_HERE
```

### 3. Configure Project Settings

The project also uses a `config.json` file to customize its behavior. Use the provided `config.example.json` as a template:

```bash
$ cp config.example.json config.json
```

Edit the `config.json` file to match your specific paths and settings. Here’s an example:

```json
{
    "csv_path": "C:\\Path\\To\\Your\\DMT_WatchlistHits.csv",
    "output_path": "C:\\Path\\To\\Your\\Output\\Directory",
    "top_items_count": 3,
    "number_of_top_stocks": 10,
    "linkedin_api_endpoint": "http://localhost:11434/api/generate",
    "linkedin_model": "mistral-small",
    "table_image_width": 8,
    "table_image_height_per_row": 0.6,
    "linkedin_dpi": 300
}
```

### 4. Install Dependencies

Set up a conda environment and install all necessary dependencies:

```bash
$ conda create --name stock_analysis_env python=3.8
$ conda activate stock_analysis_env
$ conda install --file requirements.txt
```

This will create a new conda environment called `stock_analysis_env` and install the required packages specified in `requirements.txt`.

### 5. Run the Project

You are now ready to execute the script:

```bash
$ python main.py
```

This script will analyze stock data, generate LinkedIn content, and create a stock summary table image.

## Examples

### LinkedIn Post Example

Here is a sample of the kind of LinkedIn post generated by the script:

```
🌟 Key Highlights Across Leading Stocks 🌟

📊 $PLTR: Reported a 30% revenue increase in Q3, driven by U.S. government contracts and AI demand. The company raised its full-year revenue forecast.

🔧 $FIX: Continues to secure significant contracts in the booming construction sector, contributing to steady growth.

🚓 $AXON: Innovating in public safety technology with AI-enhanced tools, including body cameras.

After an incredible post-election bump, I hope you were long the US markets!

Check out the leaders below ⬇️

#investing #stocks #stockmarket #tech #AI #growth #stocks
```

### Stock Table Image Example

The script also generates a beautiful stock summary table, ideal for sharing on LinkedIn or other platforms:

![Stock Table Example](Top_Stocks_Table.png)

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

Please use `.env.example` and `config.example.json` when contributing, to avoid exposing sensitive information.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

---

Thank you for using and contributing to the Stock Market Analysis & LinkedIn Content Automation project! If you have any questions or suggestions, feel free to reach out or open an issue.

