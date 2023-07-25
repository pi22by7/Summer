# Summer

This project demonstrates text summarization using the TextRank algorithm. Text summarization is the process of creating a shorter version (summary) of a longer piece of text while preserving its key information. The TextRank algorithm is a graph-based ranking algorithm commonly used for text summarization.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Text summarization is a challenging task in natural language processing (NLP) that finds applications in various domains, such as news aggregation, document summarization, and content extraction from web pages. This project aims to provide a simple and functional implementation of text summarization using the TextRank algorithm.

## Features

- Scrapes text from a given URL using BeautifulSoup.
- Preprocesses the text by converting it to lowercase, removing punctuation, and handling stopwords.
- Tokenizes the text into sentences and words using NLTK.
- Computes sentence vectors using GloVe word embeddings (placeholder with random vectors in the provided example).
- Generates a summary based on sentence ranking using the TextRank algorithm.
- Writes the raw text and summary to separate files.

## Installation

1. Clone the repository to your local machine:
```bash
git clone https://github.com/your_username/text-summarization.git
```
2. Navigate to the project directory:
```bash
cd text-summarization
```
3. Install the required dependencies:
```bash
pip install requests beautifulsoup4 nltk numpy networkx
```
## Usage

Just give it a URL containing the corpus which you want to summarize.

## Dependencies

The project relies on the following libraries:

- requests: For making HTTP requests to fetch the text from a URL.
- beautifulsoup4: For parsing the HTML content of the webpage.
- nltk: Natural Language Toolkit for text preprocessing and tokenization.
- numpy: For numerical computations.
- networkx: For creating and analyzing the similarity graph.

## Contributing

I would love it if you have something to contribute!
Just make a PR and I will help you.

## LICENSE

This project is licensed under the [MIT](https://choosealicense.com/licenses/mit/) License - see the [LICENSE](LICENSE) file for details.
