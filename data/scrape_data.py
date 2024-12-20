import os
import requests
from bs4 import BeautifulSoup

# Function to scrape Changi Airport Services page
def scrape_changi_airport_services():
    url = 'https://www.changiairport.com/en/airport-services.html'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract relevant information
    services = soup.find_all('div', class_='service-description')
    service_text = "\n".join([service.get_text(strip=True) for service in services])

    # Save to a text file
    output_path = './changi_jewel_docs/changi_airport_services/airport_facilities.txt'
    with open(output_path, 'w') as file:
        file.write(service_text)

    print("Scraped Changi Airport Services and saved to 'airport_facilities.txt'")

# Function to scrape Jewel Shops information
def scrape_jewel_shops():
    url = 'https://www.jewelchangiairport.com/en/directory.html'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract relevant information (assuming there's a 'shop' class for shop listings)
    shops = soup.find_all('div', class_='shop')
    shop_text = "\n".join([shop.get_text(strip=True) for shop in shops])

    # Save to a text file
    output_path = './changi_jewel_docs/jewel_shops/shopping.txt'
    with open(output_path, 'w') as file:
        file.write(shop_text)

    print("Scraped Jewel Shops and saved to 'shopping.txt'")

# Main function to trigger scraping
def main():
    # Create necessary directories
    os.makedirs('./changi_jewel_docs/changi_airport_services', exist_ok=True)
    os.makedirs('./changi_jewel_docs/jewel_shops', exist_ok=True)

    # Scrape and save data
    scrape_changi_airport_services()
    scrape_jewel_shops()

if __name__ == "__main__":
    main()
