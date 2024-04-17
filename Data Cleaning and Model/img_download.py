import aiohttp
import aiofiles
import asyncio
import csv
import os
from tqdm import tqdm
from aiohttp.client_exceptions import ServerDisconnectedError


async def download_image(session, url, folder_name, filename):
    retries = 3  # Number of retries
    for _ in range(retries):
        try:
            async with session.get(url + "?tr=w-224,h-224") as response:
                if response.status == 200:
                    async with aiofiles.open(os.path.join(folder_name, filename), 'wb') as f:
                        await f.write(await response.read())
                        # Instead of print, use tqdm's update method to update progress bar
                        pbar.update(1)
                else:
                    print(f"Failed to download {filename}")
        except (ServerDisconnectedError, aiohttp.client_exceptions.ClientPayloadError):
            print("Server disconnected. Retrying...")
            await asyncio.sleep(5)  # Wait for a short duration before retrying
            continue
        except Exception as e:
            print(f"Random error. Retrying... {e}")
            await asyncio.sleep(5)  # Wait for a short duration before retrying
            continue
        else:
            break  # Break out of the loop if download is successful

async def main():
    tasks = []
    folder_column = "product_id"

    # Create the 'images' folder if it doesn't exist
    images_folder = "images"
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    # Read CSV file
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:  # Set timeout to None
        with open('processed_data/myntra_with_ratings.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in tqdm(reader, desc="Downloading Images"):
                folder_name = os.path.join(images_folder, row[folder_column].strip())
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)

                for img_column in ['img1', 'img2', 'img3', 'img4']:  # adjust if you have more image columns
                    img_url = row[img_column].strip()
                    if img_url:
                        tasks.append(download_image(session, img_url, folder_name, os.path.basename(img_url)))

        # tqdm progress bar initialization
        global pbar
        pbar = tqdm(total=len(tasks))
        await asyncio.gather(*tasks)
        pbar.close()  # Close the progress bar when all tasks are done

if __name__ == '__main__':
    asyncio.run(main())